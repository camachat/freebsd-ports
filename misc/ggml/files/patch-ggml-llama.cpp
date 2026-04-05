diff --git src/ggml-rpc/ggml-rpc.cpp src/ggml-rpc/ggml-rpc.cpp
index 1378ba9f..4e2f1ab0 100644
--- src/ggml-rpc/ggml-rpc.cpp
+++ src/ggml-rpc/ggml-rpc.cpp
@@ -1009,8 +1009,8 @@ public:
     bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
 
     struct stored_graph {
-        ggml_context_ptr ctx_ptr;
-        ggml_cgraph *    graph;
+        std::vector<uint8_t>   buffer;
+        ggml_cgraph          * graph;
     };
 
 private:
@@ -1518,10 +1518,12 @@ bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
     LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);
 
     size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
-
+    if (stored_graphs[device].buffer.size() < buf_size) {
+        stored_graphs[device].buffer.resize(buf_size);
+    }
     struct ggml_init_params params = {
         /*.mem_size   =*/ buf_size,
-        /*.mem_buffer =*/ NULL,
+        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
         /*.no_alloc   =*/ true,
     };
     ggml_context_ptr ctx_ptr { ggml_init(params) };
@@ -1551,7 +1553,6 @@ bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
     }
     ggml_status status = ggml_backend_graph_compute(backends[device], graph);
     GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
-    stored_graphs[device].ctx_ptr.swap(ctx_ptr);
     stored_graphs[device].graph = graph;
     return true;
 }
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index 456b1699..28be4939 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -569,9 +569,15 @@ static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
     SYCL_CHECK(
         CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));
 
-    SYCL_CHECK(CHECK_TRY_ERROR((*stream)
-                                    .memset(ctx->dev_ptr, value, buffer->size)
-                                    .wait()));
+    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
+    for (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
+        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
+        SYCL_CHECK(CHECK_TRY_ERROR(
+            (*stream)
+                .memset(static_cast<char*>(ctx->dev_ptr) + off, value, chunk)
+                .wait()
+        ));
+    }
 }
 catch (sycl::exception const &exc) {
   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
diff --git src/ggml-webgpu/ggml-webgpu-shader-lib.hpp src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
index a194ce84..669d2cd5 100644
--- src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
+++ src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
@@ -95,6 +95,12 @@ struct ggml_webgpu_generic_shader_decisions {
     uint32_t wg_size = 0;
 };
 
+struct ggml_webgpu_processed_shader {
+    std::string           wgsl;
+    std::string           variant;
+    std::shared_ptr<void> decisions;
+};
+
 struct ggml_webgpu_ssm_conv_shader_decisions {
     uint32_t block_size;
     uint32_t tokens_per_wg;
@@ -384,11 +390,12 @@ struct ggml_webgpu_flash_attn_pipeline_key {
     bool      has_mask;
     bool      has_sinks;
     bool      uses_logit_softcap;
+    bool      use_vec;
 
     bool operator==(const ggml_webgpu_flash_attn_pipeline_key & other) const {
         return kv_type == other.kv_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
                kv_direct == other.kv_direct && has_mask == other.has_mask && has_sinks == other.has_sinks &&
-               uses_logit_softcap == other.uses_logit_softcap;
+               uses_logit_softcap == other.uses_logit_softcap && use_vec == other.use_vec;
     }
 };
 
@@ -402,6 +409,7 @@ struct ggml_webgpu_flash_attn_pipeline_key_hash {
         ggml_webgpu_hash_combine(seed, key.has_mask);
         ggml_webgpu_hash_combine(seed, key.has_sinks);
         ggml_webgpu_hash_combine(seed, key.uses_logit_softcap);
+        ggml_webgpu_hash_combine(seed, key.use_vec);
         return seed;
     }
 };
@@ -421,6 +429,121 @@ struct ggml_webgpu_flash_attn_shader_decisions {
     uint32_t wg_size = 0;
 };
 
+inline uint32_t ggml_webgpu_flash_attn_pick_vec_ne(const ggml_webgpu_flash_attn_pipeline_key & key) {
+    // Keep conservative defaults unless this is the f16 vec-split shape family.
+    if (key.kv_type != GGML_TYPE_F16 || key.head_dim_qk != key.head_dim_v) {
+        return 1u;
+    }
+
+    // Head-dim specializations used by the tuned vec f16 path.
+    switch (key.head_dim_qk) {
+        case 64:
+            return 2u;
+        case 96:
+            return 4u;
+        case 128:
+            return 1u;
+        case 192:
+            return 2u;
+        case 576:
+            return 2u;
+        default:
+            return 1u;
+    }
+}
+
+struct ggml_webgpu_flash_attn_vec_reduce_pipeline_key {
+    uint32_t head_dim_v;
+    uint32_t wg_size;
+};
+
+struct ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash {
+    size_t operator()(const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & key) const {
+        size_t seed = 0;
+        ggml_webgpu_hash_combine(seed, key.head_dim_v);
+        ggml_webgpu_hash_combine(seed, key.wg_size);
+        return seed;
+    }
+};
+
+inline bool operator==(const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & lhs,
+                       const ggml_webgpu_flash_attn_vec_reduce_pipeline_key & rhs) {
+    return lhs.head_dim_v == rhs.head_dim_v && lhs.wg_size == rhs.wg_size;
+}
+
+struct ggml_webgpu_flash_attn_vec_reduce_shader_lib_context {
+    ggml_webgpu_flash_attn_vec_reduce_pipeline_key key;
+    uint32_t                                       max_wg_size;
+};
+
+inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_flash_attn_vec_reduce_shader(
+    pre_wgsl::Preprocessor &                                     preprocessor,
+    const char *                                                 shader_src,
+    const ggml_webgpu_flash_attn_vec_reduce_shader_lib_context & context) {
+    std::vector<std::string> defines;
+    std::string              variant = "flash_attn_vec_reduce";
+
+    defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(context.key.head_dim_v));
+    variant += std::string("_hsv") + std::to_string(context.key.head_dim_v);
+
+    defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
+    variant += std::string("_wg") + std::to_string(context.max_wg_size);
+
+    ggml_webgpu_processed_shader result;
+    result.wgsl    = preprocessor.preprocess(shader_src, defines);
+    result.variant = variant;
+    return result;
+}
+
+struct ggml_webgpu_flash_attn_blk_pipeline_key {
+    uint32_t q_tile;
+    uint32_t kv_tile;
+
+    bool operator==(const ggml_webgpu_flash_attn_blk_pipeline_key & other) const {
+        return q_tile == other.q_tile && kv_tile == other.kv_tile;
+    }
+};
+
+struct ggml_webgpu_flash_attn_blk_pipeline_key_hash {
+    size_t operator()(const ggml_webgpu_flash_attn_blk_pipeline_key & key) const {
+        size_t seed = 0;
+        ggml_webgpu_hash_combine(seed, key.q_tile);
+        ggml_webgpu_hash_combine(seed, key.kv_tile);
+        return seed;
+    }
+};
+
+struct ggml_webgpu_flash_attn_blk_shader_lib_context {
+    ggml_webgpu_flash_attn_blk_pipeline_key key;
+    uint32_t                                max_wg_size;
+};
+
+inline ggml_webgpu_processed_shader ggml_webgpu_preprocess_flash_attn_blk_shader(
+    pre_wgsl::Preprocessor &                              preprocessor,
+    const char *                                          shader_src,
+    const ggml_webgpu_flash_attn_blk_shader_lib_context & context) {
+    std::vector<std::string> defines;
+    std::string              variant = "flash_attn_vec_blk";
+
+    defines.push_back(std::string("Q_TILE=") + std::to_string(context.key.q_tile));
+    variant += std::string("_qt") + std::to_string(context.key.q_tile);
+
+    defines.push_back(std::string("KV_TILE=") + std::to_string(context.key.kv_tile));
+    variant += std::string("_kvt") + std::to_string(context.key.kv_tile);
+
+    uint32_t wg_size = 1;
+    while ((wg_size << 1) <= context.max_wg_size) {
+        wg_size <<= 1;
+    }
+    defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
+    variant += std::string("_wg") + std::to_string(wg_size);
+
+    ggml_webgpu_processed_shader result;
+    result.wgsl    = preprocessor.preprocess(shader_src, defines);
+    result.variant = variant;
+    return result;
+}
+
 // This is exposed because it's necessary in supports_op
 inline size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
                                                   uint32_t kv_tile,
@@ -659,6 +782,14 @@ class ggml_webgpu_shader_lib {
         repeat_pipelines;           // type
     std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
         flash_attn_pipelines;
+    std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_key,
+                       webgpu_pipeline,
+                       ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash>
+        flash_attn_vec_reduce_pipelines;
+    std::unordered_map<ggml_webgpu_flash_attn_blk_pipeline_key,
+                       webgpu_pipeline,
+                       ggml_webgpu_flash_attn_blk_pipeline_key_hash>
+        flash_attn_blk_pipelines;
     std::unordered_map<ggml_webgpu_legacy_mul_mat_pipeline_key,
                        webgpu_pipeline,
                        ggml_webgpu_legacy_mul_mat_pipeline_key_hash>
@@ -1673,24 +1804,8 @@ class ggml_webgpu_shader_lib {
         return repeat_pipelines[key];
     }
 
-    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_shader_lib_context & context) {
-        const bool has_mask  = context.src3 != nullptr;
-        const bool has_sinks = context.src4 != nullptr;
-
-        bool kv_direct = (context.src1->type == GGML_TYPE_F16) && (context.src0->ne[0] % context.sg_mat_k == 0) &&
-                         (context.src1->ne[1] % context.sg_mat_n == 0);
-
-        ggml_webgpu_flash_attn_pipeline_key key = {
-            .kv_type            = context.src1->type,
-            .head_dim_qk        = (uint32_t) context.src0->ne[0],
-            .head_dim_v         = (uint32_t) context.src2->ne[0],
-            .kv_direct          = kv_direct,
-            .has_mask           = has_mask,
-            .has_sinks          = has_sinks,
-            .uses_logit_softcap = (*(float *) &context.dst->op_params[2]) != 0.0f,
-        };
-
-        auto it = flash_attn_pipelines.find(key);
+    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_flash_attn_shader_lib_context & context) {
+        auto it = flash_attn_pipelines.find(context.key);
         if (it != flash_attn_pipelines.end()) {
             return it->second;
         }
@@ -1698,7 +1813,7 @@ class ggml_webgpu_shader_lib {
         std::vector<std::string> defines;
         std::string              variant = "flash_attn";
 
-        switch (key.kv_type) {
+        switch (context.key.kv_type) {
             case GGML_TYPE_F32:
                 defines.push_back("KV_F32");
                 break;
@@ -1714,41 +1829,51 @@ class ggml_webgpu_shader_lib {
             default:
                 GGML_ABORT("Unsupported KV type for flash attention shader");
         }
-        variant += std::string("_") + ggml_type_name(key.kv_type);
+        variant += std::string("_") + ggml_type_name(context.key.kv_type);
 
-        if (key.has_mask) {
+        if (context.key.has_mask) {
             defines.push_back("MASK");
             variant += "_mask";
         }
-        if (key.has_sinks) {
+        if (context.key.has_sinks) {
             defines.push_back("SINKS");
             variant += "_sinks";
         }
-        if (key.uses_logit_softcap) {
+        if (context.key.uses_logit_softcap) {
             defines.push_back("LOGIT_SOFTCAP");
             variant += "_lgsc";
         }
-        if (key.kv_direct) {
+        if (context.key.kv_direct) {
             defines.push_back("KV_DIRECT");
             variant += "_kvdirect";
         }
+        if (context.key.has_mask && context.key.use_vec) {
+            defines.push_back("BLK");
+            variant += "_blk";
+        }
 
-        defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(key.head_dim_qk));
-        variant += std::string("_hsqk") + std::to_string(key.head_dim_qk);
+        defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(context.key.head_dim_qk));
+        variant += std::string("_hsqk") + std::to_string(context.key.head_dim_qk);
 
-        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
-        variant += std::string("_hsv") + std::to_string(key.head_dim_v);
+        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(context.key.head_dim_v));
+        variant += std::string("_hsv") + std::to_string(context.key.head_dim_v);
 
         defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
         defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
         defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
 
-        uint32_t q_tile = context.sg_mat_m;
-        uint32_t kv_tile =
-            std::min(ggml_webgpu_flash_attn_max_kv_tile({ key, context.sg_mat_m, context.sg_mat_n, context.sg_mat_k,
-                                                          context.wg_mem_limit_bytes, context.max_subgroup_size }),
-                     context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
-        if (key.kv_direct) {
+        uint32_t q_tile  = context.sg_mat_m;
+        uint32_t kv_tile = std::min(ggml_webgpu_flash_attn_max_kv_tile(context),
+                                    context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
+        if (context.key.use_vec) {
+            q_tile  = 1;
+            kv_tile = std::max(context.sg_mat_n, std::min(32u, ggml_webgpu_flash_attn_max_kv_tile(context)));
+            kv_tile = (kv_tile / context.sg_mat_n) * context.sg_mat_n;
+            const uint32_t vec_ne = ggml_webgpu_flash_attn_pick_vec_ne(context.key);
+            defines.push_back(std::string("VEC_NE=") + std::to_string(vec_ne) + "u");
+        }
+        if (context.key.kv_direct) {
+            GGML_ASSERT(kv_tile <= GGML_WEBGPU_KV_SEQ_PAD);
             while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
                 kv_tile -= context.sg_mat_n;
             }
@@ -1757,19 +1882,51 @@ class ggml_webgpu_shader_lib {
         defines.push_back(std::string("Q_TILE=") + std::to_string(q_tile));
         defines.push_back(std::string("KV_TILE=") + std::to_string(kv_tile));
 
-        uint32_t wg_size = std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);
+        uint32_t wg_size = 0;
+        if (context.key.use_vec) {
+            wg_size = std::max(1u, std::min<uint32_t>(32u, context.max_subgroup_size));
+        } else {
+            wg_size = std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);
+        }
         defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
 
-        auto processed     = preprocessor.preprocess(wgsl_flash_attn, defines);
-        auto decisions     = std::make_shared<ggml_webgpu_flash_attn_shader_decisions>();
-        decisions->q_tile  = q_tile;
-        decisions->kv_tile = kv_tile;
-        decisions->wg_size = wg_size;
+        const char *    shader_src = context.key.use_vec ? wgsl_flash_attn_vec_split : wgsl_flash_attn;
+        webgpu_pipeline pipeline =
+            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
+        auto decisions                    = std::make_shared<ggml_webgpu_flash_attn_shader_decisions>();
+        decisions->q_tile                 = q_tile;
+        decisions->kv_tile                = kv_tile;
+        decisions->wg_size                = wg_size;
+        pipeline.context                  = decisions;
+        flash_attn_pipelines[context.key] = pipeline;
+        return flash_attn_pipelines[context.key];
+    }
+
+    webgpu_pipeline get_flash_attn_blk_pipeline(const ggml_webgpu_flash_attn_blk_shader_lib_context & context) {
+        auto it = flash_attn_blk_pipelines.find(context.key);
+        if (it != flash_attn_blk_pipelines.end()) {
+            return it->second;
+        }
+
+        ggml_webgpu_processed_shader processed =
+            ggml_webgpu_preprocess_flash_attn_blk_shader(preprocessor, wgsl_flash_attn_vec_blk, context);
+        webgpu_pipeline pipeline              = ggml_webgpu_create_pipeline(device, processed.wgsl, processed.variant);
+        flash_attn_blk_pipelines[context.key] = pipeline;
+        return flash_attn_blk_pipelines[context.key];
+    }
+
+    webgpu_pipeline get_flash_attn_vec_reduce_pipeline(
+        const ggml_webgpu_flash_attn_vec_reduce_shader_lib_context & context) {
+        auto it = flash_attn_vec_reduce_pipelines.find(context.key);
+        if (it != flash_attn_vec_reduce_pipelines.end()) {
+            return it->second;
+        }
 
-        webgpu_pipeline pipeline  = ggml_webgpu_create_pipeline(device, processed, variant);
-        pipeline.context          = decisions;
-        flash_attn_pipelines[key] = pipeline;
-        return flash_attn_pipelines[key];
+        ggml_webgpu_processed_shader processed =
+            ggml_webgpu_preprocess_flash_attn_vec_reduce_shader(preprocessor, wgsl_flash_attn_vec_reduce, context);
+        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed.wgsl, processed.variant);
+        flash_attn_vec_reduce_pipelines[context.key] = pipeline;
+        return flash_attn_vec_reduce_pipelines[context.key];
     }
 
     webgpu_pipeline get_cpy_pipeline(const ggml_webgpu_shader_lib_context & context) {
diff --git src/ggml-webgpu/ggml-webgpu.cpp src/ggml-webgpu/ggml-webgpu.cpp
index 1aa15b05..5c567dc0 100644
--- src/ggml-webgpu/ggml-webgpu.cpp
+++ src/ggml-webgpu/ggml-webgpu.cpp
@@ -81,12 +81,10 @@ static inline void compute_2d_workgroups(uint32_t total_wg, uint32_t max_per_dim
 
 /* Constants */
 
-#define WEBGPU_NUM_PARAM_BUFS                96u
-#define WEBGPU_COMMAND_SUBMIT_BATCH_SIZE     32u
+#define WEBGPU_COMMAND_SUBMIT_BATCH_SIZE 32u
+#define WEBGPU_NUM_PARAM_SLOTS \
+    (WEBGPU_COMMAND_SUBMIT_BATCH_SIZE + 10)  // a few extra for safety, since some operations may need multiple slots
 #define WEBGPU_WAIT_ANY_TIMEOUT_MS           100
-// Maximum number of in-flight submissions per-thread, to avoid exhausting the
-// parameter buffer pool
-#define WEBGPU_MAX_INFLIGHT_SUBS_PER_THREAD  (WEBGPU_NUM_PARAM_BUFS / WEBGPU_COMMAND_SUBMIT_BATCH_SIZE)
 #define WEBGPU_PARAMS_BUF_SIZE_BYTES         128  // enough for 32 parameters
 #define WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES 4
 #define WEBGPU_STORAGE_BUF_BINDING_MULT      4    // a storage buffer binding size must be a multiple of 4
@@ -122,87 +120,45 @@ static void ggml_webgpu_create_buffer(wgpu::Device &    device,
                                       wgpu::BufferUsage usage,
                                       const char *      label);
 
-// Holds a pool of parameter buffers for WebGPU operations
-struct webgpu_buf_pool {
-    std::vector<wgpu::Buffer> free;
-
-    // The pool must be synchronized because
-    // 1. The memset pool is shared globally by every ggml buffer,
-    // since allocating a pool per ggml buffer would consume too much memory.
-    // 2. For the per-thread buffer pools in webgpu_context,
-    // buffers are allocated and freed in Dawn callbacks,
-    // which can run on a different thread than the calling thread.
-    std::mutex              mutex;
-    std::condition_variable cv;
-    size_t                  cur_pool_size;
-    size_t                  max_pool_size;
-    wgpu::Device            device;
-    wgpu::BufferUsage       dev_buf_usage;
-    size_t                  buf_size;
-    bool                    should_grow;
-
-    void init(wgpu::Device      device,
-              int               num_bufs,
-              size_t            buf_size,
-              wgpu::BufferUsage dev_buf_usage,
-              bool              should_grow   = false,
-              size_t            max_pool_size = WEBGPU_NUM_PARAM_BUFS * 2) {
-        this->max_pool_size = max_pool_size;
-        this->cur_pool_size = num_bufs;
-        this->device        = device;
-        this->dev_buf_usage = dev_buf_usage;
-        this->buf_size      = buf_size;
-        this->should_grow   = should_grow;
-        for (int i = 0; i < num_bufs; i++) {
-            wgpu::Buffer dev_buf;
-            ggml_webgpu_create_buffer(device, dev_buf, buf_size, dev_buf_usage, "ggml_webgpu_dev_pool_buf");
-            free.push_back(dev_buf);
+// Slot-based parameter arena for compute graph encoding. Each encoded kernel
+// gets a unique uniform-buffer slice within the current batch, and the slot
+// cursor is reset immediately after that batch is submitted.
+struct webgpu_param_arena {
+    wgpu::Buffer buffer;
+    size_t       slot_stride = 0;
+    size_t       slot_size   = 0;
+    uint32_t     slot_count  = 0;
+    uint32_t     next_slot   = 0;
+
+    void init(wgpu::Device device, size_t slot_size, uint32_t slot_count, size_t alignment) {
+        this->slot_stride = ROUNDUP_POW2(slot_size, alignment);
+        this->slot_size   = slot_size;
+        this->slot_count  = slot_count;
+        this->next_slot   = 0;
+
+        ggml_webgpu_create_buffer(device, buffer, this->slot_stride * slot_count,
+                                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform, "ggml_webgpu_param_arena");
+    }
+
+    size_t alloc_slot(size_t size) {
+        GGML_ASSERT(size <= slot_size);
+        if (next_slot >= slot_count) {
+            GGML_ABORT("ggml_webgpu: parameter arena exhausted while encoding a batch");
         }
-    }
 
-    wgpu::Buffer alloc_bufs() {
-        std::unique_lock<std::mutex> lock(mutex);
-        if (!free.empty()) {
-            wgpu::Buffer buf = free.back();
-            free.pop_back();
-            return buf;
-        }
-
-        // Try growing the pool if no free buffers
-        if (free.empty() && cur_pool_size < max_pool_size && should_grow) {
-            cur_pool_size++;
-            lock.unlock();  // avoid deadlock between this lock and Dawn's internal locks when buffers are freed in callbacks
-            wgpu::Buffer dev_buf;
-            ggml_webgpu_create_buffer(device, dev_buf, buf_size, dev_buf_usage, "ggml_webgpu_dev_pool_buf");
-
-            if (!dev_buf) {
-                GGML_ABORT("webgpu_buf_pool: failed to allocate buffers");
-            }
-            return dev_buf;
-        }
-        cv.wait(lock, [this] { return !free.empty(); });
-        wgpu::Buffer buf = free.back();
-        free.pop_back();
-        return buf;
+        return slot_stride * next_slot++;
     }
 
-    void free_bufs(std::vector<wgpu::Buffer> bufs) {
-        std::lock_guard<std::mutex> lock(mutex);
-        free.insert(free.end(), bufs.begin(), bufs.end());
-        cv.notify_all();
-    }
+    void reset() { next_slot = 0; }
 
     void cleanup() {
-        std::lock_guard<std::mutex> lock(mutex);
-        for (auto & buf : free) {
-            if (buf) {
-                buf.Destroy();
-            }
+        if (buffer) {
+            buffer.Destroy();
+            buffer = nullptr;
         }
-        free.clear();
     }
 
-    ~webgpu_buf_pool() { this->cleanup(); }
+    ~webgpu_param_arena() { this->cleanup(); }
 };
 
 #ifdef GGML_WEBGPU_GPU_PROFILE
@@ -269,10 +225,8 @@ struct webgpu_gpu_profile_buf_pool {
 };
 #endif
 
-struct webgpu_command {
-    uint32_t                  num_kernels;
-    wgpu::CommandBuffer       commands;
-    std::vector<wgpu::Buffer> params_bufs;
+struct webgpu_encoded_op {
+    uint32_t num_kernels = 0;
 #ifdef GGML_WEBGPU_GPU_PROFILE
     webgpu_gpu_profile_bufs timestamp_query_bufs;
     std::string             pipeline_name;
@@ -305,8 +259,8 @@ struct webgpu_global_context_struct {
     // Global mutex for pipeline and staging buffer, will be refactored to exclude pipeline caches.
     std::recursive_mutex mutex;
 
-    webgpu_buf_pool                memset_buf_pool;
-    std::map<int, webgpu_pipeline> memset_pipelines;  // variant or type index
+    wgpu::Buffer    memset_params_buf;
+    webgpu_pipeline memset_pipeline;
 
 #ifdef GGML_WEBGPU_CPU_PROFILE
     // Profiling: labeled CPU time in ms (total)
@@ -332,6 +286,10 @@ struct webgpu_global_context_struct {
             this->get_tensor_staging_buf.Destroy();
             this->get_tensor_staging_buf = nullptr;
         }
+        if (this->memset_params_buf) {
+            this->memset_params_buf.Destroy();
+            this->memset_params_buf = nullptr;
+        }
 #ifdef GGML_WEBGPU_DEBUG
         if (this->debug_host_buf) {
             this->debug_host_buf.Destroy();
@@ -347,13 +305,6 @@ struct webgpu_global_context_struct {
 
 typedef std::shared_ptr<webgpu_global_context_struct> webgpu_global_context;
 
-struct webgpu_submission {
-    wgpu::FutureWaitInfo submit_done;
-#ifdef GGML_WEBGPU_GPU_PROFILE
-    std::vector<wgpu::FutureWaitInfo> profile_futures;
-#endif
-};
-
 // All the base objects needed to run operations on a WebGPU device
 struct webgpu_context_struct {
     // Points to global instances owned by ggml_backend_webgpu_reg_context
@@ -361,9 +312,9 @@ struct webgpu_context_struct {
 
     std::unique_ptr<ggml_webgpu_shader_lib> shader_lib;
 
-    webgpu_buf_pool param_buf_pool;
-    wgpu::Buffer    set_rows_dev_error_buf;
-    wgpu::Buffer    set_rows_host_error_buf;
+    webgpu_param_arena param_arena;
+    wgpu::Buffer       set_rows_dev_error_buf;
+    wgpu::Buffer       set_rows_host_error_buf;
 
     size_t memset_bytes_per_thread;
 };
@@ -448,95 +399,34 @@ static void ggml_webgpu_create_buffer(wgpu::Device &    device,
 
 /** WebGPU Actions */
 
-static bool ggml_backend_webgpu_handle_wait_status(wgpu::WaitStatus status, bool allow_timeout = false) {
-    switch (status) {
-        case wgpu::WaitStatus::Success:
-            return true;
-        case wgpu::WaitStatus::TimedOut:
-            if (allow_timeout) {
-                return false;
-            }
-            GGML_LOG_ERROR("ggml_webgpu: WaitAny timed out unexpectedly\n");
-            return false;
-        case wgpu::WaitStatus::Error:
-            GGML_LOG_ERROR("ggml_webgpu: WaitAny returned an error\n");
-            return false;
-        default:
-            GGML_LOG_ERROR("ggml_webgpu: WaitAny returned an unknown status\n");
-            return false;
-    }
-}
-
 #ifdef GGML_WEBGPU_GPU_PROFILE
-static void ggml_backend_webgpu_erase_completed_futures(std::vector<wgpu::FutureWaitInfo> & futures) {
-    futures.erase(std::remove_if(futures.begin(), futures.end(),
-                                 [](const wgpu::FutureWaitInfo & info) { return info.completed; }),
-                  futures.end());
-}
-
 static void ggml_backend_webgpu_wait_profile_futures(webgpu_global_context &             ctx,
-                                                     std::vector<wgpu::FutureWaitInfo> & futures,
-                                                     bool                                block) {
+                                                     std::vector<wgpu::FutureWaitInfo> & futures) {
     if (futures.empty()) {
         return;
     }
 
-    uint64_t timeout_ms = block ? UINT64_MAX : 0;
-    if (block) {
-        while (!futures.empty()) {
-            auto waitStatus = ctx->instance.WaitAny(futures.size(), futures.data(), timeout_ms);
-            if (ggml_backend_webgpu_handle_wait_status(waitStatus)) {
-                ggml_backend_webgpu_erase_completed_futures(futures);
-            }
-        }
-    } else {
-        auto waitStatus = ctx->instance.WaitAny(futures.size(), futures.data(), timeout_ms);
-        if (ggml_backend_webgpu_handle_wait_status(waitStatus, true)) {
-            ggml_backend_webgpu_erase_completed_futures(futures);
-        }
-    }
-}
-#endif
+    constexpr size_t max_futures_per_wait = 64;
 
-// Wait for the queue to finish processing all submitted work
-static void ggml_backend_webgpu_wait(webgpu_global_context &          ctx,
-                                     std::vector<webgpu_submission> & subs,
-                                     bool                             block = true) {
-    if (subs.empty()) {
-        return;
+    while (!futures.empty()) {
+        ctx->instance.WaitAny(std::min(max_futures_per_wait, futures.size()), futures.data(), UINT64_MAX);
+        futures.erase(std::remove_if(futures.begin(), futures.end(),
+                                     [](const wgpu::FutureWaitInfo & info) { return info.completed; }),
+                      futures.end());
     }
-
-    bool blocking_wait = block || subs.size() >= WEBGPU_MAX_INFLIGHT_SUBS_PER_THREAD;
-    while (blocking_wait) {
-        auto waitStatus = ctx->instance.WaitAny(1, &subs[0].submit_done, WEBGPU_WAIT_ANY_TIMEOUT_MS * 1e6);
-        if (ggml_backend_webgpu_handle_wait_status(waitStatus, true)) {
-#ifdef GGML_WEBGPU_GPU_PROFILE
-            ggml_backend_webgpu_wait_profile_futures(ctx, subs[0].profile_futures, true);
+}
 #endif
-            subs.erase(subs.begin());
-        }
-        blocking_wait = (block && !subs.empty()) || subs.size() >= WEBGPU_MAX_INFLIGHT_SUBS_PER_THREAD;
-    }
 
-    if (subs.empty()) {
-        return;
-    }
-
-    // Poll each submit future once and remove completed submissions.
-    for (auto sub = subs.begin(); sub != subs.end();) {
-        auto waitStatus = ctx->instance.WaitAny(1, &sub->submit_done, 0);
-        bool success    = ggml_backend_webgpu_handle_wait_status(waitStatus, true);
-#ifdef GGML_WEBGPU_GPU_PROFILE
-        ggml_backend_webgpu_wait_profile_futures(ctx, sub->profile_futures, false);
-        if (success && sub->profile_futures.empty()) {
-#else
-        if (success) {
-#endif
-            sub = subs.erase(sub);
-        } else {
-            ++sub;
-        }
-    }
+static void ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
+    ctx->instance.WaitAny(
+        ctx->queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
+                                       [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
+                                           if (status != wgpu::QueueWorkDoneStatus::Success) {
+                                               GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n",
+                                                              std::string(message).c_str());
+                                           }
+                                       }),
+        UINT64_MAX);
 }
 
 static void ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
@@ -570,34 +460,10 @@ static void ggml_backend_webgpu_debug(webgpu_global_context & ctx) {
 }
 #endif
 
-static webgpu_submission ggml_backend_webgpu_submit(webgpu_global_context &       ctx,
-                                                    std::vector<webgpu_command> & commands,
-                                                    webgpu_buf_pool &             param_buf_pool) {
-    std::vector<wgpu::CommandBuffer> command_buffers;
-    std::vector<wgpu::Buffer>        params_bufs;
-    webgpu_submission                submission;
-#ifdef GGML_WEBGPU_GPU_PROFILE
-    std::vector<std::pair<std::string, webgpu_gpu_profile_bufs>> pipeline_name_and_ts_bufs;
-#endif
-
-    for (const auto & command : commands) {
-        command_buffers.push_back(command.commands);
-        params_bufs.insert(params_bufs.end(), command.params_bufs.begin(), command.params_bufs.end());
-    }
-    ctx->queue.Submit(command_buffers.size(), command_buffers.data());
-
-    wgpu::Future p_f = ctx->queue.OnSubmittedWorkDone(
-        wgpu::CallbackMode::AllowSpontaneous,
-        [&param_buf_pool, params_bufs](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
-            if (status != wgpu::QueueWorkDoneStatus::Success) {
-                GGML_LOG_ERROR("ggml_webgpu: Failed to submit commands: %s\n", std::string(message).c_str());
-            }
-            // Free the staged buffers
-            param_buf_pool.free_bufs(params_bufs);
-        });
-    submission.submit_done = { p_f };
-
 #ifdef GGML_WEBGPU_GPU_PROFILE
+static void ggml_backend_webgpu_collect_profile_futures(webgpu_global_context &             ctx,
+                                                        const std::vector<webgpu_command> & commands,
+                                                        std::vector<wgpu::FutureWaitInfo> & futures) {
     for (const auto & command : commands) {
         auto label   = command.pipeline_name;
         auto ts_bufs = command.timestamp_query_bufs;
@@ -616,15 +482,15 @@ static webgpu_submission ggml_backend_webgpu_submit(webgpu_global_context &
                 // We can't unmap in here due to WebGPU reentrancy limitations.
                 ctx->timestamp_query_buf_pool.free_bufs({ ts_bufs });
             });
-        submission.profile_futures.push_back({ f });
+        futures.push_back({ f });
     }
-#endif
-    return submission;
 }
+#endif
 
-static webgpu_command ggml_backend_webgpu_build_multi(
+static webgpu_encoded_op ggml_backend_webgpu_build_multi(
     webgpu_global_context &                                ctx,
-    webgpu_buf_pool &                                      param_buf_pool,
+    webgpu_param_arena &                                   param_arena,
+    wgpu::CommandEncoder &                                 encoder,
     const std::vector<webgpu_pipeline> &                   pipelines,
     const std::vector<std::vector<uint32_t>> &             params_list,
     const std::vector<std::vector<wgpu::BindGroupEntry>> & bind_group_entries_list,
@@ -633,16 +499,21 @@ static webgpu_command ggml_backend_webgpu_build_multi(
     GGML_ASSERT(pipelines.size() == bind_group_entries_list.size());
     GGML_ASSERT(pipelines.size() == workgroups_list.size());
 
-    std::vector<wgpu::Buffer>    params_bufs_list;
+    webgpu_encoded_op            result = {};
     std::vector<wgpu::BindGroup> bind_groups;
+    std::vector<size_t>          param_offsets;
+    result.num_kernels = pipelines.size();
 
     for (size_t i = 0; i < pipelines.size(); i++) {
-        wgpu::Buffer params_bufs = param_buf_pool.alloc_bufs();
+        const size_t param_size   = params_list[i].size() * sizeof(uint32_t);
+        const size_t param_offset = param_arena.alloc_slot(param_size);
 
         std::vector<wgpu::BindGroupEntry> entries            = bind_group_entries_list[i];
         uint32_t                          params_binding_num = entries.size();
-        entries.push_back(
-            { .binding = params_binding_num, .buffer = params_bufs, .offset = 0, .size = params_bufs.GetSize() });
+        entries.push_back({ .binding = params_binding_num,
+                            .buffer  = param_arena.buffer,
+                            .offset  = param_offset,
+                            .size    = param_arena.slot_size });
 
         wgpu::BindGroupDescriptor bind_group_desc;
         bind_group_desc.layout     = pipelines[i].pipeline.GetBindGroupLayout(0);
@@ -650,15 +521,13 @@ static webgpu_command ggml_backend_webgpu_build_multi(
         bind_group_desc.entries    = entries.data();
         bind_group_desc.label      = pipelines[i].name.c_str();
         bind_groups.push_back(ctx->device.CreateBindGroup(&bind_group_desc));
-
-        params_bufs_list.push_back(params_bufs);
+        param_offsets.push_back(param_offset);
     }
 
-    wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
-    for (size_t i = 0; i < params_bufs_list.size(); i++) {
-        ctx->queue.WriteBuffer(params_bufs_list[i], 0, params_list[i].data(), params_list[i].size() * sizeof(uint32_t));
+    for (size_t i = 0; i < param_offsets.size(); i++) {
+        ctx->queue.WriteBuffer(param_arena.buffer, param_offsets[i], params_list[i].data(),
+                               params_list[i].size() * sizeof(uint32_t));
     }
-
 #ifdef GGML_WEBGPU_GPU_PROFILE
     webgpu_gpu_profile_bufs ts_bufs = ctx->timestamp_query_buf_pool.alloc_bufs();
     if (ts_bufs.host_buf.GetMapState() == wgpu::BufferMapState::Mapped) {
@@ -683,29 +552,21 @@ static webgpu_command ggml_backend_webgpu_build_multi(
 #ifdef GGML_WEBGPU_GPU_PROFILE
     encoder.ResolveQuerySet(ts_bufs.query_set, 0, 2, ts_bufs.dev_buf, 0);
     encoder.CopyBufferToBuffer(ts_bufs.dev_buf, 0, ts_bufs.host_buf, 0, ts_bufs.host_buf.GetSize());
-#endif
-
-    wgpu::CommandBuffer commands = encoder.Finish();
-    webgpu_command      result   = {};
-    result.commands              = commands;
-    result.params_bufs           = params_bufs_list;
-    result.num_kernels           = pipelines.size();
-#ifdef GGML_WEBGPU_GPU_PROFILE
     result.timestamp_query_bufs = ts_bufs;
-    // TODO: handle multiple pipeline names
     result.pipeline_name        = pipelines.front().name;
 #endif
     return result;
 }
 
-static webgpu_command ggml_backend_webgpu_build(webgpu_global_context &           ctx,
-                                                webgpu_buf_pool &                 param_buf_pool,
-                                                webgpu_pipeline &                 pipeline,
-                                                std::vector<uint32_t>             params,
-                                                std::vector<wgpu::BindGroupEntry> bind_group_entries,
-                                                uint32_t                          wg_x,
-                                                uint32_t                          wg_y = 1) {
-    return ggml_backend_webgpu_build_multi(ctx, param_buf_pool,
+static webgpu_encoded_op ggml_backend_webgpu_build(webgpu_global_context &           ctx,
+                                                   webgpu_param_arena &              param_arena,
+                                                   wgpu::CommandEncoder &            encoder,
+                                                   webgpu_pipeline &                 pipeline,
+                                                   std::vector<uint32_t>             params,
+                                                   std::vector<wgpu::BindGroupEntry> bind_group_entries,
+                                                   uint32_t                          wg_x,
+                                                   uint32_t                          wg_y = 1) {
+    return ggml_backend_webgpu_build_multi(ctx, param_arena, encoder,
                                            {
                                                pipeline
     },
@@ -725,10 +586,28 @@ static void ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
     size_t   bytes_per_wg = WEBGPU_MAX_WG_SIZE * ctx->capabilities.memset_bytes_per_thread;
     uint32_t wg_x         = CEIL_DIV(size + 3, bytes_per_wg);
 
-    webgpu_command command =
-        ggml_backend_webgpu_build(ctx, ctx->memset_buf_pool, ctx->memset_pipelines[0], params, entries, wg_x);
-    std::vector<webgpu_command>    commands = { command };
-    std::vector<webgpu_submission> sub      = { ggml_backend_webgpu_submit(ctx, commands, ctx->memset_buf_pool) };
+    ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * sizeof(uint32_t));
+
+    entries.push_back(
+        { .binding = 1, .buffer = ctx->memset_params_buf, .offset = 0, .size = WEBGPU_PARAMS_BUF_SIZE_BYTES });
+
+    wgpu::BindGroupDescriptor bind_group_desc;
+    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
+    bind_group_desc.entryCount = entries.size();
+    bind_group_desc.entries    = entries.data();
+    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
+    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);
+
+    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
+    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
+    pass.SetPipeline(ctx->memset_pipeline.pipeline);
+    pass.SetBindGroup(0, bind_group);
+    pass.DispatchWorkgroups(wg_x, 1, 1);
+    pass.End();
+
+    wgpu::CommandBuffer              command  = encoder.Finish();
+    std::vector<wgpu::CommandBuffer> commands = { command };
+    ctx->queue.Submit(commands.size(), commands.data());
 }
 
 /** End WebGPU Actions */
@@ -841,7 +720,10 @@ static binary_overlap_flags ggml_webgpu_detect_binary_overlap(ggml_tensor * src0
     return flags;
 }
 
-static webgpu_command ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_cpy(webgpu_context &       ctx,
+                                         wgpu::CommandEncoder & encoder,
+                                         ggml_tensor *          src,
+                                         ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0        = src,
         .dst         = dst,
@@ -879,10 +761,14 @@ static webgpu_command ggml_webgpu_cpy(webgpu_context & ctx, ggml_tensor * src, g
     };
 
     uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_set(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_set(webgpu_context &       ctx,
+                                         wgpu::CommandEncoder & encoder,
+                                         ggml_tensor *          src0,
+                                         ggml_tensor *          src1,
+                                         ggml_tensor *          dst) {
     const bool inplace = ggml_webgpu_tensor_equal(src0, dst);
 
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
@@ -941,10 +827,13 @@ static webgpu_command ggml_webgpu_set(webgpu_context & ctx, ggml_tensor * src0,
                         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
 
     uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_pad(webgpu_context &       ctx,
+                                         wgpu::CommandEncoder & encoder,
+                                         ggml_tensor *          src,
+                                         ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0 = src, .dst = dst, .max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup
     };
@@ -996,13 +885,14 @@ static webgpu_command ggml_webgpu_pad(webgpu_context & ctx, ggml_tensor * src, g
     };
 
     uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_solve_tri(webgpu_context & ctx,
-                                            ggml_tensor *    src0,
-                                            ggml_tensor *    src1,
-                                            ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context &       ctx,
+                                               wgpu::CommandEncoder & encoder,
+                                               ggml_tensor *          src0,
+                                               ggml_tensor *          src1,
+                                               ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0               = src0,
         .src1               = src1,
@@ -1057,13 +947,14 @@ static webgpu_command ggml_webgpu_solve_tri(webgpu_context & ctx,
 
     const uint32_t wg_x = CEIL_DIV((uint32_t) src1->ne[0], decisions->wg_size);
     const uint32_t wg_y = (uint32_t) (dst->ne[2] * dst->ne[3]);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x, wg_y);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x, wg_y);
 }
 
-static webgpu_command ggml_webgpu_ssm_conv(webgpu_context & ctx,
-                                           ggml_tensor *    src0,
-                                           ggml_tensor *    src1,
-                                           ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context &       ctx,
+                                              wgpu::CommandEncoder & encoder,
+                                              ggml_tensor *          src0,
+                                              ggml_tensor *          src1,
+                                              ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0        = src0,
         .src1        = src1,
@@ -1113,17 +1004,18 @@ static webgpu_command ggml_webgpu_ssm_conv(webgpu_context & ctx,
 
     const uint32_t wg_x = CEIL_DIV((uint32_t) src0->ne[1], decisions->block_size);
     const uint32_t wg_y = token_tiles * (uint32_t) dst->ne[2];
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x, wg_y);
-}
-
-static webgpu_command ggml_webgpu_gated_delta_net(webgpu_context & ctx,
-                                                  ggml_tensor *    src0,
-                                                  ggml_tensor *    src1,
-                                                  ggml_tensor *    src2,
-                                                  ggml_tensor *    src3,
-                                                  ggml_tensor *    src4,
-                                                  ggml_tensor *    src5,
-                                                  ggml_tensor *    dst) {
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x, wg_y);
+}
+
+static webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context &       ctx,
+                                                     wgpu::CommandEncoder & encoder,
+                                                     ggml_tensor *          src0,
+                                                     ggml_tensor *          src1,
+                                                     ggml_tensor *          src2,
+                                                     ggml_tensor *          src3,
+                                                     ggml_tensor *          src4,
+                                                     ggml_tensor *          src5,
+                                                     ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0        = src0,
         .src1        = src1,
@@ -1198,13 +1090,14 @@ static webgpu_command ggml_webgpu_gated_delta_net(webgpu_context & ctx,
          .size    = ggml_webgpu_tensor_binding_size(ctx, dst)  }
     };
 
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, h, n_seqs);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, h, n_seqs);
 }
 
-static std::optional<webgpu_command> ggml_webgpu_set_rows(webgpu_context & ctx,
-                                                          ggml_tensor *    src,
-                                                          ggml_tensor *    idx,
-                                                          ggml_tensor *    dst) {
+static std::optional<webgpu_encoded_op> ggml_webgpu_set_rows(webgpu_context &       ctx,
+                                                             wgpu::CommandEncoder & encoder,
+                                                             ggml_tensor *          src,
+                                                             ggml_tensor *          idx,
+                                                             ggml_tensor *          dst) {
     // For set rows specifically, we need to check if src and idx are empty
     // tensors.
     if (ggml_is_empty(src) || ggml_is_empty(idx)) {
@@ -1267,7 +1160,7 @@ static std::optional<webgpu_command> ggml_webgpu_set_rows(webgpu_context & ctx,
         threads = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
     }
     uint32_t wg_x = CEIL_DIV(threads, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x, 1);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x, 1);
 }
 
 // Workgroup size is a common constant
@@ -1278,10 +1171,11 @@ static std::vector<wgpu::ConstantEntry> ggml_webgpu_wg_size_entry(uint32_t wg_si
     return constants;
 }
 
-static webgpu_command ggml_webgpu_get_rows(webgpu_context & ctx,
-                                           ggml_tensor *    src,
-                                           ggml_tensor *    idx,
-                                           ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_get_rows(webgpu_context &       ctx,
+                                              wgpu::CommandEncoder & encoder,
+                                              ggml_tensor *          src,
+                                              ggml_tensor *          idx,
+                                              ggml_tensor *          dst) {
     const bool float_parallel = src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_I32;
 
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
@@ -1333,13 +1227,14 @@ static webgpu_command ggml_webgpu_get_rows(webgpu_context & ctx,
     uint32_t total_threads  = float_parallel ? blocks_per_row * total_rows : total_rows;
     uint32_t wg_x           = CEIL_DIV(total_threads, decisions->wg_size);
 
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_mul_mat(webgpu_context & ctx,
-                                          ggml_tensor *    src0,
-                                          ggml_tensor *    src1,
-                                          ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context &       ctx,
+                                             wgpu::CommandEncoder & encoder,
+                                             ggml_tensor *          src0,
+                                             ggml_tensor *          src1,
+                                             ggml_tensor *          dst) {
     // Determine if this is a mat-vec operation
     bool is_vec = (dst->ne[1] == 1);
 
@@ -1478,17 +1373,18 @@ static webgpu_command ggml_webgpu_mul_mat(webgpu_context & ctx,
         compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
     }
 
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x, wg_y);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x, wg_y);
 }
 
 #ifndef __EMSCRIPTEN__
-static webgpu_command ggml_webgpu_flash_attn(webgpu_context & ctx,
-                                             ggml_tensor *    Q,
-                                             ggml_tensor *    K,
-                                             ggml_tensor *    V,
-                                             ggml_tensor *    mask,
-                                             ggml_tensor *    sinks,
-                                             ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context &       ctx,
+                                                wgpu::CommandEncoder & encoder,
+                                                ggml_tensor *          Q,
+                                                ggml_tensor *          K,
+                                                ggml_tensor *          V,
+                                                ggml_tensor *          mask,
+                                                ggml_tensor *          sinks,
+                                                ggml_tensor *          dst) {
     float scale = *(float *) dst->op_params;
     float max_bias;
     memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
@@ -1565,32 +1461,255 @@ static webgpu_command ggml_webgpu_flash_attn(webgpu_context & ctx,
                         .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
                         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
 
-    ggml_webgpu_shader_lib_context shader_lib_ctx = {
-        .src0               = Q,
-        .src1               = K,
-        .src2               = V,
-        .src3               = mask,
-        .src4               = sinks,
-        .dst                = dst,
-        .max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup,
-        .wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize,
+    const uint32_t k_offset_elems   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type));
+    const uint32_t v_offset_elems   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type));
+    const bool     f16_vec4_aligned = (k_offset_elems % 4u == 0u) && (v_offset_elems % 4u == 0u);
+
+    const bool kv_direct = (K->type == GGML_TYPE_F16) && f16_vec4_aligned &&
+                           (Q->ne[0] % ctx->global_ctx->capabilities.sg_mat_k == 0) &&
+                           (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
+
+    const bool kv_vec_type_supported =
+        K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0;
+    const bool use_vec = (Q->ne[1] < 20) && (Q->ne[0] % 32 == 0) && (V->ne[0] % 4 == 0) && kv_vec_type_supported &&
+                         (K->type != GGML_TYPE_F16 || f16_vec4_aligned) && (V->type == K->type);
+    const uint32_t vec_nwg_cap = std::max(1u, std::min<uint32_t>(32u, ctx->global_ctx->capabilities.max_subgroup_size));
+    const bool     use_blk     = use_vec && has_mask;
+
+    ggml_webgpu_flash_attn_pipeline_key key = {
+        .kv_type            = K->type,
+        .head_dim_qk        = (uint32_t) Q->ne[0],
+        .head_dim_v         = (uint32_t) V->ne[0],
+        .kv_direct          = kv_direct,
+        .has_mask           = static_cast<bool>(has_mask),
+        .has_sinks          = static_cast<bool>(has_sinks),
+        .uses_logit_softcap = logit_softcap != 0.0f,
+        .use_vec            = use_vec,
+    };
+
+    ggml_webgpu_flash_attn_shader_lib_context shader_lib_ctx = {
+        .key                = key,
         .sg_mat_m           = ctx->global_ctx->capabilities.sg_mat_m,
         .sg_mat_n           = ctx->global_ctx->capabilities.sg_mat_n,
         .sg_mat_k           = ctx->global_ctx->capabilities.sg_mat_k,
+        .wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize,
         .max_subgroup_size  = ctx->global_ctx->capabilities.max_subgroup_size,
     };
-
     webgpu_pipeline pipeline = ctx->shader_lib->get_flash_attn_pipeline(shader_lib_ctx);
 
     auto * decisions = static_cast<ggml_webgpu_flash_attn_shader_decisions *>(pipeline.context.get());
 
     uint32_t wg_per_head = CEIL_DIV(Q->ne[1], decisions->q_tile);
     uint32_t wg_x        = wg_per_head * Q->ne[2] * Q->ne[3];  // wg per head * number of heads * number of batches
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+
+    wgpu::Buffer blk_buf         = {};
+    uint64_t     blk_size_bytes  = 0;
+    uint32_t     blk_nblk0       = 0;
+    uint32_t     blk_nblk1       = 0;
+    uint32_t     blk_batch_count = 0;
+
+    if (use_vec) {
+        uint32_t       nwg     = 1u;
+        const uint64_t kv_span = (uint64_t) std::max(1u, decisions->kv_tile);
+        while ((2u * nwg * kv_span) < (uint64_t) K->ne[1] && nwg < vec_nwg_cap) {
+            nwg <<= 1;
+        }
+        nwg = std::min(nwg, vec_nwg_cap);
+        GGML_ASSERT(nwg <= ctx->global_ctx->capabilities.max_subgroup_size);
+        const uint64_t nrows          = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
+        const bool     use_vec_reduce = nwg > 1u;
+        GGML_ASSERT(nrows <= UINT32_MAX);
+
+        uint64_t     tmp_stats_base  = 0;
+        uint64_t     tmp_size_bytes  = 0;
+        wgpu::Buffer tmp_buf         = {};
+        uint64_t     tmp_bind_offset = 0;
+        uint64_t     tmp_bind_size   = 0;
+        const size_t align_bytes     = ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
+        const size_t dst_offset      = ggml_webgpu_tensor_offset(dst);
+        size_t       scratch_offset  = ROUNDUP_POW2(dst_offset + ggml_nbytes(dst), align_bytes);
+
+        if (use_vec_reduce) {
+            const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
+            const uint64_t tmp_stats_elems = nrows * 2u * nwg;
+            tmp_stats_base                 = tmp_data_elems;
+            tmp_size_bytes =
+                ROUNDUP_POW2((tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
+            GGML_ASSERT(tmp_stats_base <= UINT32_MAX);
+            tmp_buf         = ggml_webgpu_tensor_buf(dst);
+            tmp_bind_offset = scratch_offset;
+            tmp_bind_size   = tmp_size_bytes;
+            scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
+        } else {
+            // nwg==1 writes final dst directly in vec-split; keep tmp binding valid without extra allocation.
+            tmp_buf         = ggml_webgpu_tensor_buf(dst);
+            tmp_bind_offset = ggml_webgpu_tensor_align_offset(ctx, dst);
+            tmp_bind_size   = ggml_webgpu_tensor_binding_size(ctx, dst);
+        }
+
+        webgpu_pipeline                   blk_pipeline;
+        std::vector<uint32_t>             blk_params;
+        std::vector<wgpu::BindGroupEntry> blk_entries;
+        if (use_blk) {
+            GGML_ASSERT(has_mask);
+
+            blk_nblk0                   = CEIL_DIV((uint32_t) K->ne[1], decisions->kv_tile);
+            blk_nblk1                   = CEIL_DIV((uint32_t) Q->ne[1], decisions->q_tile);
+            blk_buf                     = ggml_webgpu_tensor_buf(dst);
+            const uint32_t stride_mask3 = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
+            blk_batch_count             = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
+            const uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
+            blk_size_bytes              = ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
+            ggml_webgpu_flash_attn_blk_shader_lib_context blk_shader_ctx = {
+                .key =
+                    {
+                        .q_tile  = decisions->q_tile,
+                        .kv_tile = decisions->kv_tile,
+                    },
+                .max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup,
+            };
+            blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx);
+
+            blk_params = {
+                (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)),  // offset_mask
+                (uint32_t) Q->ne[1],                                                                   // seq_len_q
+                (uint32_t) K->ne[1],                                                                   // seq_len_kv
+                stride_mask3,                                                                          // stride_mask3
+                blk_nblk0,                                                                             // nblk0
+                blk_nblk1,                                                                             // nblk1
+            };
+            blk_entries = {
+                { .binding = 0,
+                 .buffer  = ggml_webgpu_tensor_buf(mask),
+                 .offset  = ggml_webgpu_tensor_align_offset(ctx, mask),
+                 .size    = ggml_webgpu_tensor_binding_size(ctx, mask) },
+                { .binding = 1, .buffer = blk_buf, .offset = scratch_offset, .size = blk_size_bytes },
+            };
+            scratch_offset = ROUNDUP_POW2(scratch_offset + blk_size_bytes, align_bytes);
+        }
+
+        std::vector<uint32_t> split_params = params;
+        if (use_blk) {
+            split_params.push_back(0u);                     // blk_base
+            split_params.push_back(blk_nblk0);              // blk_nblk0
+            split_params.push_back(blk_nblk1);              // blk_nblk1
+        }
+        split_params.push_back(0u);                         // tmp_data_base
+        split_params.push_back((uint32_t) tmp_stats_base);  // tmp_stats_base
+        split_params.push_back(nwg);                        // nwg
+
+        std::vector<wgpu::BindGroupEntry> split_entries = {
+            { .binding = 0,
+             .buffer  = ggml_webgpu_tensor_buf(Q),
+             .offset  = ggml_webgpu_tensor_align_offset(ctx, Q),
+             .size    = ggml_webgpu_tensor_binding_size(ctx, Q) },
+            { .binding = 1,
+             .buffer  = ggml_webgpu_tensor_buf(K),
+             .offset  = ggml_webgpu_tensor_align_offset(ctx, K),
+             .size    = ggml_webgpu_tensor_binding_size(ctx, K) },
+            { .binding = 2,
+             .buffer  = ggml_webgpu_tensor_buf(V),
+             .offset  = ggml_webgpu_tensor_align_offset(ctx, V),
+             .size    = ggml_webgpu_tensor_binding_size(ctx, V) },
+        };
+        uint32_t split_binding_index = 3;
+        if (has_mask) {
+            split_entries.push_back({ .binding = split_binding_index++,
+                                      .buffer  = ggml_webgpu_tensor_buf(mask),
+                                      .offset  = ggml_webgpu_tensor_align_offset(ctx, mask),
+                                      .size    = ggml_webgpu_tensor_binding_size(ctx, mask) });
+        }
+        if (has_sinks) {
+            split_entries.push_back({ .binding = split_binding_index++,
+                                      .buffer  = ggml_webgpu_tensor_buf(sinks),
+                                      .offset  = ggml_webgpu_tensor_align_offset(ctx, sinks),
+                                      .size    = ggml_webgpu_tensor_binding_size(ctx, sinks) });
+        }
+        if (use_blk) {
+            split_entries.push_back({ .binding = split_binding_index++,
+                                      .buffer  = blk_buf,
+                                      .offset  = blk_entries[1].offset,
+                                      .size    = blk_size_bytes });
+        }
+        split_entries.push_back(
+            { .binding = split_binding_index++, .buffer = tmp_buf, .offset = tmp_bind_offset, .size = tmp_bind_size });
+        split_entries.push_back({ .binding = split_binding_index++,
+                                  .buffer  = ggml_webgpu_tensor_buf(dst),
+                                  .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
+                                  .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
+
+        webgpu_pipeline                   reduce_pipeline;
+        std::vector<uint32_t>             reduce_params;
+        std::vector<wgpu::BindGroupEntry> reduce_entries;
+        if (use_vec_reduce) {
+            const uint32_t reduce_wg_size = std::max(
+                32u,
+                std::min<uint32_t>(nwg * 32u, ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup));
+            ggml_webgpu_flash_attn_vec_reduce_shader_lib_context reduce_shader_ctx = {
+                .key =
+                    {
+                        .head_dim_v = (uint32_t) V->ne[0],
+                        .wg_size    = reduce_wg_size,
+                    },
+                .max_wg_size = reduce_wg_size,
+            };
+            reduce_pipeline = ctx->shader_lib->get_flash_attn_vec_reduce_pipeline(reduce_shader_ctx);
+
+            reduce_params = {
+                (uint32_t) nrows,                                                                    // nrows
+                (uint32_t) Q->ne[1],                                                                 // seq_len_q
+                (uint32_t) Q->ne[2],                                                                 // n_heads
+                (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),  // offset_dst
+                nwg,                                                                                 // nwg
+                0u,                                                                                  // tmp_data_base
+                (uint32_t) tmp_stats_base,                                                           // tmp_stats_base
+            };
+
+            reduce_entries = {
+                { .binding = 0, .buffer = tmp_buf, .offset = tmp_bind_offset, .size = tmp_size_bytes },
+                { .binding = 1,
+                 .buffer  = ggml_webgpu_tensor_buf(dst),
+                 .offset  = ggml_webgpu_tensor_align_offset(ctx, dst),
+                 .size    = ggml_webgpu_tensor_binding_size(ctx, dst) },
+            };
+        }
+
+        const uint64_t split_wg_total = (uint64_t) wg_x * nwg;
+        GGML_ASSERT(split_wg_total <= UINT32_MAX);
+        std::vector<webgpu_pipeline>                   pipelines;
+        std::vector<std::vector<uint32_t>>             params_list;
+        std::vector<std::vector<wgpu::BindGroupEntry>> entries_list;
+        std::vector<std::pair<uint32_t, uint32_t>>     workgroups_list;
+
+        if (use_blk) {
+            pipelines.push_back(blk_pipeline);
+            params_list.push_back(std::move(blk_params));
+            entries_list.push_back(std::move(blk_entries));
+            workgroups_list.push_back({ blk_nblk0, blk_nblk1 * blk_batch_count });
+        }
+        pipelines.push_back(pipeline);
+        params_list.push_back(std::move(split_params));
+        entries_list.push_back(std::move(split_entries));
+        workgroups_list.push_back({ (uint32_t) split_wg_total, 1u });
+        if (use_vec_reduce) {
+            pipelines.push_back(reduce_pipeline);
+            params_list.push_back(std::move(reduce_params));
+            entries_list.push_back(std::move(reduce_entries));
+            workgroups_list.push_back({ (uint32_t) nrows, 1u });
+        }
+
+        return ggml_backend_webgpu_build_multi(ctx->global_ctx, ctx->param_arena, encoder, pipelines, params_list,
+                                               entries_list, workgroups_list);
+    }
+
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
-#endif
+#endif  // __EMSCRIPTEN__
 
-static webgpu_command ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context &       ctx,
+                                              wgpu::CommandEncoder & encoder,
+                                              ggml_tensor *          src,
+                                              ggml_tensor *          dst) {
     bool is_unary = dst->op == GGML_OP_UNARY;
     bool inplace  = ggml_webgpu_tensor_equal(src, dst) || (dst->op == GGML_OP_FILL);
 
@@ -1665,13 +1784,14 @@ static webgpu_command ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * s
     }
 
     uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_binary_op(webgpu_context & ctx,
-                                            ggml_tensor *    src0,
-                                            ggml_tensor *    src1,
-                                            ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context &       ctx,
+                                               wgpu::CommandEncoder & encoder,
+                                               ggml_tensor *          src0,
+                                               ggml_tensor *          src1,
+                                               ggml_tensor *          dst) {
     binary_overlap_flags flags = ggml_webgpu_detect_binary_overlap(src0, src1, dst);
 
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
@@ -1767,13 +1887,14 @@ static webgpu_command ggml_webgpu_binary_op(webgpu_context & ctx,
     }
 
     uint32_t wg_x = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_concat(webgpu_context & ctx,
-                                         ggml_tensor *    src0,
-                                         ggml_tensor *    src1,
-                                         ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_concat(webgpu_context &       ctx,
+                                            wgpu::CommandEncoder & encoder,
+                                            ggml_tensor *          src0,
+                                            ggml_tensor *          src1,
+                                            ggml_tensor *          dst) {
     uint32_t ne  = (uint32_t) ggml_nelements(dst);
     uint32_t dim = (uint32_t) dst->op_params[0];
 
@@ -1823,10 +1944,13 @@ static webgpu_command ggml_webgpu_concat(webgpu_context & ctx,
     webgpu_pipeline pipeline  = ctx->shader_lib->get_concat_pipeline(shader_lib_ctx);
     auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
     uint32_t        wg_x      = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_repeat(webgpu_context &       ctx,
+                                            wgpu::CommandEncoder & encoder,
+                                            ggml_tensor *          src0,
+                                            ggml_tensor *          dst) {
     uint32_t ne = (uint32_t) ggml_nelements(dst);
 
     std::vector<uint32_t> params = { ne,
@@ -1865,10 +1989,13 @@ static webgpu_command ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor * src
     webgpu_pipeline pipeline  = ctx->shader_lib->get_repeat_pipeline(shader_lib_ctx);
     auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
     uint32_t        wg_x      = CEIL_DIV(ne, decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context &       ctx,
+                                              wgpu::CommandEncoder & encoder,
+                                              ggml_tensor *          src,
+                                              ggml_tensor *          dst) {
     bool inplace = ggml_webgpu_tensor_equal(src, dst);
 
     std::vector<uint32_t> params = {
@@ -1908,14 +2035,16 @@ static webgpu_command ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * s
     };
 
     webgpu_pipeline pipeline = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, ggml_nrows(src));
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries,
+                                     ggml_nrows(src));
 }
 
-static webgpu_command ggml_webgpu_rope(webgpu_context & ctx,
-                                       ggml_tensor *    src0,
-                                       ggml_tensor *    src1,
-                                       ggml_tensor *    src2,
-                                       ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_rope(webgpu_context &       ctx,
+                                          wgpu::CommandEncoder & encoder,
+                                          ggml_tensor *          src0,
+                                          ggml_tensor *          src1,
+                                          ggml_tensor *          src2,
+                                          ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0        = src0,
         .src1        = src1,
@@ -2012,10 +2141,14 @@ static webgpu_command ggml_webgpu_rope(webgpu_context & ctx,
     }
 
     uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_glu(webgpu_context & ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_glu(webgpu_context &       ctx,
+                                         wgpu::CommandEncoder & encoder,
+                                         ggml_tensor *          src0,
+                                         ggml_tensor *          src1,
+                                         ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0        = src0,
         .src1        = src1,
@@ -2074,10 +2207,13 @@ static webgpu_command ggml_webgpu_glu(webgpu_context & ctx, ggml_tensor * src0,
                         .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
 
     uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_scale(webgpu_context &       ctx,
+                                           wgpu::CommandEncoder & encoder,
+                                           ggml_tensor *          src,
+                                           ggml_tensor *          dst) {
     bool inplace = ggml_webgpu_tensor_equal(src, dst);
 
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
@@ -2125,14 +2261,15 @@ static webgpu_command ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src,
     }
 
     uint32_t wg_x = CEIL_DIV(ggml_nelements(dst), decisions->wg_size);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_soft_max(webgpu_context & ctx,
-                                           ggml_tensor *    src0,
-                                           ggml_tensor *    src1,
-                                           ggml_tensor *    src2,
-                                           ggml_tensor *    dst) {
+static webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context &       ctx,
+                                              wgpu::CommandEncoder & encoder,
+                                              ggml_tensor *          src0,
+                                              ggml_tensor *          src1,
+                                              ggml_tensor *          src2,
+                                              ggml_tensor *          dst) {
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
         .src0        = src0,
         .src1        = src1,
@@ -2208,10 +2345,14 @@ static webgpu_command ggml_webgpu_soft_max(webgpu_context & ctx,
                             .size    = ggml_webgpu_tensor_binding_size(ctx, dst) });
     }
 
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, ggml_nrows(dst));
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries,
+                                     ggml_nrows(dst));
 }
 
-static webgpu_command ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_argmax(webgpu_context &       ctx,
+                                            wgpu::CommandEncoder & encoder,
+                                            ggml_tensor *          src,
+                                            ggml_tensor *          dst) {
     std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                      (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                      (uint32_t) src->ne[0] };
@@ -2233,10 +2374,13 @@ static webgpu_command ggml_webgpu_argmax(webgpu_context & ctx, ggml_tensor * src
 
     webgpu_pipeline pipeline = ctx->shader_lib->get_argmax_pipeline(shader_lib_ctx);
     uint32_t        wg_x     = ggml_nelements(dst);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_argsort(webgpu_context &       ctx,
+                                             wgpu::CommandEncoder & encoder,
+                                             ggml_tensor *          src,
+                                             ggml_tensor *          dst) {
     bool is_top_k = dst->op == GGML_OP_TOP_K;
 
     ggml_webgpu_shader_lib_context shader_lib_ctx = {
@@ -2327,7 +2471,7 @@ static webgpu_command ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * sr
     workgroups_list.push_back({ wg_x_init, wg_y_init });
 
     if (merge_passes == 0) {
-        return ggml_backend_webgpu_build_multi(ctx->global_ctx, ctx->param_buf_pool, pipelines, params_list,
+        return ggml_backend_webgpu_build_multi(ctx->global_ctx, ctx->param_arena, encoder, pipelines, params_list,
                                                entries_list, workgroups_list);
     }
 
@@ -2389,11 +2533,14 @@ static webgpu_command ggml_webgpu_argsort(webgpu_context & ctx, ggml_tensor * sr
         in_is_tmp = !in_is_tmp;
     }
 
-    return ggml_backend_webgpu_build_multi(ctx->global_ctx, ctx->param_buf_pool, pipelines, params_list, entries_list,
-                                           workgroups_list);
+    return ggml_backend_webgpu_build_multi(ctx->global_ctx, ctx->param_arena, encoder, pipelines, params_list,
+                                           entries_list, workgroups_list);
 }
 
-static webgpu_command ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_cumsum(webgpu_context &       ctx,
+                                            wgpu::CommandEncoder & encoder,
+                                            ggml_tensor *          src,
+                                            ggml_tensor *          dst) {
     std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                      (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
                                      (uint32_t) src->ne[0] };
@@ -2418,10 +2565,13 @@ static webgpu_command ggml_webgpu_cumsum(webgpu_context & ctx, ggml_tensor * src
 
     webgpu_pipeline pipeline = ctx->shader_lib->get_cumsum_pipeline(shader_lib_ctx);
     uint32_t        wg_x     = ggml_nrows(dst);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
-static webgpu_command ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
+static webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context &       ctx,
+                                              wgpu::CommandEncoder & encoder,
+                                              ggml_tensor *          src,
+                                              ggml_tensor *          dst) {
     bool                  total_sum = dst->op == GGML_OP_SUM;
     std::vector<uint32_t> params = { (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
                                      (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
@@ -2450,11 +2600,13 @@ static webgpu_command ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor * s
     webgpu_pipeline pipeline = ctx->shader_lib->get_sum_rows_pipeline(shader_lib_ctx);
 
     uint32_t wg_x = total_sum ? 1 : ggml_nrows(dst);
-    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_buf_pool, pipeline, params, entries, wg_x);
+    return ggml_backend_webgpu_build(ctx->global_ctx, ctx->param_arena, encoder, pipeline, params, entries, wg_x);
 }
 
 // Returns the encoded command, or std::nullopt if the operation is a no-op
-static std::optional<webgpu_command> ggml_webgpu_encode_node(webgpu_context ctx, ggml_tensor * node) {
+static std::optional<webgpu_encoded_op> ggml_webgpu_encode_node(webgpu_context         ctx,
+                                                                wgpu::CommandEncoder & encoder,
+                                                                ggml_tensor *          node) {
     if (ggml_is_empty(node)) {
         return std::nullopt;
     }
@@ -2477,18 +2629,18 @@ static std::optional<webgpu_command> ggml_webgpu_encode_node(webgpu_context ctx,
             return std::nullopt;
         case GGML_OP_CPY:
         case GGML_OP_CONT:
-            return ggml_webgpu_cpy(ctx, src0, node);
+            return ggml_webgpu_cpy(ctx, encoder, src0, node);
         case GGML_OP_SET:
-            return ggml_webgpu_set(ctx, src0, src1, node);
+            return ggml_webgpu_set(ctx, encoder, src0, src1, node);
         case GGML_OP_SET_ROWS:
-            return ggml_webgpu_set_rows(ctx, src0, src1, node);
+            return ggml_webgpu_set_rows(ctx, encoder, src0, src1, node);
         case GGML_OP_GET_ROWS:
-            return ggml_webgpu_get_rows(ctx, src0, src1, node);
+            return ggml_webgpu_get_rows(ctx, encoder, src0, src1, node);
         case GGML_OP_MUL_MAT:
-            return ggml_webgpu_mul_mat(ctx, src0, src1, node);
+            return ggml_webgpu_mul_mat(ctx, encoder, src0, src1, node);
         case GGML_OP_FLASH_ATTN_EXT:
 #ifndef __EMSCRIPTEN__
-            return ggml_webgpu_flash_attn(ctx, src0, src1, src2, node->src[3], node->src[4], node);
+            return ggml_webgpu_flash_attn(ctx, encoder, src0, src1, src2, node->src[3], node->src[4], node);
 #else
             return std::nullopt;
 #endif
@@ -2496,22 +2648,22 @@ static std::optional<webgpu_command> ggml_webgpu_encode_node(webgpu_context ctx,
         case GGML_OP_SUB:
         case GGML_OP_MUL:
         case GGML_OP_DIV:
-            return ggml_webgpu_binary_op(ctx, src0, src1, node);
+            return ggml_webgpu_binary_op(ctx, encoder, src0, src1, node);
         case GGML_OP_CONCAT:
-            return ggml_webgpu_concat(ctx, src0, src1, node);
+            return ggml_webgpu_concat(ctx, encoder, src0, src1, node);
         case GGML_OP_REPEAT:
-            return ggml_webgpu_repeat(ctx, src0, node);
+            return ggml_webgpu_repeat(ctx, encoder, src0, node);
         case GGML_OP_RMS_NORM:
         case GGML_OP_L2_NORM:
-            return ggml_webgpu_row_norm(ctx, src0, node);
+            return ggml_webgpu_row_norm(ctx, encoder, src0, node);
         case GGML_OP_ROPE:
-            return ggml_webgpu_rope(ctx, src0, src1, src2, node);
+            return ggml_webgpu_rope(ctx, encoder, src0, src1, src2, node);
         case GGML_OP_GLU:
-            return ggml_webgpu_glu(ctx, src0, src1, node);
+            return ggml_webgpu_glu(ctx, encoder, src0, src1, node);
         case GGML_OP_SCALE:
-            return ggml_webgpu_scale(ctx, src0, node);
+            return ggml_webgpu_scale(ctx, encoder, src0, node);
         case GGML_OP_SOFT_MAX:
-            return ggml_webgpu_soft_max(ctx, src0, src1, src2, node);
+            return ggml_webgpu_soft_max(ctx, encoder, src0, src1, src2, node);
         case GGML_OP_UNARY:
         case GGML_OP_CLAMP:
         case GGML_OP_FILL:
@@ -2522,26 +2674,27 @@ static std::optional<webgpu_command> ggml_webgpu_encode_node(webgpu_context ctx,
         case GGML_OP_COS:
         case GGML_OP_DIAG:
         case GGML_OP_TRI:
-            return ggml_webgpu_unary_op(ctx, src0, node);
+            return ggml_webgpu_unary_op(ctx, encoder, src0, node);
         case GGML_OP_SOLVE_TRI:
-            return ggml_webgpu_solve_tri(ctx, src0, src1, node);
+            return ggml_webgpu_solve_tri(ctx, encoder, src0, src1, node);
         case GGML_OP_SSM_CONV:
-            return ggml_webgpu_ssm_conv(ctx, src0, src1, node);
+            return ggml_webgpu_ssm_conv(ctx, encoder, src0, src1, node);
         case GGML_OP_GATED_DELTA_NET:
-            return ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node);
+            return ggml_webgpu_gated_delta_net(ctx, encoder, src0, src1, src2, node->src[3], node->src[4], node->src[5],
+                                               node);
         case GGML_OP_PAD:
-            return ggml_webgpu_pad(ctx, src0, node);
+            return ggml_webgpu_pad(ctx, encoder, src0, node);
         case GGML_OP_ARGMAX:
-            return ggml_webgpu_argmax(ctx, src0, node);
+            return ggml_webgpu_argmax(ctx, encoder, src0, node);
         case GGML_OP_ARGSORT:
         case GGML_OP_TOP_K:
             // we reuse the same argsort implementation for top_k
-            return ggml_webgpu_argsort(ctx, src0, node);
+            return ggml_webgpu_argsort(ctx, encoder, src0, node);
         case GGML_OP_CUMSUM:
-            return ggml_webgpu_cumsum(ctx, src0, node);
+            return ggml_webgpu_cumsum(ctx, encoder, src0, node);
         case GGML_OP_SUM:
         case GGML_OP_SUM_ROWS:
-            return ggml_webgpu_sum_rows(ctx, src0, node);
+            return ggml_webgpu_sum_rows(ctx, encoder, src0, node);
         default:
             return std::nullopt;
     }
@@ -2555,31 +2708,42 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
 
     WEBGPU_CPU_PROFILE_TOTAL_START(graph_compute);
 
-    std::vector<webgpu_command>    commands;
-    std::vector<webgpu_submission> subs;
-    uint32_t                       num_batched_kernels = 0;
-    bool                           contains_set_rows   = false;
+    std::vector<webgpu_encoded_op> commands;
+#ifdef GGML_WEBGPU_GPU_PROFILE
+    std::vector<wgpu::FutureWaitInfo> profile_futures;
+#endif
+    uint32_t             num_batched_kernels = 0;
+    bool                 contains_set_rows   = false;
+    wgpu::CommandEncoder batch_encoder       = ctx->global_ctx->device.CreateCommandEncoder();
 
     for (int i = 0; i < cgraph->n_nodes; i++) {
         if (cgraph->nodes[i]->op == GGML_OP_SET_ROWS) {
             contains_set_rows = true;
         }
-        if (auto cmd = ggml_webgpu_encode_node(ctx, cgraph->nodes[i])) {
+        if (auto cmd = ggml_webgpu_encode_node(ctx, batch_encoder, cgraph->nodes[i])) {
             commands.push_back(*cmd);
             num_batched_kernels += cmd.value().num_kernels;
         }
 
         if (num_batched_kernels >= WEBGPU_COMMAND_SUBMIT_BATCH_SIZE) {
-            num_batched_kernels = 0;
-            subs.push_back(ggml_backend_webgpu_submit(ctx->global_ctx, commands, ctx->param_buf_pool));
-            // Process events and check for completed submissions
-            ctx->global_ctx->instance.ProcessEvents();
-            ggml_backend_webgpu_wait(ctx->global_ctx, subs, false);
+            num_batched_kernels                = 0;
+            wgpu::CommandBuffer batch_commands = batch_encoder.Finish();
+            ctx->global_ctx->queue.Submit(1, &batch_commands);
+#ifdef GGML_WEBGPU_GPU_PROFILE
+            ggml_backend_webgpu_collect_profile_futures(ctx->global_ctx, commands, profile_futures);
+#endif
+            ctx->param_arena.reset();
             commands.clear();
+            batch_encoder = ctx->global_ctx->device.CreateCommandEncoder();
         }
     }
     if (!commands.empty()) {
-        subs.push_back(ggml_backend_webgpu_submit(ctx->global_ctx, commands, ctx->param_buf_pool));
+        wgpu::CommandBuffer batch_commands = batch_encoder.Finish();
+        ctx->global_ctx->queue.Submit(1, &batch_commands);
+#ifdef GGML_WEBGPU_GPU_PROFILE
+        ggml_backend_webgpu_collect_profile_futures(ctx->global_ctx, commands, profile_futures);
+#endif
+        ctx->param_arena.reset();
         commands.clear();
     }
 
@@ -2590,6 +2754,11 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
                                    ctx->set_rows_host_error_buf.GetSize());
         wgpu::CommandBuffer set_rows_commands = encoder.Finish();
         ctx->global_ctx->queue.Submit(1, &set_rows_commands);
+    }
+
+    ggml_backend_webgpu_wait_queue(ctx->global_ctx);
+
+    if (contains_set_rows) {
         ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,
                                        ctx->set_rows_host_error_buf.GetSize());
         const uint32_t * error_data = (const uint32_t *) ctx->set_rows_host_error_buf.GetConstMappedRange();
@@ -2599,7 +2768,9 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
         ctx->set_rows_host_error_buf.Unmap();
     }
 
-    ggml_backend_webgpu_wait(ctx->global_ctx, subs);
+#ifdef GGML_WEBGPU_GPU_PROFILE
+    ggml_backend_webgpu_wait_profile_futures(ctx->global_ctx, profile_futures);
+#endif
     WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
     return GGML_STATUS_SUCCESS;
 }
@@ -2834,6 +3005,83 @@ static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer
                 }
             }
             break;
+        case GGML_OP_FLASH_ATTN_EXT:
+            {
+                const ggml_tensor * Q     = tensor->src[0];
+                const ggml_tensor * K     = tensor->src[1];
+                const ggml_tensor * V     = tensor->src[2];
+                const ggml_tensor * mask  = tensor->src[3];
+                const ggml_tensor * sinks = tensor->src[4];
+                if (Q && K && V) {
+                    GGML_UNUSED(sinks);
+                    const bool kv_direct = (K->type == GGML_TYPE_F16) &&
+                                           (Q->ne[0] % ctx->webgpu_global_ctx->capabilities.sg_mat_k == 0) &&
+                                           (K->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
+                    const bool kv_vec_type_supported =
+                        K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0;
+                    const bool use_vec = (Q->ne[1] < 20) && (Q->ne[0] % 32 == 0) && (V->ne[0] % 4 == 0) &&
+                                         kv_vec_type_supported && (V->type == K->type);
+                    if (use_vec) {
+                        const uint32_t sg_mat_m = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
+                        const uint32_t sg_mat_n = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
+                        const size_t   limit_bytes =
+                            ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
+                        const size_t q_tile       = sg_mat_m;
+                        const size_t base_q_bytes = (Q->ne[0] + V->ne[0]) * q_tile * GGML_WEBGPU_F16_SIZE_BYTES +
+                                                    2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
+                        size_t bytes_per_kv = 0;
+                        if (!kv_direct) {
+                            bytes_per_kv += std::max(Q->ne[0], V->ne[0]);
+                        }
+                        if (mask != nullptr) {
+                            bytes_per_kv += q_tile;
+                        }
+                        bytes_per_kv += q_tile;
+                        bytes_per_kv *= GGML_WEBGPU_F16_SIZE_BYTES;
+                        uint32_t kv_tile = ((limit_bytes - base_q_bytes) / bytes_per_kv / sg_mat_n) * sg_mat_n;
+                        kv_tile          = std::max(sg_mat_n, std::min(32u, kv_tile));
+                        kv_tile          = (kv_tile / sg_mat_n) * sg_mat_n;
+                        if (kv_direct) {
+                            GGML_ASSERT(kv_tile <= GGML_WEBGPU_KV_SEQ_PAD);
+                            while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
+                                kv_tile -= sg_mat_n;
+                            }
+                        }
+
+                        const uint32_t vec_nwg_cap = std::max(
+                            1u, std::min<uint32_t>(32u, ctx->webgpu_global_ctx->capabilities.max_subgroup_size));
+                        uint32_t       nwg     = 1u;
+                        const uint64_t kv_span = (uint64_t) std::max(1u, kv_tile);
+                        while ((2u * nwg * kv_span) < (uint64_t) K->ne[1] && nwg < vec_nwg_cap) {
+                            nwg <<= 1;
+                        }
+                        nwg = std::min(nwg, vec_nwg_cap);
+
+                        const size_t align =
+                            ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
+                        const uint64_t nrows = (uint64_t) Q->ne[1] * Q->ne[2] * Q->ne[3];
+                        if (nwg > 1u) {
+                            const uint64_t tmp_data_elems  = nrows * (uint64_t) V->ne[0] * nwg;
+                            const uint64_t tmp_stats_elems = nrows * 2u * nwg;
+                            const size_t   tmp_size_bytes  = ROUNDUP_POW2(
+                                (tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
+                            res += tmp_size_bytes + align;
+                        }
+                        if (mask != nullptr) {
+                            const uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
+                            const uint32_t blk_nblk1       = CEIL_DIV((uint32_t) Q->ne[1], 1u);
+                            const uint32_t stride_mask3    = (uint32_t) (mask->nb[3] / ggml_type_size(mask->type));
+                            const uint32_t blk_batch_count = stride_mask3 > 0 ? (uint32_t) Q->ne[3] : 1u;
+                            const uint64_t blk_elems       = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
+                            const size_t   blk_size_bytes =
+                                ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
+                            res += blk_size_bytes + align;
+                        }
+                        res = ROUNDUP_POW2(res, WEBGPU_STORAGE_BUF_BINDING_MULT);
+                    }
+                }
+            }
+            break;
         default:
             break;
     }
@@ -2900,11 +3148,11 @@ static void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
     ctx->capabilities.memset_bytes_per_thread =
         CEIL_DIV(ctx->capabilities.limits.maxStorageBufferBindingSize, max_threads);
     std::vector<wgpu::ConstantEntry> constants(2);
-    constants[0].key         = "wg_size";
-    constants[0].value       = WEBGPU_MAX_WG_SIZE;
-    constants[1].key         = "bytes_per_thread";
-    constants[1].value       = ctx->capabilities.memset_bytes_per_thread;
-    ctx->memset_pipelines[0] = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
+    constants[0].key     = "wg_size";
+    constants[0].value   = WEBGPU_MAX_WG_SIZE;
+    constants[1].key     = "bytes_per_thread";
+    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
+    ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
 }
 
 static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
@@ -3036,9 +3284,9 @@ static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
     GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);
 
     ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
-    ctx->webgpu_global_ctx->memset_buf_pool.init(ctx->webgpu_global_ctx->device, 1, WEBGPU_PARAMS_BUF_SIZE_BYTES,
-                                                 wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
-                                                 wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite);
+    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
+                              WEBGPU_PARAMS_BUF_SIZE_BYTES, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
+                              "memset_params_buf");
     ctx->webgpu_global_ctx->queue = ctx->webgpu_global_ctx->device.GetQueue();
 
 #ifdef GGML_WEBGPU_GPU_PROFILE
@@ -3062,9 +3310,8 @@ static webgpu_context initialize_webgpu_context(ggml_backend_dev_t dev) {
     webgpu_context                       webgpu_ctx = std::make_shared<webgpu_context_struct>();
     webgpu_ctx->global_ctx                          = dev_ctx->webgpu_global_ctx;
     webgpu_ctx->shader_lib = std::make_unique<ggml_webgpu_shader_lib>(dev_ctx->webgpu_global_ctx->device);
-    webgpu_ctx->param_buf_pool.init(webgpu_ctx->global_ctx->device, WEBGPU_NUM_PARAM_BUFS, WEBGPU_PARAMS_BUF_SIZE_BYTES,
-                                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
-                                    wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite, true);
+    webgpu_ctx->param_arena.init(webgpu_ctx->global_ctx->device, WEBGPU_PARAMS_BUF_SIZE_BYTES, WEBGPU_NUM_PARAM_SLOTS,
+                                 webgpu_ctx->global_ctx->capabilities.limits.minUniformBufferOffsetAlignment);
     ggml_webgpu_create_buffer(webgpu_ctx->global_ctx->device, webgpu_ctx->set_rows_dev_error_buf,
                               WEBGPU_SET_ROWS_ERROR_BUF_SIZE_BYTES,
                               wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, "set_rows_dev_error_buf");
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl
new file mode 100644
index 00000000..82d072be
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl
@@ -0,0 +1,105 @@
+diagnostic(off, subgroup_uniformity);
+enable f16;
+
+#define Q_TILE 1
+#define KV_TILE 32
+#define WG_SIZE 32
+
+struct Params {
+    offset_mask: u32,
+    seq_len_q: u32,
+    seq_len_kv: u32,
+    stride_mask3: u32,
+    // Number of KV blocks and Q blocks per batch.
+    // nblk0 = ceil(seq_len_kv / KV_TILE), nblk1 = ceil(seq_len_q / Q_TILE).
+    nblk0: u32,
+    nblk1: u32,
+};
+
+@group(0) @binding(0) var<storage, read> mask: array<f16>;
+@group(0) @binding(1) var<storage, read_write> blk: array<u32>;
+@group(0) @binding(2) var<uniform> params: Params;
+
+const MASK_MIN: f32 = -65504.0;
+const MASK_MAX: f32 = 65504.0;
+var<workgroup> wg_min: array<f32, WG_SIZE>;
+var<workgroup> wg_max: array<f32, WG_SIZE>;
+var<workgroup> wg_any: array<u32, WG_SIZE>;
+
+@compute @workgroup_size(WG_SIZE)
+fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
+        @builtin(local_invocation_id) local_id: vec3<u32>) {
+    // Dispatch mapping:
+    //  - x indexes KV blocks
+    //  - y flattens (batch_idx, q_blk) as y = batch_idx * nblk1 + q_blk
+    let kv_blk = wg_id.x;
+    let y = wg_id.y;
+    let q_blk = y % params.nblk1;
+    let batch_idx = y / params.nblk1;
+    if (kv_blk >= params.nblk0) {
+        return;
+    }
+
+    let q_start = q_blk * Q_TILE;
+    let k_start = kv_blk * KV_TILE;
+
+    let mask_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
+    let mask_batch_base = params.offset_mask + mask_batch * params.stride_mask3;
+
+    // We keep min/max to classify:
+    //  - fully masked (max <= MASK_MIN)
+    //  - all-zero mask (min == 0 && max == 0)
+    //  - mixed/general mask
+    var local_min = MASK_MAX;
+    var local_max = -MASK_MAX;
+    var local_any = 0u;
+
+    for (var q_rel = 0u; q_rel < Q_TILE; q_rel += 1u) {
+        let q_row = q_start + q_rel;
+        if (q_row >= params.seq_len_q) {
+            continue;
+        }
+        let row_base = mask_batch_base + q_row * params.seq_len_kv;
+        for (var k_rel = local_id.x; k_rel < KV_TILE; k_rel += WG_SIZE) {
+            let k_col = k_start + k_rel;
+            if (k_col >= params.seq_len_kv) {
+                continue;
+            }
+            let mv = f32(mask[row_base + k_col]);
+            local_min = min(local_min, mv);
+            local_max = max(local_max, mv);
+            local_any = 1u;
+        }
+    }
+
+    wg_min[local_id.x] = local_min;
+    wg_max[local_id.x] = local_max;
+    wg_any[local_id.x] = local_any;
+    workgroupBarrier();
+
+    // Thread 0 writes one state per block.
+    if (local_id.x == 0u) {
+        var mmin = wg_min[0];
+        var mmax = wg_max[0];
+        var many = wg_any[0];
+        for (var i = 1u; i < WG_SIZE; i += 1u) {
+            mmin = min(mmin, wg_min[i]);
+            mmax = max(mmax, wg_max[i]);
+            many = max(many, wg_any[i]);
+        }
+
+        var state = 0u;
+        if (many != 0u) {
+            if (mmax <= MASK_MIN) {
+                state = 0u;
+            } else if (mmin == 0.0 && mmax == 0.0) {
+                state = 2u;
+            } else {
+                state = 1u;
+            }
+        }
+
+        let blk_idx = (batch_idx * params.nblk1 + q_blk) * params.nblk0 + kv_blk;
+        blk[blk_idx] = state;
+    }
+}
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn_vec_reduce.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn_vec_reduce.wgsl
new file mode 100644
index 00000000..9a0de82a
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/flash_attn_vec_reduce.wgsl
@@ -0,0 +1,78 @@
+diagnostic(off, subgroup_uniformity);
+enable f16;
+enable subgroups;
+
+// Default values
+#define HEAD_DIM_V 64
+#define WG_SIZE 128
+
+struct Params {
+    nrows: u32,
+    seq_len_q: u32,
+    n_heads: u32,
+    offset_dst: u32,
+    nwg: u32,
+    tmp_data_base: u32,
+    tmp_stats_base: u32,
+};
+
+@group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
+@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
+@group(0) @binding(2) var<uniform> params: Params;
+
+const FLOAT_MIN: f32 = -1.0e9;
+
+@compute @workgroup_size(WG_SIZE)
+fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
+        @builtin(subgroup_id) subgroup_id: u32,
+        @builtin(num_subgroups) num_subgroups: u32,
+        @builtin(subgroup_size) subgroup_size: u32,
+        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
+    let rid = wg_id.x;
+    if (rid >= params.nrows) {
+        return;
+    }
+
+    let rows_per_batch = params.n_heads * params.seq_len_q;
+    let batch_idx = rid / rows_per_batch;
+    let rem = rid % rows_per_batch;
+    let head_idx = rem / params.seq_len_q;
+    let q_row = rem % params.seq_len_q;
+
+    let dst2_stride = HEAD_DIM_V * params.n_heads;
+    let dst3_stride = dst2_stride * params.seq_len_q;
+    let row_base = params.offset_dst + batch_idx * dst3_stride + q_row * dst2_stride + head_idx * HEAD_DIM_V;
+
+    let thread = sg_inv_id;
+    if (params.nwg > subgroup_size) {
+        return;
+    }
+
+    let stats_base = params.tmp_stats_base + rid * (2u * params.nwg);
+    let active_thread = thread < params.nwg;
+    let si = select(0.0, tmp[stats_base + 2u * thread + 0u], active_thread);
+    let mi = select(FLOAT_MIN, tmp[stats_base + 2u * thread + 1u], active_thread);
+    let m = subgroupMax(mi);
+    let ms = select(0.0, exp(mi - m), active_thread);
+    let s = subgroupAdd(si * ms);
+    let inv_s = select(0.0, 1.0 / s, s != 0.0);
+
+    let row_tmp_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg);
+    for (var elem_base = subgroup_id * 4u; elem_base < HEAD_DIM_V; elem_base += num_subgroups * 4u) {
+        var weighted = vec4<f32>(0.0, 0.0, 0.0, 0.0);
+        if (active_thread) {
+            let src = row_tmp_base + thread * HEAD_DIM_V + elem_base;
+            weighted = vec4<f32>(tmp[src + 0u], tmp[src + 1u], tmp[src + 2u], tmp[src + 3u]) * ms;
+        }
+
+        let sum_x = subgroupAdd(weighted.x);
+        let sum_y = subgroupAdd(weighted.y);
+        let sum_z = subgroupAdd(weighted.z);
+        let sum_w = subgroupAdd(weighted.w);
+
+        if (thread == 0u) {
+            let dst_vec_index = (row_base + elem_base) >> 2u;
+            dst[dst_vec_index] = vec4<f32>(sum_x, sum_y, sum_z, sum_w) * inv_s;
+        }
+    }
+}
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl
new file mode 100644
index 00000000..a5257587
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl
@@ -0,0 +1,729 @@
+diagnostic(off, chromium.subgroup_matrix_uniformity);
+diagnostic(off, subgroup_uniformity);
+enable f16;
+enable subgroups;
+enable chromium_experimental_subgroup_matrix;
+
+#ifdef KV_F32
+#define KV_TYPE f32
+#else
+#define KV_TYPE f16
+#endif
+
+#define HEAD_DIM_QK 64
+#define HEAD_DIM_V 64
+
+
+#define SG_MAT_M 8
+#define SG_MAT_N 8
+#define SG_MAT_K 8
+
+#define Q_TILE SG_MAT_M
+#define KV_TILE 16
+#define WG_SIZE 64
+#ifndef VEC_NE
+#define VEC_NE 4u
+#endif
+
+#define KV_BLOCKS (KV_TILE / SG_MAT_N)
+
+#define BLOCK_SIZE 32
+#define BLOCKS_K ((HEAD_DIM_QK + BLOCK_SIZE - 1) / BLOCK_SIZE)
+#define BLOCKS_V ((HEAD_DIM_V + BLOCK_SIZE - 1) / BLOCK_SIZE)
+#if defined(KV_Q4_0)
+#define NQ 16
+#define F16_PER_BLOCK 9
+#define WEIGHTS_PER_F16 4
+#elif defined(KV_Q8_0)
+#define NQ 8
+#define F16_PER_BLOCK 17
+#define WEIGHTS_PER_F16 2
+#endif
+#define F16_PER_THREAD (NQ / WEIGHTS_PER_F16)
+
+fn get_byte(value: u32, index: u32) -> u32 {
+    return (value >> (index * 8)) & 0xFF;
+}
+
+fn get_byte_i32(value: u32, index: u32) -> i32 {
+    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
+}
+
+struct Params {
+    offset_q: u32,
+    offset_k: u32,
+    offset_v: u32,
+    offset_mask: u32,
+    offset_sinks: u32,
+    offset_dst: u32,
+
+    // shapes of Q/K/V
+    n_heads: u32,
+    seq_len_q: u32,
+    seq_len_kv: u32,
+
+    // strides (in elements)
+    stride_q1: u32,
+    stride_q2: u32,
+    stride_q3: u32,
+    stride_k1: u32,
+    stride_k2: u32,
+    stride_k3: u32,
+    stride_v1: u32,
+    stride_v2: u32,
+    stride_v3: u32,
+    stride_mask3: u32,
+
+    // repeat factors for K/V, e.g., MHA vs. MQA vs. GQA
+    q_per_kv: u32,
+
+    // softmax params
+    scale: f32,
+    max_bias: f32,
+    logit_softcap: f32,
+    n_head_log2: f32,
+    m0: f32,
+    m1: f32,
+
+#ifdef BLK
+    blk_base: u32,
+    blk_nblk0: u32,
+    blk_nblk1: u32,
+#endif
+
+    tmp_data_base: u32,
+    tmp_stats_base: u32,
+    nwg: u32,
+};
+
+@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
+#if defined(KV_Q4_0) || defined(KV_Q8_0)
+@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
+#else
+@group(0) @binding(1) var<storage, read_write> K: array<vec4<KV_TYPE>>;
+#endif
+#if defined(KV_Q4_0) || defined(KV_Q8_0)
+@group(0) @binding(2) var<storage, read_write> V: array<KV_TYPE>;
+#else
+@group(0) @binding(2) var<storage, read_write> V: array<vec4<KV_TYPE>>;
+#endif
+#if defined(MASK) && defined(SINKS)
+@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
+@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
+#ifdef BLK
+#define BLK_BINDING 5
+#define TMP_BINDING 6
+#define DST_BINDING 7
+#define PARAMS_BINDING 8
+#else
+#define TMP_BINDING 5
+#define DST_BINDING 6
+#define PARAMS_BINDING 7
+#endif
+#elif defined(MASK)
+@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
+#ifdef BLK
+#define BLK_BINDING 4
+#define TMP_BINDING 5
+#define DST_BINDING 6
+#define PARAMS_BINDING 7
+#else
+#define TMP_BINDING 4
+#define DST_BINDING 5
+#define PARAMS_BINDING 6
+#endif
+#elif defined(SINKS)
+@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
+#define TMP_BINDING 4
+#define DST_BINDING 5
+#define PARAMS_BINDING 6
+#else
+#define TMP_BINDING 3
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#endif
+
+#ifdef BLK
+@group(0) @binding(BLK_BINDING) var<storage, read_write> blk: array<u32>;
+#endif
+@group(0) @binding(TMP_BINDING) var<storage, read_write> tmp: array<f32>;
+@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
+@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;
+
+// Just a very small float value.
+const FLOAT_MIN: f32 = -1.0e9;
+
+var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;
+
+#ifndef KV_DIRECT
+const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
+// we can reuse the same shmem for K and V since we only need one at a time
+var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
+#endif
+
+var<workgroup> o_shmem: array<f16, Q_TILE * HEAD_DIM_V>;
+
+#ifdef MASK
+// storage for mask values
+var<workgroup> mask_shmem: array<f16, Q_TILE * KV_TILE>;
+#endif
+
+// note that we reuse the same storage for both since we only need one at a time
+var<workgroup> inter_shmem: array<f16, Q_TILE * KV_TILE>;
+
+// Storage for row max and exp sum during online softmax
+var<workgroup> row_max_shmem: array<f32, Q_TILE>;
+var<workgroup> exp_sum_shmem: array<f32, Q_TILE>;
+var<workgroup> blk_state_wg: u32;
+
+fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32, has_bias: bool, apply_mask: bool) -> f32 {
+    var v = select(FLOAT_MIN,
+                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
+                   kv_idx < KV_TILE);
+#ifdef LOGIT_SOFTCAP
+    v = params.logit_softcap * tanh(v);
+#endif
+#ifdef MASK
+    if (apply_mask) {
+        var mask_val = select(0.0,f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
+        v += select(mask_val, slope * mask_val, has_bias);
+    }
+#endif
+    return v;
+}
+
+@compute @workgroup_size(WG_SIZE)
+fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
+    @builtin(local_invocation_id) local_id: vec3<u32>,
+    @builtin(subgroup_id) subgroup_id: u32,
+    @builtin(subgroup_size) subgroup_size: u32,
+    @builtin(num_subgroups) num_subgroups: u32,
+    @builtin(subgroup_invocation_id) sg_inv_id: u32) {
+
+    // initialize row max for online softmax
+    for (var i = local_id.x; i < Q_TILE; i += WG_SIZE) {
+        row_max_shmem[i] = FLOAT_MIN;
+        exp_sum_shmem[i] = 0.0;
+    }
+
+    for (var i = local_id.x; i < Q_TILE * HEAD_DIM_V; i += WG_SIZE) {
+        o_shmem[i] = 0.0;
+    }
+
+    // workgroups per head/batch
+    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
+    let wg_per_batch = wg_per_head * params.n_heads;
+
+    let dst2_stride = HEAD_DIM_V * params.n_heads;
+    let dst3_stride = dst2_stride * params.seq_len_q;
+
+    let iwg = wg_id.x % params.nwg;
+    let base_wg_id = wg_id.x / params.nwg;
+
+    // batch index
+    let batch_idx = base_wg_id / wg_per_batch;
+    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
+    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
+    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
+    let wg_in_batch = base_wg_id % wg_per_batch;
+
+    // head index
+    let head_idx = wg_in_batch / wg_per_head;
+    let q_head_offset = q_batch_offset + head_idx * params.stride_q2;
+    let k_head_idx = head_idx / params.q_per_kv;
+    let v_head_idx = k_head_idx;
+    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
+    let v_head_offset = v_batch_offset + v_head_idx * params.stride_v2;
+
+    // starting Q row for this workgroup
+    let wg_in_head = wg_in_batch % wg_per_head;
+    let q_row_start = wg_in_head * Q_TILE;
+
+#ifdef MASK
+    // mask offset
+    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row_start * params.seq_len_kv;
+#endif
+
+    let head = f32(head_idx);
+    let has_bias = params.max_bias > 0.0;
+    let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), has_bias);
+
+    // load q tile into shared memory
+    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
+        let q_row = elem_idx / HEAD_DIM_QK;
+        let q_col = elem_idx % HEAD_DIM_QK;
+        let head_q_row = q_row_start + q_row;
+        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
+        q_shmem[elem_idx] = f16(select(
+            0.0,
+            Q[global_q_row_offset + q_col],
+            head_q_row < params.seq_len_q && q_col < HEAD_DIM_QK));
+    }
+
+    for (var kv_tile = iwg * KV_TILE; kv_tile < params.seq_len_kv; kv_tile += KV_TILE * params.nwg) {
+#ifdef BLK
+        let q_blk = q_row_start / Q_TILE;
+        let kv_blk = kv_tile / KV_TILE;
+        let blk_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
+        let blk_idx = params.blk_base + (blk_batch * params.blk_nblk1 + q_blk) * params.blk_nblk0 + kv_blk;
+        let blk_state_local = blk[blk_idx];
+#else
+        let blk_state_local = 1u;
+#endif
+        if (local_id.x == 0u) {
+            blk_state_wg = blk_state_local;
+        }
+        workgroupBarrier();
+        let blk_state = blk_state_wg;
+        let skip_tile = blk_state == 0u;
+        for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
+            inter_shmem[elem_idx] = f16(0.0);
+        }
+
+      // load k tile into shared memory
+#if defined(KV_Q4_0)
+      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
+          let blck_idx = elem_idx / BLOCK_SIZE;
+          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
+          let k_row = blck_idx / BLOCKS_K;
+          let global_k_row = kv_tile + k_row;
+          let block_k = blck_idx % BLOCKS_K;
+          let row_offset = k_row * HEAD_DIM_QK;
+
+          if (global_k_row < params.seq_len_kv) {
+              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
+              let base_idx = global_block_idx * F16_PER_BLOCK;
+              let d = K[base_idx];
+              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
+                  let q_0 = K[base_idx + 1u + block_offset + j];
+                  let q_1 = K[base_idx + 1u + block_offset + j + 1];
+                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
+                  for (var k = 0u; k < 4u; k++) {
+                      let q_byte = get_byte(q_packed, k);
+                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
+                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
+                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
+                      kv_shmem[row_offset + idx] = q_lo;
+                      kv_shmem[row_offset + idx + 16u] = q_hi;
+                  }
+              }
+          }
+      }
+#elif defined(KV_Q8_0)
+      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
+          let blck_idx = elem_idx / BLOCK_SIZE;
+          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
+          let k_row = blck_idx / BLOCKS_K;
+          let global_k_row = kv_tile + k_row;
+          let block_k = blck_idx % BLOCKS_K;
+          let row_offset = k_row * HEAD_DIM_QK;
+
+          if (global_k_row < params.seq_len_kv) {
+              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
+              let base_idx = global_block_idx * F16_PER_BLOCK;
+              let d = K[base_idx];
+              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
+                  let q_0 = K[base_idx + 1u + block_offset + j];
+                  let q_1 = K[base_idx + 1u + block_offset + j + 1];
+                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
+                  for (var k = 0u; k < 4u; k++) {
+                      let q_byte = get_byte_i32(q_packed, k);
+                      let q_val = f16(q_byte) * d;
+                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
+                      kv_shmem[row_offset + idx] = q_val;
+                  }
+              }
+          }
+      }
+#elif defined(KV_DIRECT)
+      // Direct global loads for KV
+#else
+      for (var elem_idx = local_id.x * 4u; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * 4u) {
+          let k_row = elem_idx / HEAD_DIM_QK;
+          let k_col = elem_idx % HEAD_DIM_QK;
+          let global_k_row = kv_tile + k_row;
+          let global_k_row_offset = k_head_offset + global_k_row * params.stride_k1;
+          let in_bounds = global_k_row < params.seq_len_kv && (k_col + 3u) < HEAD_DIM_QK;
+          let vec_idx = (global_k_row_offset + k_col) >> 2u;
+          let k4 = select(vec4<KV_TYPE>(0.0), K[vec_idx], in_bounds);
+          kv_shmem[elem_idx + 0u] = f16(k4.x);
+          kv_shmem[elem_idx + 1u] = f16(k4.y);
+          kv_shmem[elem_idx + 2u] = f16(k4.z);
+          kv_shmem[elem_idx + 3u] = f16(k4.w);
+      }
+#endif
+
+      workgroupBarrier();
+
+      // accumulate q block * k block into registers across the entire KV tile
+      if (!skip_tile) {
+        let num_of_threads = subgroup_size / VEC_NE;
+        let tx = sg_inv_id % num_of_threads;
+        let ty = sg_inv_id / num_of_threads;
+          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
+              let global_q_row = q_row_start + q_tile_row;
+              if (global_q_row >= params.seq_len_q) {
+                  continue;
+              }
+              let local_q_row_offset = q_tile_row * HEAD_DIM_QK;
+
+              for (var kv_base : u32 = 0u; kv_base < KV_TILE; kv_base += VEC_NE) {
+                  let kv_idx = kv_base + ty;
+                  var partial_sum: f32 = 0.0;
+                  let kv_valid = kv_idx < KV_TILE && (kv_tile + kv_idx) < params.seq_len_kv;
+                  if (kv_valid) {
+                    for (var i = tx; i < (HEAD_DIM_QK / 4u); i += num_of_threads) {
+                        let q_off = local_q_row_offset + i * 4u;
+
+                        let qv = vec4<f32>(
+                            f32(q_shmem[q_off + 0u]),
+                            f32(q_shmem[q_off + 1u]),
+                            f32(q_shmem[q_off + 2u]),
+                            f32(q_shmem[q_off + 3u]));
+#ifdef KV_DIRECT
+                        let idx = k_head_offset + (kv_tile + kv_idx) * params.stride_k1 + (i * 4u);
+                        let kv = vec4<f32>(K[idx >> 2u]);
+#else
+                        let idx = kv_idx * HEAD_DIM_QK + (i * 4u);
+                        let kv = vec4<f32>(
+                            f32(kv_shmem[idx + 0u]),
+                            f32(kv_shmem[idx + 1u]),
+                            f32(kv_shmem[idx + 2u]),
+                            f32(kv_shmem[idx + 3u]));
+#endif
+                        partial_sum += dot(qv, kv);
+                    }
+                  }
+                  var sum = partial_sum;
+                  // Reduce over tx threads (NL) for this ty stripe.
+                  var tx_delta = num_of_threads >> 1u;
+                  loop {
+                      if (tx_delta == 0u) {
+                          break;
+                      }
+                      let sh = subgroupShuffleDown(sum, tx_delta);
+                      if (tx < tx_delta) {
+                          sum += sh;
+                      }
+                      tx_delta >>= 1u;
+                  }
+
+                  let sum_bcast = subgroupShuffle(sum, num_of_threads * ty);
+                  if (tx == 0u && kv_valid) {
+                      let dst_idx = q_tile_row * KV_TILE + kv_idx;
+                      inter_shmem[dst_idx] = f16(sum_bcast);
+                  }
+              }
+          }
+      }
+
+
+#ifdef MASK
+      let apply_mask = !skip_tile && (blk_state != 2u);
+      if (apply_mask) {
+          // load mask tile into shared memory for this KV block
+          for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
+              let mask_row = elem_idx / KV_TILE;
+              let mask_col = elem_idx % KV_TILE;
+              let global_q_row = q_row_start + mask_row;
+              let global_k_col = kv_tile + mask_col;
+              let mask_in_bounds = global_q_row < params.seq_len_q && global_k_col < params.seq_len_kv;
+              let mask_idx = mask_global_offset + mask_row * params.seq_len_kv + global_k_col;
+              mask_shmem[elem_idx] = select(0.0, mask[mask_idx], mask_in_bounds);
+          }
+      }
+#else
+      let apply_mask = false;
+#endif
+
+      workgroupBarrier();
+
+      // online softmax
+      if (!skip_tile) {
+          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
+              let global_q_row = q_row_start + q_tile_row;
+              if (global_q_row >= params.seq_len_q) {
+                  break;
+              }
+
+              var prev_max = row_max_shmem[q_tile_row];
+              var final_max = prev_max;
+              // pass 1: compute final max across the full KV tile in chunks
+              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
+                  let kv_idx = kv_offset + sg_inv_id;
+                  let kv_valid = kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE;
+                  let softmax_term = select(FLOAT_MIN,
+                                            calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask),
+                                            kv_valid);
+                  final_max = subgroupMax(max(final_max, softmax_term));
+              }
+
+              var total_exp_term: f32 = 0.0;
+              // pass 2: compute exp sum and write P using final_max
+              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
+                  let kv_idx = kv_offset + sg_inv_id;
+                  let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask);
+                  let cur_p = select(0.0,
+                                     exp(softmax_term - final_max),
+                                     kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE);
+                  total_exp_term += subgroupAdd(cur_p);
+                  if (kv_idx < KV_TILE) {
+                      inter_shmem[kv_idx + q_tile_row * KV_TILE] = f16(cur_p);
+                  }
+              }
+
+              let cur_exp = exp(prev_max - final_max);
+
+              if (sg_inv_id == 0) {
+                  row_max_shmem[q_tile_row] = final_max;
+                  exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * cur_exp + total_exp_term;
+              }
+
+              for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
+                  let idx = q_tile_row * HEAD_DIM_V + elem_idx;
+                  o_shmem[idx] = f16(f32(o_shmem[idx]) * cur_exp);
+              }
+          }
+      }
+
+      // load v tile into shared memory
+#if defined(KV_Q4_0)
+      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
+          let blck_idx = elem_idx / BLOCK_SIZE;
+          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
+          let v_row = blck_idx / BLOCKS_V;
+          let global_v_row = kv_tile + v_row;
+          let block_k = blck_idx % BLOCKS_V;
+          let row_offset = v_row * HEAD_DIM_V;
+
+          if (global_v_row < params.seq_len_kv) {
+              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
+              let base_idx = global_block_idx * F16_PER_BLOCK;
+              let d = V[base_idx];
+              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
+                  let q_0 = V[base_idx + 1u + block_offset + j];
+                  let q_1 = V[base_idx + 1u + block_offset + j + 1];
+                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
+                  for (var k = 0u; k < 4u; k++) {
+                      let q_byte = get_byte(q_packed, k);
+                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
+                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
+                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
+                      kv_shmem[row_offset + idx] = q_lo;
+                      kv_shmem[row_offset + idx + 16u] = q_hi;
+                  }
+              }
+          }
+      }
+#elif defined(KV_Q8_0)
+      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
+          let blck_idx = elem_idx / BLOCK_SIZE;
+          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
+          let v_row = blck_idx / BLOCKS_V;
+          let global_v_row = kv_tile + v_row;
+          let block_k = blck_idx % BLOCKS_V;
+          let row_offset = v_row * HEAD_DIM_V;
+
+          if (global_v_row < params.seq_len_kv) {
+              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
+              let base_idx = global_block_idx * F16_PER_BLOCK;
+              let d = V[base_idx];
+              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
+                  let q_0 = V[base_idx + 1u + block_offset + j];
+                  let q_1 = V[base_idx + 1u + block_offset + j + 1];
+                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
+                  for (var k = 0u; k < 4u; k++) {
+                      let q_byte = get_byte_i32(q_packed, k);
+                      let q_val = f16(q_byte) * d;
+                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
+                      kv_shmem[row_offset + idx] = q_val;
+                  }
+              }
+          }
+      }
+#elif defined(KV_DIRECT)
+      // Direct global loads for KV
+#else
+      for (var elem_idx = local_id.x * 4u; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * 4u) {
+          let v_row = elem_idx / HEAD_DIM_V;
+          let v_col = elem_idx % HEAD_DIM_V;
+          let global_v_row = kv_tile + v_row;
+          let global_v_row_offset = v_head_offset + global_v_row * params.stride_v1;
+          let in_bounds = global_v_row < params.seq_len_kv && (v_col + 3u) < HEAD_DIM_V;
+          let vec_idx = (global_v_row_offset + v_col) >> 2u;
+          let v4 = select(vec4<KV_TYPE>(0.0), V[vec_idx], in_bounds);
+          kv_shmem[elem_idx + 0u] = f16(v4.x);
+          kv_shmem[elem_idx + 1u] = f16(v4.y);
+          kv_shmem[elem_idx + 2u] = f16(v4.z);
+          kv_shmem[elem_idx + 3u] = f16(v4.w);
+      }
+#endif
+
+      workgroupBarrier();
+
+      if (!skip_tile) {
+          // we have P (Q_TILE x KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
+          // we want to compute O += P * V across the full KV tile
+          let ne_threads : u32 = VEC_NE;
+          let nl_threads = max(1u, subgroup_size / ne_threads);
+          let tx_pv = sg_inv_id % nl_threads;
+          let ty_pv = sg_inv_id / nl_threads;
+          for (var q_tile_row = subgroup_id;
+               q_tile_row < Q_TILE;
+               q_tile_row += num_subgroups) {
+              for (var vec_col = tx_pv; vec_col < (HEAD_DIM_V / 4u); vec_col += nl_threads) {
+                  var lo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
+                  for (var cc = 0u; cc < KV_TILE / ne_threads; cc += 1u) {
+                      let kv_idx = cc * ne_threads + ty_pv;
+                      let v_row = kv_tile + kv_idx;
+                      if (v_row >= params.seq_len_kv) {
+                          continue;
+                      }
+
+                      let p = f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]);
+#ifdef KV_DIRECT
+                      let v_idx = v_head_offset + v_row * params.stride_v1 + vec_col * 4u;
+                      let v4 = vec4<f32>(V[v_idx >> 2u]);
+#else
+                      let v_idx = kv_idx * HEAD_DIM_V + vec_col * 4u;
+                      let v4 = vec4<f32>(
+                          f32(kv_shmem[v_idx + 0u]),
+                          f32(kv_shmem[v_idx + 1u]),
+                          f32(kv_shmem[v_idx + 2u]),
+                          f32(kv_shmem[v_idx + 3u]));
+#endif
+                      lo += p * v4;
+                  }
+
+                  var lo_x = lo.x;
+                  var lo_y = lo.y;
+                  var lo_z = lo.z;
+                  var lo_w = lo.w;
+                  // Reduce over ty threads (NE) for this tx thread.
+                  var ty_delta = ne_threads >> 1u;
+                  loop {
+                      if (ty_delta == 0u) {
+                          break;
+                      }
+                      let thread_delta = ty_delta * nl_threads;
+                      let shx = subgroupShuffleDown(lo_x, thread_delta);
+                      let shy = subgroupShuffleDown(lo_y, thread_delta);
+                      let shz = subgroupShuffleDown(lo_z, thread_delta);
+                      let shw = subgroupShuffleDown(lo_w, thread_delta);
+                      if (ty_pv < ty_delta) {
+                          lo_x += shx;
+                          lo_y += shy;
+                          lo_z += shz;
+                          lo_w += shw;
+                      }
+                      ty_delta >>= 1u;
+                  }
+
+                  if (ty_pv == 0u) {
+                      let elem_base = vec_col * 4u;
+                      let o_base_idx = q_tile_row * HEAD_DIM_V + elem_base;
+                      o_shmem[o_base_idx + 0u] = f16(f32(o_shmem[o_base_idx + 0u]) + lo_x);
+                      o_shmem[o_base_idx + 1u] = f16(f32(o_shmem[o_base_idx + 1u]) + lo_y);
+                      o_shmem[o_base_idx + 2u] = f16(f32(o_shmem[o_base_idx + 2u]) + lo_z);
+                      o_shmem[o_base_idx + 3u] = f16(f32(o_shmem[o_base_idx + 3u]) + lo_w);
+                  }
+              }
+          }
+      }
+
+        workgroupBarrier();
+    }
+
+
+#ifdef SINKS
+    // Sinks are global terms and must be applied exactly once across split workgroups.
+    if (iwg == 0u) {
+        for (var q_tile_row = subgroup_id;
+             q_tile_row < Q_TILE;
+             q_tile_row += num_subgroups) {
+                let global_q_row = q_row_start + q_tile_row;
+                if (global_q_row >= params.seq_len_q) {
+                    break;
+                }
+
+                var prev_max = row_max_shmem[q_tile_row];
+
+                // for non-sink threads, exp(FLOAT_MIN) effectively zeroes out their contribution to the sum
+                let sink_val = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0);
+                let new_max = subgroupMax(max(prev_max, sink_val));
+                let max_exp = exp(prev_max - new_max);
+                let sink_exp = exp(sink_val - new_max);
+
+                let sink_exp_sum = subgroupAdd(sink_exp);
+
+                if (sg_inv_id == 0) {
+                    row_max_shmem[q_tile_row] = new_max;
+                    exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * max_exp + sink_exp_sum;
+                }
+
+            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
+                let idx = q_tile_row * HEAD_DIM_V + elem_idx;
+                o_shmem[idx] = f16(f32(o_shmem[idx]) * max_exp);
+            }
+        }
+        workgroupBarrier();
+    }
+#endif
+    let rows_per_batch = params.n_heads * params.seq_len_q;
+    for (var q_tile_row = subgroup_id;
+         q_tile_row < Q_TILE;
+         q_tile_row += num_subgroups) {
+
+        let global_q_row = q_row_start + q_tile_row;
+        if (global_q_row >= params.seq_len_q) { break; }
+
+        if (params.nwg == 1u) {
+            let exp_sum = exp_sum_shmem[q_tile_row];
+            let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
+            let row_base: u32 =
+                params.offset_dst + batch_idx * dst3_stride + global_q_row * dst2_stride + head_idx * HEAD_DIM_V;
+
+            for (var elem_base = sg_inv_id * 4u; elem_base < HEAD_DIM_V; elem_base += subgroup_size * 4u) {
+                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
+                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
+                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
+                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);
+
+                let v = vec4<f32>(
+                    f32(o_shmem[i0]) * scale,
+                    f32(o_shmem[i1]) * scale,
+                    f32(o_shmem[i2]) * scale,
+                    f32(o_shmem[i3]) * scale
+                );
+
+                let dst_vec_index: u32 = (row_base + elem_base) >> 2u;
+                dst[dst_vec_index] = v;
+            }
+        } else {
+            let rid = batch_idx * rows_per_batch + head_idx * params.seq_len_q + global_q_row;
+            let tmp_row_data_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg) + iwg * HEAD_DIM_V;
+            let tmp_row_stats_base = params.tmp_stats_base + rid * (2u * params.nwg) + 2u * iwg;
+
+            for (var elem_base = sg_inv_id * 4u;
+                elem_base < HEAD_DIM_V;
+                elem_base += subgroup_size * 4u) {
+
+                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
+                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
+                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
+                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);
+
+                let tbase = tmp_row_data_base + elem_base;
+                tmp[tbase + 0u] = f32(o_shmem[i0]);
+                tmp[tbase + 1u] = f32(o_shmem[i1]);
+                tmp[tbase + 2u] = f32(o_shmem[i2]);
+                tmp[tbase + 3u] = f32(o_shmem[i3]);
+            }
+
+            if (sg_inv_id == 0u) {
+                tmp[tmp_row_stats_base + 0u] = exp_sum_shmem[q_tile_row];
+                tmp[tmp_row_stats_base + 1u] = row_max_shmem[q_tile_row];
+            }
+        }
+    }
+}
diff --git src/ggml-zendnn/CMakeLists.txt src/ggml-zendnn/CMakeLists.txt
index 9bdb4e83..4f321a25 100644
--- src/ggml-zendnn/CMakeLists.txt
+++ src/ggml-zendnn/CMakeLists.txt
@@ -28,7 +28,7 @@ if (NOT ZENDNN_ROOT OR ZENDNN_ROOT STREQUAL "" OR ZENDNN_ROOT STREQUAL "OFF")
     ExternalProject_Add(
         zendnn
         GIT_REPOSITORY https://github.com/amd/ZenDNN.git
-        GIT_TAG a18adf8c605fb5f5e52cefd7eda08a7b18febbaf    # ZenDNN-2026-WW08
+        GIT_TAG f79f7321a1add65ced6397a6bfab7edba6e3e14e    # ZenDNN-2026-WW13
         PREFIX      ${ZENDNN_PREFIX}
         SOURCE_DIR  ${ZENDNN_SOURCE_DIR}
         BINARY_DIR  ${ZENDNN_BUILD_DIR}
diff --git src/ggml-zendnn/ggml-zendnn.cpp src/ggml-zendnn/ggml-zendnn.cpp
index c8760304..37730372 100644
--- src/ggml-zendnn/ggml-zendnn.cpp
+++ src/ggml-zendnn/ggml-zendnn.cpp
@@ -190,6 +190,170 @@ static void ggml_zendnn_compute_forward_mul_mat(
     }
 }
 
+struct mmid_row_mapping {
+    int32_t i1;
+    int32_t i2;
+};
+
+static void ggml_zendnn_compute_forward_mul_mat_id(
+    ggml_backend_zendnn_context * ctx,
+    ggml_tensor * dst) {
+
+    const ggml_tensor * src0 = dst->src[0];  // expert weights
+    const ggml_tensor * src1 = dst->src[1];  // inputs
+    const ggml_tensor * ids  = dst->src[2];  // expert ids
+
+    GGML_TENSOR_BINARY_OP_LOCALS
+
+    // exit for no tokens to process
+    if (ne2 == 0 || ne11 == 0) {
+        return;
+    }
+
+    ggml_type         const vec_dot_type = src0->type;
+    ggml_from_float_t const from_float = ggml_get_type_traits(vec_dot_type)->from_float_ref;
+
+    // we don't support permuted src0 or src1
+    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
+    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
+
+    // dst cannot be transposed or permuted
+    GGML_ASSERT(nb0 == sizeof(float));
+    GGML_ASSERT(nb0 <= nb1);
+    GGML_ASSERT(nb1 <= nb2);
+    GGML_ASSERT(nb2 <= nb3);
+
+    GGML_ASSERT(ne03 == 1);
+    GGML_ASSERT(ne13 == 1);
+    GGML_ASSERT(ne3  == 1);
+
+    // row groups
+    const int n_ids = ids->ne[0]; // n_expert_used
+    const int n_as  = ne02;       // n_experts
+
+    std::vector<int64_t> matrix_row_counts(n_as, 0);
+    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);
+
+    int64_t max_rows = 0;
+    // group rows by expert (preprocessing step)
+    for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
+        for (int id = 0; id < n_ids; ++id) {
+            const int32_t i02 = *(const int32_t *)((const char *)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
+
+            GGML_ASSERT(i02 >= 0 && i02 < n_as);
+
+            matrix_rows[i02].push_back({id, iid1});
+            matrix_row_counts[i02]++;
+            if (matrix_row_counts[i02] > max_rows) {
+                max_rows = matrix_row_counts[i02];
+            }
+        }
+    }
+
+    if (max_rows == 0) {
+        return; // no rows to process
+    }
+
+    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
+
+    // size for converting src1 rows to vec_dot_type if needed
+    const size_t nbw1 = row_size;
+    const size_t nbw2 = nbw1 * ne11;
+    const size_t nbw3 = nbw2 * ne12;
+    const size_t src1_conv_size = (src1->type != vec_dot_type) ? ne13 * nbw3 : 0;
+
+    // size for MoE gather/scatter buffers
+    const size_t wdata_cur_size = max_rows * row_size;
+    const size_t dst_cur_size = max_rows * ggml_row_size(dst->type, ne01);
+
+    // allocate single buffer for all needs
+    const size_t total_size = src1_conv_size + wdata_cur_size + dst_cur_size;
+    if (ctx->work_size < total_size) {
+        ctx->work_data.reset(new char[total_size]);
+        ctx->work_size = total_size;
+    }
+
+    // partition the buffer
+    char * work_data = ctx->work_data.get();
+    char * wdata_cur = work_data + src1_conv_size;
+    char * dst_cur = wdata_cur + wdata_cur_size;
+
+    if (src1->type != vec_dot_type) {
+        GGML_ASSERT(src1->type == GGML_TYPE_F32);
+
+        #pragma omp parallel for collapse(3) num_threads(ctx->n_threads) schedule(static)
+        for (int64_t i13 = 0; i13 < ne13; ++i13) {
+            for (int64_t i12 = 0; i12 < ne12; ++i12) {
+                for (int64_t i11 = 0; i11 < ne11; ++i11) {
+                    const float * src1_f32 = (float *)((char *)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
+                    void * src1_conv = (char *)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
+                    from_float(src1_f32, src1_conv, ne10);
+                }
+            }
+        }
+    }
+
+    const void * wdata = src1->type == vec_dot_type ? src1->data : work_data;
+
+    // process each expert with gather -> gemm -> scatter pattern
+    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
+        const int64_t cne1 = matrix_row_counts[cur_a];
+
+        if (cne1 == 0) {
+            continue;
+        }
+
+        const char * src0_cur = (const char *) src0->data + cur_a*nb02;
+
+        // gather input rows for this expert
+        #pragma omp parallel for num_threads(ctx->n_threads) schedule(static)
+        for (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
+            const mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
+            const int64_t id = row_mapping.i1;
+            const int64_t i11 = id % ne11;
+            const int64_t i12 = row_mapping.i2;
+
+            std::memcpy(
+                wdata_cur + ir1 * row_size,
+                (const char *) wdata + (i11 + i12*ne11) * row_size,
+                row_size
+            );
+        }
+
+        // batched gemm for all tokens in this expert
+        if (!ggml_zendnn_sgemm(ctx,
+                              ne01,       // m
+                              cne1,       // n
+                              ne10,       // k
+                              src0_cur,
+                              ne00,       // lda
+                              wdata_cur,
+                              ne10,       // ldb
+                              dst_cur,
+                              ne01,       // ldc
+                              src0->type,
+                              vec_dot_type,
+                              dst->type)) {
+            GGML_ABORT("%s: ZenDNN sgemm failed\n", __func__);
+        }
+
+        // scatter output rows to destination
+        #pragma omp parallel for num_threads(ctx->n_threads) schedule(static)
+        for (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
+            const mmid_row_mapping & row_mapping = matrix_rows[cur_a][ir1];
+            const int64_t id = row_mapping.i1;
+            const int64_t i1 = id;
+            const int64_t i2 = row_mapping.i2;
+
+            std::memcpy(
+                (char *) dst->data + i1*nb1 + i2*nb2,
+                dst_cur + ir1 * ggml_row_size(dst->type, ne01),
+                ggml_row_size(dst->type, ne01)
+            );
+        }
+    }
+}
+
 // backend interface
 
 static const char * ggml_backend_zendnn_get_name(ggml_backend_t backend) {
@@ -218,6 +382,9 @@ static ggml_status ggml_backend_zendnn_graph_compute(ggml_backend_t backend, ggm
             case GGML_OP_MUL_MAT:
                 ggml_zendnn_compute_forward_mul_mat(ctx, node);
                 break;
+            case GGML_OP_MUL_MAT_ID:
+                ggml_zendnn_compute_forward_mul_mat_id(ctx, node);
+                break;
             case GGML_OP_NONE:
             case GGML_OP_RESHAPE:
             case GGML_OP_VIEW:
@@ -361,6 +528,7 @@ static bool ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, const
             return true;
 
         case GGML_OP_MUL_MAT:
+        case GGML_OP_MUL_MAT_ID:
         {
             const ggml_tensor * weights = op->src[0];
             const ggml_tensor * inputs = op->src[1];
@@ -374,6 +542,17 @@ static bool ggml_backend_zendnn_device_supports_op(ggml_backend_dev_t dev, const
                 ne0 < min_batch || ne1 < min_batch || ne10 < min_batch) {
                     return false;
             }
+            // MUL_MAT_ID performs best with a moderate number of experts due to its
+            // gather + batched matmul + scatter approach. Future versions will leverage
+            // ZenDNN's grouped_gemm for better scalability with larger expert counts:
+            // https://github.com/amd/ZenDNN/blob/main/docs/operator/lowoha_group_gemm_operator.md
+            if (op->op == GGML_OP_MUL_MAT_ID) {
+                const int64_t n_experts = weights->ne[2];
+                const int64_t max_experts = 32;
+                if (n_experts > max_experts) {
+                    return false;
+                }
+            }
             switch (weights->type) {
                 case GGML_TYPE_F32:
                 case GGML_TYPE_BF16:
