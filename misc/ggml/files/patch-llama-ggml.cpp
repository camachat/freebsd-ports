diff --git src/ggml-metal/ggml-metal.cpp src/ggml-metal/ggml-metal.cpp
index 4dbf8e6f..6a836e45 100644
--- src/ggml-metal/ggml-metal.cpp
+++ src/ggml-metal/ggml-metal.cpp
@@ -918,6 +918,10 @@ ggml_backend_reg_t ggml_backend_metal_reg(void) {
         static std::vector<ggml_backend_device_ptr> devs;
 
         if (!initialized) {
+            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
+            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
+            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);
+
             static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());
 
             for (int i = 0; i < g_devices; ++i) {
diff --git src/ggml-openvino/ggml-decoder.cpp src/ggml-openvino/ggml-decoder.cpp
index 0938d227..5095e799 100644
--- src/ggml-openvino/ggml-decoder.cpp
+++ src/ggml-openvino/ggml-decoder.cpp
@@ -19,7 +19,6 @@
 #include <iomanip>
 #include <map>
 #include <memory>
-#include <mutex>
 #include <openvino/core/dimension.hpp>
 #include <openvino/core/except.hpp>
 #include <openvino/core/node.hpp>
@@ -207,8 +206,22 @@ int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
         break;
     }
     case GGML_OP_ROPE: {
+        const int mode = node->op_params[2];
+        switch (mode) {
+       case GGML_ROPE_TYPE_NEOX: {
+            op_case = 0x00010000;
+            break;
+        }
+       case GGML_ROPE_TYPE_IMROPE: {
+            op_case = 0x00020000;
+            break;
+        }
+        default:
+            op_case = 0x00000000;
+            break;
+        }
         if (node->src[0]->op == GGML_OP_VIEW) {
-            op_case = 2;
+            op_case = (op_case | 0x00000002);
         }
         break;
     }
@@ -573,9 +586,6 @@ std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() const
 }
 
 std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, bool naive) {
-    static std::mutex weights_mutex;
-    std::lock_guard<std::mutex> lock(weights_mutex);
-
     std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
     auto * nodes = cgraph->nodes;
     auto n_nodes = cgraph->n_nodes;
diff --git src/ggml-openvino/ggml-openvino-extra.cpp src/ggml-openvino/ggml-openvino-extra.cpp
index cc3cb458..4140136a 100644
--- src/ggml-openvino/ggml-openvino-extra.cpp
+++ src/ggml-openvino/ggml-openvino-extra.cpp
@@ -6,6 +6,7 @@
 #include <cstring>
 #include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
 #include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
+#include <openvino/runtime/properties.hpp>
 #include <optional>
 
 ov::Core & ov_singleton_core() {
@@ -42,11 +43,13 @@ void ggml_openvino_device_config::init() {
             {"NPUW_DQ",                           "YES"   },
             {"NPUW_DQ_FULL",                      "NO"    },
         };
-        if (cache_dir) {
+        if (cache_dir && strlen(cache_dir) > 0) {
             compile_config["NPUW_CACHE_DIR"] = cache_dir;
+            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
         }
-    } else if (cache_dir) {
-        ov_singleton_core().set_property(ov::cache_dir(cache_dir));
+    } else if (cache_dir && strlen(cache_dir) > 0) {
+        compile_config.insert(ov::cache_dir(cache_dir));
+        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
     }
 
     // Initialize remote context with queue sharing for GPU
@@ -259,10 +262,12 @@ ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_ten
             layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
             int64_t n_blocks = n_elements / layout.weights_per_block;
             layout.scales_size = n_blocks * sizeof(uint16_t);
-            // For symmetric quantization, we only need one zp value (not one per block)
-            // Zero points are stored in U4 or U8 format matching the weight type
-            size_t n_zp_elements = layout.is_symmetric ? 1 : n_blocks;
-            layout.zp_size = layout.is_u4 ? ((n_zp_elements + 1) / 2) : n_zp_elements;
+            // For symmetric quantization, no zp needed (weights stored as signed)
+            if (layout.is_symmetric) {
+                layout.zp_size = 0;
+            } else {
+                layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
+            }
 
             layout.weights_offset = 0;
             layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
@@ -313,10 +318,12 @@ ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_ten
     // Scales: F16 per block
     int64_t n_blocks = n_elements / layout.weights_per_block;
     layout.scales_size = n_blocks * sizeof(uint16_t);  // F16 = 2 bytes
-    // Zero points: U4 or U8 matching weight type
-    // For symmetric quantization, we only need one zp value (not one per block)
-    size_t n_zp_elements = layout.is_symmetric ? 1 : n_blocks;
-    layout.zp_size = layout.is_u4 ? ((n_zp_elements + 1) / 2) : n_zp_elements;
+    // For symmetric quantization, no zp needed (weights stored as signed)
+    if (layout.is_symmetric) {
+        layout.zp_size = 0;
+    } else {
+        layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
+    }
 
     // Layout in buffer: [weights | scales | zp] with alignment
     layout.weights_offset = 0;
diff --git src/ggml-openvino/ggml-openvino.cpp src/ggml-openvino/ggml-openvino.cpp
index 0c8d3508..4f3ebf25 100644
--- src/ggml-openvino/ggml-openvino.cpp
+++ src/ggml-openvino/ggml-openvino.cpp
@@ -145,13 +145,18 @@ static void * ggml_backend_openvino_buffer_get_base(ggml_backend_buffer_t buffer
     return ctx->data;
 }
 
+static bool is_stateful_enabled() {
+    static const auto * stateful = getenv("GGML_OPENVINO_STATEFUL_EXECUTION");
+    return stateful && *stateful != '\0' && strcmp(stateful, "0") != 0;
+}
+
 static enum ggml_status ggml_backend_openvino_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
     // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
     ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
 
     // Put kvcache on device memory for GPU (NPU memory is too small even for kvcache)
     if (strncmp(tensor->name, "cache_", 6) == 0 && !ctx->is_remote && ggml_openvino_get_device_name() == "GPU" &&
-        !getenv("GGML_OPENVINO_STATEFUL_EXECUTION")) {
+        !is_stateful_enabled()) {
         GGML_ASSERT(ctx->tensor_extras.empty());
         auto device = ctx->device;
         auto size = ctx->size;
@@ -600,6 +605,14 @@ bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft) {
 
 static void ggml_backend_openvino_free(ggml_backend_t backend) {
     ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
+
+    if (ctx->runtime_context) {
+        auto r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
+        if (--r_ctx->backend_count == 0) {
+            r_ctx->clear_caches();
+        }
+    }
+
     delete ctx;
     delete backend;
 }
@@ -644,7 +657,12 @@ static ggml_guid_t ggml_backend_openvino_guid(void) {
 }
 
 static std::shared_ptr<ov_runtime_context> get_ov_runtime_context_ptr() {
-    static std::shared_ptr<ov_runtime_context> r_ctx = std::make_shared<ov_runtime_context>();
+    static std::shared_ptr<ov_runtime_context> r_ctx = [] {
+        auto ctx = std::make_shared<ov_runtime_context>();
+        ctx->device = ggml_openvino_get_device_name();
+        ctx->stateful = is_stateful_enabled() && !ggml_openvino_is_npu();
+        return ctx;
+    }();
     return r_ctx;
 }
 
@@ -669,8 +687,7 @@ GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device) {
     }
 
     std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
-    r_ctx->device = ggml_openvino_get_device_name();
-    r_ctx->stateful = getenv("GGML_OPENVINO_STATEFUL_EXECUTION") && !ggml_openvino_is_npu();
+    r_ctx->backend_count++;
 
     ggml_backend_t openvino_backend = new ggml_backend{
         /* .guid      = */ ggml_backend_openvino_guid(),
@@ -883,7 +900,7 @@ static bool is_op_unsupported_case(const ggml_tensor * op) {
         const int32_t * op_params = op->op_params;
         const int n_dims = op_params[1];
         const int mode = op_params[2];
-        if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
+        if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_IMROPE) {
             // GGML_LOG_WARN("OpenVINO backend does not support ROPE with mode %d\n", mode);
             return true;
         }
@@ -896,14 +913,6 @@ static bool is_op_unsupported_case(const ggml_tensor * op) {
             // GGML_LOG_WARN("OpenVINO backend does not support ROPE with type %s\n", ggml_type_name(op->type));
             return true;
         }
-        float freq_scale;
-        float ext_factor;
-        memcpy(&freq_scale, op_params + 6, sizeof(float));
-        memcpy(&ext_factor, op_params + 7, sizeof(float));
-        if (ext_factor != 0.0f) {
-            // GGML_LOG_WARN("OpenVINO backend does not support ROPE with ext_factor %f != 0.0f\n", ext_factor);
-            return true;
-        }
         if (op->src[0]->op == GGML_OP_VIEW) {
             if (op->src[0]->view_src->ne[1] != op->src[0]->ne[2]) {
                 // GGML_LOG_WARN(
@@ -913,6 +922,12 @@ static bool is_op_unsupported_case(const ggml_tensor * op) {
                 return true;
             }
         }
+        if (mode == GGML_ROPE_TYPE_IMROPE &&
+            (op->src[2] != 0 || ((const float *) op_params)[6] != 1 || ((const float *) op_params)[7] != 0 ||
+             ((const float *) op_params)[8] != 1)) {
+            // GGML_LOG_WARN("OpenVINO backend does not support IMROPE with freq_factors, freq_scale, ext_factor, and attn_factor\n");
+            return true;
+        }
         break;
     }
     default:
@@ -942,6 +957,7 @@ static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, con
                                                  // GGML_OP_SOFT_MAX,
                                                  GGML_OP_SET_ROWS, GGML_OP_FLASH_ATTN_EXT, GGML_OP_CPY};
     static const std::set<ggml_unary_op> supported_unary_ops{
+        GGML_UNARY_OP_GELU,
         GGML_UNARY_OP_SILU,
     };
     static const std::set<ggml_glu_op> supported_glu_ops{
diff --git src/ggml-openvino/ggml-quants.cpp src/ggml-openvino/ggml-quants.cpp
index dbf38646..57d66df4 100644
--- src/ggml-openvino/ggml-quants.cpp
+++ src/ggml-openvino/ggml-quants.cpp
@@ -46,6 +46,7 @@ void unpack_32_4(const uint8_t * data, uint8_t * dst) {
 
 // Extracts (weight, scales, zp) from Q4_0 tensors.
 // Data layout is: |16 bit scale|32 x 4bit weights|.
+// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
 void extract_q4_0_data(const ggml_tensor * tensor,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
@@ -55,28 +56,32 @@ void extract_q4_0_data(const ggml_tensor * tensor,
     auto * data = static_cast<uint8_t *>(tensor->data);
     auto * weights = static_cast<uint8_t *>(weights_arr.data());
     auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
-    auto * zp = static_cast<uint8_t *>(zp_arr.data());
-
-    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization
 
-    // For Q4_0, zero point is always 8
-    if (is_scalar_zp) {
-        zp[0] = 8 | (8 << 4);  // Pack two 4-bit values
-    }
+    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path
 
-    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
-        scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
-        // For asymmetric quantization, compute per-block zero points
-        if (!is_scalar_zp) {
+    if (!is_symmetric) {
+        auto * zp = static_cast<uint8_t *>(zp_arr.data());
+        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
+            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
             // Pack two 4-bit zero points per byte
             if (i % 2 == 0) {
                 zp[i / 2] = 8;          // Lower nibble
             } else {
                 zp[i / 2] |= (8 << 4);  // Upper nibble
             }
-        }
-        unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
-    });
+            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
+        });
+    } else {
+        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
+        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
+            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
+            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
+            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
+            for (int j = 0; j < 16; ++j) {
+                weights[i * 16 + j] ^= 0x88;
+            }
+        });
+    }
 }
 
 // Extracts (weight, scales, zp) from Q4_1 tensors.
@@ -123,6 +128,7 @@ void extract_q4_1_data(const ggml_tensor * tensor,
 
 // Extracts (weight, scales, zp) from Q8_0 tensors.
 // Data layout is: |16 bit scale|32 x 8bit weights|.
+// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
 void extract_q8_0_data(const ggml_tensor * tensor,
                        ov::Tensor & weights_arr,
                        ov::Tensor & scales_arr,
@@ -133,29 +139,30 @@ void extract_q8_0_data(const ggml_tensor * tensor,
     auto * data = static_cast<uint8_t *>(tensor->data);
     auto * weights = static_cast<uint8_t *>(weights_arr.data());
     auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
-    auto * zp = static_cast<uint8_t *>(zp_arr.data());
-
-    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization
 
-    // For Q8_0, zero point is always 128
-    if (is_scalar_zp) {
-        zp[0] = 128;
-    }
+    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path
 
-    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
-        uint8_t * block_data = data + i * bytes_per_block;
-        scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
-        // For asymmetric quantization, store per-block zero points
-        if (!is_scalar_zp) {
+    if (!is_symmetric) {
+        auto * zp = static_cast<uint8_t *>(zp_arr.data());
+        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
+            uint8_t * block_data = data + i * bytes_per_block;
+            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
             zp[i] = 128;
-        }
-        for (size_t j = 0; j < weights_per_block; ++j) {
-            uint8_t x = block_data[j + 2];  // j+2 to skip the scale bytes.
-            // Original data is in int8_t, so we add a bias of -128 and invert the first bit.
-            x ^= 1 << 7;
-            weights[i * weights_per_block + j] = x;
-        }
-    });
+            for (size_t j = 0; j < weights_per_block; ++j) {
+                uint8_t x = block_data[j + 2];
+                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
+                weights[i * weights_per_block + j] = x;
+            }
+        });
+    } else {
+        // Symmetric: store original int8 values directly (no unsigned bias)
+        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
+            uint8_t * block_data = data + i * bytes_per_block;
+            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
+            // Copy int8 weights as-is (the tensor element type is i8)
+            memcpy(weights + i * weights_per_block, block_data + 2, weights_per_block);
+        });
+    }
 }
 
 void unpack_256_4(const uint8_t * data, uint8_t * dst) {
@@ -256,44 +263,62 @@ void extract_q6_k_data(const ggml_tensor * tensor,
     auto * data = static_cast<uint8_t *>(tensor->data);
     auto * weights = static_cast<uint8_t *>(weights_arr.data());
     auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
-    auto * zp = static_cast<uint8_t *>(zp_arr.data());
-
-    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization
-
-    // For Q6_K, zero point is always 32
-    if (is_scalar_zp) {
-        zp[0] = 32;
-    }
-
-    ov::parallel_for(n_super_block, [&](size_t i) {
-        uint8_t * block_data = data + i * bytes_per_block;
 
-        float scale_factor =
-            static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));  // (128+64+16)/2
+    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path
 
-        for (size_t j = 0; j < 16; j++) {
-            scales[j + i * 16] =
-                ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
-            // For asymmetric quantization, store per-block zero points
-            if (!is_scalar_zp) {
+    if (!is_symmetric) {
+        auto * zp = static_cast<uint8_t *>(zp_arr.data());
+        ov::parallel_for(n_super_block, [&](size_t i) {
+            uint8_t * block_data = data + i * bytes_per_block;
+            float scale_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
+            for (size_t j = 0; j < 16; j++) {
+                scales[j + i * 16] =
+                    ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
                 zp[j + i * 16] = 32;
             }
-        }
-
-        uint8_t * ql = block_data;
-        uint8_t * qh = block_data + 128;
-
-        for (int64_t j = 0; j < 32; ++j) {
-            weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
-            weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
-            weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
-            weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
-            weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
-            weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
-            weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
-            weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
-        }
-    });
+            uint8_t * ql = block_data;
+            uint8_t * qh = block_data + 128;
+            for (int64_t j = 0; j < 32; ++j) {
+                weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
+                weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
+                weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
+                weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
+                weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
+                weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
+                weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
+                weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
+            }
+        });
+    } else {
+        // Symmetric: subtract 32 from each weight to store as signed i8
+        ov::parallel_for(n_super_block, [&](size_t i) {
+            uint8_t * block_data = data + i * bytes_per_block;
+            float scale_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
+            for (size_t j = 0; j < 16; j++) {
+                scales[j + i * 16] =
+                    ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
+            }
+            uint8_t * ql = block_data;
+            uint8_t * qh = block_data + 128;
+            auto * signed_weights = reinterpret_cast<int8_t *>(weights);
+            for (int64_t j = 0; j < 32; ++j) {
+                signed_weights[i * 256 + j] = static_cast<int8_t>((ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 32] =
+                    static_cast<int8_t>((ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 64] = static_cast<int8_t>((ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 96] =
+                    static_cast<int8_t>((ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 128] =
+                    static_cast<int8_t>((ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 160] =
+                    static_cast<int8_t>((ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 192] =
+                    static_cast<int8_t>((ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4)) - 32;
+                signed_weights[i * 256 + j + 224] =
+                    static_cast<int8_t>((ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4)) - 32;
+            }
+        });
+    }
 }
 
 static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
@@ -389,11 +414,10 @@ ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                        size_t group_size,
                                        bool use_bias) {
     ov::Shape orig_shape = weight.get_shape();
+    bool is_signed = (weight.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP
 
     // Expand dimensions for scales and zp/bias
     auto scale_shape = scales.get_shape();
-    auto zp_shape = zp.get_shape();
-    bool is_scalar_zp = zp_shape.empty();  // Symmetric quantization
 
     ov::Shape packed_shape = {orig_shape[0], orig_shape[1] / group_size, group_size};
 
@@ -403,37 +427,48 @@ ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
     } else {
         scale_shape.push_back(1);
         scales.set_shape(scale_shape);
-        // For symmetric quantization, zp remains scalar (don't resize)
-        if (!is_scalar_zp) {
+        if (!is_signed && zp.get_size() > 0) {
+            auto zp_shape = zp.get_shape();
             zp_shape.push_back(1);
             zp.set_shape(zp_shape);
         }
     }
 
-    // Create graph nodes
-    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
-                                                               static_cast<uint8_t *>(weight.data()), nullptr);
-    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
     auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);
-    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
 
     ov::Output<ov::Node> result;
-    if (use_bias && !is_scalar_zp) {
-        // Bias path: w * s + b (zp tensor holds f16 bias values)
-        auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
-        auto w_s = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
-        result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
+    if (is_signed) {
+        // Signed path: q * s (no zero point subtraction needed)
+        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
+                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
+        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
+        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
+        result = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
     } else {
-        // Zero point path: (w - zp) * s
-        auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
-        float zp_value;
-        if (ov::op::util::get_single_value(zero_point, zp_value)) {
-            zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
+        // Unsigned path
+        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
+                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
+        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
+        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
+
+        if (use_bias && zp.get_size() > 0) {
+            // Bias path: w * s + b (zp tensor holds f16 bias values)
+            auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
+            auto w_s =
+                std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
+            result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
+        } else {
+            // Zero point path: (w - zp) * s
+            auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
+            float zp_value;
+            if (ov::op::util::get_single_value(zero_point, zp_value)) {
+                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
+            }
+            auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
+            auto w_zp =
+                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
+            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
         }
-        auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
-        auto w_zp =
-            std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
-        result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
     }
 
     if (packed_shape.size() != 2) {
@@ -452,11 +487,10 @@ ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                        size_t group_size,
                                        bool use_bias) {
     ov::Shape orig_weight_shape = weight.get_shape();
+    bool is_signed = (weight.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP
 
     // Expand dimensions for scales and zp/bias
     ov::Shape scale_shape = scales.get_shape();
-    auto zp_shape = zp.get_shape();
-    bool is_scalar_zp = zp_shape.empty();  // Symmetric quantization
 
     // Create INT4 weight tensor
     ov::Shape packed_shape = {orig_weight_shape[0], orig_weight_shape[1] / group_size, group_size};
@@ -467,36 +501,48 @@ ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
     } else {
         scale_shape.push_back(1);
         scales.set_shape(scale_shape);
-        // For symmetric quantization, zp remains scalar (don't resize)
-        if (!is_scalar_zp) {
+        if (!is_signed && zp.get_size() > 0) {
+            auto zp_shape = zp.get_shape();
             zp_shape.push_back(1);
             zp.set_shape(zp_shape);
         }
     }
 
-    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
-                                                               static_cast<uint8_t *>(weight.data()), nullptr);
-    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
-    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
     auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);
 
     ov::Output<ov::Node> result;
-    if (use_bias && !is_scalar_zp) {
-        // Bias path: w * s + b (zp tensor holds f16 bias values)
-        auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
-        auto w_s = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
-        result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
+    if (is_signed) {
+        // Signed path: q * s (no zero point subtraction needed)
+        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
+                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
+        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
+        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
+        result = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
     } else {
-        // Zero point path: (w - zp) * s
-        auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zp);
-        float zp_value;
-        if (ov::op::util::get_single_value(zero_points_node, zp_value)) {
-            zero_points_node = ov::op::v0::Constant::create(zero_points_node->get_element_type(), {}, {zp_value});
+        // Unsigned path
+        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
+                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
+        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
+        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
+
+        if (use_bias && zp.get_size() > 0) {
+            // Bias path: w * s + b (zp tensor holds f16 bias values)
+            auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
+            auto w_s =
+                std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
+            result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
+        } else {
+            // Zero point path: (w - zp) * s
+            auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zp);
+            float zp_value;
+            if (ov::op::util::get_single_value(zero_points_node, zp_value)) {
+                zero_points_node = ov::op::v0::Constant::create(zero_points_node->get_element_type(), {}, {zp_value});
+            }
+            auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);
+            auto w_zp =
+                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
+            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
         }
-        auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);
-        auto w_zp =
-            std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
-        result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
     }
 
     if (packed_shape.size() != 2) {
@@ -699,24 +745,32 @@ OvWeight process_weight_tensor(const ggml_tensor * tensor, const void * data, vo
 
     // Quantized path (normal extraction or quantized requant)
     // Create weight/scale/zp tensors - shared between both paths
-    ov::element::Type weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
+    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
+    ov::element::Type weight_type = layout.is_symmetric ? (layout.is_u4 ? ov::element::i4 : ov::element::i8) :
+                                                          (layout.is_u4 ? ov::element::u4 : ov::element::u8);
     ov::Shape scale_shape = {node_shape[0], node_shape[1] / layout.weights_per_block};
-    ov::Shape zp_shape = layout.is_symmetric ? ov::Shape{} : scale_shape;
 
     if (output_base_ptr) {
         uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
         result.weights = ov::Tensor(weight_type, node_shape, buf_base + layout.weights_offset);
         result.scales = ov::Tensor(ov::element::f16, scale_shape, buf_base + layout.scales_offset);
-        result.zp = ov::Tensor(weight_type, zp_shape, buf_base + layout.zp_offset);
+        if (!layout.is_symmetric) {
+            ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
+            result.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
+        }
+        // else: result.zp remains default-constructed (empty) for symmetric
     } else {
         result.weights = ov::Tensor(weight_type, node_shape);
         result.scales = ov::Tensor(ov::element::f16, scale_shape);
-        if (use_bias && !layout.is_symmetric) {
-            // bias only has effect for asymmetric quant
-            result.zp = ov::Tensor(ov::element::f16, zp_shape);
-        } else {
-            result.zp = ov::Tensor(weight_type, zp_shape);
+        if (!layout.is_symmetric) {
+            if (use_bias) {
+                result.zp = ov::Tensor(ov::element::f16, scale_shape);
+            } else {
+                ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
+                result.zp = ov::Tensor(zp_type, scale_shape);
+            }
         }
+        // else: result.zp remains default-constructed (empty) for symmetric
     }
 
     if (layout.is_requant && layout.requant_type.has_value()) {
@@ -741,59 +795,75 @@ void quantize_q4_0(const float * x,
 
     auto * weights = static_cast<uint8_t *>(weights_arr.data());
     auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
-    auto * zp = static_cast<uint8_t *>(zp_arr.data());
-    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization
-
-    // For Q4_0, zero point is always 8
-    if (is_scalar_zp) {
-        zp[0] = 8 | (8 << 4);  // Pack two 4-bit values
-    }
+    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path
 
-    for (int i = 0; i < nb; i++) {
-        float amax = 0.0f;  // absolute max
-        float max = 0.0f;
-
-        for (int j = 0; j < qk; j++) {
-            const float v = x[i * qk + j];
-            if (amax < fabsf(v)) {
-                amax = fabsf(v);
-                max = v;
+    if (!is_symmetric) {
+        auto * zp = static_cast<uint8_t *>(zp_arr.data());
+        for (int i = 0; i < nb; i++) {
+            float amax = 0.0f;
+            float max = 0.0f;
+            for (int j = 0; j < qk; j++) {
+                const float v = x[i * qk + j];
+                if (amax < fabsf(v)) {
+                    amax = fabsf(v);
+                    max = v;
+                }
             }
-        }
-
-        const float d = max / -8;
-
-        if (d == 0) {
-            scales[i] = ov::float16(1.0f);
-            // zp is already set to 8 for symmetric, or set per-block for asymmetric
-            if (!is_scalar_zp) {
+            const float d = max / -8;
+            if (d == 0) {
+                scales[i] = ov::float16(1.0f);
                 if (i % 2 == 0) {
                     zp[i / 2] = 8;
                 } else {
                     zp[i / 2] |= (8 << 4);
                 }
+                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
+                continue;
             }
-            memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
-            continue;
-        }
-
-        const float id = 1.0f / d;
-        scales[i] = ov::float16(d);
-        // For asymmetric quantization, store per-block zero points
-        if (!is_scalar_zp) {
+            const float id = 1.0f / d;
+            scales[i] = ov::float16(d);
             if (i % 2 == 0) {
                 zp[i / 2] = 8;
             } else {
                 zp[i / 2] |= (8 << 4);
             }
+            for (int j = 0; j < qk / 2; ++j) {
+                const float x0 = x[i * qk + 2 * j] * id;
+                const float x1 = x[i * qk + 2 * j + 1] * id;
+                const uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
+                const uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
+                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
+            }
         }
-
-        for (int j = 0; j < qk / 2; ++j) {
-            const float x0 = x[i * qk + 2 * j] * id;
-            const float x1 = x[i * qk + 2 * j + 1] * id;
-            const uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
-            const uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
-            weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
+    } else {
+        // Symmetric: produce signed i4 values in [-8, 7]
+        for (int i = 0; i < nb; i++) {
+            float amax = 0.0f;
+            float max = 0.0f;
+            for (int j = 0; j < qk; j++) {
+                const float v = x[i * qk + j];
+                if (amax < fabsf(v)) {
+                    amax = fabsf(v);
+                    max = v;
+                }
+            }
+            const float d = max / -8;
+            if (d == 0) {
+                scales[i] = ov::float16(1.0f);
+                // i4 value 0 packed: 0x00
+                memset(weights + i * qk / 2, 0, qk / 2);
+                continue;
+            }
+            const float id = 1.0f / d;
+            scales[i] = ov::float16(d);
+            for (int j = 0; j < qk / 2; ++j) {
+                const float x0 = x[i * qk + 2 * j] * id;
+                const float x1 = x[i * qk + 2 * j + 1] * id;
+                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
+                int8_t si0 = (int8_t) std::max(-8, std::min(7, (int) roundf(x0)));
+                int8_t si1 = (int8_t) std::max(-8, std::min(7, (int) roundf(x1)));
+                weights[i * qk / 2 + j] = (si0 & 0x0F) | ((si1 & 0x0F) << 4);
+            }
         }
     }
 }
@@ -809,36 +879,42 @@ void quantize_q8_0(const float * x,
 
     auto * weights = static_cast<uint8_t *>(weights_arr.data());
     auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
-    auto * zp = static_cast<uint8_t *>(zp_arr.data());
-    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization
-
-    // For Q8_0, zero point is always 128
-    if (is_scalar_zp) {
-        zp[0] = 128;
-    }
-
-    for (int i = 0; i < nb; i++) {
-        float amax = 0.0f;  // absolute max
+    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path
 
-        for (int j = 0; j < qk; j++) {
-            const float v = x[i * qk + j];
-            if (amax < fabsf(v)) {
-                amax = fabsf(v);
+    if (!is_symmetric) {
+        auto * zp = static_cast<uint8_t *>(zp_arr.data());
+        for (int i = 0; i < nb; i++) {
+            float amax = 0.0f;
+            for (int j = 0; j < qk; j++) {
+                const float v = x[i * qk + j];
+                amax = std::max(amax, fabsf(v));
             }
-        }
-
-        const float d = amax / 127.0f;
-        const float id = d ? 1.0f / d : 0.0f;
-        scales[i] = ov::float16(d);
-        // For asymmetric quantization, store per-block zero points
-        if (!is_scalar_zp) {
+            const float d = amax / 127.0f;
+            const float id = d ? 1.0f / d : 0.0f;
+            scales[i] = ov::float16(d);
             zp[i] = 128;
+            for (int j = 0; j < qk; ++j) {
+                const float x0 = x[i * qk + j] * id;
+                const int8_t xi0 = roundf(x0);
+                weights[i * qk + j] = (uint8_t) (xi0 + 128);
+            }
         }
-
-        for (int j = 0; j < qk; ++j) {
-            const float x0 = x[i * qk + j] * id;
-            const int8_t xi0 = roundf(x0);
-            weights[i * qk + j] = (uint8_t) (xi0 + 128);
+    } else {
+        // Symmetric: store signed int8 values directly
+        auto * signed_weights = reinterpret_cast<int8_t *>(weights);
+        for (int i = 0; i < nb; i++) {
+            float amax = 0.0f;
+            for (int j = 0; j < qk; j++) {
+                const float v = x[i * qk + j];
+                amax = std::max(amax, fabsf(v));
+            }
+            const float d = amax / 127.0f;
+            const float id = d ? 1.0f / d : 0.0f;
+            scales[i] = ov::float16(d);
+            for (int j = 0; j < qk; ++j) {
+                const float x0 = x[i * qk + j] * id;
+                signed_weights[i * qk + j] = (int8_t) roundf(x0);
+            }
         }
     }
 }
@@ -861,12 +937,8 @@ void quantize_q8_1(const float * x,
 
         for (int j = 0; j < qk; j++) {
             const float v = x[i * qk + j];
-            if (v < min) {
-                min = v;
-            }
-            if (v > max) {
-                max = v;
-            }
+            min = std::min(v, min);
+            max = std::max(v, max);
         }
 
         const float d = (max - min) / ((1 << 8) - 1);
diff --git src/ggml-openvino/openvino/op/rope.cpp src/ggml-openvino/openvino/op/rope.cpp
index 26dc2d24..a8db9b38 100644
--- src/ggml-openvino/openvino/op/rope.cpp
+++ src/ggml-openvino/openvino/op/rope.cpp
@@ -9,12 +9,17 @@
 #include <openvino/op/add.hpp>
 #include <openvino/op/concat.hpp>
 #include <openvino/op/constant.hpp>
+#include <openvino/op/convert.hpp>
+#include <openvino/op/cos.hpp>
+#include <openvino/op/gather.hpp>
 #include <openvino/op/multiply.hpp>
 #include <openvino/op/reshape.hpp>
 #include <openvino/op/shape_of.hpp>
+#include <openvino/op/sin.hpp>
 #include <openvino/op/slice.hpp>
 #include <openvino/op/split.hpp>
 #include <openvino/op/subtract.hpp>
+#include <openvino/op/transpose.hpp>
 #include <openvino/op/unsqueeze.hpp>
 #include <vector>
 
@@ -33,6 +38,12 @@ OutputVector translate_rope(const NodeContext & context) {
     auto data_node = context.get_input(0).get_node_shared_ptr();
     auto output_shape = context.get_output_shape().to_shape();
     int32_t * op_params = context.get_output_op_params();
+    const int mode = (op_case & 0xFFFF0000) >> 16;
+    op_case = (op_case & 0x0000FFFF);
+
+    constexpr int TYPE_NORMAL = 0;
+    constexpr int TYPE_NEOX = 1;
+    constexpr int TYPE_IMROPE = 2;
 
     Output<Node> cos_theta_node;
     Output<Node> sin_theta_node;
@@ -45,7 +56,7 @@ OutputVector translate_rope(const NodeContext & context) {
         if (context.get_input_size() == 3) {
             rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
         }
-        auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight);
+        auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode == TYPE_IMROPE);
         sin_theta_node = sin_cos.first;
         cos_theta_node = sin_cos.second;
     }
@@ -65,11 +76,7 @@ OutputVector translate_rope(const NodeContext & context) {
         }
     }
 
-    const int mode = op_params[2];
-    constexpr int ROPE_TYPE_NORMAL = 0;
-    constexpr int ROPE_TYPE_NEOX = 2;
-
-    if (mode == ROPE_TYPE_NORMAL) {
+    if (mode == TYPE_NORMAL) {
         auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
         auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
         auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
@@ -97,7 +104,7 @@ OutputVector translate_rope(const NodeContext & context) {
         auto data_shape = ov::op::v0::Constant::create(
             ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
         res = std::make_shared<ov::op::v1::Reshape>(stack, data_shape, false);
-    } else if (mode == ROPE_TYPE_NEOX) {
+    } else if (mode == TYPE_NEOX) {
         auto data_split = std::make_shared<ov::op::v1::Split>(
             data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}), 2);
         Output<Node> slice_data_node_0 = data_split->outputs()[0];
@@ -112,6 +119,25 @@ OutputVector translate_rope(const NodeContext & context) {
             std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));
 
         res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, -1);
+    } else if (mode == TYPE_IMROPE) {
+        int64_t n_dims = data_node->get_shape()[3];
+        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1,-1,1,(n_dims >> 1)});
+        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
+        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);
+
+        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
+        auto split_a = std::make_shared<ov::op::v1::Split>(data_node, split_axis, 2);
+        auto x0 = split_a->output(0);
+        auto x1 = split_a->output(1);
+        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
+        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
+        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);
+
+        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
+        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
+        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);
+
+        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
     }
 
     return rename_outputs_with_suffix({res}, context.get_name());
diff --git src/ggml-openvino/openvino/op/unary_gelu.cpp src/ggml-openvino/openvino/op/unary_gelu.cpp
new file mode 100644
index 00000000..d1e9efc3
--- /dev/null
+++ src/ggml-openvino/openvino/op/unary_gelu.cpp
@@ -0,0 +1,25 @@
+#include "../node_context.h"
+#include "../op_table.h"
+#include "../utils.h"
+
+#include <openvino/core/node_output.hpp>
+#include <openvino/op/gelu.hpp>
+
+namespace ov {
+namespace frontend {
+namespace ggml {
+namespace op {
+
+OutputVector translate_unary_gelu(const NodeContext & context) {
+    num_inputs_check(context, 1, 1);
+
+    auto input = context.get_input(0);
+    auto res = std::make_shared<ov::op::v7::Gelu>(input);
+
+    return rename_outputs_with_suffix({res}, context.get_name());
+}
+
+}  // namespace op
+}  // namespace ggml
+}  // namespace frontend
+}  // namespace ov
diff --git src/ggml-openvino/openvino/op_table.cpp src/ggml-openvino/openvino/op_table.cpp
index beadafe8..13855392 100644
--- src/ggml-openvino/openvino/op_table.cpp
+++ src/ggml-openvino/openvino/op_table.cpp
@@ -31,6 +31,7 @@ std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
         {"GGML_OP_SOFT_MAX",       op::translate_soft_max                         },
         {"GGML_OP_SUB",            op::translate_1to1_match_2_inputs<v1::Subtract>},
         {"GGML_OP_TRANSPOSE",      op::translate_transpose                        },
+        {"GGML_UNARY_OP_GELU",     op::translate_unary_gelu                       },
         {"GGML_UNARY_OP_SILU",     op::translate_unary_silu                       },
         {"GGML_OP_VIEW",           op::translate_view                             },
         {"GGML_GLU_OP_SWIGLU",     op::translate_glu_swiglu                       },
diff --git src/ggml-openvino/openvino/op_table.h src/ggml-openvino/openvino/op_table.h
index 37f76311..f546796d 100644
--- src/ggml-openvino/openvino/op_table.h
+++ src/ggml-openvino/openvino/op_table.h
@@ -21,6 +21,7 @@ GGML_OP_CONVERTER(translate_rms_norm);
 GGML_OP_CONVERTER(translate_rope);
 GGML_OP_CONVERTER(translate_scale);
 GGML_OP_CONVERTER(translate_unary_silu);
+GGML_OP_CONVERTER(translate_unary_gelu);
 GGML_OP_CONVERTER(translate_soft_max);
 GGML_OP_CONVERTER(translate_transpose);
 GGML_OP_CONVERTER(translate_view);
diff --git src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp
new file mode 100644
index 00000000..f051891c
--- /dev/null
+++ src/ggml-openvino/openvino/rt_info/weightless_caching_attributes.hpp
@@ -0,0 +1,41 @@
+// Copyright (C) 2018-2026 Intel Corporation
+// SPDX-License-Identifier: Apache-2.0
+//
+
+#pragma once
+
+#include <openvino/core/core_visibility.hpp>
+#include <openvino/core/node.hpp>
+#include <openvino/core/runtime_attribute.hpp>
+
+namespace ov {
+
+/**
+ * @brief Holds weightless caching attributes of a single constant.
+ *
+ * WeightlessCacheAttribute class represents runtime info attribute that holds
+ * the values of original size of the constant in bytes and the binary offset of the
+ * constant's data in the weights file used by the weightless caching mechanism. It's
+ * not copyable in case the data was changed (the original node was replaced by a new
+ * one produced during the tranformation pipeline) - in that case weightless caching
+ * can't be used for that constant.
+ */
+class OPENVINO_API WeightlessCacheAttribute : public RuntimeAttribute {
+public:
+    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)
+
+    WeightlessCacheAttribute() = delete;
+
+    WeightlessCacheAttribute(size_t original_size, size_t bin_offset, ov::element::Type original_dtype)
+        : original_size(original_size),
+          bin_offset(bin_offset),
+          original_dtype(original_dtype) {}
+
+    bool is_copyable() const override;
+
+    size_t original_size;
+    size_t bin_offset;
+    ov::element::Type original_dtype;
+};
+
+}  // namespace ov
diff --git src/ggml-openvino/openvino/translate_session.cpp src/ggml-openvino/openvino/translate_session.cpp
index 23a1dea2..0f68a1f5 100644
--- src/ggml-openvino/openvino/translate_session.cpp
+++ src/ggml-openvino/openvino/translate_session.cpp
@@ -3,15 +3,16 @@
 #include "ggml-openvino/openvino/node_context.h"
 #include "ggml-openvino/openvino/utils.h"
 #include "input_model.h"
-#include "pass/eliminate_zp.h"
 #include "pass/mark_decompression_convert_constant_folding.h"
 #include "pass/squeeze_matmul.h"
+#include "rt_info/weightless_caching_attributes.hpp"
 
 #include <cstdint>
 #include <cstdlib>
 #include <map>
 #include <memory>
 #include <openvino/core/node.hpp>
+#include <openvino/core/preprocess/pre_post_process.hpp>
 #include <openvino/op/add.hpp>
 #include <openvino/op/broadcast.hpp>
 #include <openvino/op/concat.hpp>
@@ -33,7 +34,6 @@
 #include <openvino/op/unsqueeze.hpp>
 #include <openvino/pass/constant_folding.hpp>
 #include <openvino/pass/make_stateful.hpp>
-#include <openvino/core/preprocess/pre_post_process.hpp>
 
 namespace ov {
 namespace frontend {
@@ -240,6 +240,31 @@ std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputMo
     resulting_model = std::make_shared<Model>(results, used_params);
 
     apply_transformations(resulting_model);
+
+    // Set WeightlessCacheAttribute on large constants to avoid unnecessary memory copies
+    // in the NPUW plugin. Without this attribute, NPUW's LazyTensor constructor
+    // (lazy_tensor.cpp, op::Const::Const) will memcpy every constant "in case export
+    // occurs", doubling memory usage per compile_model call.
+    //
+    // The bin_offset field serves as a unique key (not a real file offset) — this is
+    // the same convention the GPU plugin uses for non-IR models (see
+    // Plugin::set_weightless_cache_attributes in intel_gpu/src/plugin/plugin.cpp).
+    // Each constant must have a distinct bin_offset, otherwise GPU's weightless cache
+    // import will map multiple constants to the same data.
+    //
+    // Small constants (< 16 elements) are excluded since they may be introduced by
+    // optimization patterns and the overhead is negligible.
+    size_t offset = 0;
+    for (auto & node : resulting_model->get_ordered_ops()) {
+        if (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
+            cnst && cnst->get_byte_size() / cnst->get_element_type().size() >= 16) {
+            auto & rt_info = cnst->get_rt_info();
+            if (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
+                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
+                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
+            }
+        }
+    }
     return resulting_model;
 }
 
@@ -257,7 +282,6 @@ std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<M
         }
 
         if (ggml_model_decoder->is_static()) {
-            manager.register_pass<pass::EliminateZeroPoints>();
             manager.register_pass<pass::SqueezeMatmul>();
         }
         manager.run_passes(model);
diff --git src/ggml-openvino/openvino/utils.cpp src/ggml-openvino/openvino/utils.cpp
index 65356a51..0baaf88e 100644
--- src/ggml-openvino/openvino/utils.cpp
+++ src/ggml-openvino/openvino/utils.cpp
@@ -2,6 +2,7 @@
 
 #include "ggml-impl.h"
 
+#include <cmath>
 #include <cstddef>
 #include <ctime>
 #include <memory>
@@ -13,6 +14,7 @@
 #include <openvino/op/gather.hpp>
 #include <openvino/op/maximum.hpp>
 #include <openvino/op/multiply.hpp>
+#include <openvino/op/reshape.hpp>
 #include <openvino/op/shape_of.hpp>
 #include <openvino/op/sin.hpp>
 #include <openvino/op/squeeze.hpp>
@@ -87,8 +89,11 @@ ov::Output<ov::Node> rope_yarn_ramp_mix(int n_dims, const float corr_dims[2], fl
     auto ramp_y =
         std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
     auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
+    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
+    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
+    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
     auto ext_factor_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
-    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_clamped, ext_factor_node);
+    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_node);
     return ramp_mix;
 }
 
@@ -115,6 +120,7 @@ void ggml_rope_yarn_corr_dims(int n_dims,
 std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                            std::shared_ptr<ov::Node> inp_pos,
                                                            std::shared_ptr<ov::Node> rope_freqs_weight,
+                                                           bool imrope,
                                                            bool stateful) {
     if (stateful) {
         inp_pos = std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
@@ -122,6 +128,13 @@ std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params
         auto pos_perm =
             std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
         inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
+    } else if (imrope) {
+        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
+        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
+        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, true);
+        auto pos_transpose_shape =
+            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
+        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
     } else {
         inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
         auto pos_perm =
@@ -136,6 +149,7 @@ std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params
     float beta_fast;
     float beta_slow;
     const int n_dims = rope_params[1];
+    const size_t n_dims_half = n_dims >> 1;
     const int n_ctx_orig = rope_params[4];
     memcpy(&freq_base, rope_params + 5, sizeof(float));
     memcpy(&freq_scale, rope_params + 6, sizeof(float));
@@ -146,57 +160,74 @@ std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params
 
     const float theta_scale = powf(freq_base, -2.0f / n_dims);
 
-    float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
-
-    std::vector<float> factor(n_dims / 2);
-    factor[0] = 1.0f;
-    for (size_t i = 1; i < factor.size(); i++) {
-        factor[i] = theta_scale * factor[i - 1];
-    }
+    std::vector<float> factor(n_dims_half);
 
     Output<Node> freq_factors;
-    if (stateful) {
-        freq_factors =
-            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
-    } else {
-        freq_factors =
-            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
-    }
-    if (rope_freqs_weight) {
-        freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_freqs_weight);
-    }
-
-    auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
-    auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
-        theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
 
     Output<Node> theta;
     float mscale = attn_factor;
-    if (ext_factor == 0.0f) {
-        theta = theta_interp;
+    if (imrope) {
+        std::vector<int64_t> gather_indices(n_dims_half);
+        for (size_t j = 0; j < n_dims_half; j++) {
+            gather_indices[j] = j % 3;
+            factor[j] = std::pow(theta_scale, j);
+        }
+        auto gather_indices_const =
+            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{n_dims_half}, gather_indices);
+        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
+        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
+        auto factor_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_dims_half}, factor);
+        theta = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_const);
     } else {
-        auto ramp_mix = rope_yarn_ramp_mix(n_dims, corr_dims, ext_factor);
-        Output<Node> one;
+        float corr_dims[2];
+        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+        factor[0] = 1.0f;
+        for (size_t i = 1; i < factor.size(); i++) {
+            factor[i] = theta_scale * factor[i - 1];
+        }
         if (stateful) {
-            one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
+            freq_factors =
+                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
         } else {
-            one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
+            freq_factors =
+                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
+        }
+        if (rope_freqs_weight) {
+            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_freqs_weight);
         }
-        auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
 
-        theta = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
-                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
-        mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
+        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
+        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
+            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
+
+        if (ext_factor == 0.0f) {
+            theta = theta_interp;
+        } else {
+            auto ramp_mix = rope_yarn_ramp_mix(n_dims, corr_dims, ext_factor);
+            Output<Node> one;
+            if (stateful) {
+                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
+            } else {
+                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
+            }
+            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);
+
+            theta = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
+                                                      std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
+            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
+        }
     }
 
     Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
     Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);
 
-    auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});
+    if (!imrope) {
+        auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});
+
+        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
+        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
+    }
 
-    cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
-    sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
     return std::make_pair(sin_theta, cos_theta);
 }
 
diff --git src/ggml-openvino/openvino/utils.h src/ggml-openvino/openvino/utils.h
index 88dcad4c..767dd4c5 100644
--- src/ggml-openvino/openvino/utils.h
+++ src/ggml-openvino/openvino/utils.h
@@ -67,6 +67,7 @@ OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::
 std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t* rope_params,
                                                            std::shared_ptr<ov::Node> inp_pos,
                                                            std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
+                                                           bool imrope = false,
                                                            bool stateful = false);
 
 ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len = 0);
diff --git src/ggml-openvino/utils.cpp src/ggml-openvino/utils.cpp
index 1b553a0d..998ef7c9 100644
--- src/ggml-openvino/utils.cpp
+++ src/ggml-openvino/utils.cpp
@@ -81,8 +81,8 @@ ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
 enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
     auto & core = ov_singleton_core();
     const auto & config = ggml_openvino_get_compile_config();
-    auto device = r_ctx->device;
-    bool stateful = r_ctx->stateful;
+    const auto & device = r_ctx->device;
+    const auto & stateful = r_ctx->stateful;
     static auto is_static = false;
 
     if (is_naive(cgraph)) {
@@ -106,14 +106,26 @@ enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<
     int64_t infer_end_time;
 
     {
-        std::lock_guard<std::mutex> lock(r_ctx->ov_compute_mutex);
+        std::shared_ptr<decoder_runtime_ctx> entry;
+        ModelParams old_m_params;
 
-        auto it = r_ctx->decoder_cache.find(key);
+        {
+            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+            auto it = r_ctx->decoder_cache.find(key);
+            cache_hit = it != r_ctx->decoder_cache.end();
+            if (cache_hit) {
+                entry = it->second;
+            } else {
+                auto mutex = std::make_shared<std::mutex>();
+                entry = std::make_shared<decoder_runtime_ctx>(mutex);
+                r_ctx->decoder_cache[key] = entry;
+            }
+        }
+
+        std::lock_guard<std::mutex> lock(*(entry->mutex));
 
-        cache_hit = it != r_ctx->decoder_cache.end();
-        ModelParams old_m_params;
         if (cache_hit) {
-            ggml_decoder = it->second;
+            ggml_decoder = entry->ptr;
             old_m_params = ggml_decoder->get_model_params();
             cache_hit = old_m_params.can_reuse_dynamically(m_params);
         }
@@ -126,7 +138,10 @@ enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<
                 ggml_decoder->update_io(cgraph);
             }
             ggml_decoder->add_extra_inputs();
-            infer_request = r_ctx->infer_request_cache.at(key);
+            {
+                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+                infer_request = r_ctx->infer_request_cache.at(key);
+            }
 
             if (stateful) {
                 const auto * inp_pos = get_inp_pos_tensor(cgraph);
@@ -170,7 +185,10 @@ enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<
             conversion_end_time = decoder_end_time;
             compile_end_time = decoder_end_time;
         } else {
-            r_ctx->infer_request_cache.erase(key);
+            {
+                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+                r_ctx->infer_request_cache.erase(key);
+            }
 
             std::shared_ptr<ov::Model> model;
             auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
@@ -199,8 +217,7 @@ enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<
             }
             compile_end_time = ggml_time_us();
             infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
-            r_ctx->infer_request_cache[key] = infer_request;
-            r_ctx->decoder_cache[key] = ggml_decoder;
+            entry->ptr = ggml_decoder;
 
             std::vector<std::string> ov_input_names;
             std::vector<std::string> ov_output_names;
@@ -210,8 +227,13 @@ enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<
             for (const auto & ov_output : model->get_results()) {
                 ov_output_names.push_back(ov_output->get_friendly_name());
             }
-            r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
-            r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
+
+            {
+                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+                r_ctx->infer_request_cache[key] = infer_request;
+                r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
+                r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
+            }
 
             if (stateful) {
                 const auto * inp_pos = get_inp_pos_tensor(cgraph);
@@ -224,8 +246,13 @@ enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<
             }
         }
 
-        auto ov_input_names = r_ctx->ov_input_names_cache[key];
-        auto ov_output_names = r_ctx->ov_output_names_cache[key];
+        std::vector<std::string> ov_input_names;
+        std::vector<std::string> ov_output_names;
+        {
+            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+            ov_input_names = r_ctx->ov_input_names_cache[key];
+            ov_output_names = r_ctx->ov_output_names_cache[key];
+        }
 
         for (size_t i = 0; i < ov_input_names.size(); i++) {
             auto param_name = ov_input_names[i];
@@ -306,12 +333,26 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
     int64_t compile_end_time;
     int64_t infer_end_time;
 
-    auto it = r_ctx->decoder_cache.find(key);
-
-    cache_hit = it != r_ctx->decoder_cache.end();
+    std::shared_ptr<decoder_runtime_ctx> entry;
     ModelParams old_m_params;
+
+    {
+        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+        auto it = r_ctx->decoder_cache.find(key);
+        cache_hit = it != r_ctx->decoder_cache.end();
+        if (cache_hit) {
+            entry = it->second;
+        } else {
+            auto mutex = std::make_shared<std::mutex>();
+            entry = std::make_shared<decoder_runtime_ctx>(mutex);
+            r_ctx->decoder_cache[key] = entry;
+        }
+    }
+
+    std::lock_guard<std::mutex> lock(*(entry->mutex));
+
     if (cache_hit) {
-        ggml_decoder = it->second;
+        ggml_decoder = entry->ptr;
         old_m_params = ggml_decoder->get_model_params();
         cache_hit = old_m_params.can_reuse_statically(m_params);
     }
@@ -325,14 +366,21 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
             ggml_decoder->update_io(cgraph);
         }
         ggml_decoder->add_extra_inputs();
-        infer_request = is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
+        {
+            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+            infer_request =
+                is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
+        }
 
         decoder_end_time = ggml_time_us();
         conversion_end_time = decoder_end_time;
         compile_end_time = decoder_end_time;
     } else {
-        r_ctx->infer_request_cache.erase(key);
-        r_ctx->infer_request_cache_prefill.erase(key);
+        {
+            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+            r_ctx->infer_request_cache.erase(key);
+            r_ctx->infer_request_cache_prefill.erase(key);
+        }
 
         std::shared_ptr<ov::Model> model;
         auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);
@@ -372,16 +420,14 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
             compiled_model_decode = core.compile_model(model_decode, device, config);
         }
 
-        r_ctx->infer_request_cache_prefill[key] =
-            std::make_shared<ov::InferRequest>(compiled_model_prefill.create_infer_request());
-        r_ctx->infer_request_cache[key] =
-            std::make_shared<ov::InferRequest>(compiled_model_decode.create_infer_request());
+        auto infer_request_prefill = std::make_shared<ov::InferRequest>(compiled_model_prefill.create_infer_request());
+        auto infer_request_decode = std::make_shared<ov::InferRequest>(compiled_model_decode.create_infer_request());
         compile_end_time = ggml_time_us();
 
         model = is_prefill ? model_prefill : model_decode;
         ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
-        infer_request = is_prefill ? r_ctx->infer_request_cache_prefill[key] : r_ctx->infer_request_cache[key];
-        r_ctx->decoder_cache[key] = ggml_decoder;
+        infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
+        entry->ptr = ggml_decoder;
 
         std::vector<std::string> ov_input_names;
         std::vector<std::string> ov_output_names;
@@ -391,18 +437,29 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
         for (const auto & ov_output : model->get_results()) {
             ov_output_names.push_back(ov_output->get_friendly_name());
         }
-        r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
-        r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
+
+        {
+            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+            r_ctx->infer_request_cache_prefill[key] = infer_request_prefill;
+            r_ctx->infer_request_cache[key] = infer_request_decode;
+            r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
+            r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
+        }
     }
 
-    auto ov_input_names = r_ctx->ov_input_names_cache[key];
-    auto ov_output_names = r_ctx->ov_output_names_cache[key];
+    std::vector<std::string> ov_input_names_local;
+    std::vector<std::string> ov_output_names_local;
+    {
+        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
+        ov_input_names_local = r_ctx->ov_input_names_cache[key];
+        ov_output_names_local = r_ctx->ov_output_names_cache[key];
+    }
 
     if (is_prefill) {
         auto inp_len = inp_pos->ne[0];
         for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
-            for (size_t i = 0; i < ov_input_names.size(); i++) {
-                auto param_name = ov_input_names[i];
+            for (size_t i = 0; i < ov_input_names_local.size(); i++) {
+                auto param_name = ov_input_names_local[i];
                 auto input_tensor = get_ov_input_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                 infer_request->set_input_tensor(i, input_tensor);
 
@@ -412,8 +469,8 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
                 }
             }
 
-            for (size_t i = 0; i < ov_output_names.size(); i++) {
-                auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names[i]);
+            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
+                auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
                 auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                 infer_request->set_output_tensor(i, output_tensor);
             }
@@ -421,16 +478,16 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
             infer_request->infer();
 
             if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
-                for (size_t i = 0; i < ov_output_names.size(); i++) {
+                for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                     const auto output_tensor = infer_request->get_output_tensor(i);
-                    print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
+                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                 }
             }
         }
         infer_end_time = ggml_time_us();
     } else {
-        for (size_t i = 0; i < ov_input_names.size(); i++) {
-            auto param_name = ov_input_names[i];
+        for (size_t i = 0; i < ov_input_names_local.size(); i++) {
+            auto param_name = ov_input_names_local[i];
             auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, param_name);
             infer_request->set_input_tensor(i, input_tensor);
 
@@ -440,8 +497,8 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
             }
         }
 
-        for (size_t i = 0; i < ov_output_names.size(); i++) {
-            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names[i]);
+        for (size_t i = 0; i < ov_output_names_local.size(); i++) {
+            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
             auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
             infer_request->set_output_tensor(i, output_tensor);
         }
@@ -450,9 +507,9 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<o
         infer_end_time = ggml_time_us();
 
         if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
-            for (size_t i = 0; i < ov_output_names.size(); i++) {
+            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                 const auto output_tensor = infer_request->get_output_tensor(i);
-                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
+                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
             }
         }
     }
diff --git src/ggml-openvino/utils.h src/ggml-openvino/utils.h
index 656573d1..2c72e33c 100644
--- src/ggml-openvino/utils.h
+++ src/ggml-openvino/utils.h
@@ -3,12 +3,15 @@
 #include "ggml-impl.h"
 
 #include <algorithm>
+#include <atomic>
 #include <cstddef>
 #include <memory>
+#include <mutex>
 #include <openvino/runtime/core.hpp>
 #include <openvino/runtime/infer_request.hpp>
 #include <string>
 #include <unordered_map>
+#include <utility>
 #include <vector>
 
 struct graph_key {
@@ -40,11 +43,17 @@ struct graph_key_hash {
     }
 };
 
+struct decoder_runtime_ctx {
+    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}
+    std::shared_ptr<std::mutex> mutex;
+    std::shared_ptr<GgmlOvDecoder> ptr;
+};
+
 struct ov_runtime_context {
-    std::mutex ov_compute_mutex;
+    mutable std::mutex ctx_mutex;
     std::string device;
     bool stateful;
-    std::unordered_map<graph_key, std::shared_ptr<GgmlOvDecoder>, graph_key_hash> decoder_cache;
+    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;
     std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
     std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
     std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
@@ -53,11 +62,22 @@ struct ov_runtime_context {
     //      Simultanous stateful inference request support to be added.
     size_t stateful_kv_size;
     std::map<std::string, std::string> kv_state_input_name_map;
+    std::atomic<int> backend_count;
 
     ov_runtime_context() :
         device("CPU"),
         stateful(false),
-        stateful_kv_size(0) {}
+        stateful_kv_size(0),
+        backend_count(0) {}
+
+    void clear_caches() {
+        std::lock_guard<std::mutex> lock(ctx_mutex);
+        decoder_cache.clear();
+        infer_request_cache.clear();
+        infer_request_cache_prefill.clear();
+        ov_input_names_cache.clear();
+        ov_output_names_cache.clear();
+    }
 };
 
 enum ggml_status ov_graph_compute(struct ggml_cgraph * cgraph, ggml_backend_t backend);
diff --git src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan.cpp
index 702a249d..d4acee8b 100644
--- src/ggml-vulkan/ggml-vulkan.cpp
+++ src/ggml-vulkan/ggml-vulkan.cpp
@@ -792,6 +792,7 @@ struct vk_device_struct {
     vk_pipeline pipeline_arange_f32;
 
     vk_pipeline pipeline_fill_f32;
+    vk_pipeline pipeline_fill_f16;
 
     vk_pipeline pipeline_geglu[2];
     vk_pipeline pipeline_reglu[2];
@@ -4577,6 +4578,7 @@ static void ggml_vk_load_shaders(vk_device& device) {
     ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
 
     ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
 
 #define CREATE_GLU(name)  \
     ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
@@ -9844,6 +9846,9 @@ static vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, const
         if (dst->type == GGML_TYPE_F32) {
             return ctx->device->pipeline_fill_f32;
         }
+        if (dst->type == GGML_TYPE_F16) {
+            return ctx->device->pipeline_fill_f16;
+        }
         return nullptr;
     default:
         return nullptr;
@@ -15713,8 +15718,9 @@ static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggm
                 || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                 || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
         case GGML_OP_ARANGE:
-        case GGML_OP_FILL:
             return op->type == GGML_TYPE_F32;
+        case GGML_OP_FILL:
+            return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
         case GGML_OP_SCALE:
             return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
         case GGML_OP_PAD:
diff --git src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
index 54b9b327..ff836615 100644
--- src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
+++ src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
@@ -889,6 +889,7 @@ void process_shaders() {
     string_to_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
     string_to_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
     string_to_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
+    string_to_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
     string_to_spv("step_f16",       "step.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
     string_to_spv("step_f32",       "step.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
     string_to_spv("round_f16",      "round.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
