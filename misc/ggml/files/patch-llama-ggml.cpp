diff --git src/ggml-cpu/llamafile/sgemm.cpp src/ggml-cpu/llamafile/sgemm.cpp
index 34e320e2..e13828e3 100644
--- src/ggml-cpu/llamafile/sgemm.cpp
+++ src/ggml-cpu/llamafile/sgemm.cpp
@@ -2321,6 +2321,9 @@ class tinyBLAS_Q0_PPC {
     }
 
     void matmul(int64_t m, int64_t n) {
+    #if defined(_AIX) || defined(__BIG_ENDIAN__)
+        mnpack(0, m, 0, n);
+    #else
         const int64_t mc = 64;
         const int64_t kc = 64;
         int64_t nc = 64;
@@ -2334,7 +2337,6 @@ class tinyBLAS_Q0_PPC {
         } else {
             n_aligned = (n / 64) * 64;
         }
-
         if (n_aligned > 0) {
             if (n_aligned % 64 == 0)      nc = 64;
             else if (n_aligned == n)      nc = n;
@@ -2352,6 +2354,7 @@ class tinyBLAS_Q0_PPC {
         } else {
             mnpack(0, m, 0, n);
         }
+    #endif
     }
 
   private:
@@ -3191,12 +3194,16 @@ class tinyBLAS_PPC {
     }
 
     void matmul(int64_t m, int64_t n) {
+    #if defined(_AIX) || defined(__BIG_ENDIAN__)
+        mnpack(0, m, 0, n);
+    #else
         int64_t mc = 256; int64_t nc = 256; int64_t kc = 256;
         if (m % mc == 0 && n % nc == 0 && k % kc == 0) {
             matmul_tiled(m, n, mc, nc, kc);
         } else {
             mnpack(0, m, 0, n);
         }
+    #endif
     }
 
   private:
diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index fd8dd917..0e6f7468 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -3556,6 +3556,9 @@ static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
      && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
         const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
         const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
+        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
+            return false;
+        }
 
         if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
             return false;
@@ -3564,6 +3567,31 @@ static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
         return true;
     }
 
+    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
+     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
+        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
+        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
+        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
+        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
+            return false;
+        }
+
+        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
+            return false;
+        }
+
+        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
+        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
+        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
+            return false;
+        }
+        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
+            return false;
+        }
+
+        return true;
+    }
+
     if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
      && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
         const ggml_tensor * unary = cgraph->nodes[node_idx];
@@ -3966,8 +3994,13 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
         return 1;
     }
 
+    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
+        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
+        return 2;
+    }
+
     if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
-        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1]);
+        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
         return 1;
     }
 
diff --git src/ggml-cuda/ssm-conv.cu src/ggml-cuda/ssm-conv.cu
index b77cdc1c..4841389f 100644
--- src/ggml-cuda/ssm-conv.cu
+++ src/ggml-cuda/ssm-conv.cu
@@ -3,6 +3,7 @@
 
 template <bool apply_silu, size_t split_d_inner, size_t d_conv>
 static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
+                                    const float * __restrict__ bias,
                                     const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                     float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                     const int64_t n_t) {
@@ -27,6 +28,8 @@ static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float
         w[j] = w_block[tid * stride_w + j];
     }
 
+    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;
+
     for (int64_t i = 0; i < n_t; i++) {
         float sumf = 0.0f;
 
@@ -42,12 +45,14 @@ static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float
         for (size_t j = 0; j < d_conv; j++) {
             sumf += x[(i + j) % d_conv] * w[j];
         }
+        sumf += b;
         y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
     }
 }
 
 template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
 static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
+                                               const float * __restrict__ bias,
                                                const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                                const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                                const int dst_nb1, const int dst_nb2, const int64_t n_t) {
@@ -97,6 +102,8 @@ static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0,
         w[j] = w_block[tid * stride_w + j];
     }
 
+    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;
+
     // Compute from shared memory
     for (int64_t i = 0; i < local_n_t; i++) {
         float sumf = 0.0f;
@@ -104,12 +111,13 @@ static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0,
         for (size_t j = 0; j < d_conv; j++) {
             sumf += smem[tid * n_cols + i + j] * w[j];
         }
+        sumf += b;
         y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
     }
 }
 
 template <bool apply_silu>
-static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
+static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                               const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                               const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                               const int64_t n_s, cudaStream_t stream) {
@@ -120,14 +128,14 @@ static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int
         constexpr int kNC = decltype(NC)::value;
         if (n_t <= 32) {
             const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
-            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
+            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                        dst, dst_nb0, dst_nb1, dst_nb2, n_t);
         } else {
             const int64_t split_n_t = 32;
             dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
             const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
             ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
-                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
+                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
         }
     };
 
@@ -140,11 +148,18 @@ static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int
     }
 }
 
-void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
+void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
     const struct ggml_tensor * src0 = dst->src[0];  // conv_x
     const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
+    const bool fuse_bias = bias_add_node != nullptr;
     const bool fuse_silu = silu_dst != nullptr;
 
+    // bias always comes with silu.
+    GGML_ASSERT(!fuse_bias || fuse_silu);
+
+    // The bias (when fused) is the non-conv operand of the ADD node.
+    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;
+
     // When fusing, write to silu_dst (the node downstream references).
     const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;
 
@@ -160,16 +175,23 @@ void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, g
 
     const float * src0_d = (const float *) src0->data;
     const float * src1_d = (const float *) src1->data;
+    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
     float *       dst_d  = (float *) out->data;
     cudaStream_t  stream = ctx.stream();
 
     GGML_ASSERT(src0->type == GGML_TYPE_F32);
     GGML_ASSERT(out->type == GGML_TYPE_F32);
+    if (fuse_bias) {
+        GGML_ASSERT(bias->type == GGML_TYPE_F32);
+        GGML_ASSERT(ggml_is_contiguous(bias));
+        GGML_ASSERT(ggml_nelements(bias) == nr);
+    }
+
     if (fuse_silu) {
-        ssm_conv_f32_cuda<true>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
+        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                           out->nb[2], nc, nr, n_t, n_s, stream);
     } else {
-        ssm_conv_f32_cuda<false>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
+        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                           out->nb[2], nc, nr, n_t, n_s, stream);
     }
 }
diff --git src/ggml-cuda/ssm-conv.cuh src/ggml-cuda/ssm-conv.cuh
index f96a1cd2..8514ca84 100644
--- src/ggml-cuda/ssm-conv.cuh
+++ src/ggml-cuda/ssm-conv.cuh
@@ -1,3 +1,3 @@
 #include "common.cuh"
 
-void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr);
+void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
diff --git src/ggml-hexagon/ggml-hexagon.cpp src/ggml-hexagon/ggml-hexagon.cpp
index 0d9b5e28..9345da62 100644
--- src/ggml-hexagon/ggml-hexagon.cpp
+++ src/ggml-hexagon/ggml-hexagon.cpp
@@ -48,14 +48,16 @@ using intvec  = std::vector<int>;
 using uintvec = std::vector<unsigned int>;
 using u32vec  = std::vector<uint32_t>;
 
-static size_t opt_ndev         = 1;
-static size_t opt_nhvx         = 0; // use all
-static int    opt_arch         = 0; // autodetect
-static int    opt_etm          = 0;
-static int    opt_verbose      = 0;
-static int    opt_profile      = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
-static int    opt_hostbuf      = 1; // hostbuf ON by default
-static int    opt_use_hmx      = 1; // when set, enable HMX; when 0, use HVX only
+static int    opt_arch    = 0; // autodetect
+static size_t opt_ndev    = 1;
+static size_t opt_nhvx    = 0; // use all
+static int    opt_use_hmx = 1; // when set, enable HMX; when 0, use HVX only
+static size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
+static size_t opt_mbuf    = 1ul * 1024 * 1024 * 1024; // max buffer size
+static int    opt_etm     = 0;
+static int    opt_verbose = 0;
+static int    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
+static int    opt_hostbuf = 1; // hostbuf ON by default
 
 // Default PMU events, if profiling with PMU (mode=2) is enabled
 // See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
@@ -66,6 +68,7 @@ static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C }
 static int opt_opstage  = HTP_OPSTAGE_QUEUE | HTP_OPSTAGE_COMPUTE;
 static int opt_opbatch  = 1024; // max number of ops in a batch
 static int opt_opqueue  = 16;   // max number of pending batches
+
 static std::regex* opt_opfilter = NULL; // regex of ops to not claim
 
 #define HEX_VERBOSE(...) \
@@ -110,7 +113,7 @@ static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct
     if (!opt_verbose) return;
 
     op_desc desc(op);
-    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s : %s : %s : %s : %s : %s : %s\n", sess_name.c_str(),
+    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s: %s : %s : %s : %s : %s : %s\n", sess_name.c_str(),
                 ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, desc.buffs, supp ? "yes" : "no");
 }
 
@@ -118,8 +121,6 @@ static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const ggml_t
                                       uint32_t op_usec, uint32_t op_cycles, const uint32_t pmu[]) {
     if (!opt_profile) return;
 
-    op_desc desc(op);
-
     char pmu_str[256] = "";
     if (opt_profile > 1) {
         static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
@@ -127,6 +128,7 @@ static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const ggml_t
                 pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
     }
 
+    op_desc desc(op);
     GGML_LOG_DEBUG("ggml-hex: %s profile-op %s: %s : %s : %s : %s : usec %u cycles %u%s\n", sess_name.c_str(),
             ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, op_usec, op_cycles, pmu_str);
 }
@@ -191,33 +193,30 @@ struct ggml_hexagon_shared_buffer {
     bool                   mapped;
     bool                   pinned;
 
-    void mmap(bool pinned = false) {
-        int err = fastrpc_mmap(sess->domain_id, this->fd, (void *) this->base, 0, this->size, FASTRPC_MAP_FD_DELAYED);
+    void mmap() {
+        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : FASTRPC_MAP_FD_DELAYED;
+
+        int err = fastrpc_mmap(sess->domain_id, this->fd, (void *) this->base, 0, this->size, flags);
         if (err != 0) {
             GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                     sess->domain_id, this->size, this->fd, (unsigned) err);
             throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
         }
 
-        if (pinned) {
-            err = htp_iface_mmap(sess->handle, this->fd, this->size, pinned);
-            if (err != 0) {
-                GGML_LOG_ERROR("ggml-hex: %s buffer pinning failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
-                        sess->domain_id, this->size, this->fd, (unsigned) err);
-                throw std::runtime_error("ggml-hex: htp_iface_mmap failed (see log for details)");
-            }
-        }
-
-        this->mapped = true;
-        this->pinned = pinned;
         HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u\n",
                 sess->c_name(), (void *) this->base, this->size, this->fd, pinned);
+
+        this->mapped = true;
     }
 
     void unmap() {
         if (!this->mapped) return;
 
-        htp_iface_munmap(sess->handle, this->fd);
+        if (!this->pinned) {
+            // HTP might still hold a reference, tell it drop it
+            htp_iface_munmap(sess->handle, this->fd);
+        }
+
         fastrpc_munmap(sess->domain_id, this->fd, (void *) this->base, this->size);
 
         HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
@@ -227,7 +226,7 @@ struct ggml_hexagon_shared_buffer {
         this->fd     = -1;
     }
 
-    void alloc(size_t size, bool pinned = false) {
+    void alloc(size_t size) {
         if (this->base) return;
 
         this->base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
@@ -245,8 +244,7 @@ struct ggml_hexagon_shared_buffer {
 
         HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                     (void *) this->base, this->size, this->fd, (int) pinned);
-
-        mmap(pinned);
+        mmap();
     }
 
     void free() {
@@ -262,15 +260,14 @@ struct ggml_hexagon_shared_buffer {
     }
 
     ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
-        size += 4 * 1024;  // extra page for padding
-
         this->sess   = sess;
         this->size   = 0;
         this->base   = nullptr;
         this->fd     = -1;
         this->mapped = false;
+        this->pinned = pinned;
 
-        alloc(size, pinned);
+        alloc(size);
     }
 
     ~ggml_hexagon_shared_buffer() {
@@ -1475,6 +1472,7 @@ static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
             ggml_backend_buffer_type_t buffer_type, size_t size) {
     auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
     try {
+        size += 4 * 1024;  // guard page
         ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
         return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
     } catch (const std::exception & exc) {
@@ -1487,6 +1485,7 @@ static ggml_backend_buffer_t ggml_backend_hexagon_repack_buffer_type_alloc_buffe
             ggml_backend_buffer_type_t buffer_type, size_t size) {
     auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
     try {
+        size += 4 * 1024;  // guard page
         ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
         return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
     } catch (const std::exception & exc) {
@@ -1505,7 +1504,7 @@ static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffe
 }
 
 static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
-    return 1UL * 1024 * 1024 * 1024;  // 1GB per buffer
+    return opt_mbuf; // typically 1GB per buffer
     GGML_UNUSED(buffer_type);
 }
 
@@ -1573,14 +1572,14 @@ struct ggml_hexagon_opbatch {
         d_map.clear();
     }
 
-    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size) {
+    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
         this->sess = sess;
 
         n_bufs_max = HTP_OP_MAX_BUFS;
         n_ops_max  = batch_size;
         n_tens_max = n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS;
 
-        b_vmem_max = HTP_OP_MAX_VMEM;
+        b_vmem_max = max_vmem;
 
         ops.resize(n_ops_max);
 
@@ -1592,6 +1591,9 @@ struct ggml_hexagon_opbatch {
         t_map.reserve(n_tens_max);
         d_map.reserve(n_tens_max);
 
+        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
+                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);
+
         reset();
     }
 
@@ -1925,6 +1927,8 @@ void ggml_hexagon_session::flush_batch() {
     // Bump pending flag (cleared in the session::flush once we get the response)
     this->op_pending++;  // atomic inc
 
+    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);
+
     int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
     if (err != 0) {
         GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
@@ -1944,6 +1948,35 @@ void ggml_hexagon_session::flush(bool all) {
     flush_pending(all);
 }
 
+static size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
+    // Allocate a bunch pinned buffers till failure.
+    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
+    // Typically we're going to allocate all/most of these buffers anyway for the model weights.
+
+    std::vector<ggml_hexagon_shared_buffer *> sbufs;
+
+    const size_t MiB = 1024 * 1024;
+    const size_t GiB = MiB  * 1024;
+
+    size_t vmem = 0;
+    size_t step = 256u * MiB;
+
+    try {
+        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
+        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
+        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
+
+        while (1) {
+            sbufs.push_back(new ggml_hexagon_shared_buffer(sess, step, true));
+            vmem += step;
+        }
+    } catch (...) { }
+
+    for (auto b : sbufs) { delete b; }
+
+    return vmem - step; // backoff to account for overhead from internal mappings
+}
+
 void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
     this->valid_session = false;
     this->valid_handle  = false;
@@ -1957,7 +1990,7 @@ void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
 
     this->op_pending  = 0;
 
-    GGML_LOG_INFO("ggml-hex: allocating new session: %s\n", this->name.c_str());
+    GGML_LOG_DEBUG("ggml-hex: %s allocating new session\n", this->name.c_str());
 
     domain * my_domain = get_domain(this->domain_id);
     if (my_domain == NULL) {
@@ -2033,9 +2066,6 @@ void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
 
     this->valid_handle = true;
 
-    GGML_LOG_INFO("ggml-hex: new session: %s : session-id %d domain-id %d uri %s handle 0x%lx\n", this->name.c_str(),
-                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);
-
     // Enable FastRPC QoS mode
     {
         struct remote_rpc_control_latency l;
@@ -2047,6 +2077,9 @@ void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
         }
     }
 
+    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", this->c_name(),
+                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);
+
     const size_t req_q_size = (sizeof(htp_opbatch_req) * opt_opqueue * 2) + 1024;
     const size_t rsp_q_size = (sizeof(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;
 
@@ -2091,13 +2124,19 @@ void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
     }
 
     // Allocate buffers and state for op batching
-    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch);
     this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);
 
-    // Start processing op batch requests
-    err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx, opt_use_hmx);
+    if (!opt_vmem) {
+        opt_vmem = ggml_hexagon_measure_max_vmem(this);
+        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", this->c_name(), opt_vmem);
+    }
+
+    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch, opt_vmem);
+
+    // Start dspqueue/opbatch processing
+    err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx, opt_use_hmx, opt_vmem);
     if (err != 0) {
-        GGML_LOG_ERROR("ggml-hex: failed to start session: 0x%08x\n", (unsigned) err);
+        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", this->c_name(), (unsigned) err);
         throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
     }
     this->valid_iface = true;
@@ -2108,17 +2147,17 @@ void ggml_hexagon_session::release() noexcept(true) {
 
     int err;
 
-    delete this->op_batch;
-    delete this->op_queue;
-
-    // Stop the DSP-side service and close the queue
     if (this->valid_iface) {
+        // Stop dspqueue/opbatch processing
         err = htp_iface_stop(this->handle);
         if (err != 0) {
             GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
         }
     }
 
+    delete this->op_batch;
+    delete this->op_queue;
+
     if (opt_etm) {
         err = htp_iface_etm(this->handle, 0);
         if (err != 0) {
@@ -3380,21 +3419,6 @@ struct ggml_hexagon_registry {
 ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
     GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);
 
-    if (!opt_arch) {
-        int err = get_hex_arch_ver(CDSP_DOMAIN_ID, &opt_arch);
-        if (err != 0) {
-            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
-            opt_arch = 73;
-        }
-    }
-
-#if defined(__ANDROID__)
-    if (opt_arch < 75) {
-        opt_ndev = 1;
-        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
-    }
-#endif
-
     GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d\n", opt_arch);
 
     // Create devices / sessions
@@ -3480,32 +3504,67 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
     static_assert((unsigned int) HTP_TYPE_IQ4_NL == (unsigned int) GGML_TYPE_IQ4_NL,
                   "please update hexagon_type to match ggml_type");
 
-    const char * str_verbose = getenv("GGML_HEXAGON_VERBOSE");
-    const char * str_hostbuf = getenv("GGML_HEXAGON_HOSTBUF");
-    const char * str_opstage = getenv("GGML_HEXAGON_OPSTAGE");
-    const char * str_opbatch = getenv("GGML_HEXAGON_OPBATCH");
-    const char * str_opqueue = getenv("GGML_HEXAGON_OPQUEUE");
-    const char * str_opfilter= getenv("GGML_HEXAGON_OPFILTER");
-    const char * str_profile = getenv("GGML_HEXAGON_PROFILE");
-    const char * str_etm     = getenv("GGML_HEXAGON_ETM");
-    const char * str_nhvx    = getenv("GGML_HEXAGON_NHVX");
-    const char * str_use_hmx = getenv("GGML_HEXAGON_USE_HMX");
-    const char * str_ndev    = getenv("GGML_HEXAGON_NDEV");
-    const char * str_arch    = getenv("GGML_HEXAGON_ARCH");
+    const char * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
+    const char * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
+    const char * str_opstage  = getenv("GGML_HEXAGON_OPSTAGE");
+    const char * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
+    const char * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
+    const char * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
+    const char * str_profile  = getenv("GGML_HEXAGON_PROFILE");
+    const char * str_etm      = getenv("GGML_HEXAGON_ETM");
+    const char * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
+    const char * str_use_hmx  = getenv("GGML_HEXAGON_USE_HMX");
+    const char * str_ndev     = getenv("GGML_HEXAGON_NDEV");
+    const char * str_arch     = getenv("GGML_HEXAGON_ARCH");
+    const char * str_vmem     = getenv("GGML_HEXAGON_VMEM");
+    const char * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
+
+    // Init Arch first since it affects other defaults
+    if (!str_arch) {
+        int err = get_hex_arch_ver(CDSP_DOMAIN_ID, &opt_arch);
+        if (err != 0) {
+            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
+            opt_arch = 73;
+        }
+    } else {
+        if (str_arch[0] == 'v' || str_arch[0] == 'V') {
+            str_arch++;
+        }
+        opt_arch = strtoul(str_arch, NULL, 0);
+    }
+
+    size_t MiB = 1024 * 1024;
+
+    // Update vmem default
+    opt_vmem = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
 
     auto RE_ICASE = std::regex_constants::icase;
 
-    opt_opfilter     = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
-    opt_verbose      = str_verbose  ? atoi(str_verbose)             : 0;
-    opt_hostbuf      = str_hostbuf  ? atoi(str_hostbuf)             : opt_hostbuf;
-    opt_opstage      = str_opstage  ? strtoul(str_opstage, NULL, 0) : opt_opstage;
-    opt_opbatch      = str_opbatch  ? strtoul(str_opbatch, NULL, 0) : opt_opbatch;
-    opt_opqueue      = str_opqueue  ? strtoul(str_opqueue, NULL, 0) : opt_opqueue;
-    opt_etm          = str_etm      ? atoi(str_etm)                 : 0;
-    opt_nhvx         = str_nhvx     ? strtoul(str_nhvx, NULL, 0)    : opt_nhvx;
-    opt_use_hmx      = str_use_hmx  ? atoi(str_use_hmx)             : opt_use_hmx;
-    opt_ndev         = str_ndev     ? strtoul(str_ndev, NULL, 0)    : opt_ndev;
-    opt_hostbuf      = str_hostbuf  ? atoi(str_hostbuf)             : opt_hostbuf;
+    opt_opfilter  = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
+    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
+    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf)                      : opt_hostbuf;
+    opt_opstage   = str_opstage  ? strtoul(str_opstage, NULL, 0)          : opt_opstage;
+    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, NULL, 0)          : opt_opbatch;
+    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, NULL, 0)          : opt_opqueue;
+    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
+    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
+    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, NULL, 0)             : opt_nhvx;
+    opt_use_hmx   = str_use_hmx  ? atoi(str_use_hmx)                      : opt_use_hmx;
+    opt_ndev      = str_ndev     ? strtoul(str_ndev, NULL, 0)             : opt_ndev;
+    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf)                      : opt_hostbuf;
+    opt_mbuf      = str_mbuf     ? strtoul(str_mbuf, NULL, 0) * MiB       : opt_mbuf;
+    opt_vmem      = str_vmem     ? strtoul(str_vmem, NULL, 0) * MiB       : opt_vmem;
+
+    if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
+        opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
+    }
+
+#if defined(__ANDROID__)
+    if (opt_arch < 75) {
+        opt_ndev = 1;
+        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
+    }
+#endif
 
     if (str_profile) {
         opt_pmu_evt = [&]() -> std::vector<uint32_t> {
@@ -3520,17 +3579,6 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
                 vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
     }
 
-    if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
-        opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
-    }
-
-    if (str_arch) {
-        if (str_arch[0] == 'v') {
-            str_arch++;
-        }
-        opt_arch = strtoul(str_arch, NULL, 0);
-    }
-
     reg->context = new ggml_hexagon_registry(reg);
 }
 
diff --git src/ggml-hexagon/htp/htp-ctx.h src/ggml-hexagon/htp/htp-ctx.h
index d704fede..e9c563ca 100644
--- src/ggml-hexagon/htp/htp-ctx.h
+++ src/ggml-hexagon/htp/htp-ctx.h
@@ -20,7 +20,7 @@ struct htp_mmap {
     uint64_t size;
     uint64_t base;
     uint32_t fd;
-    uint32_t pinned;
+    uint32_t reserved;
 };
 
 // Scratchpad state
@@ -77,6 +77,8 @@ struct htp_context {
     atomic_bool            vtcm_valid;
     atomic_bool            vtcm_needs_release;
 
+    uint64_t               max_vmem;
+
     struct htp_ops_context octx;
 
 #ifdef HTP_HAS_HMX
diff --git src/ggml-hexagon/htp/htp-ops.h src/ggml-hexagon/htp/htp-ops.h
index 4397245c..66a3150c 100644
--- src/ggml-hexagon/htp/htp-ops.h
+++ src/ggml-hexagon/htp/htp-ops.h
@@ -90,15 +90,11 @@ enum htp_op_code {
 #define HTP_OP_MAX_INPUTS  6    // aka GGML_MAX_SRCS
 #define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
 
-#define HTP_OP_MAX_BUFS    8
+#define HTP_OP_MAX_BUFS    16
 #define HTP_OP_MAX_REQS    256
 #define HTP_OP_MAX_TENSORS (HTP_OP_MAX_REQS * HTP_OP_MAX_INPUTS + HTP_OP_MAX_REQS)
 
-#if __HVX_ARCH__ < 75
-#define HTP_OP_MAX_VMEM    (3167538380u)
-#else
-#define HTP_OP_MAX_VMEM    (3221225472u)
-#endif
+#define HTP_OP_MAX_VMEM_DEFAULT (3355443200u)
 
 #define HTP_MMAP_MAX_VMEM  (2147483648u)
 
diff --git src/ggml-hexagon/htp/htp_iface.idl src/ggml-hexagon/htp/htp_iface.idl
index dbcafd1d..d696a5fb 100644
--- src/ggml-hexagon/htp/htp_iface.idl
+++ src/ggml-hexagon/htp/htp_iface.idl
@@ -11,9 +11,9 @@ struct htp_iface_pmu_conf {
 };
 
 interface htp_iface : remote_handle64 {
-    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx, in uint32 use_hmx);
+    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx, in uint32 use_hmx, in uint64 max_vmem);
     AEEResult stop();
-    AEEResult mmap(in uint32 fd, in uint32 size, in uint32 pinned);
+    AEEResult mmap(in uint32 fd, in uint32 size);
     AEEResult munmap(in uint32 fd);
     AEEResult profiler(in uint32 mode, in htp_iface_pmu_conf pmu);
     AEEResult etm(in uint32 enable);
diff --git src/ggml-hexagon/htp/main.c src/ggml-hexagon/htp/main.c
index f5834730..49c1a15b 100644
--- src/ggml-hexagon/htp/main.c
+++ src/ggml-hexagon/htp/main.c
@@ -210,7 +210,7 @@ AEEResult htp_iface_close(remote_handle64 handle) {
     return AEE_SUCCESS;
 }
 
-AEEResult htp_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size, uint32 pinned) {
+AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint32_t size) {
     struct htp_context * ctx = (struct htp_context *) handle;
     if (!ctx) {
         return AEE_EBADPARM;
@@ -220,7 +220,6 @@ AEEResult htp_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size, uint32
     for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
         struct htp_mmap *m = &ctx->mmap[i];
         if (m->fd == fd) {
-            m->pinned = pinned;
             return AEE_SUCCESS;
         }
     }
@@ -229,7 +228,7 @@ AEEResult htp_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size, uint32
     for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) {
         struct htp_mmap *m = &ctx->mmap[i];
         if (!m->size) {
-            FARF(HIGH, "mmap : fd %u size %u pinned %u", fd, size, pinned);
+            FARF(HIGH, "mmap : fd %u size %u", fd, size);
 #if __HVX_ARCH__ > 73
             void *va = HAP_mmap2(NULL, size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
 #else
@@ -248,7 +247,6 @@ AEEResult htp_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size, uint32
             m->base   = (uint64_t) va;
             m->fd     = fd;
             m->size   = size;
-            m->pinned = pinned;
 
             return AEE_SUCCESS;
         }
@@ -275,7 +273,6 @@ AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
             m->size   = 0;
             m->base   = NULL;
             m->fd     = -1;
-            m->pinned = 0;
         }
     }
 
@@ -358,7 +355,7 @@ static void vtcm_free(struct htp_context * ctx) {
 static void htp_packet_callback(dspqueue_t queue, int error, void * context);
 static void htp_error_callback(dspqueue_t queue, int error, void * context);
 
-AEEResult htp_iface_start(remote_handle64 handle, uint32 sess_id, uint64 dsp_queue_id, uint32 n_hvx, uint32 use_hmx) {
+AEEResult htp_iface_start(remote_handle64 handle, uint32 sess_id, uint64 dsp_queue_id, uint32 n_hvx, uint32 use_hmx, uint64_t max_vmem) {
     struct htp_context * ctx = (struct htp_context *) handle;
 
     if (!ctx) {
@@ -376,12 +373,12 @@ AEEResult htp_iface_start(remote_handle64 handle, uint32 sess_id, uint64 dsp_que
                               htp_error_callback,   // Error callback; no errors expected on the DSP
                               (void *) ctx,         // Callback context
                               &ctx->queue);
-
     if (err) {
         FARF(ERROR, "Queue import failed with 0x%08x", (unsigned) err);
         return err;
     }
 
+    ctx->max_vmem    = max_vmem;
     ctx->thread_id   = qurt_thread_get_id();
     ctx->thread_prio = qurt_thread_get_priority(ctx->thread_id);
 
@@ -622,8 +619,8 @@ static inline bool reuse_buf(struct htp_context *ctx, uint32_t *m_reuse, struct
 }
 
 static inline void drop_mmap(struct htp_context *ctx, struct htp_mmap *m) {
-    if (m->size && !m->pinned) {
-        FARF(HIGH, "unmap : fd %u base %p size %u pinned %u", m->fd, (void*) m->base, (uint32_t) m->size, m->pinned);
+    if (m->size) {
+        FARF(HIGH, "unmap : fd %u base %p size %u", m->fd, (void*) m->base, (uint32_t) m->size);
 #if __HVX_ARCH__ > 73
         HAP_munmap2((void *) m->base, m->size);
 #else
@@ -660,9 +657,8 @@ static inline void mmap_buf(struct htp_context *ctx, struct htp_buf_desc *b) {
             m->base   = b->base = (uint64_t) va;
             m->fd     = b->fd;
             m->size   = b->size;
-            m->pinned = 0;
 
-            FARF(HIGH, "mmap : fd %u base %p size %u pinned %u", m->fd, (void*) m->base, (uint32_t) m->size, m->pinned);
+            FARF(HIGH, "mmap : fd %u base %p size %u", m->fd, (void*) m->base, (uint32_t) m->size);
             return;
         }
     }
@@ -672,8 +668,8 @@ static void prep_op_bufs(struct htp_context *ctx, struct htp_buf_desc *bufs, uin
     uint32_t m_reuse = 0; // mmap reuse mask (index from ctx->mmap array)
     uint32_t b_reuse = 0; // buf reuse count
 
-    size_t   m_vmem  = 0; // mapped vmem
-    size_t   e_vmem  = 0; // extra  vmem
+    uint64_t m_vmem  = 0; // mapped vmem
+    uint64_t e_vmem  = 0; // extra  vmem
 
     // See what we can reuse
     for (uint32_t i=0; i < n_bufs; i++) {
@@ -687,9 +683,10 @@ static void prep_op_bufs(struct htp_context *ctx, struct htp_buf_desc *bufs, uin
     // See how much vmem we have mmaped right now
     for (uint32_t i=0; i<HTP_MAX_MMAPS; i++) { m_vmem += ctx->mmap[i].size; }
 
-    FARF(HIGH, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu n-bufs %u b-reuse %u", m_vmem, e_vmem, n_bufs, b_reuse);
+    FARF(HIGH, "prep-bufs : pass1 mmap-vmem %zu extra-vmem %zu max-vmem %zu : n-bufs %u b-reuse %u",
+            (size_t) m_vmem, (size_t) e_vmem, (size_t) ctx->max_vmem, n_bufs, b_reuse);
 
-    if ((m_vmem + e_vmem) > HTP_OP_MAX_VMEM) {
+    if ((m_vmem + e_vmem) > ctx->max_vmem) {
         // Drop unused mappings
         for (uint32_t i=0; i < HTP_MAX_MMAPS; i++) {
             bool used = m_reuse & (1<<i);
