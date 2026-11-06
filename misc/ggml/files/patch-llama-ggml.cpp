diff --git include/ggml.h include/ggml.h
index 3ed41814..5a5ba966 100644
--- include/ggml.h
+++ include/ggml.h
@@ -2034,10 +2034,6 @@ extern "C" {
             float                 beta_slow),
         "use ggml_rope_ext_inplace instead");
 
-    // compute correction dims for YaRN RoPE scaling
-    GGML_API void ggml_rope_yarn_corr_dims(
-        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);
-
     // rotary position embedding backward, i.e compute dx from dy
     // a - dy
     GGML_API struct ggml_tensor * ggml_rope_ext_back(
@@ -2080,6 +2076,11 @@ extern "C" {
             struct ggml_tensor  * a,
             int                   n_offs);
 
+    // control rounding of YaRN correction bounds (default: true)
+    GGML_API struct ggml_tensor * ggml_rope_set_truncate(
+            struct ggml_tensor * a,
+            bool                 truncate);
+
     // im2col
     // converts data into a format that effectively results in a convolution when combined with matrix multiplication
     GGML_API struct ggml_tensor * ggml_im2col(
diff --git src/ggml-cann/aclnn_ops.cpp src/ggml-cann/aclnn_ops.cpp
index 902d2eda..66e011ef 100644
--- src/ggml-cann/aclnn_ops.cpp
+++ src/ggml-cann/aclnn_ops.cpp
@@ -2934,7 +2934,7 @@ void ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     const float theta_scale = powf(freq_base, -2.0f / n_dims);
 
     float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, true, corr_dims);
 
     bool       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
     const bool is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl apply interleaved mrope
@@ -3289,7 +3289,7 @@ void ggml_cann_rope_cache_preload(ggml_backend_cann_context & ctx, ggml_tensor *
     const float theta_scale = powf(freq_base, -2.0f / n_dims);
 
     float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, true, corr_dims);
 
     bool       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
     const bool is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;
diff --git src/ggml-cann/ggml-cann.cpp src/ggml-cann/ggml-cann.cpp
index d3a47f83..fd2c80bd 100644
--- src/ggml-cann/ggml-cann.cpp
+++ src/ggml-cann/ggml-cann.cpp
@@ -2547,6 +2547,9 @@ static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, const ggml_ten
             }
         case GGML_OP_ROPE:
             {
+                if (!ggml_rope_get_truncate(op)) {
+                    return false;
+                }
                 if (((const int32_t *) op->op_params)[15] != 0) {
                     return false; // FIXME: support ggml_rope_set_offset
                 }
diff --git src/ggml-cpu/ops.cpp src/ggml-cpu/ops.cpp
index 00712833..55f62555 100644
--- src/ggml-cpu/ops.cpp
+++ src/ggml-cpu/ops.cpp
@@ -6310,8 +6310,8 @@ static void ggml_compute_forward_rope_flt(
     //const int n_past     = ((int32_t *) dst->op_params)[0];
     const int n_dims     = ((int32_t *) dst->op_params)[1];
     const int mode       = ((int32_t *) dst->op_params)[2];
-    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
     const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
+    const bool truncate = ggml_rope_get_truncate(dst);
 
     memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
     memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
@@ -6356,7 +6356,7 @@ static void ggml_compute_forward_rope_flt(
     const float theta_scale = powf(freq_base, -2.0f/n_dims);
 
     float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, truncate, corr_dims);
 
     const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
     const bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
diff --git src/ggml-cuda/common.cuh src/ggml-cuda/common.cuh
index 00476f0f..bc83ea37 100644
--- src/ggml-cuda/common.cuh
+++ src/ggml-cuda/common.cuh
@@ -1585,6 +1585,7 @@ struct ggml_cuda_mm_fusion_args_host {
     const ggml_tensor * gate_scale = nullptr;
     ggml_glu_op glu_op;
     float glu_limit = 0.0f;
+    ggml_prec prec_src1 = GGML_PREC_UNDEFINED; // src1 precision requested by the fused MUL_MATs (MMQ)
     const ggml_tensor * shared_up = nullptr;
     const ggml_tensor * shared_gate = nullptr;
     ggml_tensor * shared_dst = nullptr;
diff --git src/ggml-cuda/fattn-common.cuh src/ggml-cuda/fattn-common.cuh
index beb87794..26a432d8 100644
--- src/ggml-cuda/fattn-common.cuh
+++ src/ggml-cuda/fattn-common.cuh
@@ -428,17 +428,18 @@ static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict
 
     const int8_t * q8 = (const int8_t *) &q;
 
-#ifdef FP16_AVAILABLE
     if constexpr (std::is_same_v<T, half>) {
+#ifdef FP16_AVAILABLE
         const half2 d = __half2half2(x[ib].d);
 
 #pragma unroll
         for (int l0 = 0; l0 < ne; l0 += 2) {
             ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
         }
-    } else
+#else
+        NO_DEVICE_CODE;
 #endif // FP16_AVAILABLE
-    if constexpr (std::is_same_v<T, float>) {
+    } else if constexpr (std::is_same_v<T, float>) {
         const float d = x[ib].d;
 
 #pragma unroll
@@ -466,8 +467,8 @@ static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict
 
     const int8_t * q8 = (const int8_t *) &q;
 
-#ifdef FP16_AVAILABLE
     if constexpr (std::is_same_v<T, half>) {
+#ifdef FP16_AVAILABLE
         const half2 dm = x[ib].dm;
         const half2 d  = __half2half2( __low2half(dm));
         const half2 m  = __half2half2(__high2half(dm));
@@ -476,9 +477,10 @@ static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict
         for (int l0 = 0; l0 < ne; l0 += 2) {
             ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
         }
-    } else
+#else
+        NO_DEVICE_CODE;
 #endif // FP16_AVAILABLE
-    if constexpr (std::is_same_v<T, float>) {
+    } else if constexpr (std::is_same_v<T, float>) {
         const float2 dm = __half22float2(x[ib].dm);
 
 #pragma unroll
@@ -518,17 +520,18 @@ static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict
 
     const int8_t * q8 = (const int8_t *) &q;
 
-#ifdef FP16_AVAILABLE
     if constexpr (std::is_same_v<T, half>) {
+#ifdef FP16_AVAILABLE
         const half2 d = __half2half2(x[ib].d);
 
 #pragma unroll
         for (int l0 = 0; l0 < ne; l0 += 2) {
             ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
         }
-    } else
+#else
+        NO_DEVICE_CODE;
 #endif // FP16_AVAILABLE
-    if constexpr (std::is_same_v<T, float>) {
+    } else if constexpr (std::is_same_v<T, float>) {
         const float d = x[ib].d;
 
 #pragma unroll
@@ -566,8 +569,8 @@ static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict
 
     const int8_t * q8 = (const int8_t *) &q;
 
-#ifdef FP16_AVAILABLE
     if constexpr (std::is_same_v<T, half>) {
+#ifdef FP16_AVAILABLE
         const half2 dm = x[ib].dm;
         const half2 d  = __half2half2( __low2half(dm));
         const half2 m  = __half2half2(__high2half(dm));
@@ -576,9 +579,10 @@ static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict
         for (int l0 = 0; l0 < ne; l0 += 2) {
             ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
         }
-    } else
+#else
+        NO_DEVICE_CODE;
 #endif // FP16_AVAILABLE
-    if constexpr (std::is_same_v<T, float>) {
+    } else if constexpr (std::is_same_v<T, float>) {
         const float2 dm = __half22float2(x[ib].dm);
 
 #pragma unroll
@@ -601,17 +605,18 @@ static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict
     int8_t qs[ne];
     ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);
 
-#ifdef FP16_AVAILABLE
     if constexpr (std::is_same<T, half>::value) {
+#ifdef FP16_AVAILABLE
         const half2 d = __half2half2(x[ib].d);
 
 #pragma unroll
         for (int l0 = 0; l0 < ne; l0 += 2) {
             ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
         }
-    } else
+#else
+        NO_DEVICE_CODE;
 #endif // FP16_AVAILABLE
-    if constexpr (std::is_same<T, float>::value) {
+    } else if constexpr (std::is_same<T, float>::value) {
         const float d = x[ib].d;
 
 #pragma unroll
diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index ee28955f..9c467ce9 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -2151,8 +2151,50 @@ static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_
     return true;
 }
 
+// The NVFP4 weight scale that scale_node (MUL) applies to the output of mm_node (MUL_MAT), or nullptr.
+// Shared by ggml_cuda_try_fuse and graph_optimize so that both accept the same patterns.
+static const ggml_tensor * ggml_cuda_get_mul_mat_scale(const ggml_tensor * scale_node, const ggml_tensor * mm_node) {
+    const bool scale_lhs_mm = scale_node->src[0] == mm_node;
+    const bool scale_rhs_mm = scale_node->src[1] == mm_node;
+    if (!scale_lhs_mm && !scale_rhs_mm) {
+        return nullptr;
+    }
+
+    const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
+    if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
+            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
+            !ggml_are_same_shape(scale_node, mm_node)) {
+        return nullptr;
+    }
+
+    return scale;
+}
+
+// The same for MUL_MAT_ID with one scale per expert: RESHAPE, REPEAT and GET_ROWS (by the ids) of the scales, then the MUL.
+static const ggml_tensor * ggml_cuda_get_mul_mat_id_scale(const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
+        const ggml_tensor * scale_node, const ggml_tensor * mm_node) {
+    if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
+        return nullptr;
+    }
+    if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
+            (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
+        return nullptr;
+    }
+
+    const ggml_tensor * scale = reshape->src[0];
+    if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
+            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
+            !ggml_are_same_shape(scale_node, mm_node)) {
+        return nullptr;
+    }
+
+    return scale;
+}
+
 // Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (see #29986).
-static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device) {
+// up_scale/gate_scale: the MUL nodes that apply the NVFP4 weight scales (per tensor or per expert) between the MUL_MATs and the GLU.
+static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device,
+        const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
     const ggml_tensor * x_up = up->src[0];
     const ggml_tensor * y    = up->src[1];
 
@@ -2165,7 +2207,12 @@ static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggm
         return false;
     }
 
-    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu)) {
+    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu, nullptr, nullptr, up_scale, gate_scale)) {
+        return false;
+    }
+
+    // The fused kernel runs both MUL_MATs at one src1 precision, see ggml_cuda_mul_mat_q.
+    if (ggml_prec_get_src(up, 1) != ggml_prec_get_src(gate, 1)) {
         return false;
     }
 
@@ -2178,7 +2225,8 @@ static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggm
     return turing_mma_available(ggml_cuda_info().devices[device].cc);
 }
 
-static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu) {
+static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu,
+        const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
     // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
     if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
         return false;
@@ -2190,7 +2238,7 @@ static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_t
             ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
     };
     const int device = ggml_cuda_get_device();
-    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
+    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device, up_scale, gate_scale) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
         return false;
     }
 
@@ -2209,7 +2257,7 @@ static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_t
     }
 
     // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
-    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, x_up->ne[1], ncols, nchannels, device);
+    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, ggml_prec_get_src(up, 1), x_up->ne[1], ncols, nchannels, device);
 }
 
 static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
@@ -4284,43 +4332,6 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
     bool fused_mul_mat_vec = false;
     int  fused_node_count  = 0;
 
-    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
-        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
-        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
-        if (!scale_lhs_mm && !scale_rhs_mm) {
-            return nullptr;
-        }
-
-        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
-        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
-                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
-                !ggml_are_same_shape(scale_node, mm_node)) {
-            return nullptr;
-        }
-
-        return scale;
-    };
-
-    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
-            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
-        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
-            return nullptr;
-        }
-        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
-                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
-            return nullptr;
-        }
-
-        const ggml_tensor * scale = reshape->src[0];
-        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
-                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
-                !ggml_are_same_shape(scale_node, mm_node)) {
-            return nullptr;
-        }
-
-        return scale;
-    };
-
     auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
         if (op_bias == GGML_OP_ADD) {
             if (bias_node->src[0] == mul_node) {
@@ -4387,8 +4398,8 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
                     continue;
                 }
 
-                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
-                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
+                const ggml_tensor * gate_scale = ggml_cuda_get_mul_mat_scale(gate_scale_n, gate_n);
+                const ggml_tensor * up_scale   = ggml_cuda_get_mul_mat_scale(up_scale_n, up_n);
                 if (!gate_scale || !up_scale) {
                     continue;
                 }
@@ -4419,6 +4430,14 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
                     fused_node_count  = n_ops;
                     break;
                 }
+
+                if (!with_bias && ggml_cuda_should_fuse_mul_mat_q(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
+                    fusion_data.prec_src1 = ggml_prec_get_src(up_n, 1);
+                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
+                    fused_mul_mat_vec = true;
+                    fused_node_count  = n_ops;
+                    break;
+                }
             }
 
             if (fused_mul_mat_vec) {
@@ -4483,9 +4502,9 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
                     continue;
                 }
 
-                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
+                const ggml_tensor * gate_scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                         cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
-                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
+                const ggml_tensor * up_scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                         cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                 if (!gate_scale || !up_scale) {
                     continue;
@@ -4513,6 +4532,14 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
                     fused_node_count  = n_ops;
                     break;
                 }
+
+                if (!with_bias && ggml_cuda_should_fuse_mul_mat_q(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
+                    fusion_data.prec_src1 = ggml_prec_get_src(up_n, 1);
+                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
+                    fused_mul_mat_vec = true;
+                    fused_node_count  = n_ops;
+                    break;
+                }
             }
 
             if (fused_mul_mat_vec) {
@@ -4628,6 +4655,7 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
                 fusion_data.gate      = gate->src[0];
                 fusion_data.glu_op    = ggml_get_glu_op(glu);
                 fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);
+                fusion_data.prec_src1 = ggml_prec_get_src(up, 1);
 
                 ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                 fused_mul_mat_vec = true;
@@ -4689,9 +4717,9 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
 
             const ggml_tensor * scale = nullptr;
             if (op == GGML_OP_MUL_MAT) {
-                scale = get_mul_mat_scale(scale_node, mm_node);
+                scale = ggml_cuda_get_mul_mat_scale(scale_node, mm_node);
             } else {
-                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
+                scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
             }
             if (!scale) {
                 continue;
@@ -5242,6 +5270,33 @@ static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph
                     continue;
                 }
             }
+            // Same with the NVFP4 scales: MUL_MAT(gate), MUL(gate scale), MUL_MAT(up), MUL(up scale), GLU.
+            if (i + 4 < cgraph->n_nodes && cgraph->nodes[i + 4]->op == GGML_OP_GLU) {
+                ggml_tensor * glu        = cgraph->nodes[i + 4];
+                ggml_tensor * gate_scale = cgraph->nodes[i + 1];
+                ggml_tensor * up_scale   = cgraph->nodes[i + 3];
+                if (glu->src[0] == gate_scale && glu->src[1] == up_scale &&
+                        ggml_cuda_mul_mat_q_fusion_matches(cgraph->nodes[i + 2], cgraph->nodes[i], glu, cuda_ctx->device, up_scale, gate_scale) &&
+                        ggml_cuda_get_mul_mat_scale(gate_scale, cgraph->nodes[i]) && ggml_cuda_get_mul_mat_scale(up_scale, cgraph->nodes[i + 2])) {
+                    add_alloc_deps(i, i + 4);
+                    i += 4;
+                    continue;
+                }
+            }
+            // Same with per-expert NVFP4 scales: MUL_MAT_ID, RESHAPE, REPEAT, GET_ROWS, MUL for gate and for up, GLU.
+            if (i + 10 < cgraph->n_nodes && cgraph->nodes[i + 10]->op == GGML_OP_GLU) {
+                ggml_tensor * glu        = cgraph->nodes[i + 10];
+                ggml_tensor * gate_scale = cgraph->nodes[i + 4];
+                ggml_tensor * up_scale   = cgraph->nodes[i + 9];
+                if (glu->src[0] == gate_scale && glu->src[1] == up_scale &&
+                        ggml_cuda_mul_mat_q_fusion_matches(cgraph->nodes[i + 5], cgraph->nodes[i], glu, cuda_ctx->device, up_scale, gate_scale) &&
+                        ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], gate_scale, cgraph->nodes[i]) &&
+                        ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 6], cgraph->nodes[i + 7], cgraph->nodes[i + 8], up_scale, cgraph->nodes[i + 5])) {
+                    add_alloc_deps(i, i + 10);
+                    i += 10;
+                    continue;
+                }
+            }
 
             ggml_cuda_moe_weighted_reduction_match match;
             if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
diff --git src/ggml-cuda/lightning-indexer.cu src/ggml-cuda/lightning-indexer.cu
index 429d2e20..402477d2 100644
--- src/ggml-cuda/lightning-indexer.cu
+++ src/ggml-cuda/lightning-indexer.cu
@@ -2,21 +2,22 @@
 #include "lightning-indexer.cuh"
 #include "fattn-common.cuh"
 #include "convert.cuh"
+#include "mma.cuh"
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
-#if defined(TURING_MMA_AVAILABLE)
+// tokens scored per block by the tile kernel
+#define LIGHTNING_INDEXER_TILE_TOKENS 8
 
-typedef union {
-    int2 i2;
-    half2 h2[2];
-} half4;
+// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
+// at 28 KB and the queries of four heads do not fit there next to the key tile
+#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
+#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
+#else
+#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
+#endif
 
-// TODO add support for AMD cards via rocWMMA
-#include <mma.h>
-namespace wmma = nvcuda::wmma;
 
 template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
-static __global__ void lightning_indexer_kernel_wmma(
+static __global__ void lightning_indexer_kernel_mma(
         const float * Q, const char * K, const float * W, const half * M, float * dst,
         int64_t n_stream, int64_t n_batch, int64_t n_kv,
         size_t nb1, size_t nb2, size_t nb3,
@@ -26,227 +27,164 @@ static __global__ void lightning_indexer_kernel_wmma(
         size_t nbm1, size_t nbm2, size_t nbm3,
         int64_t nem3
     ) {
-
-    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
-    constexpr int HEADS_PER_INNER_LOOP = 8;
-    constexpr int K_EMBD_PER_INNER_LOOP = 16;
+    constexpr int MMA_DIM = 16;
+    constexpr int WAVE_SIZE = ggml_cuda_get_physical_warp_size();
+    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WAVE_SIZE;
     constexpr int N_EMBD_PADDED = N_EMBD + 8;
+    constexpr int HEAD_CHUNK    = MMA_DIM;
+    constexpr int N_HEAD_CHUNKS = N_HEAD / HEAD_CHUNK;
+    constexpr int N_KV_TILES    = K_VECS_PER_BLOCK / MMA_DIM;
+    constexpr int N_EMBD_TILES  = N_EMBD / MMA_DIM;
+
+    static_assert(N_HEAD           % MMA_DIM == 0, "N_HEAD must be a multiple of 16");
+    static_assert(N_EMBD           % MMA_DIM == 0, "N_EMBD must be a multiple of 16");
+    static_assert(K_VECS_PER_BLOCK % MMA_DIM == 0, "K_VECS_PER_BLOCK must be a multiple of 16");
+
+#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
+    using tile_qk = ggml_cuda_mma::tile<MMA_DIM, MMA_DIM, float, ggml_cuda_mma::DATA_LAYOUT_J_MAJOR>;
+#else
+    using tile_qk = ggml_cuda_mma::tile<MMA_DIM, MMA_DIM, float>;
+#endif
 
     const int i_batch  = blockIdx.y;
     const int i_stream = blockIdx.z;
     const int i_warp   = threadIdx.y;
     const int i_lane   = threadIdx.x;
-    const int tid      = i_warp * WARP_SIZE + i_lane;
+    const int tid      = i_warp * WAVE_SIZE + i_lane;
 
-    // each block processes K_VECS_PER_BLOCK K vectors
     const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;
 
     const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
     const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);
 
-    // phase 1 - load weights and first Q tile to shared memory
-
     __shared__ float w_shared[N_HEAD];
-    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];
+    __shared__ half  q_shared[2][HEAD_CHUNK][N_EMBD_PADDED];
+    __shared__ half  k_shared[K_VECS_PER_BLOCK][N_EMBD_PADDED];
+    __shared__ float qk_chunk[HEAD_CHUNK][K_VECS_PER_BLOCK];
+    __shared__ float score_shared[K_VECS_PER_BLOCK];
 
-    if (tid < N_HEAD) {
-        w_shared[tid] = w_base[tid];
+    for (int i = tid; i < N_HEAD; i += THREADS_PER_BLOCK) {
+        w_shared[i] = w_base[i];
     }
-
-    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
-    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
-    // number of registers needed in each thread to store Q tile in thread block
-    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
-
-#pragma unroll
-    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
-        const int i_head = i_q / (N_EMBD / 4);
-        const int i_embd = i_q % (N_EMBD / 4);
-        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
-        half4 q_packed;
-        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
-        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
-        q_shared_h[i_head][i_embd] = q_packed.i2;
+    for (int i = tid; i < K_VECS_PER_BLOCK; i += THREADS_PER_BLOCK) {
+        score_shared[i] = 0.0f;
     }
 
-    // phase 2 - load (and dequantize if needed) K to shared mem
-
-    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];
-
-    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);
-
+    constexpr int N_K_VEC = K_VECS_PER_BLOCK * (N_EMBD / 4);
     if constexpr (TYPE_K == GGML_TYPE_F16) {
-#pragma unroll
-        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
-            const int i_k_vec = i_k / (N_EMBD / 4);
-            const int i_embd = i_k % (N_EMBD / 4);
-            const int i_kv = start_kv + i_k_vec;
+        for (int i = tid; i < N_K_VEC; i += THREADS_PER_BLOCK) {
+            const int i_k_vec = i / (N_EMBD / 4);
+            const int i_embd4 = i % (N_EMBD / 4);
+            const int i_kv    = start_kv + i_k_vec;
             if (i_kv < n_kv) {
-                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
-                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
+                const int2 * k_ptr = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
+                *(int2 *) &k_shared[i_k_vec][i_embd4*4] = k_ptr[i_embd4];
             } else {
-                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
+                *(int2 *) &k_shared[i_k_vec][i_embd4*4] = make_int2(0, 0);
             }
         }
     } else {
         constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
-#pragma unroll
-        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
-            const int i_k_vec = i_k / (N_EMBD / 4);
-            const int i_embd = i_k % (N_EMBD / 4);
-            const int i_kv = start_kv + i_k_vec;
+        for (int i = tid; i < N_K_VEC; i += THREADS_PER_BLOCK) {
+            const int i_k_vec = i / (N_EMBD / 4);
+            const int i_embd4 = i % (N_EMBD / 4);
+            const int i_kv    = start_kv + i_k_vec;
             if (i_kv < n_kv) {
-                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
-                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
+                const void * k_ptr = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
+                dequantize_k(k_ptr, &k_shared[i_k_vec][i_embd4*4], i_embd4*4);
             } else {
-                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
+                *(int2 *) &k_shared[i_k_vec][i_embd4*4] = make_int2(0, 0);
             }
         }
     }
 
-    __syncthreads();
-
-    // phase 3 - calculate lightning indexer scores
-
-    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];
-
-    // load K fragment
-    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
-    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);
-
-    float score_k = 0.0f;
+    constexpr int N_Q_CHUNK_VEC = HEAD_CHUNK * (N_EMBD / 4);
+    constexpr int N_Q_NEXT = (N_Q_CHUNK_VEC + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
+
+    for (int i = tid; i < N_Q_CHUNK_VEC; i += THREADS_PER_BLOCK) {
+        const int i_head  = i / (N_EMBD / 4);
+        const int i_embd4 = i % (N_EMBD / 4);
+        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd4*sizeof(float4));
+        q_shared[0][i_head][i_embd4*4 + 0] = (half) q.x;
+        q_shared[0][i_head][i_embd4*4 + 1] = (half) q.y;
+        q_shared[0][i_head][i_embd4*4 + 2] = (half) q.z;
+        q_shared[0][i_head][i_embd4*4 + 3] = (half) q.w;
+    }
 
-    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
-        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;
+    __syncthreads();
 
-        // we don't use accumulator for anything, fill it with zeros
-        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
-        wmma::fill_fragment(frag_acc, 0.0f);
+    constexpr ggml_cuda_mma::data_layout input_layout = ggml_cuda_mma::get_input_data_layout();
 
-        // load Q fragment
-        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
-        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);
+    for (int hc = 0; hc < N_HEAD_CHUNKS; ++hc) {
+        const int buf  = hc & 1;
+        const int nbuf = (hc + 1) & 1;
+        const bool has_next = (hc + 1) < N_HEAD_CHUNKS;
 
-        // preload next Q tile to registers during matrix multiplication
         float4 q_next[N_Q_NEXT];
-
-        if (i_head_next < N_HEAD) {
+        if (has_next) {
 #pragma unroll
-            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
-                const int i_head = i_head_next + i_q / (N_EMBD / 4);
-                const int i_embd =               i_q % (N_EMBD / 4);
-                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
+            for (int i = tid, j = 0; i < N_Q_CHUNK_VEC; i += THREADS_PER_BLOCK, ++j) {
+                const int i_head  = (hc + 1) * HEAD_CHUNK + i / (N_EMBD / 4);
+                const int i_embd4 =                         i % (N_EMBD / 4);
+                q_next[j] = *(const float4 *) (q_base + i_head*nbq1 + i_embd4*sizeof(float4));
             }
         }
 
-        // perform matrix multiplication
-        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
-        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);
-
-        // make sure all threads finished using q_shared_h so we can store next tile
-        __syncthreads();
-
-        // write preloaded Q tile to shared memory
-        if (i_head_next < N_HEAD) {
+        for (int kt = i_warp; kt < N_KV_TILES; kt += WARPS_PER_BLOCK) {
+            tile_qk Dqk;
 #pragma unroll
-            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
-                const int i_head = i_q / (N_EMBD / 4);
-                const int i_embd = i_q % (N_EMBD / 4);
-                half4 q_packed;
-                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
-                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
-                q_shared_h[i_head][i_embd] = q_packed.i2;
-                ++i_q_next;
+            for (int et = 0; et < N_EMBD_TILES; ++et) {
+                ggml_cuda_mma::tile<MMA_DIM, MMA_DIM/2, half2, input_layout> Aq;
+                ggml_cuda_mma::tile<MMA_DIM, MMA_DIM/2, half2, input_layout> Bk;
+                ggml_cuda_mma::load_generic(Aq, (const half2 *) &q_shared[buf][0][et*MMA_DIM], N_EMBD_PADDED/2);
+                ggml_cuda_mma::load_generic(Bk, (const half2 *) &k_shared[kt*MMA_DIM][et*MMA_DIM], N_EMBD_PADDED/2);
+                ggml_cuda_mma::mma(Dqk, Aq, Bk);
             }
-        }
-
-        // accumulate QK multiplication results from all block warps
-        // (there are 256 threads in block and 256 matmul outputs)
-        // TODO it will break if WARP_SIZE is not 32
-        const int h = tid / K_VECS_PER_BLOCK;
-        const int k = tid % K_VECS_PER_BLOCK;
-        const float w_val = w_shared[i_head_0 + h];
-
-        float sum = 0.0f;
 #pragma unroll
-        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
-            sum += qk_shared[w][h][k];
+            for (int l = 0; l < Dqk.ne; ++l) {
+                const int head_local = Dqk.get_i(l);
+                const int kv_local   = kt*MMA_DIM + Dqk.get_j(l);
+                qk_chunk[head_local][kv_local] = Dqk.x[l];
+            }
         }
 
-        // ReLU, weight
-        sum = sum > 0.0f ? sum : 0.0f;
-        sum *= w_val;
-
-        // wait until qk_shared[0] is no longer used
         __syncthreads();
 
-        // reuse qk_shared[0] for storing partial results
-        qk_shared[0][h][k] = sum;
-
-        // wait until all threads write their results
-        __syncthreads();
+        if (has_next) {
+#pragma unroll
+            for (int i = tid, j = 0; i < N_Q_CHUNK_VEC; i += THREADS_PER_BLOCK, ++j) {
+                const int i_head  = i / (N_EMBD / 4);
+                const int i_embd4 = i % (N_EMBD / 4);
+                q_shared[nbuf][i_head][i_embd4*4 + 0] = (half) q_next[j].x;
+                q_shared[nbuf][i_head][i_embd4*4 + 1] = (half) q_next[j].y;
+                q_shared[nbuf][i_head][i_embd4*4 + 2] = (half) q_next[j].z;
+                q_shared[nbuf][i_head][i_embd4*4 + 3] = (half) q_next[j].w;
+            }
+        }
 
-        // accumulate result over heads
-        if (tid < K_VECS_PER_BLOCK) {
+        for (int kv = tid; kv < K_VECS_PER_BLOCK; kv += THREADS_PER_BLOCK) {
+            float s = 0.0f;
 #pragma unroll
-            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
-                score_k += qk_shared[0][i_head][tid];
+            for (int h = 0; h < HEAD_CHUNK; ++h) {
+                const float qk = qk_chunk[h][kv];
+                s += (qk > 0.0f ? qk : 0.0f) * w_shared[hc*HEAD_CHUNK + h];
             }
+            score_shared[kv] += s;
         }
 
-        // make sure all threads finished using qk_shared
         __syncthreads();
     }
 
-    // phase 4 - store output to VRAM
-
-    if (tid < K_VECS_PER_BLOCK) {
-        const int i_kv = start_kv + tid;
+    for (int i_kv_local = tid; i_kv_local < K_VECS_PER_BLOCK; i_kv_local += THREADS_PER_BLOCK) {
+        const int i_kv = start_kv + i_kv_local;
         if (i_kv < n_kv) {
-            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
-            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
-            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
+            const half  * m_base   = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
+            float       * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
+            dst_base[i_kv] = score_shared[i_kv_local] + __half2float(m_base[i_kv]);
         }
     }
 }
 
-#else // defined(TURING_MMA_AVAILABLE)
-
-template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
-static __global__ void lightning_indexer_kernel_wmma(
-        const float * Q, const char * K, const float * W, const half * M, float * dst,
-        int64_t n_stream, int64_t n_batch, int64_t n_kv,
-        size_t nb1, size_t nb2, size_t nb3,
-        size_t nbq1, size_t nbq2, size_t nbq3,
-        size_t nbk1, size_t nbk2, size_t nbk3,
-        size_t nbw1, size_t nbw2, size_t nbw3,
-        size_t nbm1, size_t nbm2, size_t nbm3,
-        int64_t nem3
-    ) {
-    GGML_UNUSED_VARS(Q, K, W, M, dst,
-        n_stream, n_batch, n_kv,
-        nb1, nb2, nb3,
-        nbq1, nbq2, nbq3,
-        nbk1, nbk2, nbk3,
-        nbw1, nbw2, nbw3,
-        nem3);
-    NO_DEVICE_CODE;
-}
-
-#endif // defined(TURING_MMA_AVAILABLE)
-#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
-
-// tokens scored per block by the tile kernel
-#define LIGHTNING_INDEXER_TILE_TOKENS 8
-
-// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
-// at 28 KB and the queries of four heads do not fit there next to the key tile
-#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
-#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
-#else
-#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
-#endif
-
 // TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
 // thanks to that one warp operating on float4 processes whole indexer K/Q vectors
 // 32 * 4 = 128 (N_EMBD)
@@ -616,27 +554,22 @@ void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor *
     const int cc     = ggml_cuda_info().devices[device].cc;
 
     if (n_embd == 128 && n_head == 64) {
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
-        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
-            // use wmma kernel
+        if ((turing_mma_available(cc) || amd_mfma_available(cc) || amd_wmma_available(cc)) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
             constexpr int K_VECS_PER_BLOCK = 32;
-            constexpr int WARPS_PER_BLOCK = 8;
+            constexpr int WARPS_PER_BLOCK  = 4;
 
-            dim3 block(32, WARPS_PER_BLOCK);
+            dim3 block(ggml_cuda_info().devices[device].warp_size, WARPS_PER_BLOCK);
             int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
             dim3 grid(num_kv_blocks, n_batch, n_stream);
 
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 64, k, GGML_TYPE_F16)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 64, k, GGML_TYPE_Q4_0)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 64, k, GGML_TYPE_Q4_1)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 64, k, GGML_TYPE_Q5_0)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 64, k, GGML_TYPE_Q5_1)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 64, k, GGML_TYPE_Q8_0)
             GGML_ABORT("fatal error");
         } else {
-#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
-        {
-#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
             // use vector kernel
             constexpr int K_VECS_PER_WARP = 8;
             constexpr int WARPS_PER_BLOCK = 8;
@@ -657,27 +590,22 @@ void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor *
             GGML_ABORT("fatal error");
         }
     } else if (n_embd == 128 && n_head == 32) {
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
-        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
-            // use wmma kernel
+        if ((turing_mma_available(cc) || amd_mfma_available(cc) || amd_wmma_available(cc)) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
             constexpr int K_VECS_PER_BLOCK = 32;
-            constexpr int WARPS_PER_BLOCK = 8;
+            constexpr int WARPS_PER_BLOCK  = 4;
 
-            dim3 block(32, WARPS_PER_BLOCK);
+            dim3 block(ggml_cuda_info().devices[device].warp_size, WARPS_PER_BLOCK);
             int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
             dim3 grid(num_kv_blocks, n_batch, n_stream);
 
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
-            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 32, k, GGML_TYPE_F16)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 32, k, GGML_TYPE_Q4_0)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 32, k, GGML_TYPE_Q4_1)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 32, k, GGML_TYPE_Q5_0)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 32, k, GGML_TYPE_Q5_1)
+            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_mma, 128, 32, k, GGML_TYPE_Q8_0)
             GGML_ABORT("fatal error");
         } else {
-#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
-        {
-#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
             // use vector kernel
             constexpr int K_VECS_PER_WARP = 8;
             constexpr int WARPS_PER_BLOCK = 8;
diff --git src/ggml-cuda/mmf.cu src/ggml-cuda/mmf.cu
index 646a5899..040d390e 100644
--- src/ggml-cuda/mmf.cu
+++ src/ggml-cuda/mmf.cu
@@ -50,7 +50,8 @@ void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * sr
     mmf_ids_data * ids_info_ptr = nullptr;
     ggml_cuda_pool_alloc<int32_t> ids_src_compact_dev;
     ggml_cuda_pool_alloc<int32_t> ids_dst_compact_dev;
-    ggml_cuda_pool_alloc<int32_t> expert_bounds_dev;
+
+    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec_dev;
 
     // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
     const int64_t ncols_dst          = ids ? ne2  : ne1;
@@ -74,25 +75,28 @@ void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * sr
         const int64_t n_experts     = ne02;
         const int64_t n_tokens      = ne12;
         const int64_t ne_get_rows   = n_tokens * n_expert_used;
+        const int64_t max_tiles     = (ne_get_rows + 16 - 1) / 16 + ne02;
 
         ids_src_compact_dev.alloc(ctx.pool(), ne_get_rows);
         ids_dst_compact_dev.alloc(ctx.pool(), ne_get_rows);
-        expert_bounds_dev.alloc(ctx.pool(), n_experts + 1);
+        tile_spec_dev.alloc(ctx.pool(), max_tiles);
 
         const int si1  = static_cast<int>(ids_s1);
         const int sis1 = static_cast<int>(src1->nb[2] / src1->nb[1]);
 
         GGML_ASSERT(sis1 > 0);
 
-        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), expert_bounds_dev.get(),
-            static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), si1, sis1, /*write_inverse =*/ false, ctx.stream());
+        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), tile_spec_dev.get(),
+            static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), 16,
+            si1, sis1, /*write_inverse =*/ false, ctx.stream());
         CUDA_CHECK(cudaGetLastError());
 
-        ids_info.ids_src_compact   = ids_src_compact_dev.get();
-        ids_info.ids_dst_compact   = ids_dst_compact_dev.get();
-        ids_info.expert_bounds_dev = expert_bounds_dev.get();
-        ids_info.n_experts         = static_cast<int>(n_experts);
-        ids_info.sis1              = sis1;
+        ids_info.ids_src_compact = ids_src_compact_dev.get();
+        ids_info.ids_dst_compact = ids_dst_compact_dev.get();
+        ids_info.tile_spec_dev   = tile_spec_dev.get();
+        ids_info.n_experts       = static_cast<int>(n_experts);
+        ids_info.max_tiles       = static_cast<int>(max_tiles);
+        ids_info.sis1            = sis1;
         ids_info_ptr = &ids_info;
     }
 
diff --git src/ggml-cuda/mmf.cuh src/ggml-cuda/mmf.cuh
index 429c8874..c49f6c47 100644
--- src/ggml-cuda/mmf.cuh
+++ src/ggml-cuda/mmf.cuh
@@ -1,7 +1,8 @@
 #pragma once
 
-#include "mma.cuh"
 #include "common.cuh"
+#include "mma.cuh"
+#include "mmid.cuh"
 #include "convert.cuh"
 
 using namespace ggml_cuda_mma;
@@ -36,8 +37,9 @@ static constexpr __device__ int mmf_get_padding() {
 struct mmf_ids_data {
     const int32_t * ids_src_compact = nullptr;
     const int32_t * ids_dst_compact = nullptr;
-    const int32_t * expert_bounds_dev = nullptr;
+    const ggml_cuda_mm_id_tile_spec * tile_spec_dev = nullptr;
     int n_experts = 0;
+    int max_tiles = 0;
     int sis1 = 0;
 };
 
@@ -308,7 +310,7 @@ __launch_bounds__(ggml_cuda_get_physical_warp_size()*nwarps, 1)
 static __global__ void mul_mat_f_ids(
         const T * __restrict__ x, const float * __restrict__ y,
         const int32_t * __restrict__ ids_src_compact, const int32_t * __restrict__ ids_dst_compact,
-        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
+        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst,
         const int ncols, const int ncols_dst_total, const int nchannels_dst, const int stride_row, const int stride_col_y, const int stride_col_dst,
         const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
         const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
@@ -349,21 +351,20 @@ static __global__ void mul_mat_f_ids(
     constexpr int ntA = rows_per_block / tile_A::I;
     constexpr int ntB = (cols_per_block + tile_B::I - 1) / tile_B::I;
 
-    const int row0        = blockIdx.x * rows_per_block;
+    const int row0     = blockIdx.x * rows_per_block;
+    const int tile_idx = blockIdx.y;
 
-    const int expert_idx = blockIdx.y;
-    const int expert_start = expert_bounds[expert_idx];
-    const int expert_end   = expert_bounds[expert_idx + 1];
+    const ggml_cuda_mm_id_tile_spec ts = tile_spec[tile_idx];
+
+    const int expert_idx   = ts.expert;
+    const int expert_start = ts.col_low;
+    const int expert_end   = ts.col_high;
     const int ncols_expert = expert_end - expert_start;
 
-    const int tiles_for_expert = (ncols_expert + cols_per_block - 1) / cols_per_block;
-    const int tile_idx = blockIdx.z;
-    if (tile_idx >= tiles_for_expert) {
+    if (ncols_expert == 0) {
         return;
     }
 
-    const int col_base = tile_idx * cols_per_block;
-
     GGML_UNUSED(channel_ratio);
 
     const int channel_x   = expert_idx;
@@ -409,10 +410,9 @@ static __global__ void mul_mat_f_ids(
 #pragma unroll
                 for (int j0 = 0; j0 < tile_B::I; ++j0) {
                     const int j = j0 + tile_idx_local*tile_B::I;
-                    const int global_j = col_base + j;
                     float val = 0.0f;
-                    if (j < cols_per_block && global_j < ncols_expert) {
-                        const int src_entry = ids_src_expert[global_j];
+                    if (j < cols_per_block && j < ncols_expert) {
+                        const int src_entry = ids_src_expert[j];
                         const uint2 qrm = fast_div_modulo((uint32_t) src_entry, sis1_fd);
                         const int token   = (int) qrm.x;
                         const int channel = (int) qrm.y;
@@ -462,10 +462,9 @@ static __global__ void mul_mat_f_ids(
 #pragma unroll
                 for (int j0 = 0; j0 < tile_B::I; ++j0) {
                     const int j = j0 + tile_idx_local*tile_B::I;
-                    const int global_j = col_base + j;
                     float2 tmp = make_float2(0.0f, 0.0f);
-                    if (j < cols_per_block && global_j < ncols_expert) {
-                        const int src_entry = ids_src_expert[global_j];
+                    if (j < cols_per_block && j < ncols_expert) {
+                        const int src_entry = ids_src_expert[j];
                         const uint2 qrm = fast_div_modulo((uint32_t) src_entry, sis1_fd);
                         const int token   = (int) qrm.x;
                         const int channel = (int) qrm.y;
@@ -564,9 +563,8 @@ static __global__ void mul_mat_f_ids(
             }
         }
 
-        const int global_j = col_base + j;
-        if (j < cols_per_block && global_j < ncols_expert && nchannels_dst > 0) {
-            const int dst_entry = ids_dst_expert[global_j];
+        if (j < cols_per_block && j < ncols_expert && nchannels_dst > 0) {
+            const int dst_entry = ids_dst_expert[j];
             const uint2 qrm = fast_div_modulo((uint32_t) dst_entry, nch_fd);
             const int token = (int) qrm.x;
             if (token < ncols_dst_total) {
@@ -580,7 +578,7 @@ static __global__ void mul_mat_f_ids(
     }
     }
 #else
-    GGML_UNUSED_VARS(x, y, ids_src_compact, ids_dst_compact, expert_bounds, dst,
+    GGML_UNUSED_VARS(x, y, ids_src_compact, ids_dst_compact, tile_spec, dst,
         ncols, ncols_dst_total, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
         channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
         sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, sis1_fd, nch_fd);
@@ -603,17 +601,16 @@ static inline void mul_mat_f_switch_ids(
     // Use the compact-ids kernel only for larger tiles; for small ncols_dst (< 16)
     // we prefer the normal mul_mat_f path with has_ids=true.
     if (has_ids_data && ncols_dst > 16) {
-        const int max_tiles = (int) ((ncols_dst + cols_per_block - 1) / cols_per_block);
-        if (max_tiles == 0) {
+        if (ids_data->max_tiles == 0) {
             return;
         }
-        dim3 block_nums_ids(block_nums.x, ids_data->n_experts, max_tiles);
+        dim3 block_nums_ids(block_nums.x, ids_data->max_tiles, 1);
 
         const uint3 sis1_fd = ids_data->sis1 > 0 ? init_fastdiv_values((uint32_t) ids_data->sis1) : make_uint3(0, 0, 1);
         const uint3 nch_fd  = init_fastdiv_values((uint32_t) nchannels_dst);
 
         mul_mat_f_ids<T, rows_per_block, cols_per_block, nwarps><<<block_nums_ids, block_dims, nbytes_shared_total, stream>>>
-            (x, y, ids_data->ids_src_compact, ids_data->ids_dst_compact, ids_data->expert_bounds_dev, dst,
+            (x, y, ids_data->ids_src_compact, ids_data->ids_dst_compact, ids_data->tile_spec_dev, dst,
             ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
diff --git src/ggml-cuda/mmid.cu src/ggml-cuda/mmid.cu
index 12ae14ce..64de4fb2 100644
--- src/ggml-cuda/mmid.cu
+++ src/ggml-cuda/mmid.cu
@@ -27,12 +27,16 @@ template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };
 // Helper function for mul_mat_id, converts ids to a more convenient format.
 // ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
 // ids_dst describes the same mapping but for the dst tensor.
-// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
+// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
+//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
+// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
 template <int n_expert_used_template>
 __launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
 static __global__ void mm_ids_helper(
-        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
-        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
+        const int32_t * __restrict__ ids,
+        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
+        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
+        const int si1, const int sis1, const bool write_inverse) {
     constexpr int warp_size = ggml_cuda_get_physical_warp_size();
     const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
     const int expert = blockIdx.x;
@@ -118,19 +122,29 @@ static __global__ void mm_ids_helper(
         return;
     }
 
-    expert_bounds[expert] = nex_prev;
+    int col_low  = nex_prev;
+    int col_high = nex_prev + min(int(tile_width.z), it_compact);
 
-    if (expert < static_cast<int>(gridDim.x) - 1) {
-        return;
+    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
+    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
+        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
+            = {expert, col_low, col_high, /*padding =*/ 0};
+
+        col_low  = col_high;
+        col_high = min(col_high + int(tile_width.z), nex_prev + it_compact);
     }
 
-    expert_bounds[gridDim.x] = nex_prev + it_compact;
+    // Finally, at most one extra tile_spec between the seams per expert.
+    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
+        = {expert, col_low, col_high, /*padding =*/ 0};
 }
 
 template <int n_expert_used_template>
 static void launch_mm_ids_helper(
-        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
-        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
+        const int32_t * __restrict__ ids,
+        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
+        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
+        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
     GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
     GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");
 
@@ -139,41 +153,53 @@ static void launch_mm_ids_helper(
     const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
     CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);
 
+    const uint3 tile_width_fd = init_fastdiv_values(tile_width);
+
     const dim3 num_blocks(n_experts, 1, 1);
     const dim3 block_size(warp_size, 1, 1);
     const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
     GGML_ASSERT(nbytes_shared <= smpbo);
     mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
-        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
+        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
 }
 
 void ggml_cuda_launch_mm_ids_helper(
-        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
-        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
+        const int32_t * __restrict__ ids,
+        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
+        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
+        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
     switch (n_expert_used) {
         case  2:
-            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         case  4:
-            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         case  6:
-            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         case  8:
-            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         case 10:
-            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         case 16:
-            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         case 32:
-            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
         default:
-            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
+            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
+                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
             break;
     }
 }
diff --git src/ggml-cuda/mmid.cuh src/ggml-cuda/mmid.cuh
index 74c2db43..dc878a71 100644
--- src/ggml-cuda/mmid.cuh
+++ src/ggml-cuda/mmid.cuh
@@ -1,5 +1,16 @@
 #pragma once
 
+struct ggml_cuda_mm_id_tile_spec {
+    int32_t expert;
+    int32_t col_low;
+    int32_t col_high;
+
+    int32_t padding;
+};
+static_assert(sizeof(ggml_cuda_mm_id_tile_spec) == 16, "bad ggml_cuda_mm_id_tile_spec size");
+
 void ggml_cuda_launch_mm_ids_helper(
-        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
-        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
+        const int32_t * __restrict__ ids,
+        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
+        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
+        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream);
diff --git src/ggml-cuda/mmq-load-tiles.cuh src/ggml-cuda/mmq-load-tiles.cuh
index 4951db72..8b4c1e8b 100644
--- src/ggml-cuda/mmq-load-tiles.cuh
+++ src/ggml-cuda/mmq-load-tiles.cuh
@@ -1724,10 +1724,8 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
     }
 }
 
-template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, bool has_fusion> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
         const char * __restrict__ x, const char * __restrict__ x_gate, int * __restrict__ x_tile, const int kb0, const int i_max, const int stride) {
-    GGML_UNUSED(x_gate);
-
     constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
     constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
     constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
@@ -1755,7 +1753,7 @@ template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __de
             i = min(i, i_max);
         }
 
-        const block_nvfp4 * bxi = (const block_nvfp4 *) x + kb0 + i * stride + kbx;
+        const block_nvfp4 * bxi = ggml_cuda_mmq_x_row<block_nvfp4, has_fusion>(x, x_gate, kb0, i, stride) + kbx;
         const uint32_t * __restrict__ src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
         const int kqs = 16 * kbx;
         const int ksc = 4 * kbx;
@@ -1782,10 +1780,8 @@ template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __de
     }
 }
 
-template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4_nvfp4(
+template <ggml_type type, int J, bool fallback, bool has_fusion> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4_nvfp4(
         const char * __restrict__ x, const char * __restrict__ x_gate, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
-    GGML_UNUSED(x_gate);
-
     constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
     constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q4) / warp_size;
     constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q4);
@@ -1800,7 +1796,6 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
     const int kbx = txi % threads_per_row;
     const int row_in_warp = txi / threads_per_row;
 
-    const block_nvfp4 * bxi_base = (const block_nvfp4 *) x + kbx0 + kbx;
     uint32_t * x_u32_scale = x_u32 + 64 + kbx;
 
 #pragma unroll
@@ -1811,7 +1806,7 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
             i = min(i, i_max);
         }
 
-        const block_nvfp4 * bxi = bxi_base + i * stride;
+        const block_nvfp4 * bxi = ggml_cuda_mmq_x_row<block_nvfp4, has_fusion>(x, x_gate, kbx0, i, stride) + kbx;
 
         const uint32_t * src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
 
diff --git src/ggml-cuda/mmq.cu src/ggml-cuda/mmq.cu
index 49e0ce90..36765750 100644
--- src/ggml-cuda/mmq.cu
+++ src/ggml-cuda/mmq.cu
@@ -114,17 +114,17 @@ static ggml_prec ggml_cuda_mmq_get_prec_env() {
 
 // src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
 // default to the native W4A4 instructions on Blackwell.
-static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
+static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_type type, const ggml_prec prec_requested, const int cc) {
     static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();
 
     ggml_prec prec = prec_env;
     if (prec == GGML_PREC_UNDEFINED) {
-        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
+        prec = prec_requested;
     }
 
     // Q4 only for the FP4 types on Blackwell
     GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
-    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
+    const bool can_use_q4 = (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
     if (prec == GGML_PREC_Q8 || !can_use_q4) {
         return GGML_PREC_Q8;
     }
@@ -172,11 +172,18 @@ void ggml_cuda_mul_mat_q(
     const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
     ggml_cuda_mm_fusion_args_device fusion_device = {};
     if (fusion) {
-        GGML_ASSERT(gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
+        GGML_ASSERT(gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale == !fusion->gate_scale);
         GGML_ASSERT(gate->type == src0->type && ggml_are_same_shape(gate, src0) && ggml_are_same_stride(gate, src0));
-        fusion_device.gate      = gate->data;
-        fusion_device.glu_op    = fusion->glu_op;
-        fusion_device.glu_limit = fusion->glu_limit;
+        // NVFP4 weight scales (only applied for NVFP4): one value per tensor, or one per expert with ids.
+        for (const ggml_tensor * scale : { fusion->x_scale, fusion->gate_scale }) {
+            GGML_ASSERT(!scale || (src0->type == GGML_TYPE_NVFP4 && scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
+                ggml_nelements(scale) == (ids ? src0->ne[2] : 1)));
+        }
+        fusion_device.gate       = gate->data;
+        fusion_device.x_scale    = fusion->x_scale    ? fusion->x_scale->data    : nullptr;
+        fusion_device.gate_scale = fusion->gate_scale ? fusion->gate_scale->data : nullptr;
+        fusion_device.glu_op     = fusion->glu_op;
+        fusion_device.glu_limit  = fusion->glu_limit;
     }
 
     GGML_TENSOR_BINARY_OP_LOCALS;
@@ -224,9 +231,8 @@ void ggml_cuda_mul_mat_q(
 
     const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);
 
-    // With fusion dst is the GLU output, not a MUL_MAT; the fused types always quantize src1 to Q8_1.
-    // TODO support for Q4 precision if both tensors allow it
-    const ggml_prec prec_src1 = gate ? GGML_PREC_Q8 : ggml_cuda_mmq_get_prec_src1(src0, dst, cc);
+    // With fusion dst is the GLU output, not a MUL_MAT; the src1 precision requested by both fused MUL_MATs is in the fusion args.
+    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0->type, gate ? fusion->prec_src1 : ggml_prec_get_src(dst, 1), cc);
 
     const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
     const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
@@ -241,10 +247,7 @@ void ggml_cuda_mul_mat_q(
             ncols_opt = ne12;
 
             // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
-            // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
-            if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
-                ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
-            }
+            ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
         }
 
         J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);
@@ -258,6 +261,8 @@ void ggml_cuda_mul_mat_q(
         * src1_load_chunk_size;
 
     if (!ids) {
+        const int ntj = (ne1 + J_best - 1) / J_best;
+
         const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
         ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
         ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
@@ -295,7 +300,7 @@ void ggml_cuda_mul_mat_q(
             ne00, ne01, ne1, s01, ne11, s1,
             ne02, ne12, s02, s12, s2,
             ne03, ne13, s03, s13, s3,
-            ne1, J_best, fusion_device};
+            ntj, J_best, fusion_device};
         ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
         return;
     }
@@ -310,7 +315,9 @@ void ggml_cuda_mul_mat_q(
 
     ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
     ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.
-    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
+
+    const int ntj = (ne_get_rows + J_best - 1) / J_best + ne02;
+    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);
 
     // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
     // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
@@ -321,8 +328,8 @@ void ggml_cuda_mul_mat_q(
         const int si1  = ids->nb[1] / ggml_element_size(ids);
         const int sis1 = nb12 / nb11;
 
-        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
-            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
+        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), tile_spec.get(),
+            ne02, ne12, n_expert_used, ne11, J_best, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
         CUDA_CHECK(cudaGetLastError());
     }
 
@@ -369,12 +376,12 @@ void ggml_cuda_mul_mat_q(
 
     // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
     const mmq_args args = {
-        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
+        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), tile_spec.get(), dst_d,
         src1_scale.ptr,
         ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
-        ne02, ne02, s02, s12, s2,
+        ne02,    1, s02, s12, s2,
         ne03, ne13, s03, s13, s3,
-        ne12, J_best, fusion_device};
+        ntj, J_best, fusion_device};
 
     ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
 }
@@ -513,18 +520,20 @@ bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t
     return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
 }
 
-bool ggml_cuda_mmq_fusion_is_efficient(const ggml_type type, const int64_t nrows_x, const int64_t ncols, const int64_t nchannels, const int device) {
+bool ggml_cuda_mmq_fusion_is_efficient(
+        const ggml_type type, const ggml_prec prec_requested, const int64_t nrows_x, const int64_t ncols, const int64_t nchannels, const int device) {
     const int    cc    = ggml_cuda_info().devices[device].cc;
     const int    nsm   = ggml_cuda_info().devices[device].nsm;
     const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
 
-    // Same tile width as ggml_cuda_mul_mat_q; the fused kernel has no fallback, quantizes src1 to Q8_1 and writes I/2 output rows per tile.
-    const bool fallback = false;
-    const int  J_best   = ggml_cuda_mmq_get_J_best(type, ncols, fallback, cc, smpbo, GGML_PREC_Q8);
+    // Same tile width as ggml_cuda_mul_mat_q; the fused kernel has no fallback and writes I/2 output rows per tile.
+    const bool      fallback  = false;
+    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(type, prec_requested, cc);
+    const int       J_best    = ggml_cuda_mmq_get_J_best(type, ncols, fallback, cc, smpbo, prec_src1);
     if (J_best == 0) {
         return false;
     }
-    const int64_t nrows_tile = ggml_cuda_mmq_get_config(type, J_best, fallback, cc, GGML_PREC_Q8).I / 2;
+    const int64_t nrows_tile = ggml_cuda_mmq_get_config(type, J_best, fallback, cc, prec_src1).I / 2;
     const int64_t ntiles     = (ncols + J_best - 1)/J_best * ((nrows_x + nrows_tile - 1)/nrows_tile) * nchannels;
     // Same condition as launch_mul_mat_q uses to run the stream-k kernel with one block per tile.
     const int64_t tiles_nwaves             = (ntiles + nsm - 1) / nsm;
diff --git src/ggml-cuda/mmq.cuh src/ggml-cuda/mmq.cuh
index 308968be..4d67555e 100644
--- src/ggml-cuda/mmq.cuh
+++ src/ggml-cuda/mmq.cuh
@@ -1,6 +1,7 @@
 #pragma once
 
 #include "common.cuh"
+#include "mmid.cuh"
 #include "unary.cuh"
 
 #include <climits>
@@ -14,9 +15,9 @@
 typedef void (*ggml_cuda_mmq_load_tiles_t)(const char * __restrict__ x, const char * __restrict__ x_gate, int * x_tile,
     const int kbx0, const int i_max, const int stride);
 typedef void (*ggml_cuda_mmq_vec_dot_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
-typedef void (*ggml_cuda_mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ get_rows_to_sorted,
-    float * __restrict__ dst, const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max,
-    const ggml_glu_op glu_op, const float glu_limit);
+typedef void (*ggml_cuda_mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ ids_dst,
+    float * __restrict__ dst, const float * __restrict__ y_scale, const int stride, const int i_max, const int j_sup,
+    const ggml_glu_op glu_op, const float glu_limit, const float * __restrict__ x_scale, const float * __restrict__ gate_scale);
 
 enum mmq_q8_1_ds_layout {
     MMQ_Q8_1_DS_LAYOUT_D4,
@@ -436,7 +437,8 @@ static __host__ int ggml_cuda_mmq_get_nbytes_shared_x(const ggml_cuda_mmq_config
 
 // Weight types with a fused gate/up/GLU path (see ggml_cuda_mmq_get_util_funcs).
 static constexpr __host__ __device__ bool ggml_cuda_mmq_fusion_supported(const ggml_type type) {
-    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_XS;
+    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_XS ||
+        type == GGML_TYPE_NVFP4;
 }
 
 // Fused gate/up/GLU: the weight tile alternates chunks of this many up and gate rows, so rows r and r + 8 of an MMA tile (same thread) pair up.
@@ -461,22 +463,21 @@ static __device__ __forceinline__ const block_t * ggml_cuda_mmq_x_row(
 
 template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __device__ __forceinline__ void ggml_cuda_mmq_write_back_dp4a(
         const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
-        const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max,
-        const ggml_glu_op glu_op, const float glu_limit) {
-    GGML_UNUSED_VARS(glu_op, glu_limit);
-
+        const float * __restrict__ y_scale, const int stride, const int i_max, const int j_sup,
+        const ggml_glu_op glu_op, const float glu_limit, const float * __restrict__ x_scale, const float * __restrict__ gate_scale) {
+    GGML_UNUSED_VARS(glu_op, glu_limit, x_scale, gate_scale);
     constexpr int warp_size = ggml_cuda_get_physical_warp_size();
     constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
     constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
     static_assert(I % warp_size == 0, "DP4A MMQ tile height must be a multiple of the physical warp size");
 
-    const bool y_scale_used = y_scale != nullptr;
+    const bool y_scale_used = type == GGML_TYPE_NVFP4 && y_scale != nullptr;
 
 #pragma unroll
     for (int j0 = 0; j0 < J; j0 += nwarps) {
         const int j = j0 + threadIdx.y;
 
-        if (j > j_max) {
+        if (j >= j_sup) {
             return;
         }
 
@@ -488,26 +489,22 @@ template <ggml_type type, int J, bool fallback, ggml_prec prec_src1> static __de
                 continue;
             }
 
-            if constexpr (type == GGML_TYPE_NVFP4) {
-                if (y_scale_used) {
-                    dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
-                } else {
-                    dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
-                }
+            if (y_scale_used) {
+                dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
             } else {
                 dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (I/warp_size) + i0/warp_size];
-                GGML_UNUSED(y_scale_used);
             }
         }
     }
 }
 
 // has_fusion: accumulators l and l + ne/2 hold the up and gate values of one output element, the GLU of the two is written.
+// x_scale/gate_scale: optional NVFP4 scales of the up and gate weights (per tensor, or of the expert of this tile).
 template<ggml_type type, int J, bool fallback, ggml_prec prec_src1, bool has_fusion>
 static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
             const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
-            const float * __restrict__ y_scale, const int stride, const int i_max, const int j_max,
-            const ggml_glu_op glu_op, const float glu_limit) {
+            const float * __restrict__ y_scale, const int stride, const int i_max, const int j_sup,
+            const ggml_glu_op glu_op, const float glu_limit, const float * __restrict__ x_scale, const float * __restrict__ gate_scale) {
 
 #if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
     typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
@@ -521,7 +518,12 @@ static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
 
     const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
 
-    const bool y_scale_used = y_scale != nullptr;
+    const bool y_scale_used = type == GGML_TYPE_NVFP4 && y_scale != nullptr;
+
+    // Fused NVFP4: load the weight scales once, 1 without scales.
+    constexpr bool fused_nvfp4 = has_fusion && type == GGML_TYPE_NVFP4;
+    const float scale_up   = fused_nvfp4 && x_scale ? *x_scale    : 1.0f;
+    const float scale_gate = fused_nvfp4 && gate_scale ? *gate_scale : 1.0f;
 
 #pragma unroll
     for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
@@ -531,7 +533,7 @@ static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
             for (int l = 0; l < (has_fusion ? tile_C::ne/2 : tile_C::ne); ++l) {
                 const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
 
-                if (j > j_max) {
+                if (j >= j_sup) {
                     continue;
                 }
 
@@ -541,24 +543,29 @@ static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
                     continue;
                 }
 
-                if constexpr (has_fusion) {
+                if (has_fusion) {
                     const int sum_index = (j0/tile_C::J + n)*tile_C::ne + l;
-                    dst[ids_dst[j]*stride + i] = ggml_cuda_op_glu_single(glu_op, sum[sum_index + tile_C::ne/2], sum[sum_index], glu_limit);
-                } else if constexpr (type == GGML_TYPE_NVFP4) {
-                    if (y_scale_used) {
-                        dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/tile_C::J + n)*tile_C::ne + l];
-                    } else {
-                        dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
+                    float up   = sum[sum_index];
+                    float gate = sum[sum_index + tile_C::ne/2];
+                    if constexpr (type == GGML_TYPE_NVFP4) {
+                        if (y_scale_used) {
+                            up   *= y_scale[j];
+                            gate *= y_scale[j];
+                        }
+                        up   *= scale_up;
+                        gate *= scale_gate;
                     }
+                    dst[ids_dst[j]*stride + i] = ggml_cuda_op_glu_single(glu_op, gate, up, glu_limit);
+                } else if (y_scale_used) {
+                    dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/tile_C::J + n)*tile_C::ne + l];
                 } else {
                     dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
-                    GGML_UNUSED(y_scale_used);
                 }
             }
         }
     }
 
-    GGML_UNUSED_VARS(y_scale_used, glu_op, glu_limit);
+    GGML_UNUSED_VARS(y_scale_used, glu_op, glu_limit, x_scale, gate_scale, scale_up, scale_gate);
 }
 
 // -------------------------------------------------------------------------------------------------------------------------------------
@@ -712,7 +719,7 @@ static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_func
             case GGML_TYPE_NVFP4:
                 return ggml_cuda_mmq_util_funcs(
                     VDR_NVFP4_Q8_1_MMQ,
-                    ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1>,
+                    ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1, false>,
                     ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,
                     ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
             default:
@@ -737,9 +744,9 @@ static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_func
             if (prec_src1 == GGML_PREC_Q4) {
                 return ggml_cuda_mmq_util_funcs(
                     -1,
-                    ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
+                    ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback, has_fusion>,
                     ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
-                    ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1, false>);
+                    ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1, has_fusion>);
             }
             break;
         default:
@@ -882,9 +889,9 @@ static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_func
         case GGML_TYPE_NVFP4:
             return ggml_cuda_mmq_util_funcs(
                 -1,
-                ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1>,
+                ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1, has_fusion>,
                 ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,
-                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1, false>);
+                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1, has_fusion>);
         default:
             return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
     }
@@ -918,8 +925,9 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
         const char * __restrict__ x, const char * __restrict__ x_gate, const int offset_x, const int * __restrict__ y,
         const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
         const float * __restrict__ y_scale, const ggml_glu_op glu_op, const float glu_limit,
+        const float * __restrict__ x_scale, const float * __restrict__ gate_scale,
         const int stride_row_x, const int ncols_y, const int stride_col_dst,
-        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
+        const int tile_x_i_max, const int tile_y_j_sup, const int kb0_start, const int kb0_stop) {
     static_assert(!(fixup && has_fusion), "the fused gate/up path does not split K");
     static_assert(!(fallback && has_fusion), "the fused gate/up path needs nrows_x % 128 == 0");
 
@@ -952,7 +960,7 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
     constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
 
     for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
-        load_tiles(x, x_gate, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
+        load_tiles(x, x_gate, tile_x, offset_x + kb0, tile_x_i_max, stride_row_x);
         {
             const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
 #pragma unroll
@@ -987,9 +995,9 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
     }
 
     if (fixup) {
-        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J, glu_op, glu_limit);
+        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J, glu_op, glu_limit, x_scale, gate_scale);
     } else {
-        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j, glu_op, glu_limit);
+        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_i_max, tile_y_j_sup, glu_op, glu_limit, x_scale, gate_scale);
     }
 }
 
@@ -1000,21 +1008,28 @@ template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, bool has_fu
 __launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
 static __global__ void mul_mat_q(
         const char * __restrict__ x, const char * __restrict__ x_gate, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
-        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
+        const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst, float * __restrict__ tmp_fixup,
         const float * __restrict__ y_scale, const ggml_glu_op glu_op, const float glu_limit,
+        const float * __restrict__ x_scale, const float * __restrict__ gate_scale,
         const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
         const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
         const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
-        const uint3 ntx) {
+        const uint3 ntj) {
 
 #ifdef TURING_MMA_AVAILABLE
     constexpr bool fusion_available = true;
 #else
     constexpr bool fusion_available = false;
 #endif // TURING_MMA_AVAILABLE
+#ifdef BLACKWELL_MMA_AVAILABLE
+    constexpr bool native_fp4_available = true;
+#else
+    constexpr bool native_fp4_available = false;
+#endif // BLACKWELL_MMA_AVAILABLE
 
-    // Skip unused template specializations for faster compilation:
-    if (ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type == GGML_TYPE_COUNT || (has_fusion && !fusion_available)) {
+    // Skip unused template specializations for faster compilation (the host only uses 4-bit src1 with native FP4):
+    if (ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type == GGML_TYPE_COUNT ||
+            (has_fusion && (!fusion_available || (prec_src1 == GGML_PREC_Q4 && !native_fp4_available)))) {
         NO_DEVICE_CODE;
         return;
     }
@@ -1024,7 +1039,7 @@ static __global__ void mul_mat_q(
     constexpr int qk        = ggml_cuda_type_traits<type>::qk;
     constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
 
-    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y
+    const uint32_t nti = (nrows_x + I - 1) / I; // Number of tiles in I direction
 
     // Initialize the ids for writing back data with just the index.
     // For regular matrix multiplications this is never changed.
@@ -1049,35 +1064,28 @@ static __global__ void mul_mat_q(
 
         const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
         const int wt = tmp2.x;
-        const int zt = tmp2.y;
+        int       zt = tmp2.y;
         const int jt = blockIdx.x;
         const int it = blockIdx.y;
 
         // Defaults for regular matrix multiplication:
-        int col_low    = 0;
-        int col_high   = ncols_dst;
-        int col_diff   = ncols_dst;
+        int col_low        = jt*J;
+        int col_high       = min(col_low + J, ncols_dst);
         int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
+        int offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
         int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
-        int offset_y_scale;
-        if constexpr (type == GGML_TYPE_NVFP4) {
-            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
-        } else {
-            GGML_UNUSED(offset_y_scale);
-        }
 
         if (ids_dst) {
-            col_low  = expert_bounds[zt + 0];
-            col_high = expert_bounds[zt + 1];
-            col_diff = col_high - col_low;
-
-            offset_y   = 0;
-            offset_dst = 0;
-            if constexpr (type == GGML_TYPE_NVFP4) {
-                offset_y_scale = 0;
-            }
+            const ggml_cuda_mm_id_tile_spec tsj = tile_spec[jt];
+            zt       = tsj.expert;
+            col_low  = tsj.col_low;
+            col_high = tsj.col_high;
 
-            if (jt*J >= col_diff) {
+            offset_y       = 0;
+            offset_y_scale = 0;
+            offset_dst     = 0;
+
+            if (col_low == col_high) {
                 return;
             }
 
@@ -1090,29 +1098,30 @@ static __global__ void mul_mat_q(
                     break;
                 }
 
-                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
+                ids_dst_shared[j] = ids_dst[col_low + j];
             }
             __syncthreads();
         }
 
-        offset_y   += (col_low + jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
-        offset_dst += it*I_dst;
-        const float * y_scale_tile = nullptr;
-        if constexpr (type == GGML_TYPE_NVFP4) {
-            offset_y_scale += col_low + jt*J;
-            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
-        }
+        offset_y       += col_low * (sizeof(block_q8_1_mmq)/sizeof(int));
+        offset_y_scale += col_low;
+        offset_dst     += it*I_dst;
 
-        const int tile_x_max_i = nrows_x  - it*I_dst - 1;
-        const int tile_y_max_j = col_diff - jt*J - 1;
+        const float * y_scale_tile = type == GGML_TYPE_NVFP4 && y_scale ? y_scale + offset_y_scale : nullptr;
+
+        const int tile_x_i_max = nrows_x - it*I_dst - 1;
 
         const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I_dst*stride_row_x;
 
+        // With ids the NVFP4 weight scales are per expert.
+        const float * x_scale_tile    = x_scale    && ids_dst ? x_scale    + zt : x_scale;
+        const float * gate_scale_tile = gate_scale && ids_dst ? gate_scale + zt : gate_scale;
+
         constexpr bool fixup = false;
         mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1, has_fusion>
             (x, x_gate, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
-             glu_op, glu_limit, stride_row_x, ncols_y, stride_col_dst,
-             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
+             glu_op, glu_limit, x_scale_tile, gate_scale_tile, stride_row_x, ncols_y, stride_col_dst,
+             tile_x_i_max, col_high - col_low, 0, blocks_per_ne00.z);
         return;
     }
 
@@ -1120,8 +1129,8 @@ static __global__ void mul_mat_q(
     constexpr int blocks_per_iter = ITER_K / qk;
 
     // kbc == k block continuous, current index in continuous ijk space.
-    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
-    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
+    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntj.z*nti*blocks_per_ne00.z) / gridDim.x;
+    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntj.z*nti*blocks_per_ne00.z) / gridDim.x;
 
     kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
     kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;
@@ -1131,41 +1140,34 @@ static __global__ void mul_mat_q(
     int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
     while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
         int tmp = fastdiv(kbc, blocks_per_ne00);
-        uint2 tmp2 = fast_div_modulo(tmp, ntx);
+        uint2 tmp2 = fast_div_modulo(tmp, ntj);
         const int jt = tmp2.y;
         tmp = tmp2.x;
         tmp2 = fast_div_modulo(tmp, nchannels_y);
-        const int zt = tmp2.y;
+        int zt = tmp2.y;
         tmp = tmp2.x;
         tmp2 = fast_div_modulo(tmp, nsamples_y);
         const int wt = tmp2.y;
         const int it = tmp2.x;
 
         // Defaults for regular matrix multiplication:
-        int col_low    = 0;
-        int col_high   = ncols_dst;
-        int col_diff   = ncols_dst;
+        int col_low        = jt*J;
+        int col_high       = min(col_low + J, ncols_dst);
         int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
+        int offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
         int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
-        int offset_y_scale;
-        if constexpr (type == GGML_TYPE_NVFP4) {
-            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
-        } else {
-            GGML_UNUSED(offset_y_scale);
-        }
 
         if (ids_dst) {
-            col_low  = expert_bounds[zt + 0];
-            col_high = expert_bounds[zt + 1];
-            col_diff = col_high - col_low;
-
-            offset_y   = 0;
-            offset_dst = 0;
-            if constexpr (type == GGML_TYPE_NVFP4) {
-                offset_y_scale = 0;
-            }
+            const ggml_cuda_mm_id_tile_spec tsj = tile_spec[jt];
+            zt       = tsj.expert;
+            col_low  = tsj.col_low;
+            col_high = tsj.col_high;
 
-            if (jt*J >= col_diff) {
+            offset_y       = 0;
+            offset_y_scale = 0;
+            offset_dst     = 0;
+
+            if (col_low == col_high) {
                 kbc += blocks_per_ne00.z;
                 kbc -= fastmodulo(kbc, blocks_per_ne00);
 
@@ -1184,29 +1186,26 @@ static __global__ void mul_mat_q(
                     break;
                 }
 
-                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
+                ids_dst_shared[j] = ids_dst[col_low + j];
             }
             __syncthreads();
         }
 
-        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
-        offset_dst += it*I;
-        const float * y_scale_tile = nullptr;
-        if constexpr (type == GGML_TYPE_NVFP4) {
-            offset_y_scale += col_low + jt * J;
-            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
-        }
+        offset_y       += col_low * (sizeof(block_q8_1_mmq) / sizeof(int));
+        offset_y_scale += col_low;
+        offset_dst     += it*I;
 
-        const int tile_x_max_i = nrows_x  - it*I - 1;
-        const int tile_y_max_j = col_diff - jt*J - 1;
+        const float * y_scale_tile = type == GGML_TYPE_NVFP4 && y_scale ? y_scale + offset_y_scale : nullptr;
+
+        const int tile_x_i_max = nrows_x - it*I - 1;
 
         const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;
 
         constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
         mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1, false>
             (x, x_gate, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
-             glu_op, glu_limit, stride_row_x, ncols_y, stride_col_dst,
-             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
+             glu_op, glu_limit, x_scale, gate_scale, stride_row_x, ncols_y, stride_col_dst,
+             tile_x_i_max, col_high - col_low, kb0_start, kb0_stop);
 
         kbc += blocks_per_ne00.z;
         kbc -= fastmodulo(kbc, blocks_per_ne00);
@@ -1220,41 +1219,34 @@ static __global__ void mul_mat_q(
     }
 
     int tmp = fastdiv(kbc, blocks_per_ne00);
-    uint2 tmp2 = fast_div_modulo(tmp, ntx);
+    uint2 tmp2 = fast_div_modulo(tmp, ntj);
     const int jt = tmp2.y;
     tmp = tmp2.x;
     tmp2 = fast_div_modulo(tmp, nchannels_y);
-    const int zt = tmp2.y;
+    int zt = tmp2.y;
     tmp = tmp2.x;
     tmp2 = fast_div_modulo(tmp, nsamples_y);
     const int wt = tmp2.y;
     const int it = tmp2.x;
 
     // Defaults for regular matrix multiplication:
-    int col_low    = 0;
-    int col_high   = ncols_dst;
-    int col_diff   = ncols_dst;
+    int col_low        = jt*J;
+    int col_high       = min(col_low + J, ncols_dst);
     int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
+    int offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
     int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
-    int offset_y_scale;
-    if constexpr (type == GGML_TYPE_NVFP4) {
-        offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
-    } else {
-        GGML_UNUSED(offset_y_scale);
-    }
 
     if (ids_dst) {
-        col_low  = expert_bounds[zt + 0];
-        col_high = expert_bounds[zt + 1];
-        col_diff = col_high - col_low;
+        const ggml_cuda_mm_id_tile_spec tsj = tile_spec[jt];
+        zt       = tsj.expert;
+        col_low  = tsj.col_low;
+        col_high = tsj.col_high;
 
-        offset_y   = 0;
-        offset_dst = 0;
-        if constexpr (type == GGML_TYPE_NVFP4) {
-            offset_y_scale = 0;
-        }
+        offset_y       = 0;
+        offset_y_scale = 0;
+        offset_dst     = 0;
 
-        if (jt*J >= col_diff) {
+        if (col_low == col_high) {
             return;
         }
 
@@ -1273,33 +1265,30 @@ static __global__ void mul_mat_q(
         __syncthreads();
     }
 
-    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
-    offset_dst += it*I;
-    const float * y_scale_tile = nullptr;
-    if constexpr (type == GGML_TYPE_NVFP4) {
-        offset_y_scale += col_low + jt * J;
-        y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
-    }
+    offset_y       += col_low * (sizeof(block_q8_1_mmq) / sizeof(int));
+    offset_y_scale += col_low;
+    offset_dst     += it*I;
+
+    const float * y_scale_tile = type == GGML_TYPE_NVFP4 && y_scale ? y_scale + offset_y_scale : nullptr;
 
-    const int tile_x_max_i = nrows_x  - it*I - 1;
-    const int tile_y_max_j = col_diff - jt*J - 1;
+    const int tile_x_i_max = nrows_x - it*I - 1;
 
     const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;
 
     constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
     mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1, false>
         (x, x_gate, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
-         glu_op, glu_limit, stride_row_x, ncols_y, stride_col_dst,
-         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
+         glu_op, glu_limit, x_scale, gate_scale, stride_row_x, ncols_y, stride_col_dst,
+         tile_x_i_max, col_high - col_low, kb0_start, kb0_stop);
 }
 
 template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
 __launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1)/2, 1)
 static __global__ void mul_mat_q_stream_k_fixup(
-        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
+        const int32_t * __restrict__ ids_dst, const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst,
         float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
         const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
-        const int stride_sample_dst, const uint3 ntx) {
+        const int stride_sample_dst, const uint3 ntj) {
     constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
     constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / 2) / warp_size;
     constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
@@ -1310,13 +1299,13 @@ static __global__ void mul_mat_q_stream_k_fixup(
     float sum[J / nwarps] = {0.0f};
     const int i = blockIdx.y*warp_size + threadIdx.x;
 
-    const int nty = (nrows_x + I - 1) / I;
+    const int nti = (nrows_x + I - 1) / I;
 
     const int bidx0 = blockIdx.x;
 
     // kbc == k block continuous, current index in continuous ijk space.
-    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
-    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
+    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntj.z*nti*blocks_per_ne00.z) / gridDim.x;
+    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntj.z*nti*blocks_per_ne00.z) / gridDim.x;
 
     kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
     kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;
@@ -1335,7 +1324,7 @@ static __global__ void mul_mat_q_stream_k_fixup(
     int bidx = bidx0 - 1;
     int kbc_stop = kbc0;
     while(true) {
-        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
+        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntj.z*nti*blocks_per_ne00.z) / gridDim.x;
         kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;
 
         if (kbc == kbc_stop) { // Did not have any data.
@@ -1367,11 +1356,11 @@ static __global__ void mul_mat_q_stream_k_fixup(
     }
 
     int tmp = fastdiv(kbc0, blocks_per_ne00);
-    uint2 tmp2 = fast_div_modulo(tmp, ntx);
+    uint2 tmp2 = fast_div_modulo(tmp, ntj);
     const int jt = tmp2.y;
     tmp = tmp2.x;
     tmp2 = fast_div_modulo(tmp, nchannels_y);
-    const int zt = tmp2.y;
+    int zt = tmp2.y;
     tmp = tmp2.x;
     tmp2 = fast_div_modulo(tmp, nsamples_y);
     const int wt = tmp2.y;
@@ -1382,7 +1371,7 @@ static __global__ void mul_mat_q_stream_k_fixup(
         dst += offset_dst;
 
         const int i_max = nrows_x   - it*I - 1;
-        const int j_max = ncols_dst - jt*J - 1;
+        const int j_sup = ncols_dst - jt*J;
         if (fallback && i > i_max) {
             return;
         }
@@ -1391,7 +1380,7 @@ static __global__ void mul_mat_q_stream_k_fixup(
         for (int j0 = 0; j0 < J; j0 += nwarps) {
             const int j = j0 + threadIdx.y;
 
-            if (j > j_max) {
+            if (j >= j_sup) {
                 return;
             }
 
@@ -1401,12 +1390,14 @@ static __global__ void mul_mat_q_stream_k_fixup(
     }
 
     __shared__ int ids_dst_shared[J];
-    const int col_low  = expert_bounds[zt + 0];
-    const int col_high = expert_bounds[zt + 1];
-    const int col_diff = col_high - col_low;
+    const ggml_cuda_mm_id_tile_spec tsj = tile_spec[jt];
+    zt                 = tsj.expert;
+    const int col_low  = tsj.col_low;
+    const int col_high = tsj.col_high;
+    const int j_sup    = col_high - col_low;
 
     for (int j = threadIdx.y*warp_size + threadIdx.x; j < J; j += nwarps*warp_size) {
-        ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
+        ids_dst_shared[j] = ids_dst[col_low + j];
     }
     __syncthreads();
 
@@ -1414,7 +1405,6 @@ static __global__ void mul_mat_q_stream_k_fixup(
     dst += offset_dst;
 
     const int i_max = nrows_x  - it*I - 1;
-    const int j_max = col_diff - jt*J - 1;
     if (fallback && i > i_max) {
         return;
     }
@@ -1423,7 +1413,7 @@ static __global__ void mul_mat_q_stream_k_fixup(
     for (int j0 = 0; j0 < J; j0 += nwarps) {
         const int j = j0 + threadIdx.y;
 
-        if (j > j_max) {
+        if (j >= j_sup) {
             return;
         }
 
@@ -1432,12 +1422,12 @@ static __global__ void mul_mat_q_stream_k_fixup(
 }
 
 struct mmq_args {
-    const char * x; ggml_type type_x; const int * y; const int32_t * ids_dst; const int32_t * expert_bounds; float * dst;
-    const float * y_scale;
+    const char * x; ggml_type type_x; const int * y; const int32_t * ids_dst;
+    const ggml_cuda_mm_id_tile_spec * tile_spec; float * dst; const float * y_scale;
     int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
     int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
     int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
-    int64_t ncols_max;
+    int64_t ntj;
     int J_best; // Tile width in ne11(dense)/ne12(MoE) direction to use for optimal performance.
     ggml_cuda_mm_fusion_args_device fusion = {}; // gate != nullptr selects the fused gate/up/GLU path
 };
@@ -1469,38 +1459,38 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
     }
 
     const int I_dst = has_fusion ? config.I/2 : config.I; // Rows of dst per tile, see mul_mat_q.
-    const int nty  = (args.nrows_x   + I_dst - 1) / I_dst;
-    const int ntx  = (args.ncols_max + config.J - 1) / config.J;
-    const int ntzw = args.nchannels_y * args.nsamples_y;
+    const int nti   = (args.nrows_x + I_dst - 1) / I_dst;
+    const int ntzw  = args.nchannels_y * args.nsamples_y;
     // Column tiles innermost so that the concurrently running blocks share the same tile of x and read it from L2.
-    const dim3 block_nums_xy_tiling(ntx, nty, ntzw);
+    const dim3 block_nums_xy_tiling(args.ntj, nti, ntzw);
 
-    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
-    GGML_ASSERT(args.nsamples_y  % args.nsamples_x  == 0);
-    const int channel_ratio = args.nchannels_y / args.nchannels_x;
-    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;
+    GGML_ASSERT(args.ids_dst || args.nchannels_y % args.nchannels_x == 0);
+    GGML_ASSERT(                args.nsamples_y  % args.nsamples_x  == 0);
+    const int channel_ratio = args.ids_dst ? 1 : args.nchannels_y / args.nchannels_x;
+    const int sample_ratio  =                    args.nsamples_y  / args.nsamples_x;
 
     const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
-    const uint3 ntx_fd             = init_fastdiv_values(ntx);
+    const uint3 ntj_fd             = init_fastdiv_values(args.ntj);
     const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
     const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
     const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
     const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);
 
     if (has_fusion || !config.stream_k) {
-        GGML_ASSERT(nty <= 65535); // gridDim.y limit
+        GGML_ASSERT(nti <= 65535); // gridDim.y limit
         mul_mat_q<type, J, fallback, prec_src1, has_fusion><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
-            (args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale, args.fusion.glu_op, args.fusion.glu_limit,
+            (args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, nullptr,
+             args.y_scale, args.fusion.glu_op, args.fusion.glu_limit, (const float *) args.fusion.x_scale, (const float *) args.fusion.gate_scale,
              blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
              channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
              sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
-             ntx_fd);
+             ntj_fd);
         return;
     }
 
     // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
     // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
-    const int ntiles_dst = ntx * nty * ntzw;
+    const int ntiles_dst = args.ntj * nti * ntzw;
     const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
     const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
 
@@ -1521,11 +1511,12 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
     const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);
 
     mul_mat_q<type, J, fallback, prec_src1, has_fusion><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
-        (args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale, args.fusion.glu_op, args.fusion.glu_limit,
+        (args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, tmp_fixup.ptr,
+         args.y_scale, args.fusion.glu_op, args.fusion.glu_limit, (const float *) args.fusion.x_scale, (const float *) args.fusion.gate_scale,
          blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
          channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
          sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
-         ntx_fd);
+         ntj_fd);
 
     if (!fixup_needed) {
         return;
@@ -1533,9 +1524,9 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
 
     CUDA_CHECK(cudaGetLastError());
     mul_mat_q_stream_k_fixup<type, J, fallback, prec_src1><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
-        (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
+        (args.ids_dst, args.tile_spec, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
          args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
-         ntx_fd);
+         ntj_fd);
 }
 
 template <ggml_type type, bool fallback, ggml_prec prec_src1, bool has_fusion>
@@ -1661,4 +1652,5 @@ void ggml_cuda_mul_mat_q(
 
 bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);
 
-bool ggml_cuda_mmq_fusion_is_efficient(ggml_type type, int64_t nrows_x, int64_t ncols, int64_t nchannels, int device);
+// prec_requested: the src1 precision requested by the fused MUL_MATs.
+bool ggml_cuda_mmq_fusion_is_efficient(ggml_type type, ggml_prec prec_requested, int64_t nrows_x, int64_t ncols, int64_t nchannels, int device);
diff --git src/ggml-cuda/mmvq.cu src/ggml-cuda/mmvq.cu
index 1a9cf55c..8f79123a 100644
--- src/ggml-cuda/mmvq.cu
+++ src/ggml-cuda/mmvq.cu
@@ -285,6 +285,7 @@ static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type
 int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
     // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
     if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
+        cc = ggml_cuda_highest_compiled_arch(cc);
         if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
             return MMVQ_MAX_BATCH_SIZE;
         }
diff --git src/ggml-cuda/rope.cu src/ggml-cuda/rope.cu
index 45e81340..2604d014 100644
--- src/ggml-cuda/rope.cu
+++ src/ggml-cuda/rope.cu
@@ -591,7 +591,6 @@ void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
     //const int n_past     = ((int32_t *) dst->op_params)[0];
     const int n_dims     = ((int32_t *) dst->op_params)[1];
     const int mode       = ((int32_t *) dst->op_params)[2];
-    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
     const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
     const int n_offs     = ((int32_t *) dst->op_params)[15];
     mrope_sections sections;
@@ -637,7 +636,7 @@ void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
     }
 
     rope_corr_dims corr_dims;
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(dst), corr_dims.v);
 
     // compute
     if (is_neox) {
@@ -929,7 +928,7 @@ void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
     const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;
 
     rope_corr_dims corr_dims;
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, ggml_rope_get_truncate(rope), corr_dims.v);
 
     const size_t ts0 = ggml_type_size(x->type);
     GGML_ASSERT(x->nb[0] == ts0);
diff --git src/ggml-cuda/topk-moe.cu src/ggml-cuda/topk-moe.cu
index a93d3a97..e99b2e84 100644
--- src/ggml-cuda/topk-moe.cu
+++ src/ggml-cuda/topk-moe.cu
@@ -217,6 +217,9 @@ __global__ void topk_moe_cuda(const float *         logits,
             const int expert = i + threadIdx.x;
             selection_wt[i / WARP_SIZE] =
                 (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
+            if (__isnanf(selection_wt[i / WARP_SIZE])) {
+                selection_wt[i / WARP_SIZE] = -FLT_MAX;
+            }
         }
     }
 
diff --git src/ggml-et/ggml-et.cpp src/ggml-et/ggml-et.cpp
index cdd947c5..466e6890 100644
--- src/ggml-et/ggml-et.cpp
+++ src/ggml-et/ggml-et.cpp
@@ -1069,7 +1069,7 @@ static bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, const ggm
                 const bool zero_rot_offset  = ggml_get_op_params_i32(op, 15) == 0;
 
                 supported =
-                    zero_view_offset && zero_rot_offset && ndims <= 512 &&
+                    ggml_rope_get_truncate(op) && zero_view_offset && zero_rot_offset && ndims <= 512 &&
                     (is_normal || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
             } else {
                 supported = false;
diff --git src/ggml-hexagon/ggml-hexagon.cpp src/ggml-hexagon/ggml-hexagon.cpp
index 47e90b03..9b2600c6 100644
--- src/ggml-hexagon/ggml-hexagon.cpp
+++ src/ggml-hexagon/ggml-hexagon.cpp
@@ -90,7 +90,7 @@ size_t opt_vmem       = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for
 int    opt_etm        = 0;
 int    opt_verbose    = 0;
 int    opt_profile    = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
-bool   opt_hostbuf    = false;
+bool   opt_hostbuf    = true;
 bool   opt_dma64      = false;
 int    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start
 
@@ -150,7 +150,7 @@ int    opt_opqueue   = 8;    // max number of pending batches
 int    opt_optrace   = 0;    // trace buffer size per thread (0 means default)
 int    opt_oppoll    = 0;    // polling for batch completions
 int    opt_opfusion  = 1;    // enable/disable op fusion
-int    opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always
+int    opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = all tileable weights (default)
 
 static std::regex* opt_opfilter = NULL; // regex of ops to not claim
 
@@ -463,7 +463,6 @@ struct ggml_hexagon_rpcmem_block {
 struct ggml_hexagon_shared_buffer {
     ggml_hexagon_session *                     sess;
     std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
-    std::vector<htp_tensor_extra *>            tensor_extra;
     bool     mapped;
     bool     pinned;
     bool     extended;
@@ -486,6 +485,9 @@ struct ggml_hexagon_shared_buffer {
         if (err != 0) {
             GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                     sess->domain_id, size(), fd(), (unsigned) err);
+            if (!this->pinned) {
+                GGML_LOG_ERROR("ggml-hex: %s possibly out of IOVA space, try --no-host\n", sess->c_name());
+            }
             throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
         }
 
@@ -561,9 +563,6 @@ struct ggml_hexagon_shared_buffer {
 
     ~ggml_hexagon_shared_buffer() {
         free();
-        for (auto * extra : tensor_extra) {
-            delete extra;
-        }
     }
 };
 
@@ -622,11 +621,9 @@ inline void ggml_hexagon_session::free_fence(void * ptr, uint32_t n_slots) {
 }
 
 inline ggml_tensor ggml_hexagon_session::make_fence_tensor(const volatile void * slot, uint32_t n_slots) const {
-    static htp_tensor_extra fence_extra { {}, 0, HTP_TENSOR_FENCE };
-
     ggml_tensor t {};
     t.buffer = &this->fence_buf->backend_buffer;
-    t.extra  = &fence_extra;
+    t.extra  = (void *) (uintptr_t) HTP_TENSOR_FENCE;
     t.data   = const_cast<void *>(slot);
     t.type   = GGML_TYPE_I8;
     t.ne[0]  = HTP_FENCE_SLOT_SIZE;
@@ -659,10 +656,7 @@ static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buf
     HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                 tensor->name, (void *) sbuf->base(), tensor->data, ggml_nbytes(tensor));
 
-    auto extra = new htp_tensor_extra();
-    sbuf->tensor_extra.push_back(extra);
-
-    tensor->extra = extra;
+    tensor->extra = (void *) (uintptr_t) 0;
 
     return GGML_STATUS_SUCCESS;
 }
@@ -672,42 +666,19 @@ static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    const void *          data,
                                                    size_t                offset,
                                                    size_t                size) {
-    auto extra = (htp_tensor_extra *)           tensor->extra;
-    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
-    auto sess  = sbuf->sess;
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+
+    sess->flush_sync();
 
     if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
-        extra->flags |= HTP_TENSOR_WEIGHT;
-        if (htp_is_tiled_weight_tensor(tensor)) {
-            extra->flags |= HTP_TENSOR_TILED;
-        }
+        ggml_hexagon_tensor_set_flags(tensor, HTP_TENSOR_WEIGHT);
     }
 
     HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
-        sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, extra->flags);
-
-    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
-        memcpy((char *) tensor->data + offset, data, size);
-        return;
-    }
-
-    if (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
-        htp_repack_tensor_tiled(tensor, data, size);
-        return;
-    }
-
-    if (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
-        extra->shadow_buf.resize(ggml_nbytes(tensor));
-    }
-    memcpy(extra->shadow_buf.data() + offset, data, size);
-    extra->shadow_size += size;
+        sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, ggml_hexagon_tensor_get_flags(tensor));
 
-    if (extra->shadow_size >= ggml_nbytes(tensor)) {
-        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
-        extra->shadow_buf.clear();
-        extra->shadow_buf.shrink_to_fit();
-        extra->shadow_size = 0;
-    }
+    memcpy((char *) tensor->data + offset, data, size);
 }
 
 static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
@@ -715,14 +686,14 @@ static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    void *                data,
                                                    size_t                offset,
                                                    size_t                size) {
-    auto extra = (htp_tensor_extra *)           tensor->extra;
-    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
-    auto sess  = sbuf->sess;
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+    const uint32_t flags = ggml_hexagon_tensor_get_flags(tensor);
 
     HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
-            sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, extra->flags);
+            sess->c_name(), tensor->name, data, offset, size, (int) buffer->usage, flags);
 
-    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
+    if ((flags & HTP_TENSOR_TILED) == 0) {
         memcpy(data, (const char *) tensor->data + offset, size);
         return;
     }
@@ -742,47 +713,27 @@ static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      bu
 }
 
 static void ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
-                                                          ggml_tensor *         tensor,
-                                                          const void *          data,
-                                                          size_t                offset,
-                                                          size_t                size,
-                                                          size_t                n_copies,
-                                                          size_t                stride_tensor,
-                                                          size_t                stride_data) {
-    auto extra = (htp_tensor_extra *)           tensor->extra;
-    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
-    auto sess  = sbuf->sess;
+                                                      ggml_tensor *         tensor,
+                                                      const void *          data,
+                                                      size_t                offset,
+                                                      size_t                size,
+                                                      size_t                n_copies,
+                                                      size_t                stride_tensor,
+                                                      size_t                stride_data) {
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+
+    sess->flush_sync();
 
     if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
-        extra->flags |= HTP_TENSOR_WEIGHT;
-        if (htp_is_tiled_weight_tensor(tensor)) {
-            extra->flags |= HTP_TENSOR_TILED;
-        }
+        ggml_hexagon_tensor_set_flags(tensor, HTP_TENSOR_WEIGHT);
     }
 
     HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
-                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage, extra->flags);
-
-    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
-        for (size_t i = 0; i < n_copies; i++) {
-            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
-        }
-        return;
-    }
+                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage, ggml_hexagon_tensor_get_flags(tensor));
 
-    if (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
-        extra->shadow_buf.resize(ggml_nbytes(tensor));
-    }
     for (size_t i = 0; i < n_copies; i++) {
-        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
-    }
-    extra->shadow_size += n_copies * size;
-
-    if (extra->shadow_size >= ggml_nbytes(tensor)) {
-        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
-        extra->shadow_buf.clear();
-        extra->shadow_buf.shrink_to_fit();
-        extra->shadow_size = 0;
+        memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
     }
 }
 
@@ -794,14 +745,14 @@ static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buff
                                                       size_t                n_copies,
                                                       size_t                stride_tensor,
                                                       size_t                stride_data) {
-    auto extra = (htp_tensor_extra *)           tensor->extra;
-    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
-    auto sess  = sbuf->sess;
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+    const uint32_t flags = ggml_hexagon_tensor_get_flags(tensor);
 
     HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                 sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (int) buffer->usage);
 
-    if ((extra->flags & HTP_TENSOR_TILED) == 0) {
+    if ((flags & HTP_TENSOR_TILED) == 0) {
         for (size_t i = 0; i < n_copies; i++) {
             memcpy((uint8_t *)data + i * stride_data, (const uint8_t *)tensor->data + offset + i * stride_tensor, size);
         }
@@ -825,6 +776,29 @@ static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buff
     }
 }
 
+static void ggml_backend_hexagon_buffer_set_tensor_n(ggml_backend_buffer_t buffer,
+                                                     struct ggml_tensor * const * tensors,
+                                                     const void * const * data,
+                                                     const size_t * offsets,
+                                                     const size_t * sizes,
+                                                     size_t n_tensors) {
+    const bool is_weights = ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+
+    sess->flush_sync();
+
+    for (size_t i = 0; i < n_tensors; i++) {
+        if (sizes[i] == 0) {
+            continue;
+        }
+        if (is_weights) {
+            ggml_hexagon_tensor_set_flags(tensors[i], HTP_TENSOR_WEIGHT);
+        }
+        memcpy((char *) tensors[i]->data + offsets[i], data[i], sizes[i]);
+    }
+}
+
 static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
     auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
     auto sess = sbuf->sess;
@@ -844,7 +818,7 @@ static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
     /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
     /* .clear           = */ ggml_backend_hexagon_buffer_clear,
     /* .reset           = */ NULL,
-    /* .set_tensor_n    = */ NULL,
+    /* .set_tensor_n    = */ ggml_backend_hexagon_buffer_set_tensor_n,
     /* .get_tensor_n    = */ NULL,
     /* .cpy_tensor_n    = */ NULL,
 };
@@ -856,8 +830,12 @@ static void ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t bu
                                                         const void *          data,
                                                         size_t                offset,
                                                         size_t                size) {
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+
+    sess->flush_sync();
+
     memcpy((char *) tensor->data + offset, data, size);
-    GGML_UNUSED(buffer);
 }
 
 static void ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
@@ -869,6 +847,25 @@ static void ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t bu
     GGML_UNUSED(buffer);
 }
 
+static void ggml_backend_hexagon_host_buffer_set_tensor_n(ggml_backend_buffer_t buffer,
+                                                          struct ggml_tensor * const * tensors,
+                                                          const void * const * data,
+                                                          const size_t * offsets,
+                                                          const size_t * sizes,
+                                                          size_t n_tensors) {
+    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
+    auto sess = sbuf->sess;
+
+    sess->flush_sync();
+
+    for (size_t i = 0; i < n_tensors; i++) {
+        if (sizes[i] == 0) {
+            continue;
+        }
+        memcpy((char *) tensors[i]->data + offsets[i], data[i], sizes[i]);
+    }
+}
+
 static ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
     /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
     /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
@@ -881,7 +878,7 @@ static ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
     /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
     /* .clear           = */ ggml_backend_hexagon_buffer_clear,
     /* .reset           = */ NULL,
-    /* .set_tensor_n    = */ NULL,
+    /* .set_tensor_n    = */ ggml_backend_hexagon_host_buffer_set_tensor_n,
     /* .get_tensor_n    = */ NULL,
     /* .cpy_tensor_n    = */ NULL,
 };
@@ -896,7 +893,7 @@ static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
             ggml_backend_buffer_type_t buffer_type, size_t size) {
     auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
     auto sess    = dev_ctx->session();
-    if (sess && sess->max_vmem && size > sess->max_vmem) {
+    if (size > sess->max_vmem) {
         GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                        dev_ctx->c_name(), size, sess->max_vmem);
         return nullptr;
@@ -914,7 +911,7 @@ static ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
             ggml_backend_buffer_type_t buffer_type, size_t size) {
     auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
     auto sess    = dev_ctx->session();
-    if (sess && sess->max_vmem && size > sess->max_vmem) {
+    if (size > sess->max_vmem) {
         GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                        dev_ctx->c_name(), size, sess->max_vmem);
         return nullptr;
@@ -934,7 +931,7 @@ static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer
 }
 
 static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
-    if (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
+    if (opt_rt_repack != 0 && !ggml_backend_buft_is_host(buft) && ggml_op_is_empty(t->op) && htp_is_tiled_weight_tensor(t)) {
         int64_t ne0 = hex_round_up(t->ne[0], 32);
         int64_t ne1 = hex_round_up(t->ne[1], 32);
         int64_t ne2 = t->ne[2];
@@ -1212,8 +1209,7 @@ struct ggml_hexagon_opbatch {
     std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
     std::vector<htp_op_desc>         h_ops;     // htp op descriptors
 
-    std::unordered_map<const ggml_tensor*, int> t_map; // tensor ptr  to index
-    std::unordered_multimap<void*, int>         d_map; // tensor data to index
+    std::unordered_map<uint64_t, int> t_map; // tensor hash to index
 
     unsigned int n_bufs;     // num buffers in the batch
     unsigned int n_tens;     // num tensors ...
@@ -1232,7 +1228,6 @@ struct ggml_hexagon_opbatch {
         b_vmem = 0;
 
         t_map.clear();
-        d_map.clear();
         ops.resize(n_ops_max);
     }
 
@@ -1252,7 +1247,6 @@ struct ggml_hexagon_opbatch {
         h_ops.resize(n_ops_max);
 
         t_map.reserve(n_tens_max);
-        d_map.reserve(n_tens_max);
 
         GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                 sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);
@@ -1298,7 +1292,7 @@ struct ggml_hexagon_opbatch {
     }
 
     static void init_tensor_shape(htp_tensor & h, const ggml_tensor * t) {
-        auto extra = (htp_tensor_extra *) t->extra;
+        const uint32_t flags = ggml_hexagon_tensor_get_flags(t);
 
         h.type  = t->type;
         h.ne[0] = t->ne[0];
@@ -1306,7 +1300,7 @@ struct ggml_hexagon_opbatch {
         h.ne[2] = t->ne[2];
         h.ne[3] = t->ne[3];
 
-        if (extra->flags & HTP_TENSOR_TILED) {
+        if (flags & HTP_TENSOR_TILED) {
             if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                 h.type  = GGML_TYPE_F16;
                 h.nb[0] = sizeof(ggml_fp16_t);
@@ -1327,31 +1321,42 @@ struct ggml_hexagon_opbatch {
             h.size  = ggml_nbytes(t);
         }
 
-        h.flags = extra->flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_TILED | HTP_TENSOR_FENCE);
+        h.flags = flags & (HTP_TENSOR_WEIGHT | HTP_TENSOR_TILED | HTP_TENSOR_FENCE);
     }
 
-    static bool same_shape(const htp_tensor & a, const htp_tensor & b) {
-        return (a.type == b.type) &&
-               (a.ne[0] == b.ne[0]) && (a.ne[1] == b.ne[1]) && (a.ne[2] == b.ne[2]) && (a.ne[3] == b.ne[3]) &&
-               (a.nb[0] == b.nb[0]) && (a.nb[1] == b.nb[1]) && (a.nb[2] == b.nb[2]) && (a.nb[3] == b.nb[3]);
+    static uint64_t hash_tensor(const ggml_tensor * t, const htp_tensor & shape) {
+        uint64_t hash = 1469598103934665603ull;
+        auto hash_value = [&hash](uint64_t value) {
+            hash ^= value;
+            hash *= 1099511628211ull;
+        };
+
+        hash_value((uintptr_t) t->data);
+        hash_value(shape.type);
+        hash_value(shape.size);
+        hash_value(shape.flags);
+        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
+            hash_value(shape.ne[i]);
+            hash_value(shape.nb[i]);
+        }
+
+        // avalanche high bits down to low bits for power-of-two table indexing
+        hash ^= hash >> 33;
+        hash *= 0xff51afd7ed558ccdull;
+        hash ^= hash >> 33;
+
+        return hash;
     }
 
     // add tensor and return its index
     int add_tensor(const ggml_tensor * t) {
-        // Lookup by tensor ptr
-        auto it = t_map.find(t);
-        if (it != t_map.end()) { return it->second; }
-
         htp_tensor shape {};
         init_tensor_shape(shape, t);
 
-        // Lookup by tensor data and shape
-        auto range = d_map.equal_range(t->data);
-        for (auto dit = range.first; dit != range.second; ++dit) {
-            if (same_shape(h_tens[dit->second], shape)) {
-                t_map.insert({t, dit->second});
-                return dit->second;
-            }
+        const uint64_t hash = hash_tensor(t, shape);
+        auto it = t_map.find(hash);
+        if (it != t_map.end()) {
+            return it->second;
         }
 
         auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
@@ -1360,8 +1365,7 @@ struct ggml_hexagon_opbatch {
         int ti = n_tens++;
         GGML_ASSERT(n_tens <= n_tens_max);
 
-        t_map.insert({t,       ti});
-        d_map.insert({t->data, ti});
+        t_map[hash] = ti;
 
         uint64_t t_offset = (uint8_t *) t->data - sbuf->base();
 
@@ -1391,8 +1395,11 @@ struct ggml_hexagon_opbatch {
 
         auto fit_tensor = [&](const ggml_tensor * t) {
             if (!t) return;
+
             if (check_tens) {
-                if (t_map.count(t)) return;
+                htp_tensor shape {};
+                init_tensor_shape(shape, t);
+                if (t_map.count(hash_tensor(t, shape))) return;
                 extra_tens++;
             }
 
@@ -2352,6 +2359,7 @@ ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
 
 // ** backend interface
 
+
 static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
     auto sess = static_cast<ggml_hexagon_session *>(backend->context);
     return sess->c_name();
@@ -2363,11 +2371,33 @@ static void ggml_backend_hexagon_free(ggml_backend_t backend) {
     delete backend;
 }
 
-static inline bool op_is_compute(ggml_tensor *node)
-{
+static inline bool ggml_hexagon_op_is_compute(const struct ggml_tensor * node) {
     return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
 }
 
+static void ggml_hexagon_tensor_repack(ggml_tensor * tensor, ggml_hexagon_session * sess) {
+    if (tensor->data == nullptr || ggml_hexagon_tensor_is_tiled(tensor) || !ggml_hexagon_tensor_can_tile(tensor)) {
+        return;
+    }
+    HEX_VERBOSE("ggml-hex: %s rt-repack %s|size %zu\n", sess->c_name(), htp_format_tensor(tensor, HTP_OPFMT_TENSOR).c_str(), ggml_nbytes(tensor));
+    std::vector<uint8_t> raw(ggml_nbytes(tensor));
+    memcpy(raw.data(), tensor->data, raw.size());
+    htp_repack_tensor_tiled(tensor, raw.data(), raw.size());
+    ggml_hexagon_tensor_set_flags(tensor, HTP_TENSOR_TILED);
+}
+
+static void ggml_hexagon_graph_repack(const ggml_cgraph * graph, ggml_hexagon_session * sess) {
+    for (int i = 0; i < graph->n_nodes; ++i) {
+        ggml_tensor * n = graph->nodes[i];
+        if (!ggml_hexagon_op_is_compute(n)) {
+            continue;
+        }
+        if (n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) {
+            ggml_hexagon_tensor_repack(n->src[0], sess);
+        }
+    }
+}
+
 static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
     auto sess = static_cast<ggml_hexagon_session *>(backend->context);
 
@@ -2384,13 +2414,15 @@ static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, gg
     if (cache_hit) {
         nodes_ptr = &sess->cached_nodes;
     } else {
+        ggml_hexagon_graph_repack(graph, sess);
+
         htp_fusion_tag_graph(graph);
 
         computed_nodes.reserve(graph->n_nodes);
 
         for (int i = 0; i < graph->n_nodes; ++i) {
             ggml_tensor * n = graph->nodes[i];
-            if (!op_is_compute(n)) {
+            if (!ggml_hexagon_op_is_compute(n)) {
                 continue;
             }
 
@@ -2400,22 +2432,7 @@ static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, gg
             }
 
             htp_opnode node(opcode, n);
-            if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_GET_ROWS) {
-                ggml_tensor * src0 = node.node->src[0];
-                if (htp_needs_repack(src0)) {
-                    HEX_VERBOSE("ggml-hex: %s rt-repack %s|size %zu\n", sess->c_name(), htp_format_tensor(src0, HTP_OPFMT_TENSOR).c_str(), ggml_nbytes(src0));
-                    std::vector<uint8_t> raw(ggml_nbytes(src0));
-                    memcpy(raw.data(), src0->data, raw.size());
-                    htp_repack_tensor_tiled(src0, raw.data(), raw.size());
-                    auto * extra = (htp_tensor_extra *) src0->extra;
-                    extra->flags |= HTP_TENSOR_TILED;
-                }
-            }
-
             GGML_ASSERT(htp_kparams_solve(sess, node));
-            if (node.opcode == HTP_OP_CPY && node.kparams<htp_copy_kernel_params>()->total_elems == 0) {
-                continue;
-            }
 
             computed_nodes.push_back(std::move(node));
         }
@@ -2625,7 +2642,11 @@ static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<no
 }
 
 static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
-    GGML_UNUSED(params);
+    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
+
+    HEX_VERBOSE("ggml-hex: %s graph-optimize n_nodes %d\n", sess->c_name(), gf->n_nodes);
+
+    ggml_hexagon_graph_repack(gf, sess);
 
     const int n = gf->n_nodes;
 
@@ -2660,7 +2681,7 @@ static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgr
         }
     }
 
-    GGML_UNUSED(backend);
+    GGML_UNUSED(params);
 }
 
 static uint64_t ggml_hexagon_session_key(const ggml_hexagon_session * sess) {
@@ -2722,14 +2743,7 @@ static bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, gg
         return false;
     }
 
-    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
-    if (!dst->extra) {
-        ggml_backend_buffer_init_tensor(dst->buffer, dst);
-    }
-
-    auto * dst_extra = static_cast<htp_tensor_extra *>(dst->extra);
-    const auto * src_extra = static_cast<const htp_tensor_extra *>(src->extra);
-    dst_extra->flags = src_extra->flags & ~HTP_TENSOR_FUSEABLE;
+    dst->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(src) & ~HTP_TENSOR_FUSEABLE);
 
     auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
     auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);
@@ -3002,18 +3016,18 @@ static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, cons
     }
 
     bool supp = false;
-    switch (op->op) {
-        case GGML_OP_NONE:
-        case GGML_OP_RESHAPE:
-        case GGML_OP_VIEW:
-        case GGML_OP_PERMUTE:
-        case GGML_OP_TRANSPOSE:
-            supp = true;
-            break;
-
-        default:
+    if (ggml_op_is_empty(op->op) || ggml_is_empty(op)) {
+        supp = true;
+    } else {
+        supp = ggml_backend_hexagon_buffer_type_get_alloc_size(&dev_ctx->buffer_type, op) <= sess->max_vmem;
+        for (int i = 0; supp && i < GGML_MAX_SRC; i++) {
+            if (op->src[i] && ggml_backend_hexagon_buffer_type_get_alloc_size(&dev_ctx->buffer_type, op->src[i]) > sess->max_vmem) {
+                supp = false;
+            }
+        }
+        if (supp) {
             supp = htp_kparams_check(sess, op);
-            break;
+        }
     }
 
     ggml_hexagon_dump_op_supp(sess->c_name(), op, supp);
diff --git src/ggml-hexagon/htp-common.h src/ggml-hexagon/htp-common.h
index 472892be..815a021b 100644
--- src/ggml-hexagon/htp-common.h
+++ src/ggml-hexagon/htp-common.h
@@ -4,12 +4,6 @@
 #include "htp-opnode.h"
 #include "htp-repack.h"
 
-struct htp_tensor_extra {
-    std::vector<uint8_t> shadow_buf;
-    size_t               shadow_size { 0 };
-    uint32_t             flags { 0 };
-};
-
 extern int    opt_arch;
 extern size_t opt_ndev;
 extern size_t opt_nhvx;
@@ -46,36 +40,41 @@ extern int    opt_rt_repack;
     if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)
 #endif
 
-static inline bool htp_tensor_is_fuseable(const struct ggml_tensor * t) {
-    if (!t || !t->extra) return false;
-    const auto * extra = (const struct htp_tensor_extra *) t->extra;
-    return (extra->flags & HTP_TENSOR_FUSEABLE) != 0;
+static inline uint32_t ggml_hexagon_tensor_get_flags(const struct ggml_tensor * t) {
+    return (uint32_t) (uintptr_t) t->extra;
 }
 
-static inline bool htp_can_rt_repack(const struct ggml_tensor * t) {
-    if (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
-        return false;
-    }
-    if (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
-        return false;
-    }
-    return opt_rt_repack > 1 || ggml_is_quantized(t->type);
+static inline void ggml_hexagon_tensor_set_flags(struct ggml_tensor * t, uint32_t flags) {
+    t->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(t) | flags);
 }
 
-static inline bool htp_tensor_is_tiled(const struct ggml_tensor * t) {
-    const auto * extra = (const struct htp_tensor_extra *) t->extra;
-    if (extra && (extra->flags & HTP_TENSOR_TILED)) {
-        return true;
-    }
-    return htp_can_rt_repack(t);
+static inline void ggml_hexagon_tensor_clear_flags(struct ggml_tensor * t, uint32_t flags) {
+    t->extra = (void *) (uintptr_t) (ggml_hexagon_tensor_get_flags(t) & ~flags);
 }
 
-static inline bool htp_needs_repack(const struct ggml_tensor * t) {
-    const auto * extra = (const struct htp_tensor_extra *) t->extra;
-    if (extra && (extra->flags & HTP_TENSOR_TILED)) {
+static inline bool ggml_hexagon_tensor_is_fuseable(const struct ggml_tensor * t) {
+    return (ggml_hexagon_tensor_get_flags(t) & HTP_TENSOR_FUSEABLE) != 0;
+}
+
+static inline bool ggml_hexagon_tensor_is_tiled(const struct ggml_tensor * t) {
+    return (ggml_hexagon_tensor_get_flags(t) & HTP_TENSOR_TILED) != 0;
+}
+
+static inline bool ggml_hexagon_tensor_can_tile(const struct ggml_tensor * t) {
+    if (opt_rt_repack == 0) {
         return false;
     }
-    return htp_can_rt_repack(t);
+    if (t->buffer) {
+        if (ggml_backend_buft_is_host(t->buffer->buft) ||
+            ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
+            return false;
+        }
+    }
+    return htp_is_tiled_weight_tensor(t);
+}
+
+static inline bool ggml_hexagon_tensor_will_tile(const struct ggml_tensor * t) {
+    return ggml_hexagon_tensor_is_tiled(t) || ggml_hexagon_tensor_can_tile(t);
 }
 
 static inline bool htp_tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
diff --git src/ggml-hexagon/htp-fusion.cpp src/ggml-hexagon/htp-fusion.cpp
index ba2dd641..4de6dd8b 100644
--- src/ggml-hexagon/htp-fusion.cpp
+++ src/ggml-hexagon/htp-fusion.cpp
@@ -35,7 +35,7 @@ static inline bool htp_match_consumer(
         return false;
     }
     const ggml_tensor * prev_out = prev.dst();
-    if (!htp_tensor_is_fuseable(prev_out)) {
+    if (!ggml_hexagon_tensor_is_fuseable(prev_out)) {
         return false;
     }
     if (htp_same_tensor_data(cur.src0(), prev_out)) {
@@ -53,14 +53,14 @@ static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struc
     if (kparams->n_hmx) {
         return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
     }
-    return htp_tensor_is_tiled(src0) && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW);
+    return ggml_hexagon_tensor_is_tiled(src0) && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW);
 }
 
 static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
     if (kparams->n_hmx) {
         return kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
     }
-    return htp_tensor_is_tiled(src0) &&
+    return ggml_hexagon_tensor_is_tiled(src0) &&
            (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
 }
 
@@ -74,7 +74,7 @@ static bool fuse_allreduce_add(const ggml_hexagon_session * sess, htp_opnode & p
     const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
 
     const ggml_tensor * ar_local = prev.inputs[rank];
-    if (!htp_tensor_is_fuseable(ar_local)) return false;
+    if (!ggml_hexagon_tensor_is_fuseable(ar_local)) return false;
 
     const ggml_tensor * add_dst    = cur.dst();
     const ggml_tensor * res_tensor = nullptr;
@@ -250,7 +250,7 @@ static bool fuse_mul_mat_nx(const ggml_hexagon_session * sess, htp_opnode & prev
 
     if (x_in != x_orig || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
         w0->ne[2] != 1 || w0->ne[3] != 1 || w_in->ne[2] != 1 || w_in->ne[3] != 1 ||
-        htp_tensor_is_tiled(w_in) != htp_tensor_is_tiled(w0) ||
+        ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0) ||
         orig_kparams->n_hmx != cur_kparams->n_hmx) {
         return false;
     }
@@ -295,7 +295,7 @@ static bool fuse_mul_mat_id_nx(const ggml_hexagon_session * sess, htp_opnode & p
 
     if (x_in != x || ids_in != ids || w_in->type != w0->type ||
         w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
-        htp_tensor_is_tiled(w_in) != htp_tensor_is_tiled(w0) ||
+        ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0) ||
         orig_kparams->n_hmx != cur_kparams->n_hmx) {
         return false;
     }
@@ -598,19 +598,17 @@ int htp_fusion_max(const struct ggml_cgraph * graph, int idx) {
 
 void htp_fusion_tag_graph(struct ggml_cgraph * graph) {
     for (int i = 0; i < graph->n_nodes; i++) {
-        auto * extra = (htp_tensor_extra *) graph->nodes[i]->extra;
-        if (!extra) continue;
+        ggml_tensor * node = graph->nodes[i];
+        ggml_hexagon_tensor_clear_flags(node, HTP_TENSOR_FUSEABLE);
 
-        extra->flags &= ~HTP_TENSOR_FUSEABLE;
-
-        const ggml_op op = graph->nodes[i]->op;
+        const ggml_op op = node->op;
         if ((op == GGML_OP_RMS_NORM || op == GGML_OP_SCALE || op == GGML_OP_REPEAT ||
              op == GGML_OP_UNARY    || op == GGML_OP_ADD   || op == GGML_OP_SUB    ||
              op == GGML_OP_MUL      || op == GGML_OP_DIV   || op == GGML_OP_MUL_MAT_ID ||
              op == GGML_OP_SSM_CONV ||
-             (op == GGML_OP_MUL_MAT && !htp_is_hadamard(graph->nodes[i]))) &&
+             (op == GGML_OP_MUL_MAT && !htp_is_hadamard(node))) &&
             ggml_node_has_n_uses(graph, i, 1)) {
-            extra->flags |= HTP_TENSOR_FUSEABLE;
+            ggml_hexagon_tensor_set_flags(node, HTP_TENSOR_FUSEABLE);
         }
     }
 }
diff --git src/ggml-hexagon/htp-kparams.cpp src/ggml-hexagon/htp-kparams.cpp
index 48bd9327..4fa900be 100644
--- src/ggml-hexagon/htp-kparams.cpp
+++ src/ggml-hexagon/htp-kparams.cpp
@@ -628,7 +628,7 @@ static bool htp_matmul_is_hmx_eligible(
         return false;
     }
 
-    if (!htp_is_hmx_weight_type((ggml_type) wtype)) {
+    if (!htp_is_tiled_weight_type((ggml_type) wtype)) {
         return false;
     }
 
@@ -636,7 +636,7 @@ static bool htp_matmul_is_hmx_eligible(
         return false;
     }
 
-    const bool is_tiled = htp_tensor_is_tiled(src0);
+    const bool is_tiled = ggml_hexagon_tensor_will_tile(src0);
     if (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
         return false;
     }
@@ -674,7 +674,7 @@ static bool htp_kparams_solve_hmx_mm(
     size_t vtcm_budget,
     struct htp_mm_kernel_params * kparams
 ) {
-    const bool     is_tiled          = htp_tensor_is_tiled(src0);
+    const bool     is_tiled          = ggml_hexagon_tensor_will_tile(src0);
     const int      aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
     const int      n_threads         = (int) htp_session_n_threads(sess);
     const uint32_t mdev_count        = htp_session_mdev_count(sess);
@@ -893,7 +893,7 @@ static bool htp_kparams_solve_matmul_impl(
     const int ne12 = src1->ne[2];
     const int ne13 = src1->ne[3];
 
-    const bool is_tiled    = htp_tensor_is_tiled(src0);
+    const bool is_tiled    = ggml_hexagon_tensor_will_tile(src0);
     const int  wtype       = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
     const int  ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
     const int  ne11_padded = hex_round_up(ne11, 32);
@@ -1109,7 +1109,7 @@ bool htp_kparams_solve_mmnx(
     const int ne12 = act->ne[2];
     const int ne13 = act->ne[3];
 
-    const bool is_tiled    = htp_tensor_is_tiled(src0);
+    const bool is_tiled    = ggml_hexagon_tensor_will_tile(src0);
     const int  wtype       = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
     const int  ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
     const int  ne11_padded = hex_round_up(ne11, 32);
@@ -1683,7 +1683,7 @@ bool htp_kparams_solve_repeat_get_rows(
     const uint32_t ne12 = src1->ne[2];
     const uint32_t nr   = ne10 * ne11 * ne12;
 
-    const bool tiled = htp_tensor_is_tiled(src0);
+    const bool tiled = ggml_hexagon_tensor_is_tiled(src0);
     const int  wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
 
     if (tiled) {
@@ -1874,6 +1874,10 @@ static bool htp_kparams_solve_rope(
     const struct ggml_tensor * src2 = op->src[2];
     const struct ggml_tensor * dst  = op;
 
+    if (!ggml_rope_get_truncate(op)) {
+        return false;
+    }
+
     if (!ggml_are_same_shape(src0, dst)) {
         return false;
     }
diff --git src/ggml-hexagon/htp-repack.h src/ggml-hexagon/htp-repack.h
index f656e9b8..f3206bc8 100644
--- src/ggml-hexagon/htp-repack.h
+++ src/ggml-hexagon/htp-repack.h
@@ -8,31 +8,30 @@
 #include "htp/htp-tensor.h"
 
 static inline bool htp_is_tiled_weight_type(enum ggml_type type) {
-    return type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
-           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
-           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
-           type == GGML_TYPE_Q6_K   ||
+    return type == GGML_TYPE_Q1_0  || type == GGML_TYPE_Q2_0   ||
+           type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
+           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_IQ4_NL ||
+           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_Q6_K   ||
+           type == GGML_TYPE_Q2_K  || type == GGML_TYPE_Q3_K   ||
            type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
-           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
-           type == GGML_TYPE_Q2_0  ||
            type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
            type == GGML_TYPE_BF16;
 }
 
 static inline bool htp_is_tiled_weight_tensor(const struct ggml_tensor * t) {
-    if (!ggml_is_contiguous(t) || ggml_is_view(t)) {
+    if (!htp_is_tiled_weight_type(t->type)) {
         return false;
     }
-    if (!htp_is_tiled_weight_type(t->type)) {
+    if (!ggml_is_contiguous(t) || ggml_is_view(t)) {
         return false;
     }
-    if (t->ne[0] < 32) {
+    if (t->ne[0] < 32 || (t->ne[0] % 32 != 0) || t->ne[3] != 1) {
         return false;
     }
     if (!ggml_is_quantized(t->type)) {
-        return (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
+        return t->ne[1] > 1;
     }
-    return t->ne[3] == 1;
+    return true;
 }
 
 // Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
@@ -59,10 +58,6 @@ static inline size_t htp_tiled_row_size(enum ggml_type type, int64_t ne0) {
     return ggml_row_size(type, ne0);
 }
 
-static inline bool htp_is_hmx_weight_type(enum ggml_type type) {
-    return htp_is_tiled_weight_type(type);
-}
-
 void htp_repack_tensor_tiled(struct ggml_tensor * t, const void * data, size_t size);
 void htp_unpack_tensor_tiled(void * data, const struct ggml_tensor * t, size_t offset, size_t size);
 
diff --git src/ggml-impl.h src/ggml-impl.h
index 4dc55e78..7c6a605a 100644
--- src/ggml-impl.h
+++ src/ggml-impl.h
@@ -107,6 +107,13 @@ static inline bool ggml_impl_is_view(const struct ggml_tensor * t) {
 static inline float ggml_compute_softplus_f32(float input) {
     return (input > 20.0f) ? input : logf(1 + expf(input));
 }
+
+// compute correction dims for YaRN RoPE scaling
+GGML_API void ggml_rope_yarn_corr_dims(
+    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, bool truncate, float dims[2]);
+
+GGML_API bool ggml_rope_get_truncate(const struct ggml_tensor * a);
+
 //
 // logging
 //
diff --git src/ggml-metal/ggml-metal-common.cpp src/ggml-metal/ggml-metal-common.cpp
index 70fd551e..f3631225 100644
--- src/ggml-metal/ggml-metal-common.cpp
+++ src/ggml-metal/ggml-metal-common.cpp
@@ -38,7 +38,7 @@ bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgr
 
     return !ggml_is_transposed(op->src[0]) &&
            !ggml_is_transposed(op->src[1]) &&
-           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
+           has_simdgroup_mm && ne11 > 8 && ne00 >= 32;
 }
 
 bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
diff --git src/ggml-metal/ggml-metal-context.m src/ggml-metal/ggml-metal-context.m
index 597c126c..e12c55a3 100644
--- src/ggml-metal/ggml-metal-context.m
+++ src/ggml-metal/ggml-metal-context.m
@@ -53,9 +53,9 @@ struct ggml_metal {
 
     // command buffer state
     int n_cb;           // number of extra threads used to submit the command buffers
-    int n_nodes_0;      // number of nodes submitted by the main thread
-    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
-    int n_nodes_per_cb;
+
+    // first node of each extra command buffer, then the end of the graph
+    int cb_start[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
 
     struct ggml_cgraph * gf;
 
@@ -491,7 +491,7 @@ enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph *
     ggml_metal_device_rsets_keep_alive(ctx->dev);
 
     // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
-    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
+    // the first n_main nodes are encoded and submitted for processing directly by the calling thread
     // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
     // each thread creates it's own command buffer and enqueues the ops in parallel
     //
@@ -500,16 +500,22 @@ enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph *
     @autoreleasepool {
         ctx->gf = gf;
 
-        if (ctx->n_cb == 0) {
-            // single-threaded encoding: the whole graph is encoded by one command buffer
-            ctx->n_nodes_0      = gf->n_nodes;
-            ctx->n_nodes_1      = 0;
-            ctx->n_nodes_per_cb = 0;
-        } else {
-            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);
-            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;
-
-            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;
+        // the main thread encodes [0, cb_start[0]) and the extra command buffer i encodes [cb_start[i], cb_start[i + 1])
+        // a fusion cannot span two command buffers, so each boundary moves to the end of the fusable group it falls
+        // into and the encoders fuse the same nodes as with a single command buffer
+        {
+            const int n_rest   = gf->n_nodes - MIN(n_main, gf->n_nodes);
+            const int n_per_cb = n_cb > 0 ? (n_rest + n_cb - 1) / n_cb : 0;
+
+            int i = 0;
+            for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
+                const int target = MIN(n_main + cb_idx*n_per_cb, gf->n_nodes);
+                while (i < target) {
+                    i += ggml_metal_fusion_max(gf, i);
+                }
+                ctx->cb_start[cb_idx] = MIN(i, gf->n_nodes);
+            }
+            ctx->cb_start[n_cb] = gf->n_nodes;
         }
 
         if (ctx->capture_compute >= 0) {
@@ -744,17 +750,12 @@ void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
         const int cb_idx = iter;
         const int n_cb_l = ctx->n_cb;
 
-        const int n_nodes_0 = ctx->n_nodes_0;
-        const int n_nodes_1 = ctx->n_nodes_1;
-
-        const int n_nodes_per_cb = ctx->n_nodes_per_cb;
-
         int idx_start = 0;
-        int idx_end   = n_nodes_0;
+        int idx_end   = ctx->cb_start[0];
 
         if (cb_idx < n_cb_l) {
-            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
-            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
+            idx_start = ctx->cb_start[cb_idx];
+            idx_end   = ctx->cb_start[cb_idx + 1];
         }
 
         id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
diff --git src/ggml-metal/ggml-metal-device.cpp src/ggml-metal/ggml-metal-device.cpp
index 463d57c7..72133d86 100644
--- src/ggml-metal/ggml-metal-device.cpp
+++ src/ggml-metal/ggml-metal-device.cpp
@@ -1738,12 +1738,13 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge(ggml
 }
 
 ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
-        ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm) {
+        ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm, int32_t gating, bool pbias) {
     char base[256];
     char name[256];
 
     snprintf(base, 256, "kernel_topk_moe_f32");
-    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", base, n_expert, top_k, with_norm ? 1 : 0);
+    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d_gating=%d_pbias=%d",
+            base, n_expert, top_k, with_norm ? 1 : 0, gating, pbias ? 1 : 0);
 
     ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
     if (!res.pipeline) {
@@ -1751,6 +1752,8 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
         ggml_metal_cv_set_bool (cv, with_norm, FC_TOPK_MOE + 0);
         ggml_metal_cv_set_int32(cv, n_expert,  FC_TOPK_MOE + 1);
         ggml_metal_cv_set_int32(cv, top_k,     FC_TOPK_MOE + 2);
+        ggml_metal_cv_set_int32(cv, gating,    FC_TOPK_MOE + 3);
+        ggml_metal_cv_set_bool (cv, pbias,     FC_TOPK_MOE + 4);
 
         res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
 
diff --git src/ggml-metal/ggml-metal-device.h src/ggml-metal/ggml-metal-device.h
index e096d611..4053a624 100644
--- src/ggml-metal/ggml-metal-device.h
+++ src/ggml-metal/ggml-metal-device.h
@@ -154,7 +154,7 @@ struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k             (ggml_metal_library_t lib, const struct ggml_tensor * op);
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix       (ggml_metal_library_t lib, const struct ggml_tensor * op);
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge       (ggml_metal_library_t lib, const struct ggml_tensor * op);
-struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm);
+struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm, int32_t gating, bool pbias);
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, enum ggml_op op);
diff --git src/ggml-metal/ggml-metal-device.m src/ggml-metal/ggml-metal-device.m
index 68a9addb..63fbe273 100644
--- src/ggml-metal/ggml-metal-device.m
+++ src/ggml-metal/ggml-metal-device.m
@@ -1696,7 +1696,7 @@ bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_te
             return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
         case GGML_OP_ROPE:
         case GGML_OP_ROPE_BACK:
-            return true;
+            return ggml_rope_get_truncate(op);
         case GGML_OP_IM2COL:
             return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
         case GGML_OP_CONV_2D:
diff --git src/ggml-metal/ggml-metal-fusion.cpp src/ggml-metal/ggml-metal-fusion.cpp
index 020cad29..6766b4bd 100644
--- src/ggml-metal/ggml-metal-fusion.cpp
+++ src/ggml-metal/ggml-metal-fusion.cpp
@@ -345,9 +345,6 @@ static bool ggml_metal_fusion_check_snake(
 
 #define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024
 
-// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
-// This is a multi-output elision chain: the fused kernel writes both the selected
-// expert ids and the gathered/normalized routing weights.
 static const std::vector<ggml_op> ops_topk_moe = {
     GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
 };
@@ -362,6 +359,60 @@ static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
     GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
     GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
 };
+static const std::vector<ggml_op> ops_topk_moe_norm_bias = {
+    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
+};
+static const std::vector<ggml_op> ops_topk_moe_norm_scale_bias = {
+    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
+};
+
+static const std::vector<ggml_op> ops_topk_moe_sp = {
+    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
+};
+static const std::vector<ggml_op> ops_topk_moe_sp_scale = {
+    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
+};
+static const std::vector<ggml_op> ops_topk_moe_sp_norm = {
+    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
+};
+static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale = {
+    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
+};
+static const std::vector<ggml_op> ops_topk_moe_sp_norm_bias = {
+    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
+};
+static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale_bias = {
+    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
+};
+
+static const std::vector<ggml_op> ops_topk_moe_sig = {
+    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
+};
+static const std::vector<ggml_op> ops_topk_moe_sig_scale = {
+    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
+};
+static const std::vector<ggml_op> ops_topk_moe_sig_norm = {
+    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
+};
+static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale = {
+    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
+};
+static const std::vector<ggml_op> ops_topk_moe_sig_norm_bias = {
+    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
+};
+static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale_bias = {
+    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
+    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
+};
 
 static bool ggml_metal_fusion_check_topk_moe(
         const ggml_metal_fusion       * fusion,
@@ -375,8 +426,11 @@ static bool ggml_metal_fusion_check_topk_moe(
 
     const int n_ops = (int) fusion->ops.size();
 
-    const bool with_norm  = n_ops >= 6;
-    const bool with_scale = n_ops == 4 || n_ops == 7;
+    bool with_norm = false;
+    for (int j = 0; j < n_ops; ++j) {
+        with_norm |= fusion->ops[j] == GGML_OP_SUM_ROWS;
+    }
+    const bool with_scale = fusion->ops[n_ops - 1] == GGML_OP_SCALE;
 
     // the fusion table operates on the non-empty node sequence; the raw graph also
     // contains the RESHAPE/VIEW nodes that the fused kernel elides.
@@ -411,24 +465,70 @@ static bool ggml_metal_fusion_check_topk_moe(
         }
     }
 
-    const ggml_tensor * softmax        = gf->nodes[raw_start];
-    const ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
-    const ggml_tensor * argsort        = gf->nodes[raw_start + 2];
-    const ggml_tensor * ids            = gf->nodes[raw_start + 3];
-    const ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
-    const ggml_tensor * out            = gf->nodes[raw_end];
-    const ggml_tensor * logits         = softmax->src[0];
-
-    // the fused kernel implements plain softmax only
-    float scale   = 1.0f;
-    float max_bias = 0.0f;
-    memcpy(&scale,    ((const int32_t *) softmax->op_params) + 0, sizeof(scale));
-    memcpy(&max_bias, ((const int32_t *) softmax->op_params) + 1, sizeof(max_bias));
-    if (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
+    const ggml_tensor * first = gf->nodes[raw_start];
+    const ggml_tensor * out   = gf->nodes[raw_end];
+
+    // the unary chains start at the sigmoid or the softplus node and rank biased scores
+    const bool with_unary = first->op == GGML_OP_UNARY;
+    const bool with_sp    = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS;
+    const bool with_sig   = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID;
+
+    if (with_unary && !with_sp && !with_sig) {
         return false;
     }
 
-    if (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
+    const ggml_tensor * probs;          // the scoring output
+    const ggml_tensor * scores;         // the node the argsort ranks (adds the bias for the unary chains)
+    const ggml_tensor * probs_reshaped;
+    const ggml_tensor * argsort;
+    const ggml_tensor * ids;
+    const ggml_tensor * get_rows;
+    const ggml_tensor * logits = first->src[0];
+    int ids_off;
+
+    if (with_unary) {
+        // the softplus chains add a SQRT node, the sigmoid chains start at the sigmoid node
+        const int base = with_sp ? 1 : 0;
+
+        probs          = gf->nodes[raw_start + base + 0];
+        probs_reshaped = gf->nodes[raw_start + base + 1];
+        scores         = gf->nodes[raw_start + base + 2];
+        argsort        = gf->nodes[raw_start + base + 3];
+        ids            = gf->nodes[raw_start + base + 4];
+        get_rows       = gf->nodes[raw_start + base + 5];
+        ids_off        = base + 4;
+
+        if (scores->op != GGML_OP_ADD || scores->src[0] != probs ||
+            (with_sp && (probs->op != GGML_OP_SQRT || probs->src[0] != first))) {
+            return false;
+        }
+
+        // the selection bias is a per-expert vector shared by all rows
+        const ggml_tensor * bias = scores->src[1];
+        if (bias->type != GGML_TYPE_F32 || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
+            bias->ne[0] != probs->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
+            return false;
+        }
+    } else {
+        probs          = first;
+        scores         = first;
+        probs_reshaped = gf->nodes[raw_start + 1];
+        argsort        = gf->nodes[raw_start + 2];
+        ids            = gf->nodes[raw_start + 3];
+        get_rows       = gf->nodes[raw_start + 4];
+        ids_off        = 3;
+
+        // the fused kernel implements plain softmax only
+        float scale    = 1.0f;
+        float max_bias = 0.0f;
+        memcpy(&scale,    ((const int32_t *) first->op_params) + 0, sizeof(scale));
+        memcpy(&max_bias, ((const int32_t *) first->op_params) + 1, sizeof(max_bias));
+        if (scale != 1.0f || max_bias != 0.0f || first->src[1] || first->src[2]) {
+            return false;
+        }
+    }
+
+    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
         out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
         return false;
     }
@@ -455,38 +555,40 @@ static bool ggml_metal_fusion_check_topk_moe(
         return false;
     }
 
-    if (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
+    // the top-k gathers the unbiased scores, the argsort ranks the biased ones
+    if (probs_reshaped->src[0] != probs || argsort->src[0] != scores ||
         ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
         return false;
     }
 
     if (with_norm) {
-        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
-        const ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
-        const ggml_tensor * clamp            = gf->nodes[raw_start + 7];
-        const ggml_tensor * div              = gf->nodes[raw_start + 8];
-        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];
+        const int norm_off = with_unary ? 6 + (with_sp ? 1 : 0) : 5;
+
+        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + norm_off + 0];
+        const ggml_tensor * sum_rows         = gf->nodes[raw_start + norm_off + 1];
+        const ggml_tensor * denom            = gf->nodes[raw_start + norm_off + 2];
+        const ggml_tensor * div              = gf->nodes[raw_start + norm_off + 3];
+        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + norm_off + 4];
 
         if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
-            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
+            denom->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != denom ||
             out_reshaped->src[0] != div) {
             return false;
         }
 
-        if (with_scale) {
-            const ggml_tensor * scale_node = gf->nodes[raw_start + 10];
-            if (scale_node->src[0] != out_reshaped) {
-                return false;
-            }
+        // the kernel applies the sum adjustment as a lower clamp or an additive bias, never a scale
+        if (denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) != 1.0f) {
+            return false;
         }
-    } else if (with_scale) {
-        const ggml_tensor * scale_node = gf->nodes[raw_start + 5];
-        if (scale_node->src[0] != get_rows) {
+
+        if (with_scale && out->src[0] != out_reshaped) {
             return false;
         }
+    } else if (with_scale && out->src[0] != get_rows) {
+        return false;
     }
 
-    const int outputs[2] = { raw_start + 3, raw_end };
+    const int outputs[2] = { raw_start + ids_off, raw_end };
     if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
         return false;
     }
@@ -763,33 +865,54 @@ static const std::vector<ggml_op> ops_moe_reduce_8 = {
 static const std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };
 
 static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
-    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     false, ggml_metal_fusion_check_norm },
-    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     false, ggml_metal_fusion_check_norm },
-    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     false, ggml_metal_fusion_check_norm },
-    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     false, ggml_metal_fusion_check_norm },
-    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     false, ggml_metal_fusion_check_norm },
-    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     false, ggml_metal_fusion_check_norm },
-    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     false, ggml_metal_fusion_check_add_chain },
-    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     false, ggml_metal_fusion_check_add_chain },
-    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     false, ggml_metal_fusion_check_add_chain },
-    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     false, ggml_metal_fusion_check_add_chain },
-    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     false, ggml_metal_fusion_check_add_chain },
-    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     false, ggml_metal_fusion_check_add_chain },
-    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     false, ggml_metal_fusion_check_snake },
-    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     true,  ggml_metal_fusion_check_gdn_cache },
-    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    true,  ggml_metal_fusion_check_topk_moe },
-    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    true,  ggml_metal_fusion_check_topk_moe },
-    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    true,  ggml_metal_fusion_check_topk_moe },
-    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    true,  ggml_metal_fusion_check_topk_moe },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     true,  ggml_metal_fusion_check_moe_reduce },
-    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     false, ggml_metal_fusion_check_ssm_conv_silu },
-    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     false, ggml_metal_fusion_check_mul_mat_add },
+    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         {}, false, ggml_metal_fusion_check_norm },
+    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     {}, false, ggml_metal_fusion_check_norm },
+    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       {}, false, ggml_metal_fusion_check_norm },
+    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     {}, false, ggml_metal_fusion_check_norm },
+    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, {}, false, ggml_metal_fusion_check_norm },
+    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   {}, false, ggml_metal_fusion_check_norm },
+
+    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_2, {}, false, ggml_metal_fusion_check_add_chain },
+    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_3, {}, false, ggml_metal_fusion_check_add_chain },
+    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_4, {}, false, ggml_metal_fusion_check_add_chain },
+    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_5, {}, false, ggml_metal_fusion_check_add_chain },
+    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_6, {}, false, ggml_metal_fusion_check_add_chain },
+    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_7, {}, false, ggml_metal_fusion_check_add_chain },
+
+    { GGML_METAL_FUSION_SNAKE, ops_snake, {}, false, ggml_metal_fusion_check_snake },
+
+    { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },
+
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe,                     {1}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_scale,               {1}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm,                {1}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale,          {1}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_bias,           {1}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale_bias,     {1}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp,                  {3}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_scale,            {3}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm,             {3}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale,       {3}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_bias,        {3}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale_bias,  {3}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig,                 {2}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_scale,           {2}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm,            {2}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale,      {2}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_bias,       {2}, true, ggml_metal_fusion_check_topk_moe },
+    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale_bias, {2}, true, ggml_metal_fusion_check_topk_moe },
+
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, {}, true, ggml_metal_fusion_check_moe_reduce },
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, {}, true, ggml_metal_fusion_check_moe_reduce },
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, {}, true, ggml_metal_fusion_check_moe_reduce },
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, {}, true, ggml_metal_fusion_check_moe_reduce },
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, {}, true, ggml_metal_fusion_check_moe_reduce },
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, {}, true, ggml_metal_fusion_check_moe_reduce },
+    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, {}, true, ggml_metal_fusion_check_moe_reduce },
+
+    { GGML_METAL_FUSION_SSM_CONV_SILU, ops_ssm_conv_silu, {}, false, ggml_metal_fusion_check_ssm_conv_silu },
+
+    { GGML_METAL_FUSION_MUL_MAT_ADD, ops_mul_mat_add, {}, false, ggml_metal_fusion_check_mul_mat_add },
 };
 
 // ---- alloc deps -----------------------------------------------------------
diff --git src/ggml-metal/ggml-metal-impl.h src/ggml-metal/ggml-metal-impl.h
index 11c15fef..5fc49f9b 100644
--- src/ggml-metal/ggml-metal-impl.h
+++ src/ggml-metal/ggml-metal-impl.h
@@ -1270,6 +1270,7 @@ typedef struct {
     uint64_t nb01;      // logits row stride
     uint64_t nb1_ids;   // ids row stride
     float    clamp;
+    float    bias;
     float    scale;
 } ggml_metal_kargs_topk_moe;
 
diff --git src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal-ops.cpp
index cace34bd..a42f6608 100644
--- src/ggml-metal/ggml-metal-ops.cpp
+++ src/ggml-metal/ggml-metal-ops.cpp
@@ -821,6 +821,16 @@ int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
 int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
     ggml_tensor * op = ctx->node(idx);
 
+    // the sigmoid and sqrt(softplus) TOPK_MOE fusions anchor at the gating unary node
+    if (op->op == GGML_OP_UNARY && ctx->use_fusion() &&
+        (ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID)) {
+        int n = 1;
+        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
+        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
+            return ggml_metal_op_topk_moe(ctx, idx);
+        }
+    }
+
     ggml_metal_library_t lib = ctx->lib;
     ggml_metal_encoder_t enc = ctx->enc;
 
@@ -6023,29 +6033,48 @@ int ggml_metal_op_topk_moe(ggml_metal_op_t ctx, int idx) {
         return 1;
     }
 
-    ggml_tensor * softmax  = ctx->node(idx);
-    ggml_tensor * logits   = softmax->src[0];
-    ggml_tensor * get_rows = ctx->node(idx + 2);
-    ggml_tensor * ids      = get_rows->src[1];
-    ggml_tensor * weights  = ctx->node(idx + n_fuse - 1);
+    ggml_tensor * first  = ctx->node(idx);
+    ggml_tensor * logits = first->src[0];
+
+    // the unary chains score with sigmoid or sqrt(softplus) and rank biased scores
+    const bool with_unary = first->op == GGML_OP_UNARY;
+    const int  gating     = !with_unary ? 0 : ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS ? 1 : 2;
+
+    ggml_tensor * get_rows  = nullptr;
+    ggml_tensor * sum_rows  = nullptr;
+    ggml_tensor * prob_bias = nullptr;
+    int sum_rows_pos = -1;
+    for (int j = 0; j < n_fuse; ++j) {
+        ggml_tensor * node = ctx->node(idx + j);
+        if (node->op == GGML_OP_GET_ROWS) { get_rows = node; }
+        if (node->op == GGML_OP_ADD)      { prob_bias = node->src[1]; }
+        if (node->op == GGML_OP_SUM_ROWS) { sum_rows = node; sum_rows_pos = j; }
+    }
+
+    ggml_tensor * ids     = get_rows->src[1];
+    ggml_tensor * weights = ctx->node(idx + n_fuse - 1);
+
+    const bool with_norm  = sum_rows != nullptr;
+    const bool with_scale = weights->op == GGML_OP_SCALE;
 
     const int64_t n_expert      = logits->ne[0];
     const int64_t n_tokens      = logits->ne[1];
     const int64_t n_expert_used = ids->ne[0];
 
-    const bool with_norm  = n_fuse >= 6;
-    const bool with_scale = n_fuse == 4 || n_fuse == 7;
-
     float clamp = -INFINITY;
+    float bias  = 0.0f;
     if (with_norm) {
-        ggml_tensor * clamp_node = ctx->node(idx + 4);
-        clamp = ggml_get_op_params_f32(clamp_node, 0);
+        ggml_tensor * denom_node = ctx->node(idx + sum_rows_pos + 1);
+        if (denom_node->op == GGML_OP_SCALE) {
+            bias = ggml_get_op_params_f32(denom_node, 1);
+        } else {
+            clamp = ggml_get_op_params_f32(denom_node, 0);
+        }
     }
 
     float scale = 1.0f;
     if (with_scale) {
-        ggml_tensor * scale_node = ctx->node(idx + n_fuse - 1);
-        scale = ggml_get_op_params_f32(scale_node, 0);
+        scale = ggml_get_op_params_f32(weights, 0);
     }
 
     ggml_metal_kargs_topk_moe args = {
@@ -6053,23 +6082,27 @@ int ggml_metal_op_topk_moe(ggml_metal_op_t ctx, int idx) {
         /*.nb01      =*/ logits->nb[1],
         /*.nb1_ids   =*/ ids->nb[1],
         /*.clamp =*/ clamp,
+        /*.bias  =*/ bias,
         /*.scale =*/ scale,
     };
 
-    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_used, with_norm);
+    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_used,
+            with_norm, gating, prob_bias != nullptr);
 
     ggml_metal_encoder_set_pipeline(enc, pipeline);
     ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
     ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(logits),  1);
     ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
     ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),     3);
+    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(prob_bias ? prob_bias : logits), 4);
 
     ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) n_tokens, 1, 1, 32, 1, 1);
 
     ctx->count_fusions(fusion);
 
     if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
-        GGML_LOG_DEBUG("%s: fuse: SOFT_MAX + ARGSORT + GET_ROWS\n", __func__);
+        GGML_LOG_DEBUG("%s: fuse: %s + ARGSORT + GET_ROWS\n", __func__,
+                gating == 1 ? "SOFTPLUS + SQRT" : gating == 2 ? "SIGMOID" : "SOFT_MAX");
     }
 
     return n_fuse;
diff --git src/ggml-metal/kernels/argsort.metal src/ggml-metal/kernels/argsort.metal
index 5231b839..6328528a 100644
--- src/ggml-metal/kernels/argsort.metal
+++ src/ggml-metal/kernels/argsort.metal
@@ -3,6 +3,8 @@
 constant bool FC_topk_moe_with_norm [[function_constant(FC_TOPK_MOE + 0)]];
 constant int  FC_topk_moe_n_expert  [[function_constant(FC_TOPK_MOE + 1)]];
 constant int  FC_topk_moe_top_k     [[function_constant(FC_TOPK_MOE + 2)]];
+constant int  FC_topk_moe_gating    [[function_constant(FC_TOPK_MOE + 3)]];
+constant bool FC_topk_moe_pbias     [[function_constant(FC_TOPK_MOE + 4)]];
 
 constant int  FC_moe_reduce_n_expert_used [[function_constant(FC_MOE_REDUCE + 0)]];
 
@@ -342,13 +344,14 @@ kernel void kernel_top_k_f32_i32(
     }
 }
 
-// fused SOFT_MAX + top-k + GET_ROWS (+ optional norm/scale) for MoE routing.
+// fused SOFT_MAX or sqrt(softplus) + top-k + GET_ROWS (+ optional norm/scale) for MoE routing.
 // One SIMDgroup handles one token row; n_expert is limited to 1024 by the host.
 kernel void kernel_topk_moe_f32(
         constant   ggml_metal_kargs_topk_moe & args,
         device const char * src0,
         device       float * weights,
         device      int32_t * ids,
+        device const float * probs_bias,
         uint3   tgpig[[threadgroup_position_in_grid]],
         ushort  tiisg[[thread_index_in_simdgroup]]) {
     const int row = (int) tgpig.x;
@@ -377,50 +380,78 @@ kernel void kernel_topk_moe_f32(
         wt[i / 32] = isnan(v) ? -FLT_MAX : v;
     }
 
-    // softmax over the expert logits
-    float max_val = -INFINITY;
-    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
-        max_val = max(max_val, wt[i]);
-    }
-    max_val = simd_max(max_val);
+    if (FC_topk_moe_gating == 1) {
+        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
+            const float v = wt[i];
+            wt[i] = sqrt(v > 20.0f ? v : log(1.0f + exp(v)));
+        }
+    } else if (FC_topk_moe_gating == 2) {
+        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
+            wt[i] = 1.0f / (1.0f + exp(-wt[i]));
+        }
+    } else {
+        // softmax over the expert logits
+        float max_val = -INFINITY;
+        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
+            max_val = max(max_val, wt[i]);
+        }
+        max_val = simd_max(max_val);
 
-    float sum_val = 0.0f;
-    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
-        wt[i] = exp(wt[i] - max_val);
-        sum_val += wt[i];
+        float sum_val = 0.0f;
+        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
+            wt[i] = exp(wt[i] - max_val);
+            sum_val += wt[i];
+        }
+        sum_val = simd_sum(sum_val);
+
+        const float inv_sum = 1.0f / sum_val;
+        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
+            wt[i] *= inv_sum;
+        }
     }
-    sum_val = simd_sum(sum_val);
 
-    const float inv_sum = 1.0f / sum_val;
-    FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
-        wt[i] *= inv_sum;
+    // the ranking keys add the selection bias, the output weights stay unbiased
+    float sel[32];
+    if (FC_topk_moe_pbias) {
+        FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
+            const int expert = lane + i * 32;
+            sel[i] = expert < n_expert ? wt[i] + probs_bias[expert] : -INFINITY;
+        }
     }
 
     float wt_sum = 0.0f;
 
     for (int k = 0; k < top_k; ++k) {
-        float best_val = -INFINITY;
+        float best_val  = -INFINITY;
+        float best_key  = -INFINITY;
         int   best_expert = -1;
 
         FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
-            const int expert = lane + i * 32;
-            if (expert < n_expert && (wt[i] > best_val || (wt[i] == best_val && expert < best_expert))) {
+            const int   expert = lane + i * 32;
+            const float key    = FC_topk_moe_pbias ? sel[i] : wt[i];
+            if (expert < n_expert && (key > best_key || (key == best_key && expert < best_expert))) {
                 best_val    = wt[i];
+                best_key    = key;
                 best_expert = expert;
             }
         }
 
         FOR_UNROLL (int mask = 16; mask > 0; mask >>= 1) {
             const float val    = simd_shuffle_xor(best_val, mask);
+            const float key    = simd_shuffle_xor(best_key, mask);
             const int   expert = simd_shuffle_xor(best_expert, mask);
-            if (val > best_val || (val == best_val && expert < best_expert)) {
+            if (key > best_key || (key == best_key && expert < best_expert)) {
                 best_val    = val;
+                best_key    = key;
                 best_expert = expert;
             }
         }
 
         if ((best_expert & 31) == lane) {
             wt[best_expert / 32] = -INFINITY;
+            if (FC_topk_moe_pbias) {
+                sel[best_expert / 32] = -INFINITY;
+            }
         }
 
         if ((k & 31) == lane) {
@@ -437,7 +468,7 @@ kernel void kernel_topk_moe_f32(
 
     if (FC_topk_moe_with_norm) {
         wt_sum = simd_sum(wt_sum);
-        wt_sum = max(wt_sum, args.clamp);
+        wt_sum = max(wt_sum, args.clamp) + args.bias;
         const float inv = 1.0f / wt_sum;
         FOR_UNROLL (int i = 0; i < n_per_lane; ++i) {
             output_weights[i] *= inv;
diff --git src/ggml-metal/kernels/mul_mm.metal src/ggml-metal/kernels/mul_mm.metal
index edcc061d..fab72009 100644
--- src/ggml-metal/kernels/mul_mm.metal
+++ src/ggml-metal/kernels/mul_mm.metal
@@ -895,7 +895,8 @@ typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4,
 template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
 #if defined(GGML_METAL_HAS_BF16)
-template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
+template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float,  float2x4>;
+template [[host_name("kernel_mul_mm_bf16_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, bfloat, bfloat2x4>;
 #endif
 template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
 template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
diff --git src/ggml-opencl/CMakeLists.txt src/ggml-opencl/CMakeLists.txt
index 20dd595c..58a6d438 100644
--- src/ggml-opencl/CMakeLists.txt
+++ src/ggml-opencl/CMakeLists.txt
@@ -78,6 +78,7 @@ set(GGML_OPENCL_KERNELS
     diag_mask_inf
     diag
     div
+    dsv4_hc
     gelu
     get_rows
     glu
diff --git src/ggml-opencl/ggml-opencl.cpp src/ggml-opencl/ggml-opencl.cpp
index 22cf6f67..e0ee5518 100644
--- src/ggml-opencl/ggml-opencl.cpp
+++ src/ggml-opencl/ggml-opencl.cpp
@@ -853,6 +853,7 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_gelu_quick, kernel_gelu_quick_4;
     cl_kernel kernel_relu;
     cl_kernel kernel_sigmoid_f32, kernel_sigmoid_f16;
+    cl_kernel kernel_sigmoid_f32_nc, kernel_sigmoid_f16_nc;
     cl_kernel kernel_tri;
     cl_kernel kernel_fill;
     cl_kernel kernel_clamp, kernel_clamp_nc;
@@ -1104,8 +1105,12 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_argmax_f32_i32;
     cl_kernel kernel_argsort_f32_i32;
     cl_kernel kernel_sum_rows_f32, kernel_sum_rows_f32_4;
+    cl_kernel kernel_dsv4_hc_pre = nullptr;
+    cl_kernel kernel_dsv4_hc_post[2] = {};
+    size_t local_dsv4_hc_pre = 0;
+    size_t local_dsv4_hc_post[2] = {};
     cl_kernel kernel_cumsum_blk, kernel_cumsum_add;
-    cl_kernel kernel_repeat_f32;
+    cl_kernel kernel_repeat_f32, kernel_repeat_f16, kernel_repeat_i32;
     cl_kernel kernel_pad;
     cl_kernel kernel_tanh_f32, kernel_tanh_f32_4, kernel_tanh_f32_nc;
     cl_kernel kernel_tanh_f16, kernel_tanh_f16_4, kernel_tanh_f16_nc;
@@ -1418,7 +1423,7 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_gemv_noshuffle_q6_K_f32_o4;
     cl_kernel kernel_gemv_noshuffle_q6_K_f32_o4_global;  // weights via __global (opt-in)
     cl_kernel kernel_gemv_noshuffle_q6_K_f32_tiled;      // tiled-wide layout (opt-in)
-    cl_kernel kernel_gemv_noshuffle_q6_K_f32_tiled_mc3;  // tiled multi-column (N=3) verify lm_head
+    cl_kernel kernel_gemv_noshuffle_q6_K_f32_tiled_mc3;  // tiled multi-column (N 2..4) verify lm_head
     cl_kernel kernel_gemm_noshuffle_q6_K_f32_tiled;      // batched (N>1) over the tiled layout
     cl_kernel kernel_convert_block_q6_k_tiled_ns;        // tiled-wide convert (opt-in)
     cl_kernel kernel_gemv_noshuffle_q6_K_f32_mc3;        // multi-column (N=3) verify GEMV
@@ -1463,6 +1468,14 @@ struct ggml_backend_opencl_context {
 #endif // GGML_OPENCL_USE_ADRENO_KERNELS
 
     ~ggml_backend_opencl_context() {
+        if (kernel_dsv4_hc_pre) {
+            CL_CHECK(clReleaseKernel(kernel_dsv4_hc_pre));
+        }
+        for (int i = 0; i < 2; ++i) {
+            if (kernel_dsv4_hc_post[i]) {
+                CL_CHECK(clReleaseKernel(kernel_dsv4_hc_post[i]));
+            }
+        }
         if (iq1_s_grid_image) {
             CL_CHECK(clReleaseMemObject(iq1_s_grid_image));
         }
@@ -4227,6 +4240,28 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         GGML_LOG_CONT(".");
     }
 
+    // dsv4_hc
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "dsv4_hc.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("dsv4_hc.cl");
+#endif
+        for (int i = 0; i < 2; ++i) {
+            cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts + (i == 0 ? " -DHC_VEC=1" : " -DHC_VEC=4"));
+            if (i == 0) {
+                CL_CHECK((backend_ctx->kernel_dsv4_hc_pre = clCreateKernel(prog, "kernel_dsv4_hc_pre", &err), err));
+                backend_ctx->local_dsv4_hc_pre = MIN((size_t) 64, backend_ctx->get_kernel_workgroup_size(backend_ctx->kernel_dsv4_hc_pre));
+            }
+            CL_CHECK((backend_ctx->kernel_dsv4_hc_post[i] = clCreateKernel(prog, "kernel_dsv4_hc_post", &err), err));
+            backend_ctx->local_dsv4_hc_post[i] = MIN((size_t) 64, backend_ctx->get_kernel_workgroup_size(backend_ctx->kernel_dsv4_hc_post[i]));
+            CL_CHECK(clReleaseProgram(prog));
+        }
+        GGML_LOG_CONT(".");
+    }
+
     // sum_rows
     {
 #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -4276,6 +4311,8 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
 
         CL_CHECK((backend_ctx->kernel_sigmoid_f32 = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f32", &err), err));
         CL_CHECK((backend_ctx->kernel_sigmoid_f16 = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f16", &err), err));
+        CL_CHECK((backend_ctx->kernel_sigmoid_f32_nc = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f32_nc", &err), err));
+        CL_CHECK((backend_ctx->kernel_sigmoid_f16_nc = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f16_nc", &err), err));
         GGML_LOG_CONT(".");
     }
 
@@ -4308,6 +4345,8 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         cl_program prog =
             build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
         CL_CHECK((backend_ctx->kernel_repeat_f32 = clCreateKernel(prog, "kernel_repeat_f32", &err), err));
+        CL_CHECK((backend_ctx->kernel_repeat_f16 = clCreateKernel(prog, "kernel_repeat_f16", &err), err));
+        CL_CHECK((backend_ctx->kernel_repeat_i32 = clCreateKernel(prog, "kernel_repeat_i32", &err), err));
         CL_CHECK(clReleaseProgram(prog));
         GGML_LOG_CONT(".");
     }
@@ -6411,16 +6450,13 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::A6X) {
             CL_gemv_compile_opts += " -DGGML_CL_A6X_CONSTFOLD_FIX";
         }
-        // Opt-in: dequant-once-per-block mc3 verify GEMV (factors q4_K dequant
-        // out of the 3-column loop; byte-identical, lower spill). A/B vs the
-        // shipped inline mc3 in the same binary.
-        if (getenv("GGML_OPENCL_Q4K_MC3_DQ")) {
+        // dequant-once-per-block mc3 verify GEMV
+        if (getenv("GGML_OPENCL_Q4_K_MC3_DQ")) {
             CL_gemv_compile_opts += " -DQ4K_MC3_DEQUANT_ONCE ";
         }
-        // Opt-in: LDS-staged dequant mc3 verify GEMV (stages the dequantized
-        // q4_K weights in __local instead of private regs that spill to slow
-        // global on Adreno; byte-identical). A/B vs inline + dequant-once.
-        if (getenv("GGML_OPENCL_Q4K_MC3_LDS")) {
+        // LM-staged dequant mc3 verify GEMV, stages the dequantized
+        // q4_K weights in __local instead of regs
+        if (getenv("GGML_OPENCL_Q4_K_MC3_LDS")) {
             CL_gemv_compile_opts += " -DQ4K_MC3_DEQUANT_LDS ";
         }
 
@@ -7342,17 +7378,6 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
 
         CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q6_K_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32", &err), err));
         CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q6_K_f32_mc3 = clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32_mc3", &err), err));
-        if (getenv("GGML_OPENCL_MC3_PROBE")) {
-            cl_ulong pm6 = 0, pm4 = 0; size_t wg6 = 0, wg4 = 0, mult = 0;
-            clGetKernelWorkGroupInfo(backend_ctx->kernel_gemv_noshuffle_q6_K_f32_mc3, backend_ctx->device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pm6), &pm6, NULL);
-            clGetKernelWorkGroupInfo(backend_ctx->kernel_gemv_noshuffle_q6_K_f32_mc3, backend_ctx->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg6), &wg6, NULL);
-            clGetKernelWorkGroupInfo(backend_ctx->kernel_gemv_noshuffle_q4_k_f32_mc3, backend_ctx->device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pm4), &pm4, NULL);
-            clGetKernelWorkGroupInfo(backend_ctx->kernel_gemv_noshuffle_q4_k_f32_mc3, backend_ctx->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg4), &wg4, NULL);
-            clGetKernelWorkGroupInfo(backend_ctx->kernel_gemv_noshuffle_q6_K_f32_mc3, backend_ctx->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(mult), &mult, NULL);
-            fprintf(stderr, "[MC3-PROBE] q4K_mc3 private=%llu wg_cap=%zu | q6K_mc3 private=%llu wg_cap=%zu | pref_mult=%zu\n",
-                          (unsigned long long)pm4, wg4, (unsigned long long)pm6, wg6, mult);
-            fflush(stderr);
-        }
         GGML_LOG_CONT(".");
     }
 
@@ -7835,7 +7860,8 @@ static void ggml_opencl_ensure_fa_f32_f16_mq_decode(ggml_backend_opencl_context
     }
 
     const bool cluster = dk == 64 || dk == 128;
-    const int head_sub = cluster ? 2 : (gqa == 8 ? (dk == 512 ? 4 : 2) : 1);
+    const int head_sub = gqa == 16 ? (cluster ? 4 : 8) :
+                         cluster   ? 2 : (gqa == 8 ? (dk == 512 ? 4 : 2) : 1);
     const int nsg_max = dk == 64 ? 1 : (dk == 128 || (gqa == 8 && dk == 256) ? 2 : 4);
     const char * kernel_name = cluster ? "flash_attn_f32_f16_q1_vec_mq_split_c8" : "flash_attn_f32_f16_q1_vec_mq_split";
     const std::string src = ggml_opencl_fa_kernel_src(FA_VARIANT_F32_F16);
@@ -7843,7 +7869,7 @@ static void ggml_opencl_ensure_fa_f32_f16_mq_decode(ggml_backend_opencl_context
                              " -D FA_MQ_ONLY -D MQ_GQA=" + std::to_string(gqa / head_sub) +
                              " -D FA_HEAD_SUB=" + std::to_string(head_sub) +
                              (cluster ? " -D MQ_NSG=" + std::to_string(nsg_max) + " -D FA_CL_C=16" : " -D FA_MQ_SPLIT_ONLY") +
-                             (dk == 64 ? " -D FA_CL_MHRED -D FA_CL_MASK_BCAST" : "") +
+                             (dk == 64 ? " -D FA_CL_MHRED -D FA_CL_MASK_BCAST" : (cluster && gqa == 16 ? " -D FA_CL_MHRED" : "")) +
                              (gqa == 8 && dk == 256 ? " -D FA_Q1_Q_REG" : "");
     for (int nsg = nsg_max; nsg >= 1; nsg /= 2) {
         const size_t wg = 64 * nsg;
@@ -10839,7 +10865,8 @@ static void ggml_cl_mul_mat_q4_k_glu_fused(ggml_backend_t backend, ggml_tensor *
     // byte-identical to the per-op path. Keep the maxwg query as a further floor
     // for any driver that reports < 512.
     size_t maxwg = backend_ctx->get_kernel_workgroup_size(kernel);
-    size_t nsg_y = 8;
+    // X2E: 4 is faster than 8 on every q4_K FFN shape measured there
+    size_t nsg_y = backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E ? 4 : 8;
     while (nsg_y > 1 && 64 * nsg_y > maxwg) { nsg_y >>= 1; }
     size_t local_work_size[3]  = { 64, nsg_y, 1 };
     size_t global_work_size[3] = { (size_t)CEIL_DIV(M / 2, 64) * 64, nsg_y, 1 };
@@ -11723,7 +11750,8 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
                 case GGML_UNARY_OP_GELU_QUICK:
                     return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
                 case GGML_UNARY_OP_SIGMOID:
-                    return ggml_is_contiguous(op->src[0]);
+                    return op->src[0]->type == op->type &&
+                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16);
                 case GGML_UNARY_OP_TANH:
                 case GGML_UNARY_OP_NEG:
                 case GGML_UNARY_OP_EXP:
@@ -11774,7 +11802,16 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
         case GGML_OP_L2_NORM:
             return ggml_is_contiguous_rows(op->src[0]);
         case GGML_OP_REPEAT:
-            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32; // Assuming F32 for now, can be expanded
+            switch (op->type) {
+                case GGML_TYPE_F32:
+                case GGML_TYPE_F16:
+                case GGML_TYPE_I32:
+                case GGML_TYPE_I16:
+                case GGML_TYPE_BF16:
+                    return op->src[0]->type == op->type;
+                default:
+                    return false;
+            }
         case GGML_OP_PAD:
             // TODO: add circular padding support for opencl, see https://github.com/ggml-org/llama.cpp/pull/16985
             if (ggml_get_op_params_i32(op, 8) != 0) {
@@ -11985,6 +12022,9 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
             return op->ne[3] == 1;
         case GGML_OP_ROPE: {
             const int mode = ((const int32_t *) op->op_params)[2];
+            if (!ggml_rope_get_truncate(op)) {
+                return false;
+            }
             const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
             const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
             if (is_mrope && !is_vision) {
@@ -12046,6 +12086,14 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
         case GGML_OP_SUM_ROWS:
         case GGML_OP_CUMSUM:
             return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
+        case GGML_OP_DSV4_HC_PRE:
+        case GGML_OP_DSV4_HC_POST:
+            for (int i = 0; i < (op->op == GGML_OP_DSV4_HC_PRE ? 2 : 4); ++i) {
+                if (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
+                    return false;
+                }
+            }
+            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
         case GGML_OP_MEAN:
             return op->src[0]->type == GGML_TYPE_F32;
         case GGML_OP_FLASH_ATTN_EXT: {
@@ -12148,9 +12196,13 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
                 }
                 if (q->ne[1] == 1) {
                     const char * decode_env = getenv("GGML_OPENCL_FA_DK512_DECODE");
+                    const int64_t gqa = k->ne[2] > 0 ? q->ne[2] / k->ne[2] : 0;
+                    // GQA 2 and 16 decode on the X2E MQ kernels (see mq_decode_x2e in ggml_cl_flash_attn)
+                    const bool gqa_ok = gqa == 4 || gqa == 8 ||
+                                        (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E && (gqa == 2 || gqa == 16));
                     if ((decode_env && decode_env[0] == '0') ||
                         backend_ctx->gpu_family != ADRENO || k->ne[2] <= 0 ||
-                        (q->ne[2] / k->ne[2] != 4 && q->ne[2] / k->ne[2] != 8) || q->ne[2] % k->ne[2] != 0 ||
+                        !gqa_ok || q->ne[2] % k->ne[2] != 0 ||
                         !ggml_opencl_ensure_fa_f32_f16_vec_512(backend_ctx)) {
                         return false;
                     }
@@ -18268,7 +18320,7 @@ static void ggml_cl_add(ggml_backend_t backend, const ggml_tensor * src0, const
 
     cl_kernel kernel;
 
-    const bool bcast_row = ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0;
+    const bool bcast_row = ggml_is_contiguous(src0) && ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0;
 
     if (bcast_row) {
         GGML_ASSERT(ggml_is_contiguous(src0));
@@ -19416,6 +19468,33 @@ static void ggml_cl_sigmoid(ggml_backend_t backend, const ggml_tensor * src0, co
     cl_ulong offset0 = extra0->offset + src0->view_offs;
     cl_ulong offsetd = extrad->offset + dst->view_offs;
 
+    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
+        GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
+        GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
+        GGML_TENSOR_LOCALS(cl_ulong, nb,  dst,  nb);
+
+        cl_kernel kernel = src0->type == GGML_TYPE_F32 ? backend_ctx->kernel_sigmoid_f32_nc : backend_ctx->kernel_sigmoid_f16_nc;
+
+        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
+        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
+        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
+        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
+        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
+        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &nb00));
+        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb01));
+        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb02));
+        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb03));
+        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb0));
+        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb1));
+        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb2));
+        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb3));
+
+        size_t global_work_size[] = {(size_t)ne01 * 64, (size_t)ne02, (size_t)ne03};
+        size_t local_work_size[] = {64, 1, 1};
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+        return;
+    }
+
     cl_kernel kernel;
     if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
         kernel = backend_ctx->kernel_sigmoid_f32;
@@ -20751,7 +20830,22 @@ static void ggml_cl_repeat(ggml_backend_t backend, const ggml_tensor * src0, con
     const cl_ulong nb2 = dst->nb[2];
     const cl_ulong nb3 = dst->nb[3];
 
-    cl_kernel kernel = backend_ctx->kernel_repeat_f32;
+    cl_kernel kernel;
+    switch (src0->type) {
+        case GGML_TYPE_F32:
+            kernel = backend_ctx->kernel_repeat_f32;
+            break;
+        case GGML_TYPE_F16:
+        case GGML_TYPE_I16:
+        case GGML_TYPE_BF16:
+            kernel = backend_ctx->kernel_repeat_f16;
+            break;
+        case GGML_TYPE_I32:
+            kernel = backend_ctx->kernel_repeat_i32;
+            break;
+        default:
+            GGML_ABORT("unsupported type");
+    }
 
     CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
     CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
@@ -22596,11 +22690,18 @@ static void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, co
     const std::pair<int, int> dk_dv = {d_head_q, d_head_v};
     const int mq_gqa = n_head_kv > 0 ? n_head / n_head_kv : 0;
     const std::tuple<int, int, int> mq_decode_key = {d_head_q, d_head_v, mq_gqa};
+    // X2E also takes GQA 2 and 6 at head size 256 (gemma-4 SWA layers, Qwen3.5-27B),
+    // GQA 16 at 512 and 128 (gemma-4-12b global layers, muse-glimmer) and GQA 2 at 512 (gemma-4-E4B MTP drafter)
+    const bool mq_decode_x2e = backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E &&
+                               ((d_head_q == 256 && (mq_gqa == 2 || mq_gqa == 6)) ||
+                                (d_head_q == 512 && (mq_gqa == 2 || mq_gqa == 16)) ||
+                                (d_head_q == 128 && mq_gqa == 16 && backend_ctx->has_subgroup_shuffle));
     const bool mq_decode_shape = backend_ctx->gpu_family == ADRENO && is_mixed && n_q == 1 &&
                                  d_head_q == d_head_v && n_head_kv > 0 && n_head % n_head_kv == 0 &&
                                  ((((d_head_q == 64 && mq_gqa == 8) || (d_head_q == 128 && mq_gqa == 4)) &&
                                    backend_ctx->has_subgroup_shuffle) ||
-                                  ((d_head_q == 256 || d_head_q == 512) && (mq_gqa == 4 || mq_gqa == 8)));
+                                  ((d_head_q == 256 || d_head_q == 512) && (mq_gqa == 4 || mq_gqa == 8)) ||
+                                  mq_decode_x2e);
     if (mq_decode_shape && n_kv >= 32) {
         ggml_opencl_ensure_fa_f32_f16_mq_decode(backend_ctx, d_head_q, d_head_v, mq_gqa);
     }
@@ -24626,14 +24727,8 @@ static void ggml_cl_mul_mat_q4_0_f32_adreno(ggml_backend_t backend, const ggml_t
     int N = ne1;
     int K = ne00;
 
-    // Multi-column (N=3) verify GEMV for q4_0: route the spec/MTP verify batch
-    // (ne1==3) onto the efficient GEMV path instead of the transposed-GEMM dead-
-    // zone (gemm_noshuffle_q4_0 is ~50% of MTP decode on a Q4_0 model since q4_0
-    // weights have no cok/mc3, unlike q4_K). Reuses the ne1==1 GEMV image setup
-    // (activation image already sized by N=ne1). Byte-identical. Opt-in via
-    // GGML_OPENCL_Q40_MC3=1. Per-layer only (ne01 < 32768); q4_0 lm_head doesn't
-    // occur (token_embd/output stay Q6_K), guard kept for parity with q4_K mc3.
-    static const bool q40_mc3 = (getenv("GGML_OPENCL_Q40_MC3") != nullptr);
+    // Multi-column (N=3) verify GEMV for q4_0
+    static const bool q40_mc3 = (getenv("GGML_OPENCL_Q4_0_MC3") != nullptr);
     const bool use_q40_mc3 = q40_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);
 
     const bool use_bin = use_q4_0_bin_kernels(backend_ctx, src0);
@@ -24642,7 +24737,7 @@ static void ggml_cl_mul_mat_q4_0_f32_adreno(ggml_backend_t backend, const ggml_t
         if (use_q40_mc3) {
             static bool warned = false;
             if (!warned) {
-                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q40_MC3 is bypassed by Q4_0 binary kernels\n");
+                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q4_0_MC3 is bypassed by Q4_0 binary kernels\n");
                 warned = true;
             }
         }
@@ -25020,11 +25115,8 @@ static void ggml_cl_mul_mat_q4_1_f32_adreno(ggml_backend_t backend, const ggml_t
     int N = ne1;
     int K = ne00;
 
-    // Multi-column (N=3) verify GEMV for q4_1: route the spec/MTP verify batch
-    // (ne1==3) onto the efficient GEMV path instead of the transposed-GEMM dead-
-    // zone (gemm_noshuffle_q4_1). Reuses the ne1==1 GEMV image setup. Opt-in via
-    // GGML_OPENCL_Q41_MC3=1. Per-layer only (ne01 < 32768).
-    static const bool q41_mc3 = (getenv("GGML_OPENCL_Q41_MC3") != nullptr);
+    // Multi-column (N=3) verify GEMV for q4_1
+    static const bool q41_mc3 = (getenv("GGML_OPENCL_Q4_1_MC3") != nullptr);
     const bool use_q41_mc3 = q41_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);
 
     if (ne1 == 1 || use_q41_mc3) {
@@ -26841,15 +26933,8 @@ static void ggml_cl_mul_mat_q4_k_f32_adreno(ggml_backend_t backend, const ggml_t
     cl_uchar mask_d4 = 0x0F;
     cl_uchar mask_hi2 = 0xC0;
 
-    // Multi-column verify GEMV: route the spec/MTP verify batch (ne1==3 = 2
-    // drafts + 1 bonus) onto the efficient GEMV path (subgroup-broadcast, no
-    // transpose) instead of the transposed-GEMM dead-zone. Reuses the ne1==1
-    // GEMV setup (the activation image is already sized by N=ne1). Byte-
-    // identical. Opt-in via GGML_OPENCL_Q4K_MC3=1 while validating.
-    static const bool q4k_mc3 = (getenv("GGML_OPENCL_Q4K_MC3") != nullptr);
-    // Per-layer only (ne01 < 32768): the batched large-vocab lm_head at ne1==3
-    // is left to the existing routing (corrupts on the Adreno GEMV path; x2-
-    // unified routes batched Q6_K lm_head to CPU). Per-layer mc3 is byte-identical.
+    // Multi-column verify GEMV
+    static const bool q4k_mc3 = (getenv("GGML_OPENCL_Q4_K_MC3") != nullptr);
     const bool use_mc3 = q4k_mc3 && (ne1 == 3) && (ne01 < 32768);
 
     const bool use_bin = use_q4_k_bin_kernels(backend_ctx, src0);
@@ -26858,7 +26943,7 @@ static void ggml_cl_mul_mat_q4_k_f32_adreno(ggml_backend_t backend, const ggml_t
         if (use_mc3) {
             static bool warned = false;
             if (!warned) {
-                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q4K_MC3 is bypassed by Q4_K binary kernels\n");
+                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q4_K_MC3 is bypassed by Q4_K binary kernels\n");
                 warned = true;
             }
         }
@@ -27683,20 +27768,15 @@ static void ggml_cl_mul_mat_q6_K_f32_adreno(ggml_backend_t backend, const ggml_t
     cl_image_format  img_fmt;
     cl_image_desc    img_desc;
 
-    // subbuffer and image for activation
-    // Multi-column verify GEMV: route the spec/MTP verify q6_K matmuls (ne1==3)
-    // onto the efficient GEMV path instead of the transposed-GEMM dead-zone.
-    // Reuses the ne1==1 image setup (activation image sized by N=ne1). Byte-
-    // identical. Opt-in via GGML_OPENCL_Q6K_MC3=1 while validating.
-    static const bool q6k_mc3 = (getenv("GGML_OPENCL_Q6K_MC3") != nullptr);
-    // Per-layer only (ne01 < 32768): batched large-vocab lm_head stays on the
-    // existing path (x2-unified routes batched Q6_K lm_head to CPU; the Adreno
-    // GEMV corrupts it). Per-layer mc3 is byte-identical.
+    // Multi-column verify GEMV
+    static const bool q6k_mc3 = (getenv("GGML_OPENCL_Q6_K_MC3") != nullptr);
     const bool use_q6k_mc3 = q6k_mc3 && (ne1 == 3) && (ne01 < 32768);
-    // Batched verify lm_head/embed (ne1==3, tiled layout): multi-column tiled
-    // GEMV — streams the large lm_head weight once across the 3 verify columns
-    // (the #1 MTP bottleneck; mc3 above can't, it reads the noshuffle layout).
-    const bool use_q6k_tiled_mc = q6k_mc3 && (ne1 == 3) && (ne01 >= 32768) && use_q6k_tiled(backend_ctx, src0);
+    // Multi-column tiled vocab GEMV for verify
+    // streams the large lm_head weight once across the verify columns
+    static const char * q6k_tiled_mc_env = getenv("GGML_OPENCL_Q6_K_TILED_MC");
+    const bool q6k_tiled_mc = q6k_tiled_mc_env ? atoi(q6k_tiled_mc_env) != 0
+                                               : (q6k_mc3 || backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
+    const bool use_q6k_tiled_mc = q6k_tiled_mc && (ne1 >= 2 && ne1 <= 4) && (ne01 >= 32768) && use_q6k_tiled(backend_ctx, src0);
 
     const bool use_bin = use_q6_k_bin_kernels(backend_ctx, src0);
 
@@ -27704,7 +27784,7 @@ static void ggml_cl_mul_mat_q6_K_f32_adreno(ggml_backend_t backend, const ggml_t
         if (use_q6k_mc3 || use_q6k_tiled_mc) {
             static bool warned = false;
             if (!warned) {
-                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q6K_MC3 is bypassed by Q6_K binary kernels\n");
+                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q6_K_MC3 is bypassed by Q6_K binary kernels\n");
                 warned = true;
             }
         }
@@ -27792,6 +27872,9 @@ static void ggml_cl_mul_mat_q6_K_f32_adreno(ggml_backend_t backend, const ggml_t
         CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
         CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
         CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));
+        if (use_q6k_tiled_mc) {
+            CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_int), &ne1));
+        }
 
         const size_t gws_x = use_tiled
                                  ? (size_t) CEIL_DIV(ne01, 64) * 64
@@ -28359,8 +28442,9 @@ static void ggml_cl_mul_mat_q5_K_f32_adreno(ggml_backend_t backend, const ggml_t
     // (ne1==3) onto the efficient GEMV path instead of the transposed-GEMM dead-
     // zone (gemm_noshuffle_q5_k, the #2 chunk of MTP decode on a Q4_0-mix model
     // after q4_0 mc3). Reuses the ne1==1 GEMV image setup (q + qh + activations).
-    // Opt-in via GGML_OPENCL_Q5K_MC3=1. Per-layer only (ne01 < 32768).
-    static const bool q5k_mc3 = (getenv("GGML_OPENCL_Q5K_MC3") != nullptr);
+    // Default on for X2E, GGML_OPENCL_Q5_K_MC3=0/1 forces it. Per-layer only (ne01 < 32768).
+    static const char * q5k_mc3_env = getenv("GGML_OPENCL_Q5_K_MC3");
+    const bool q5k_mc3 = q5k_mc3_env ? atoi(q5k_mc3_env) != 0 : backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E;
     const bool use_q5k_mc3 = q5k_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);
 
     const bool use_bin = use_q5_k_bin_kernels(backend_ctx, src0);
@@ -28369,7 +28453,7 @@ static void ggml_cl_mul_mat_q5_K_f32_adreno(ggml_backend_t backend, const ggml_t
         if (use_q5k_mc3) {
             static bool warned = false;
             if (!warned) {
-                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q5K_MC3 is bypassed by Q5_K binary kernels\n");
+                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q5_K_MC3 is bypassed by Q5_K binary kernels\n");
                 warned = true;
             }
         }
@@ -30926,8 +31010,9 @@ static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, co
                 // skinny f32 weight (GDN ssm_alpha/ssm_beta, M=32) it launches one
                 // under-occupied WG at ~2.3% tile utilization. Route to a per-output
                 // (m,n) GEMV (64-thread WG, K-split + __local reduce) instead.
-                // Opt-in GGML_OPENCL_F32_MC=1; 2D contiguous, small N + skinny M only.
-                static const bool f32_mc = (getenv("GGML_OPENCL_F32_MC") != nullptr);
+                // 2D contiguous, small N + skinny M only. Default on for X2E, GGML_OPENCL_F32_MC=0/1 forces it.
+                static const char * f32_mc_env = getenv("GGML_OPENCL_F32_MC");
+                const bool f32_mc = f32_mc_env ? atoi(f32_mc_env) != 0 : backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E;
                 if (f32_mc && ne11 >= 2 && ne11 <= 8 && ne01 <= 512 && (ne00 % 4 == 0) &&
                     ne02 == 1 && ne12 == 1 && ne13 == 1 &&
                     ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
@@ -37826,6 +37911,47 @@ static void ggml_cl_argsort(ggml_backend_t backend, const ggml_tensor * src0, co
 #endif // GGML_OPENCL_USE_ADRENO_KERNELS
 }
 
+static void ggml_cl_dsv4_hc(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
+    auto * backend_ctx = (ggml_backend_opencl_context *) backend->context;
+    const bool pre = dst->op == GGML_OP_DSV4_HC_PRE;
+    const bool vec = !pre && src0->ne[0] % 4 == 0 && src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float);
+    cl_kernel kernel = pre ? backend_ctx->kernel_dsv4_hc_pre : backend_ctx->kernel_dsv4_hc_post[vec];
+
+    cl_uint arg = 0;
+    for (int i = 0; i < (pre ? 2 : 4); ++i) {
+        const ggml_tensor * src = dst->src[i];
+        auto * extra = src ? (ggml_tensor_extra_cl *) src->extra : nullptr;
+        const cl_mem buffer = extra ? extra->data_device : nullptr;
+        const cl_ulong offset = src ? extra->offset + src->view_offs : 0;
+        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer));
+        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset));
+        for (int j = 0; j < 3; ++j) {
+            const cl_ulong stride = src ? src->nb[j] : 0;
+            CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &stride));
+        }
+    }
+
+    auto * extra = (ggml_tensor_extra_cl *) dst->extra;
+    const cl_ulong offset = extra->offset + dst->view_offs;
+    const cl_long n_embd = src0->ne[0];
+    const cl_long n_hc = pre ? src0->ne[1] : src1->ne[1];
+    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &extra->data_device));
+    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset));
+    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_long), &n_embd));
+    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_long), &n_hc));
+    if (pre) {
+        const float scale = ggml_get_op_params_f32(dst, 0);
+        const cl_int gated = ggml_get_op_params_i32(dst, 1);
+        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &scale));
+        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_int), &gated));
+    }
+
+    const size_t n_work = (size_t) n_embd / (vec ? 4 : 1);
+    size_t local_work_size[] = { pre ? backend_ctx->local_dsv4_hc_pre : backend_ctx->local_dsv4_hc_post[vec], 1, 1 };
+    size_t global_work_size[] = { CEIL_DIV(n_work, local_work_size[0]) * local_work_size[0], (size_t) dst->ne[1], (size_t) dst->ne[2] };
+    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+}
+
 static void ggml_cl_sum_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
     GGML_ASSERT(src0);
     GGML_ASSERT(src0->extra);
@@ -38713,6 +38839,10 @@ bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor
             }
             func = ggml_cl_sum_rows;
             break;
+        case GGML_OP_DSV4_HC_PRE:
+        case GGML_OP_DSV4_HC_POST:
+            func = ggml_cl_dsv4_hc;
+            break;
         case GGML_OP_CUMSUM:
             if (!any_on_device) {
                 return false;
diff --git src/ggml-opencl/kernels/dsv4_hc.cl src/ggml-opencl/kernels/dsv4_hc.cl
new file mode 100644
index 00000000..164ab1da
--- /dev/null
+++ src/ggml-opencl/kernels/dsv4_hc.cl
@@ -0,0 +1,101 @@
+#if HC_VEC == 4
+typedef float4 hc_float;
+#define HC_LOAD(p) vload4(0, (global const float *) (p))
+#define HC_STORE(v, p) vstore4(v, 0, (global float *) (p))
+#else
+typedef float hc_float;
+#define HC_LOAD(p) (*(global const float *) (p))
+#define HC_STORE(v, p) (*(global float *) (p) = (v))
+#endif
+
+#if HC_VEC == 1
+kernel void kernel_dsv4_hc_pre(
+        global const char * x,
+        ulong offset_x,
+        ulong nbx0,
+        ulong nbx1,
+        ulong nbx2,
+        global const char * weights,
+        ulong offset_w,
+        ulong nbw0,
+        ulong nbw1,
+        ulong nbw2,
+        global char * dst,
+        ulong offset_dst,
+        long n_embd,
+        long n_hc,
+        float scale,
+        int gated
+) {
+    const size_t i = get_global_id(0);
+    const size_t t = get_global_id(1);
+    if (i >= n_embd) {
+        return;
+    }
+
+    x += offset_x + i*nbx0 + t*nbx2;
+    weights += offset_w + (gated ? i*nbw0 + t*nbw2 : t*nbw1);
+
+    float sum = 0.0f;
+    for (long h = 0; h < n_hc; ++h) {
+        const float xv = *(global const float *) (x + h*nbx1);
+        float wv = *(global const float *) (weights + h*(gated ? nbw1 : nbw0));
+        if (gated) {
+            wv = 1.0f / (1.0f + exp(-wv));
+        }
+        sum += xv * wv;
+    }
+    ((global float *) (dst + offset_dst))[i + t*n_embd] = scale * sum;
+}
+#endif
+
+kernel void kernel_dsv4_hc_post(
+        global const char * x,
+        ulong offset_x,
+        ulong nbx0,
+        ulong nbx1,
+        ulong nbx2,
+        global const char * residual,
+        ulong offset_r,
+        ulong nbr0,
+        ulong nbr1,
+        ulong nbr2,
+        global const char * post,
+        ulong offset_p,
+        ulong nbp0,
+        ulong nbp1,
+        ulong nbp2,
+        global const char * comb,
+        ulong offset_c,
+        ulong nbc0,
+        ulong nbc1,
+        ulong nbc2,
+        global char * dst,
+        ulong offset_dst,
+        long n_embd,
+        long n_hc
+) {
+    const size_t i = get_global_id(0) * HC_VEC;
+    const size_t h = get_global_id(1);
+    const size_t t = get_global_id(2);
+    if (i >= n_embd) {
+        return;
+    }
+
+    const hc_float xv = HC_LOAD(x + offset_x + i*nbx0 + t*nbx1);
+    const float pv = *(global const float *) (post + offset_p + h*nbp0 + t*nbp1);
+    residual += offset_r + i*nbr0 + t*nbr2;
+
+    hc_float sum = xv * pv;
+    if (comb) {
+        comb += offset_c + h*nbc0 + t*nbc2;
+        for (long s = 0; s < n_hc; ++s) {
+            const hc_float rv = HC_LOAD(residual + s*nbr1);
+            const float cv = *(global const float *) (comb + s*nbc1);
+            sum += rv * cv;
+        }
+    } else {
+        sum += HC_LOAD(residual + h*nbr1);
+    }
+    HC_STORE(sum, dst + offset_dst + (i + (h + t*n_hc)*n_embd)*sizeof(float));
+}
diff --git src/ggml-opencl/kernels/gemv_noshuffle_q6_k_f32_tiled.cl src/ggml-opencl/kernels/gemv_noshuffle_q6_k_f32_tiled.cl
index c5049f39..a9ddc113 100644
--- src/ggml-opencl/kernels/gemv_noshuffle_q6_k_f32_tiled.cl
+++ src/ggml-opencl/kernels/gemv_noshuffle_q6_k_f32_tiled.cl
@@ -104,11 +104,11 @@ kernel void kernel_gemv_noshuffle_q6_K_f32_tiled(
     }
 }
 
-// Multi-column (N=3) variant of the tiled q6_K decode GEMV, for the speculative/
-// MTP VERIFY lm_head/embed (ne1=3 = 2 drafts + 1 bonus). Identical tiled weight
-// layout + unpack as the ne1=1 kernel above; each WI computes 3 output columns,
+// Multi-column (ne1 in [2..4]) variant of the tiled q6_K decode GEMV, for the
+// speculative/MTP VERIFY lm_head/embed (drafts + 1 bonus). Identical tiled weight
+// layout + unpack as the ne1=1 kernel above; each WI computes up to 4 output columns,
 // streaming the (large) lm_head weight ONCE per superblock and reusing it across
-// the 3 verify activation columns (dequant once per code, MAC into 3 accs). This
+// the verify activation columns (dequant once per code, MAC into 4 accs). This
 // is the lm_head analogue of the per-layer mc3 GEMV; the multiply order matches
 // the ne1=1 kernel, so each column is byte-identical to a standalone tiled GEMV.
 #if defined(ADRENO_GPU)
@@ -123,7 +123,8 @@ kernel void kernel_gemv_noshuffle_q6_K_f32_tiled_mc3(
     global float * dst,
     ulong offsetd,
     int ne00,
-    int ne01
+    int ne01,
+    int ne1
 ) {
     int grp = get_local_id(1);
     int row = get_global_id(0);
@@ -133,7 +134,12 @@ kernel void kernel_gemv_noshuffle_q6_K_f32_tiled_mc3(
     int nb = ne00 / 256;
     int col_stride = ne00 / 4;          // activation float4 pixels per column
 
-    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
+    // columns past ne1 re-read the last column and are not stored
+    int col1 = min(1, ne1 - 1) * col_stride;
+    int col2 = min(2, ne1 - 1) * col_stride;
+    int col3 = min(3, ne1 - 1) * col_stride;
+
+    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
 
     for (int sb = grp; sb < nb; sb += NSUBGROUPS) {
         int tile_blk = rt * nb + sb;
@@ -157,9 +163,10 @@ kernel void kernel_gemv_noshuffle_q6_K_f32_tiled_mc3(
         int act_base = sb * 64;
         #pragma unroll
         for (int e4 = 0; e4 < 64; ++e4) {
-            float4 a0 = read_imagef(src1, 0*col_stride + act_base + e4);
-            float4 a1 = read_imagef(src1, 1*col_stride + act_base + e4);
-            float4 a2 = read_imagef(src1, 2*col_stride + act_base + e4);
+            float4 a0 = read_imagef(src1, act_base + e4);
+            float4 a1 = read_imagef(src1, col1 + act_base + e4);
+            float4 a2 = read_imagef(src1, col2 + act_base + e4);
+            float4 a3 = read_imagef(src1, col3 + act_base + e4);
             #pragma unroll
             for (int t = 0; t < 4; ++t) {
                 int  e    = e4 * 4 + t;
@@ -171,15 +178,17 @@ kernel void kernel_gemv_noshuffle_q6_K_f32_tiled_mc3(
                 float av0 = (t == 0) ? a0.x : (t == 1) ? a0.y : (t == 2) ? a0.z : a0.w;
                 float av1 = (t == 0) ? a1.x : (t == 1) ? a1.y : (t == 2) ? a1.z : a1.w;
                 float av2 = (t == 0) ? a2.x : (t == 1) ? a2.y : (t == 2) ? a2.z : a2.w;
+                float av3 = (t == 0) ? a3.x : (t == 1) ? a3.y : (t == 2) ? a3.z : a3.w;
                 acc0 += w * av0;
                 acc1 += w * av1;
                 acc2 += w * av2;
+                acc3 += w * av3;
             }
         }
     }
 
     local float4 reduce_lm[NSUBGROUPS * TILE_ROWS];
-    reduce_lm[grp * TILE_ROWS + rit] = (float4)(acc0, acc1, acc2, 0.0f);
+    reduce_lm[grp * TILE_ROWS + rit] = (float4)(acc0, acc1, acc2, acc3);
     barrier(CLK_LOCAL_MEM_FENCE);
 
     if (grp == 0) {
@@ -188,9 +197,10 @@ kernel void kernel_gemv_noshuffle_q6_K_f32_tiled_mc3(
                      + reduce_lm[2 * TILE_ROWS + rit]
                      + reduce_lm[3 * TILE_ROWS + rit];
         dst = (global float*)((global char*)dst + offsetd);
-        // dst column-major [ne01 rows x 3 cols]: (row, col) at col*ne01 + row
+        // dst column-major [ne01 rows x ne1 cols]: (row, col) at col*ne01 + row
         dst[0*ne01 + row] = total.x;
         dst[1*ne01 + row] = total.y;
-        dst[2*ne01 + row] = total.z;
+        if (ne1 > 2) dst[2*ne01 + row] = total.z;
+        if (ne1 > 3) dst[3*ne01 + row] = total.w;
     }
 }
diff --git src/ggml-opencl/kernels/repeat.cl src/ggml-opencl/kernels/repeat.cl
index 53951a55..88dee31c 100644
--- src/ggml-opencl/kernels/repeat.cl
+++ src/ggml-opencl/kernels/repeat.cl
@@ -1,38 +1,43 @@
-kernel void kernel_repeat_f32(
-        global const char * src0,
-        ulong               offset0,
-        global       char * dst,
-        ulong               offsetd,
-        int     ne00,
-        int     ne01,
-        int     ne02,
-        int     ne03,
-        ulong   nb00,
-        ulong   nb01,
-        ulong   nb02,
-        ulong   nb03,
-        int     ne0,
-        ulong   nb0,
-        ulong   nb1,
-        ulong   nb2,
-        ulong   nb3
-) {
-    src0 = src0 + offset0;
-    dst  = dst  + offsetd;
-
-    const int i3 = get_group_id(2);
-    const int i2 = get_group_id(1);
-    const int i1 = get_group_id(0);
-
-    const int i03 = i3%ne03;
-    const int i02 = i2%ne02;
-    const int i01 = i1%ne01;
-
-    global const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
-    global       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1;
-
-    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
-        const int i00 = i0%ne00;
-        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i00*nb00));
-    }
+#define KERNEL_REPEAT(SUFFIX, T)                                                  \
+kernel void kernel_repeat_##SUFFIX(                                               \
+        global const char * src0,                                                 \
+        ulong               offset0,                                              \
+        global       char * dst,                                                  \
+        ulong               offsetd,                                              \
+        int     ne00,                                                             \
+        int     ne01,                                                             \
+        int     ne02,                                                             \
+        int     ne03,                                                             \
+        ulong   nb00,                                                             \
+        ulong   nb01,                                                             \
+        ulong   nb02,                                                             \
+        ulong   nb03,                                                             \
+        int     ne0,                                                              \
+        ulong   nb0,                                                              \
+        ulong   nb1,                                                              \
+        ulong   nb2,                                                              \
+        ulong   nb3                                                               \
+) {                                                                               \
+    src0 = src0 + offset0;                                                        \
+    dst  = dst  + offsetd;                                                        \
+                                                                                  \
+    const int i3 = get_group_id(2);                                               \
+    const int i2 = get_group_id(1);                                               \
+    const int i1 = get_group_id(0);                                               \
+                                                                                  \
+    const int i03 = i3%ne03;                                                      \
+    const int i02 = i2%ne02;                                                      \
+    const int i01 = i1%ne01;                                                      \
+                                                                                  \
+    global const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;         \
+    global       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1;          \
+                                                                                  \
+    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {           \
+        const int i00 = i0%ne00;                                                  \
+        *((global T *)(dst_ptr + i0*nb0)) = *((global T *)(src0_ptr + i00*nb00)); \
+    }                                                                             \
 }
+
+KERNEL_REPEAT(f32, float)
+KERNEL_REPEAT(f16, ushort)
+KERNEL_REPEAT(i32, int)
diff --git src/ggml-opencl/kernels/sigmoid.cl src/ggml-opencl/kernels/sigmoid.cl
index e3f669dd..cfe9f729 100644
--- src/ggml-opencl/kernels/sigmoid.cl
+++ src/ggml-opencl/kernels/sigmoid.cl
@@ -27,3 +27,63 @@ kernel void kernel_sigmoid_f16(
 
     dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
 }
+
+kernel void kernel_sigmoid_f32_nc(
+        global const char * src0,
+        ulong               offset0,
+        global       char * dst,
+        ulong               offsetd,
+        int   ne00,
+        ulong nb00,
+        ulong nb01,
+        ulong nb02,
+        ulong nb03,
+        ulong nb0,
+        ulong nb1,
+        ulong nb2,
+        ulong nb3
+) {
+    src0 = src0 + offset0;
+    dst  = dst + offsetd;
+
+    const int i3 = get_group_id(2);
+    const int i2 = get_group_id(1);
+    const int i1 = get_group_id(0);
+
+    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
+        global const float * x = (global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
+        global       float * y = (global       float *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
+
+        *y = 1.0f / (1.0f + exp(-*x));
+    }
+}
+
+kernel void kernel_sigmoid_f16_nc(
+        global const char * src0,
+        ulong               offset0,
+        global       char * dst,
+        ulong               offsetd,
+        int   ne00,
+        ulong nb00,
+        ulong nb01,
+        ulong nb02,
+        ulong nb03,
+        ulong nb0,
+        ulong nb1,
+        ulong nb2,
+        ulong nb3
+) {
+    src0 = src0 + offset0;
+    dst  = dst + offsetd;
+
+    const int i3 = get_group_id(2);
+    const int i2 = get_group_id(1);
+    const int i1 = get_group_id(0);
+
+    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
+        global const half * x = (global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
+        global       half * y = (global       half *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
+
+        *y = 1.0f / (1.0f + exp(-(float)*x));
+    }
+}
diff --git src/ggml-openvino/ggml-openvino.cpp src/ggml-openvino/ggml-openvino.cpp
index 88634557..a63f69bf 100644
--- src/ggml-openvino/ggml-openvino.cpp
+++ src/ggml-openvino/ggml-openvino.cpp
@@ -1636,6 +1636,9 @@ static ggml_openvino_op_support is_op_supported_case(const ggml_tensor * op) {
         break;
     }
     case GGML_OP_ROPE: {
+        if (!ggml_rope_get_truncate(op)) {
+            return {false, "ROPE without YaRN truncation is not supported"};
+        }
         if (op->view_src != nullptr && !ggml_is_contiguous(op->src[0])) {
             return {false, "ROPE on VIEW / non-contiguous input is not supported"};
         }
diff --git src/ggml-rpc/ggml-rpc.cpp src/ggml-rpc/ggml-rpc.cpp
index 966bd997..d346bb12 100644
--- src/ggml-rpc/ggml-rpc.cpp
+++ src/ggml-rpc/ggml-rpc.cpp
@@ -3055,7 +3055,10 @@ static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_b
 
 static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
     GGML_UNUSED(dev);
-    GGML_UNUSED(op);
+    if ((op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_BACK) &&
+        !ggml_rope_get_truncate(op)) {
+        return false;
+    }
     //TODO: call the remote backend and cache the results
     return true;
 }
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index 9ee16423..ac752f9b 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -7569,6 +7569,7 @@ static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, cons
         }
         case GGML_OP_ROPE:
         case GGML_OP_ROPE_BACK:
+            return ggml_rope_get_truncate(op);
         case GGML_OP_IM2COL:
         case GGML_OP_IM2COL_3D:
         case GGML_OP_UPSCALE:
diff --git src/ggml-sycl/rope.cpp src/ggml-sycl/rope.cpp
index b6d22559..f2ca6034 100644
--- src/ggml-sycl/rope.cpp
+++ src/ggml-sycl/rope.cpp
@@ -547,7 +547,7 @@ void ggml_sycl_op_rope_impl(ggml_backend_sycl_context &ctx, ggml_tensor *dst,
 
     rope_corr_dims corr_dims;
     ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
-                             beta_slow, corr_dims.v);
+                             beta_slow, true, corr_dims.v);
 
     // compute
     if (is_neox) {
diff --git src/ggml-virtgpu/ggml-backend-device.cpp src/ggml-virtgpu/ggml-backend-device.cpp
index 987ce9dd..86df4cc5 100644
--- src/ggml-virtgpu/ggml-backend-device.cpp
+++ src/ggml-virtgpu/ggml-backend-device.cpp
@@ -30,11 +30,14 @@ static void ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size
 }
 
 static bool ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
+    if ((op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_BACK) &&
+        !ggml_rope_get_truncate(op)) {
+        return false;
+    }
 #if USE_ALWAYS_TRUE_SUPPORTS_OP == 1
     /* ggml-rpc cheats it like this */
     /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
     UNUSED(dev);
-    UNUSED(op);
 
     return true;
 #else
diff --git src/ggml-vulkan/ggml-vulkan-buffers.cpp src/ggml-vulkan/ggml-vulkan-buffers.cpp
index 4e60df41..3e04eef6 100644
--- src/ggml-vulkan/ggml-vulkan-buffers.cpp
+++ src/ggml-vulkan/ggml-vulkan-buffers.cpp
@@ -229,7 +229,7 @@ vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
             buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                        vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
-        } else if (device->disable_host_visible_vidmem) {
+        } else if (!device->rebar || device->disable_host_visible_vidmem) {
             if (device->allow_sysmem_fallback) {
                 buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
diff --git src/ggml-vulkan/ggml-vulkan-comm.cpp.orig src/ggml-vulkan/ggml-vulkan-comm.cpp.orig
new file mode 100644
index 00000000..a2370b8c
--- /dev/null
+++ src/ggml-vulkan/ggml-vulkan-comm.cpp.orig
@@ -0,0 +1,591 @@
+#include "ggml-vulkan-common.h"
+
+#if !defined(_WIN32)
+#include <fcntl.h>
+#include <unistd.h>
+#if defined(__linux__)
+#include <pthread.h>
+#endif
+
+static constexpr size_t ggml_vk_comm_f16_min_size = 1024 * 1024;
+
+// follows ops ggml-backend-meta may delay a reduction across
+// TODO: the required precision should come from the meta backend instead.
+static bool ggml_vk_comm_needs_f32(const ggml_tensor * t, int depth = 0) {
+    if (t == nullptr || depth > 64) {
+        return false;
+    }
+    switch (t->op) {
+        case GGML_OP_MUL_MAT:
+        case GGML_OP_MUL_MAT_ID:
+            return ggml_get_op_params_i32(t, 0) == GGML_PREC_F32;
+        case GGML_OP_ADD:
+            return ggml_vk_comm_needs_f32(t->src[0], depth + 1) || ggml_vk_comm_needs_f32(t->src[1], depth + 1);
+        case GGML_OP_MUL:
+        case GGML_OP_ADD_ID:
+            return ggml_vk_comm_needs_f32(t->src[0], depth + 1);
+        case GGML_OP_VIEW:
+            return ggml_vk_comm_needs_f32(t->src[0], depth + 1);
+        default:
+            return false;
+    }
+}
+
+// host_bridge: device i signals done[i], a host thread signals go[i] once all peers of i are done
+// sync_fd: a host thread exchanges SYNC_FD semaphores between peers, the wait resolves in the kernel
+enum class vk_comm_sync {
+    host_bridge,
+    sync_fd,
+};
+
+struct vk_comm_device {
+    ggml_backend_t                     backend  = nullptr;
+    ggml_backend_vk_context *          ctx      = nullptr;
+    void *                             host_ptr = nullptr;
+    std::vector<ggml_backend_buffer_t> host_buf; // host_ptr imported into each peer
+
+    struct {
+        PFN_vkGetSemaphoreFdKHR    get_fd    = nullptr;
+        PFN_vkImportSemaphoreFdKHR import_fd = nullptr;
+        vk::Semaphore              sem;
+        std::vector<vk::Semaphore> imp;      // [peer], pooled, reused only after ggml_vk_graph_cleanup
+    } sfd;
+
+    struct {
+        vk::Semaphore done;
+        vk::Semaphore go;
+    } bridge;
+};
+
+struct ggml_backend_vk_comm_context {
+    std::vector<vk_comm_device> devs;
+    vk_comm_sync                mode     = vk_comm_sync::host_bridge;
+    size_t                      align    = 4096;
+    size_t                      cap      = 0;
+    uint64_t                    round    = 0;
+    ggml_context *              tctx     = nullptr;
+    ggml_tensor *               slot     = nullptr;
+    bool                        disabled = false;
+
+    std::thread                 thread;
+    std::chrono::nanoseconds    spin{ 200'000 };
+    std::atomic<uint64_t>       submitted{ 0 };
+    std::atomic<uint64_t>       imported{ 0 };
+    std::atomic<bool>           sleeping{ false };
+    std::atomic<bool>           stop{ false };
+    std::mutex                  mutex;
+    std::condition_variable     cv;
+};
+
+static void ggml_vk_comm_free_buffers(ggml_backend_vk_comm_context * comm) {
+    for (vk_comm_device & d : comm->devs) {
+        for (ggml_backend_buffer_t & buf : d.host_buf) {
+            ggml_backend_buffer_free(buf);
+            buf = nullptr;
+        }
+        std::free(d.host_ptr);
+        d.host_ptr = nullptr;
+    }
+    comm->cap = 0;
+}
+
+static bool ggml_backend_vk_comm_ensure(ggml_backend_vk_comm_context * comm, size_t nbytes) {
+    if (comm->disabled) {
+        return false;
+    }
+    if (nbytes <= comm->cap) {
+        return true;
+    }
+    auto fail = [comm]() {
+        GGML_LOG_WARN("ggml_vulkan: shared host buffer allocation failed, using generic all-reduce\n");
+        comm->disabled = true;
+        return false;
+    };
+    for (vk_comm_device & d : comm->devs) {
+        ggml_backend_synchronize(d.backend);
+    }
+    ggml_vk_comm_free_buffers(comm);
+
+    const size_t n   = comm->devs.size();
+    const size_t cap = GGML_PAD(nbytes, comm->align);
+    for (size_t i = 0; i < n; i++) {
+        vk_comm_device & d = comm->devs[i];
+        if (posix_memalign(&d.host_ptr, comm->align, 2 * cap) != 0) {
+            d.host_ptr = nullptr;
+            return fail();
+        }
+        for (size_t j = 0; j < n; j++) {
+            d.host_buf[j] = ggml_backend_vk_device_buffer_from_host_ptr(
+                ggml_backend_get_device(comm->devs[j].backend), d.host_ptr, 2 * cap, 2 * cap);
+            if (!d.host_buf[j]) {
+                return fail();
+            }
+        }
+    }
+    comm->cap = cap;
+    return true;
+}
+
+static void ggml_vk_comm_set_slot(ggml_tensor * slot, const ggml_tensor * src, ggml_type type, ggml_backend_buffer_t buf, size_t offs) {
+    slot->type  = type;
+    slot->nb[0] = ggml_type_size(type);
+    for (int d = 0; d < GGML_MAX_DIMS; d++) {
+        slot->ne[d] = src->ne[d];
+        if (d > 0) {
+            slot->nb[d] = slot->nb[d - 1] * slot->ne[d - 1];
+        }
+    }
+    slot->buffer = buf;
+    slot->data   = (char *) ggml_backend_buffer_get_base(buf) + offs;
+}
+
+static void ggml_vk_comm_submit(ggml_backend_vk_context * ctx, vk_context & c, vk::Semaphore signal, uint64_t value) {
+    ggml_vk_ctx_end(c);
+    for (auto & cpy : c->in_memcpys) {
+        memcpy(cpy.dst, cpy.src, cpy.n);
+    }
+    for (auto & mset : c->memsets) {
+        memset(mset.dst, mset.val, mset.n);
+    }
+    c->in_memcpys.clear();
+    c->memsets.clear();
+    if (signal) {
+        c->seqs.back().back().signal_semaphores.push_back({ signal, value });
+    }
+    ggml_vk_submit(c, {});
+    ctx->submit_pending = true;
+    ctx->compute_ctx.reset();
+}
+
+static void ggml_vk_comm_notify_submitted(ggml_backend_vk_comm_context * comm, uint64_t round) {
+    comm->submitted.store(round);
+    if (comm->sleeping.load()) {
+        std::lock_guard<std::mutex> lock(comm->mutex);
+        comm->cv.notify_one();
+    }
+}
+
+// false on stop
+static bool ggml_vk_comm_wait_submitted(ggml_backend_vk_comm_context * comm, uint64_t r) {
+    using clock = std::chrono::steady_clock;
+    for (const auto t0 = clock::now(); comm->submitted.load() < r;) {
+        if (comm->stop.load()) {
+            return false;
+        }
+        if (clock::now() - t0 < comm->spin) {
+            YIELD();
+            continue;
+        }
+        comm->sleeping.store(true);
+        if (comm->submitted.load() < r) {
+            std::unique_lock<std::mutex> lock(comm->mutex);
+            comm->cv.wait(lock, [&] { return comm->submitted.load() >= r || comm->stop.load(); });
+        }
+        comm->sleeping.store(false);
+    }
+    return true;
+}
+
+// signals go[i] once all peers of i are done; false on stop
+static bool ggml_vk_comm_bridge_release(ggml_backend_vk_comm_context * comm, uint64_t r, std::vector<uint64_t> & seen, std::vector<bool> & sent) {
+    using clock = std::chrono::steady_clock;
+    const size_t n = comm->devs.size();
+    std::fill(sent.begin(), sent.end(), false);
+    size_t left = n;
+    for (const auto t0 = clock::now();;) {
+        size_t pending = n;
+        for (size_t i = 0; i < n; i++) {
+            if (sent[i]) {
+                continue;
+            }
+            bool ready = true;
+            for (size_t j = 0; j < n && ready; j++) {
+                if (j != i && seen[j] < r) {
+                    const vk_comm_device & p = comm->devs[j];
+                    seen[j] = p.ctx->device->device.getSemaphoreCounterValue(p.bridge.done);
+                    ready   = seen[j] >= r;
+                    pending = ready ? pending : j;
+                }
+            }
+            if (ready) {
+                const vk_comm_device & d = comm->devs[i];
+                d.ctx->device->device.signalSemaphore({ d.bridge.go, r });
+                sent[i] = true;
+                left--;
+            }
+        }
+        if (left == 0) {
+            return true;
+        }
+        if (clock::now() - t0 < comm->spin) {
+            YIELD();
+            continue;
+        }
+        const vk_comm_device &      p = comm->devs[pending];
+        const vk::SemaphoreWaitInfo wi{ {}, 1, &p.bridge.done, &r };
+        while (p.ctx->device->device.waitSemaphores(wi, 100'000'000) != vk::Result::eSuccess) {
+            if (comm->stop.load()) {
+                return false;
+            }
+        }
+    }
+}
+
+static void ggml_vk_comm_bridge_thread(ggml_backend_vk_comm_context * comm) {
+#if defined(__linux__)
+    pthread_setname_np(pthread_self(), "ggml-vk-comm");
+#endif
+    std::vector<uint64_t> seen(comm->devs.size(), 0);
+    std::vector<bool>     sent(comm->devs.size());
+    try {
+        for (uint64_t r = 1;; r++) {
+            if (!ggml_vk_comm_wait_submitted(comm, r) || !ggml_vk_comm_bridge_release(comm, r, seen, sent)) {
+                return;
+            }
+        }
+    } catch (const vk::SystemError & e) {
+        GGML_ABORT("ggml_vulkan: all-reduce bridge failed: %s", e.what());
+    }
+}
+
+static void ggml_vk_comm_bridge_init(ggml_backend_vk_comm_context * comm) {
+    vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
+    vk::SemaphoreCreateInfo     sci{};
+    sci.pNext = &stci;
+    for (vk_comm_device & d : comm->devs) {
+        d.bridge.done = d.ctx->device->device.createSemaphore(sci);
+        d.bridge.go   = d.ctx->device->device.createSemaphore(sci);
+    }
+    comm->thread = std::thread(ggml_vk_comm_bridge_thread, comm);
+}
+
+static void ggml_vk_comm_bridge_free(vk_comm_device & d) {
+    if (d.bridge.done) {
+        d.ctx->device->device.destroySemaphore(d.bridge.done);
+    }
+    if (d.bridge.go) {
+        d.ctx->device->device.destroySemaphore(d.bridge.go);
+    }
+}
+
+static void ggml_vk_comm_barrier_to_host(vk_context & c) {
+    const vk::MemoryBarrier to_host{ vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead };
+    c->s->buffer->buf.pipelineBarrier(c->p->q->stage_flags, vk::PipelineStageFlagBits::eHost, {}, { to_host }, {}, {});
+}
+
+// the peers' slots were made available to the host before the wait resolved, the submit predates that
+static void ggml_vk_comm_barrier_from_host(vk_context & c) {
+    const vk::MemoryBarrier from_host{ vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead };
+    c->s->buffer->buf.pipelineBarrier(vk::PipelineStageFlagBits::eHost, c->p->q->stage_flags, {}, { from_host }, {}, {});
+}
+
+static void ggml_vk_comm_bridge_signal(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
+    const vk_comm_device & d = comm->devs[i];
+    ggml_vk_comm_barrier_to_host(c);
+    ggml_vk_comm_submit(d.ctx, c, d.bridge.done, round);
+}
+
+static void ggml_vk_comm_bridge_wait(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
+    c->s->wait_semaphores.push_back({ comm->devs[i].bridge.go, round });
+    ggml_vk_comm_barrier_from_host(c);
+}
+
+static bool ggml_vk_comm_sfd_supported(vk_comm_device & d) {
+    const vk_device & dev = d.ctx->device;
+    const bool mesa = dev->driver_id == vk::DriverId::eMesaRadv ||
+                      dev->driver_id == vk::DriverId::eIntelOpenSourceMESA ||
+                      dev->driver_id == vk::DriverId::eMesaNvk;
+    if (!mesa || !dev->external_semaphore) {
+        return false;
+    }
+    d.sfd.get_fd    = (PFN_vkGetSemaphoreFdKHR)    dev->device.getProcAddr("vkGetSemaphoreFdKHR");
+    d.sfd.import_fd = (PFN_vkImportSemaphoreFdKHR) dev->device.getProcAddr("vkImportSemaphoreFdKHR");
+    if (!d.sfd.get_fd || !d.sfd.import_fd) {
+        return false;
+    }
+    vk::PhysicalDeviceExternalSemaphoreInfo esi{ vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd };
+    vk::ExternalSemaphoreProperties esp = dev->physical_device.getExternalSemaphoreProperties(esi);
+    const vk::ExternalSemaphoreFeatureFlags need =
+        vk::ExternalSemaphoreFeatureFlagBits::eExportable | vk::ExternalSemaphoreFeatureFlagBits::eImportable;
+    return (esp.externalSemaphoreFeatures & need) == need &&
+           (esp.compatibleHandleTypes & vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd);
+}
+
+// Mesa moves a queue to its submit thread on the first wait-before-signal, taking the CS ioctls off this thread
+static void ggml_vk_comm_sfd_enable_submit_thread(const vk_device & dev) {
+    vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
+    vk::SemaphoreCreateInfo     sci{};
+    sci.pNext = &stci;
+    const vk::Semaphore sem   = dev->device.createSemaphore(sci);
+    const vk::Fence     fence = dev->device.createFence({});
+    const uint64_t      one   = 1;
+    vk::TimelineSemaphoreSubmitInfo tsi{ 1, &one, 0, nullptr };
+    const vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
+    vk::SubmitInfo si{ 1, &sem, &stage, 0, nullptr, 0, nullptr };
+    si.setPNext(&tsi);
+    dev->compute_queue->handle->submit(si, fence);
+    dev->device.signalSemaphore({ sem, 1 });
+    VK_CHECK(dev->device.waitForFences({ fence }, true, UINT64_MAX), "ggml_vk_comm_sfd_enable_submit_thread", dev);
+    dev->device.destroyFence(fence);
+    dev->device.destroySemaphore(sem);
+}
+
+// Requires a submitted signal on sem; the export resets it.
+static int ggml_vk_comm_sfd_export(vk_comm_device & d) {
+    int fd = -1;
+    VkSemaphoreGetFdInfoKHR gi{};
+    gi.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
+    gi.semaphore  = d.sfd.sem;
+    gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
+    const VkResult result = d.sfd.get_fd((VkDevice) d.ctx->device->device, &gi, &fd);
+    if (result != VK_SUCCESS) {
+        GGML_ABORT("ggml_vulkan: vkGetSemaphoreFdKHR failed: %d", (int) result);
+    }
+    return fd;
+}
+
+// fd -1 imports an already signalled payload
+static void ggml_vk_comm_sfd_import(vk_comm_device & d, vk::Semaphore sem, int fd) {
+    VkImportSemaphoreFdInfoKHR isi{};
+    isi.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
+    isi.semaphore  = sem;
+    isi.flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
+    isi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
+    isi.fd         = fd;
+    const VkResult result = d.sfd.import_fd((VkDevice) d.ctx->device->device, &isi);
+    if (result != VK_SUCCESS) {
+        GGML_ABORT("ggml_vulkan: vkImportSemaphoreFdKHR failed: %d", (int) result);
+    }
+}
+
+static void ggml_vk_comm_sfd_exchange(ggml_backend_vk_comm_context * comm, uint64_t r) {
+    const size_t n = comm->devs.size();
+    for (size_t j = 0; j < n; j++) {
+        const int    fd   = ggml_vk_comm_sfd_export(comm->devs[j]);
+        const size_t last = j == n - 1 ? n - 2 : n - 1;
+        for (size_t i = 0; i < n; i++) {
+            if (i == j) {
+                continue;
+            }
+            const int imp_fd = fd < 0 || i == last ? fd : fcntl(fd, F_DUPFD_CLOEXEC, 0);
+            GGML_ASSERT(imp_fd >= 0 || fd < 0);
+            ggml_vk_comm_sfd_import(comm->devs[i], comm->devs[i].sfd.imp[j], imp_fd);
+        }
+    }
+    comm->imported.store(r, std::memory_order_release);
+}
+
+static void ggml_vk_comm_sfd_thread(ggml_backend_vk_comm_context * comm) {
+#if defined(__linux__)
+    pthread_setname_np(pthread_self(), "ggml-vk-comm");
+#endif
+    for (uint64_t r = 1; ggml_vk_comm_wait_submitted(comm, r); r++) {
+        ggml_vk_comm_sfd_exchange(comm, r);
+    }
+}
+
+static void ggml_vk_comm_sfd_init(ggml_backend_vk_comm_context * comm) {
+    vk::ExportSemaphoreCreateInfo esci{ vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd };
+    vk::SemaphoreCreateInfo       sci{};
+    sci.pNext = &esci;
+    for (vk_comm_device & d : comm->devs) {
+        d.sfd.sem = d.ctx->device->device.createSemaphore(sci);
+        d.sfd.imp.assign(comm->devs.size(), {});
+        ggml_vk_comm_sfd_enable_submit_thread(d.ctx->device);
+    }
+    comm->thread = std::thread(ggml_vk_comm_sfd_thread, comm);
+}
+
+static void ggml_vk_comm_sfd_free(vk_comm_device & d) {
+    if (d.sfd.sem) {
+        d.ctx->device->device.destroySemaphore(d.sfd.sem);
+    }
+}
+
+static void ggml_vk_comm_sfd_signal(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c) {
+    vk_comm_device & d = comm->devs[i];
+    ggml_vk_comm_barrier_to_host(c);
+    ggml_vk_comm_submit(d.ctx, c, d.sfd.sem, 0);
+    for (size_t j = 0; j < comm->devs.size(); j++) {
+        d.sfd.imp[j] = j == i ? vk::Semaphore{} : ggml_vk_create_binary_semaphore(d.ctx);
+    }
+}
+
+static void ggml_vk_comm_sfd_wait(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
+    for (size_t j = 0; j < comm->devs.size(); j++) {
+        if (j != i) {
+            c->s->wait_semaphores.push_back({ comm->devs[i].sfd.imp[j], 0 });
+        }
+    }
+    c->s->ready       = &comm->imported;
+    c->s->ready_value = round;
+    ggml_vk_comm_barrier_from_host(c);
+}
+
+void ggml_backend_vk_comm_free(void * comm_ctx) {
+    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
+    // pending reductions need the thread
+    for (vk_comm_device & d : comm->devs) {
+        ggml_backend_synchronize(d.backend);
+    }
+    if (comm->thread.joinable()) {
+        {
+            std::lock_guard<std::mutex> lock(comm->mutex);
+            comm->stop.store(true);
+        }
+        comm->cv.notify_all();
+        comm->thread.join();
+    }
+    for (vk_comm_device & d : comm->devs) {
+        ggml_vk_comm_sfd_free(d);
+        ggml_vk_comm_bridge_free(d);
+    }
+    ggml_vk_comm_free_buffers(comm);
+    if (comm->tctx) {
+        ggml_free(comm->tctx);
+    }
+    delete comm;
+}
+
+void * ggml_backend_vk_comm_init(ggml_backend_t * backends, size_t n_backends) {
+    if (vk_instance.allreduce_disabled || n_backends < 2) {
+        return nullptr;
+    }
+    for (size_t i = 0; i < n_backends; i++) {
+        if (!ggml_backend_is_vk(backends[i])) {
+            return nullptr;
+        }
+    }
+    ggml_backend_vk_comm_context * comm = new ggml_backend_vk_comm_context;
+    comm->devs.resize(n_backends);
+    for (size_t i = 0; i < n_backends; i++) {
+        vk_comm_device & d = comm->devs[i];
+        d.backend = backends[i];
+        d.ctx     = (ggml_backend_vk_context *) backends[i]->context;
+        d.host_buf.assign(n_backends, nullptr);
+    }
+
+    for (size_t i = 0; i < n_backends; i++) {
+        const vk_device & dev = comm->devs[i].ctx->device;
+        comm->align = std::max(comm->align, (size_t) dev->min_imported_host_pointer_alignment);
+        if (!dev->external_memory_host) {
+            GGML_LOG_WARN("ggml_vulkan: device %zu: VK_EXT_external_memory_host missing, using generic all-reduce\n", i);
+            ggml_backend_vk_comm_free(comm);
+            return nullptr;
+        }
+        if (dev->serialize_submissions) {
+            GGML_LOG_WARN("ggml_vulkan: device %zu: GGML_VK_SERIALIZE_SUBMISSIONS set, using generic all-reduce\n", i);
+            ggml_backend_vk_comm_free(comm);
+            return nullptr;
+        }
+    }
+
+    if (std::all_of(comm->devs.begin(), comm->devs.end(), ggml_vk_comm_sfd_supported)) {
+        comm->mode = vk_comm_sync::sync_fd;
+    }
+
+    if (const char * spin = getenv("GGML_VK_ALLREDUCE_SPIN_US")) {
+        comm->spin = std::chrono::microseconds(std::max(0, atoi(spin)));
+    }
+    const ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
+    comm->tctx = ggml_init(ip);
+    comm->slot = ggml_new_tensor_1d(comm->tctx, GGML_TYPE_F32, 1);
+
+    try {
+        if (comm->mode == vk_comm_sync::sync_fd) {
+            ggml_vk_comm_sfd_init(comm);
+        } else {
+            ggml_vk_comm_bridge_init(comm);
+        }
+    } catch (const vk::SystemError & e) {
+        GGML_LOG_WARN("ggml_vulkan: semaphore initialization failed: %s, using generic all-reduce\n", e.what());
+        ggml_backend_vk_comm_free(comm);
+        return nullptr;
+    }
+    return comm;
+}
+
+// A: partial to own host slot (round & 1), signal; B: wait on peers' A, add their slots; all A precede any B
+// slot reuse: A(k) follows own B(k-1), which waited on peers' A(k-1), submitted after their B(k-2), the last reader
+bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
+    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
+    const size_t n = comm->devs.size();
+
+    const int64_t ne = ggml_nelements(tensors[0]);
+    for (size_t i = 0; i < n; i++) {
+        const ggml_tensor * t = tensors[i];
+        if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne || !ggml_is_contiguous(t) || !ggml_backend_buffer_is_vk(t->buffer)) {
+            return false;
+        }
+        const ggml_backend_vk_context * ctx = comm->devs[i].ctx;
+        if (!ctx->compute_ctx.expired() && !ctx->compute_ctx.lock()->out_memcpys.empty()) {
+            return false;
+        }
+    }
+    if (ne == 0) {
+        return true;
+    }
+    const size_t nbytes = ggml_nbytes(tensors[0]);
+    if (!ggml_backend_vk_comm_ensure(comm, nbytes)) {
+        return false;
+    }
+
+    const bool      f32_wire  = nbytes < ggml_vk_comm_f16_min_size || ggml_vk_comm_needs_f32(tensors[0]);
+    const ggml_type wire      = f32_wire ? GGML_TYPE_F32 : GGML_TYPE_F16;
+    const uint64_t  round     = ++comm->round;
+    const size_t    slot_offs = (round & 1) * comm->cap;
+    const size_t    slot_size = ne * ggml_type_size(wire);
+
+    for (size_t i = 0; i < n; i++) {
+        const vk_comm_device &    d   = comm->devs[i];
+        ggml_backend_vk_context * ctx = d.ctx;
+        ggml_tensor *             t   = tensors[i];
+        vk_buffer & tbuf = ((ggml_backend_vk_buffer_context *) t->buffer->context)->dev_buffer;
+        vk_buffer & hbuf = ((ggml_backend_vk_buffer_context *) d.host_buf[i]->context)->dev_buffer;
+        const size_t toffs = vk_tensor_offset(t) + t->view_offs;
+
+        vk_context c = ggml_vk_get_compute_ctx(ctx);
+        ggml_vk_sync_buffers(ctx, c);
+        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
+            c->s->buffer->buf.fillBuffer(hbuf->buffer, slot_offs, GGML_PAD(slot_size, 4), 0);
+            c->s->buffer->buf.fillBuffer(tbuf->buffer, toffs, nbytes, 0);
+        } else if (wire == GGML_TYPE_F32) {
+            ggml_vk_buffer_copy_async(c, hbuf, slot_offs, tbuf, toffs, nbytes);
+        } else {
+            ggml_vk_comm_set_slot(comm->slot, t, wire, d.host_buf[i], slot_offs);
+            ggml_vk_cpy(ctx, c, t, comm->slot);
+        }
+        if (comm->mode == vk_comm_sync::sync_fd) {
+            ggml_vk_comm_sfd_signal(comm, i, c);
+        } else {
+            ggml_vk_comm_bridge_signal(comm, i, c, round);
+        }
+    }
+    ggml_vk_comm_notify_submitted(comm, round);
+
+    for (size_t i = 0; i < n; i++) {
+        ggml_backend_vk_context * ctx = comm->devs[i].ctx;
+        vk_context c = ggml_vk_get_compute_ctx(ctx);
+        ctx->do_add_rms_partials = false;
+        if (comm->mode == vk_comm_sync::sync_fd) {
+            ggml_vk_comm_sfd_wait(comm, i, c, round);
+        } else {
+            ggml_vk_comm_bridge_wait(comm, i, c, round);
+        }
+        ggml_vk_sync_buffers(ctx, c);
+        for (size_t j = 0; j < n; j++) {
+            if (j == i) {
+                continue;
+            }
+            ggml_vk_comm_set_slot(comm->slot, tensors[i], wire, comm->devs[j].host_buf[i], slot_offs);
+            ggml_vk_add(ctx, c, tensors[i], comm->slot, tensors[i]);
+            ggml_vk_sync_buffers(ctx, c);
+        }
+        // small reductions go out with the next graph's first submit
+        if (wire != GGML_TYPE_F32 || vk_perf_logger_enabled) {
+            ggml_vk_comm_submit(ctx, c, {}, 0);
+        }
+    }
+    return true;
+}
+#endif
diff --git src/ggml-vulkan/ggml-vulkan-common.h src/ggml-vulkan/ggml-vulkan-common.h
index 0ac0f8b8..b5491fc4 100644
--- src/ggml-vulkan/ggml-vulkan-common.h
+++ src/ggml-vulkan/ggml-vulkan-common.h
@@ -245,6 +245,12 @@ bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tenso
 #endif
 
 // debug
+void ggml_vk_count_fusion(const vk_device & device, const char * label);
+void * ggml_backend_vk_fusion_get(ggml_backend_dev_t dev);
+void ggml_backend_vk_fusion_stats_init(void * finfo);
+void ggml_backend_vk_fusion_stats_reset(void * finfo);
+int ggml_backend_vk_fusion_stats_get(void * finfo, const char ** labels, uint64_t * counts, int n);
+void ggml_backend_vk_fusion_set_enabled(void * finfo, bool enabled);
 int64_t ggml_vk_get_op_batch_size(const ggml_tensor * op);
 #ifdef GGML_VULKAN_RUN_TESTS
 template <typename X_TYPE, typename Y_TYPE>
diff --git src/ggml-vulkan/ggml-vulkan-debug.cpp src/ggml-vulkan/ggml-vulkan-debug.cpp
index 97747e86..5fa91951 100644
--- src/ggml-vulkan/ggml-vulkan-debug.cpp
+++ src/ggml-vulkan/ggml-vulkan-debug.cpp
@@ -12,6 +12,58 @@ uint32_t vk_perf_logger_frequency = 1;
 
 std::string vk_pipeline_stats_filter;
 
+void ggml_vk_count_fusion(const vk_device & device, const char * label) {
+    if (!device->fusion_stats_enabled.load(std::memory_order_relaxed)) {
+        return;
+    }
+    std::lock_guard<std::mutex> lock(device->fusion_mutex);
+    device->fusion_counts[label]++;
+}
+
+void * ggml_backend_vk_fusion_get(ggml_backend_dev_t dev) {
+    auto * ctx = (ggml_backend_vk_device_context *) dev->context;
+    return ggml_vk_get_device(ctx->device).get();
+}
+
+void ggml_backend_vk_fusion_stats_init(void * finfo) {
+    auto * device = (vk_device_struct *) finfo;
+    device->fusion_stats_enabled.store(true, std::memory_order_relaxed);
+}
+
+void ggml_backend_vk_fusion_stats_reset(void * finfo) {
+    auto * device = (vk_device_struct *) finfo;
+    std::lock_guard<std::mutex> lock(device->fusion_mutex);
+    // Keep labels and their order stable between fused and unfused runs.
+    for (auto & entry : device->fusion_counts) {
+        entry.second = 0;
+    }
+}
+
+int ggml_backend_vk_fusion_stats_get(void * finfo, const char ** labels, uint64_t * counts, int n) {
+    auto * device = (vk_device_struct *) finfo;
+    std::lock_guard<std::mutex> lock(device->fusion_mutex);
+    if (labels == nullptr) {
+        return (int) device->fusion_counts.size();
+    }
+    int i = 0;
+    for (const auto & entry : device->fusion_counts) {
+        if (i >= n) {
+            break;
+        }
+        labels[i] = entry.first.c_str();
+        if (counts) {
+            counts[i] = entry.second;
+        }
+        i++;
+    }
+    return i;
+}
+
+void ggml_backend_vk_fusion_set_enabled(void * finfo, bool enabled) {
+    auto * device = (vk_device_struct *) finfo;
+    device->disable_fusion = !enabled;
+}
+
 void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
     if (!vk_memory_logger_enabled) {
         return;
diff --git src/ggml-vulkan/ggml-vulkan-types.h src/ggml-vulkan/ggml-vulkan-types.h
index d3b1a8dd..5b30f026 100644
--- src/ggml-vulkan/ggml-vulkan-types.h
+++ src/ggml-vulkan/ggml-vulkan-types.h
@@ -486,6 +486,15 @@ static constexpr uint32_t num_topk_moe_pipelines = 10;
 
 static constexpr uint32_t num_topk_pipelines = 11;
 
+// FWHT block widths with a dedicated pipeline. The subgroup shader keeps n/subgroup_size
+// values per invocation, so widths above fwht_max_subgroup_n, or a subgroup narrow enough to
+// exceed fwht_max_subgroup_el_w values, use the shared-memory shader with one row per
+// workgroup instead.
+static constexpr uint32_t num_fwht_pipelines     = 8;
+static constexpr uint32_t fwht_max_subgroup_n    = 512;
+static constexpr uint32_t fwht_max_subgroup_el_w = 64;
+static constexpr uint32_t fwht_rows              = 4;
+
 static constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                              GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                              GGML_OP_SUM_ROWS, GGML_OP_CLAMP,    GGML_OP_DIV,
@@ -698,6 +707,7 @@ struct vk_device_struct {
     uint32_t subgroup_size_log2;
     uint32_t shader_core_count;
     bool uma;
+    bool rebar;
     bool prefer_host_memory;
     bool float_controls_rte_fp16;
     bool float_controls_denorm_preserve_fp16;
@@ -935,7 +945,9 @@ struct vk_device_struct {
     vk_pipeline pipeline_sum_rows_f32;
     vk_pipeline pipeline_cross_entropy_loss_f32, pipeline_cross_entropy_loss_f32_wg512;
     vk_pipeline pipeline_cross_entropy_loss_back_f32, pipeline_cross_entropy_loss_back_f32_wg512;
-    vk_pipeline pipeline_fwht_f32[4];
+    vk_pipeline pipeline_fwht_f32[num_fwht_pipelines];
+    vk_pipeline pipeline_fwht_f16[num_fwht_pipelines];
+    uint32_t    fwht_rows_per_wg[num_fwht_pipelines] = {};
     vk_pipeline pipeline_cumsum_f32;
     vk_pipeline pipeline_cumsum_small_f32;
     vk_pipeline pipeline_cumsum_multipass1_f32;
@@ -1007,6 +1019,9 @@ struct vk_device_struct {
     ggml_backend_buffer_type buffer_type;
 
     bool disable_fusion;
+    std::atomic<bool> fusion_stats_enabled {false};
+    std::mutex fusion_mutex;
+    std::map<std::string, uint64_t> fusion_counts;
     bool disable_descriptor_reuse;
     std::atomic<uint64_t> buffer_destroy_count {};
     bool disable_host_visible_vidmem;
diff --git src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan.cpp
index 89f0470c..8347cd1b 100644
--- src/ggml-vulkan/ggml-vulkan.cpp
+++ src/ggml-vulkan/ggml-vulkan.cpp
@@ -1678,6 +1678,17 @@ static bool ggml_vk_fa_scalar_uses_mmq(const vk_device& device, ggml_type k_type
 #endif
 }
 
+// AMD proprietary drivers before 2.0.416 (Adrenalin 26.10.1) run the large coopmat matmul tile much slower than the medium one.
+static bool ggml_vk_amd_large_tile_supported(const vk_device& device) {
+    if (device->driver_id != vk::DriverId::eAmdProprietary) {
+        return true;
+    }
+    // AMD encodes driverVersion as major [31:22], build [21:0], not VK_MAKE_API_VERSION.
+    const uint32_t major = device->properties.driverVersion >> 22;
+    const uint32_t build = device->properties.driverVersion & 0x3fffff;
+    return major > 2 || (major == 2 && build >= 416);
+}
+
 void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
     VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");
 
@@ -1850,7 +1861,7 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
         if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
             m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
             m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
-        } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
+        } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && ggml_vk_amd_large_tile_supported(device)) {
             // This is intentionally using tx_m values, slight performance increase
             l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
             l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
@@ -3581,21 +3592,31 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
     // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
     const bool can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
         !ggml_vk_intel_windows_driver_in_range(device->properties.driverVersion, 101, 8509, 101, 8860);
-    if (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
+    if (can_use_fwht) {
+        const bool use_subgroup = device->subgroup_basic && device->subgroup_shuffle;
+        const uint32_t sg = std::max(device->subgroup_size, 1u);
         int idx = 0;
-        for (uint32_t n : {64, 128, 256, 512}) {
-            if (device->subgroup_size <= n) {
-                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n }, 1, true, true, device->subgroup_size);
+        for (uint32_t n : {64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
+            const bool wide = n > fwht_max_subgroup_n || n / sg > fwht_max_subgroup_el_w;
+            if (use_subgroup && !wide) {
+                if (device->subgroup_size <= n) {
+                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, fwht_rows }, 1, true, true, device->subgroup_size);
+                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f16[idx], "fwht_f16", fwht_f16_len, fwht_f16_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, fwht_rows }, 1, true, true, device->subgroup_size);
+                    device->fwht_rows_per_wg[idx] = fwht_rows;
+                }
+            } else {
+                // wide blocks trade rows per workgroup for a smaller register block
+                const uint32_t block_size = wide ? std::min(n, 256u) : std::min(sg, n);
+                const uint32_t rows       = wide ? 1u : fwht_rows;
+                if ((uint64_t)rows * n * sizeof(float) <= device->properties.limits.maxComputeSharedMemorySize &&
+                    block_size * rows <= device->properties.limits.maxComputeWorkGroupInvocations) {
+                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);
+                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f16[idx], "fwht_shmem_f16", fwht_shmem_f16_len, fwht_shmem_f16_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);
+                    device->fwht_rows_per_wg[idx] = rows;
+                }
             }
             ++idx;
         }
-    } else if (can_use_fwht) {
-        int idx = 0;
-        for (uint32_t n : {64, 128, 256, 512}) {
-            const uint32_t block_size = std::min(device->subgroup_size, n);
-            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n }, 1);
-            ++idx;
-        }
     }
 
     const uint32_t cumsum_elem_per_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
@@ -4020,6 +4041,24 @@ static bool ggml_vk_host_buffer_import_usable(uint32_t vendor_id) {
     return vendor_id != VK_VENDOR_ID_AMD;
 }
 
+static bool ggml_vk_has_rebar(const vk::PhysicalDevice & physical_device) {
+    const vk::PhysicalDeviceMemoryProperties mem_props = physical_device.getMemoryProperties();
+    const vk::MemoryPropertyFlags hvv = vk::MemoryPropertyFlagBits::eDeviceLocal |
+                                        vk::MemoryPropertyFlagBits::eHostVisible |
+                                        vk::MemoryPropertyFlagBits::eHostCoherent;
+    uint32_t vram_heap = UINT32_MAX;
+    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
+        const vk::MemoryType & t = mem_props.memoryTypes[i];
+        if (vram_heap == UINT32_MAX && (t.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
+            vram_heap = t.heapIndex;
+        }
+        if ((t.propertyFlags & hvv) == hvv && t.heapIndex == vram_heap) {
+            return true;
+        }
+    }
+    return false;
+}
+
 vk_device ggml_vk_get_device(size_t idx) {
     VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");
 
@@ -4254,6 +4293,7 @@ vk_device ggml_vk_get_device(size_t idx) {
         device->subgroup_size = subgroup_props.subgroupSize;
         device->subgroup_size_log2 = uint32_t(log2f(float(device->subgroup_size)));
         device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
+        device->rebar = ggml_vk_has_rebar(device->physical_device);
         if (sm_builtins) {
             device->shader_core_count = sm_props.shaderSMCount;
         } else if (amd_shader_core_properties2) {
@@ -4816,7 +4856,7 @@ vk_device ggml_vk_get_device(size_t idx) {
             switch (device->vendor_id) {
 #ifndef GGML_VULKAN_RUN_TESTS
             case VK_VENDOR_ID_AMD:
-                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
+                device->mul_mat_l[i]    = device->coopmat_support && ggml_vk_amd_large_tile_supported(device);
                 device->mul_mat_m[i]    = true;
                 device->mul_mat_s[i]    = true;
                 device->mul_mat_id_l[i] = false;
@@ -4943,8 +4983,7 @@ vk_device ggml_vk_get_device(size_t idx) {
 
         device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != nullptr;
 
-        device->add_rms_fusion = !device->disable_fusion &&
-                                 device->subgroup_arithmetic &&
+        device->add_rms_fusion = device->subgroup_arithmetic &&
                                  device->vendor_id != VK_VENDOR_ID_INTEL;
         device->partials_binding_alignment =
             std::max(4u, (uint32_t)device->properties.limits.minStorageBufferOffsetAlignment);
@@ -5156,6 +5195,7 @@ static void ggml_vk_print_gpu_info(size_t idx) {
     uint32_t default_subgroup_size = get_subgroup_size("", device_architecture);
     const size_t subgroup_size = (default_subgroup_size != 0) ? default_subgroup_size : subgroup_props.subgroupSize;
     const bool uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
+    const bool rebar = ggml_vk_has_rebar(physical_device);
 
     integer_dot_product = integer_dot_product
                        && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated
@@ -5202,8 +5242,8 @@ static void ggml_vk_print_gpu_info(size_t idx) {
 #endif
 
     std::string device_name = props2.properties.deviceName.data();
-    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
-              idx, device_name.c_str(), driver_props.driverName.data(), uma, fp16_str, bf16, fp4, subgroup_size,
+    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | rebar: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
+              idx, device_name.c_str(), driver_props.driverName.data(), uma, rebar, fp16_str, bf16, fp4, subgroup_size,
               props2.properties.limits.maxComputeSharedMemorySize, integer_dot_product, matrix_cores.c_str());
 
     if (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
@@ -5955,6 +5995,11 @@ static uint32_t ggml_vk_guess_split_k(ggml_backend_vk_context * ctx, uint32_t m,
                 split_k = ctx->device->shader_core_count / (m_tiles * n_tiles);
             } else if (m_tiles * n_tiles <= ctx->device->shader_core_count * 2 / 3) {
                 split_k = 3;
+            } else if ((ctx->device->architecture == AMD_RDNA3 || ctx->device->architecture == AMD_RDNA4) &&
+                       m_tiles * n_tiles > ctx->device->shader_core_count &&
+                       m_tiles * n_tiles <= ctx->device->shader_core_count * 4 / 3) {
+                // two waves, the second one at most 1/3 full
+                split_k = 3;
             }
             // Cap the split at 8x. Unless k is huge this is a lot of overhead.
             split_k = std::min(split_k, 8u);
@@ -7200,6 +7245,10 @@ static int ggml_vk_fwht_pipeline_idx(int64_t n) {
         case 128: return 1;
         case 256: return 2;
         case 512: return 3;
+        case 1024: return 4;
+        case 2048: return 5;
+        case 4096: return 6;
+        case 8192: return 7;
         default:  return -1;
     }
 }
@@ -7213,12 +7262,18 @@ bool ggml_vk_can_use_fwht(const ggml_backend_vk_context * ctx, const ggml_tensor
         return false;
     }
 
-    const int idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
-    if (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == nullptr) {
+    if (dst->type != GGML_TYPE_F32) {
         return false;
     }
 
-    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
+    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
+        return false;
+    }
+
+    const bool src_f16 = src1->type == GGML_TYPE_F16;
+
+    const int idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
+    if (idx < 0 || (src_f16 ? ctx->device->pipeline_fwht_f16[idx] : ctx->device->pipeline_fwht_f32[idx]) == nullptr) {
         return false;
     }
 
@@ -7232,9 +7287,10 @@ bool ggml_vk_can_use_fwht(const ggml_backend_vk_context * ctx, const ggml_tensor
 
 void ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src, ggml_tensor * dst) {
     const int idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
-    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];
+    vk_pipeline pipeline = src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_f16[idx] : ctx->device->pipeline_fwht_f32[idx];
 
-    const uint32_t rows_per_workgroup = 4;
+    const uint32_t rows_per_workgroup = ctx->device->fwht_rows_per_wg[idx];
+    GGML_ASSERT(rows_per_workgroup > 0);
     const uint32_t n_rows = (uint32_t)ggml_nrows(src);
     const uint32_t max_workgroups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];
 
@@ -8221,10 +8277,14 @@ void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const
                                  v->nb[1] != (uint64_t)HSV * ggml_type_size(v->type)) &&
                                 (HSK % 8) == 0 && (HSV % 8) == 0;
     const size_t kv_scratch_ts = kv_raw_strided ? ggml_type_size(k->type) : sizeof(ggml_fp16_t);
+    // K and V share one scratch buffer, so the sum must fit in one allocation
+    const uint64_t dequant_kv_sz = ((uint64_t)ggml_nelements(k) + (uint64_t)ggml_nelements(v)) * kv_scratch_ts;
     const bool use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                 is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                 (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.maxStorageBufferRange &&
                                 (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.maxStorageBufferRange &&
+                                dequant_kv_sz <= ctx->device->max_buffer_size &&
+                                dequant_kv_sz <= ctx->device->max_memory_allocation_size &&
                                 (kv_raw_strided ||
                                  (ctx->device->pipeline_dequant_transpose[k->type] != nullptr &&
                                   ctx->device->pipeline_dequant_transpose[v->type] != nullptr)) &&
@@ -10904,7 +10964,7 @@ static vk_op_rope_push_constants ggml_vk_make_rope_constants(const ggml_tensor *
     const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
 
     float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, true, corr_dims);
 
     const float theta_scale = powf(freq_base, -2.0f/n_dims);
 
@@ -11402,7 +11462,7 @@ void ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_
     }
 
     float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, true, corr_dims);
 
     uint32_t set_rows_stride = 0;
     // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
@@ -12407,11 +12467,12 @@ bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, in
             cgraph->nodes[next_node_idx]->op == GGML_OP_RMS_NORM &&
             cgraph->nodes[next_node_idx]->src[0] == cgraph->nodes[next_node_idx - 1] &&
             ggml_nrows(cgraph->nodes[next_node_idx]) == 1 &&
-            ctx->device->add_rms_fusion) {
+            ctx->device->add_rms_fusion && !ctx->device->disable_fusion) {
             uint32_t size = ggml_vk_rms_partials_size(ctx, cgraph->nodes[node_idx]);
             ctx->do_add_rms_partials_offset_calculation = true;
             if (ctx->prealloc_size_add_rms_partials_offset + size <= ctx->prealloc_size_add_rms_partials) {
                 ctx->do_add_rms_partials = true;
+                ggml_vk_count_fusion(ctx->device, "ADD_RMS_NORM_PARTIALS");
             }
         }
     }
@@ -14179,7 +14240,7 @@ bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct
         return false;
     }
 
-    return true;
+    return ggml_rope_get_truncate(rope);
 }
 
 bool ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
@@ -14317,7 +14378,7 @@ bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, const str
         return false;
     }
 
-    return true;
+    return ggml_rope_get_truncate(rope);
 }
 
 uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
@@ -14819,6 +14880,16 @@ static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cg
         }
 
         if (enqueued) {
+            if (fusion_string && ctx->device->fusion_stats_enabled.load(std::memory_order_relaxed)) {
+                std::string label = fusion_string;
+                if (cgraph->nodes[i]->op == GGML_OP_ADD) {
+                    label += "_" + std::to_string(ctx->num_additional_fused_ops + 1);
+                }
+                if (ctx->fused_topk_moe_scale) {
+                    label += "_SCALE";
+                }
+                ggml_vk_count_fusion(ctx->device, label.c_str());
+            }
             ++submitted_nodes;
 
 #ifndef GGML_VULKAN_CHECK_RESULTS
@@ -15866,8 +15937,12 @@ static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggm
         case GGML_OP_REPEAT_BACK:
             return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
         case GGML_OP_ROPE:
+            if (!ggml_rope_get_truncate(op)) {
+                return false;
+            }
             return ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0]);
         case GGML_OP_ROPE_BACK:
+            return ggml_rope_get_truncate(op);
         case GGML_OP_NONE:
         case GGML_OP_RESHAPE:
         case GGML_OP_VIEW:
@@ -16383,9 +16458,22 @@ static void * ggml_backend_vk_reg_get_proc_address(ggml_backend_reg_t reg, const
     if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
         return (void *) ggml_backend_vk_comm_allreduce_tensor;
     }
-#else
-    UNUSED(name);
 #endif
+    if (strcmp(name, "ggml_backend_fusion_get") == 0) {
+        return (void *) ggml_backend_vk_fusion_get;
+    }
+    if (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
+        return (void *) ggml_backend_vk_fusion_stats_init;
+    }
+    if (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
+        return (void *) ggml_backend_vk_fusion_stats_reset;
+    }
+    if (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
+        return (void *) ggml_backend_vk_fusion_stats_get;
+    }
+    if (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
+        return (void *) ggml_backend_vk_fusion_set_enabled;
+    }
     return nullptr;
 }
 
diff --git src/ggml-vulkan/vulkan-shaders/fwht.comp src/ggml-vulkan/vulkan-shaders/fwht.comp
index a2069964..d279d430 100644
--- src/ggml-vulkan/vulkan-shaders/fwht.comp
+++ src/ggml-vulkan/vulkan-shaders/fwht.comp
@@ -5,11 +5,13 @@
 #extension GL_KHR_shader_subgroup_basic : enable
 #extension GL_KHR_shader_subgroup_shuffle : enable
 #endif
+#extension GL_EXT_shader_16bit_storage : require
 
 layout(constant_id = 0) const uint BLOCK_SIZE = 32;
 layout(constant_id = 1) const uint N = 128;
+layout(constant_id = 2) const uint ROWS = 4;
 
-layout(local_size_x_id = 0, local_size_y = 4, local_size_z = 1) in;
+layout(local_size_x_id = 0, local_size_y_id = 2, local_size_z = 1) in;
 
 layout(push_constant) uniform parameter
 {
@@ -19,13 +21,13 @@ layout(push_constant) uniform parameter
     float scale;
 };
 
-layout(binding = 0, std430) readonly buffer A { float data_a[]; };
+layout(binding = 0, std430) readonly buffer A { A_TYPE data_a[]; };
 layout(binding = 1, std430) writeonly buffer D { float data_d[]; };
 
 const uint EL_W = N / BLOCK_SIZE;
 
 #ifdef FWHT_SHMEM
-shared float shmem[4 * N];
+shared float shmem[ROWS * N];
 #endif
 
 void main() {
@@ -54,7 +56,7 @@ void main() {
 
         [[unroll]]
         for (uint i = 0; i < EL_W; ++i) {
-            reg[i] = row < n_rows ? data_a[src_offset + row_offset + i * BLOCK_SIZE + tid] * scale : 0.0;
+            reg[i] = row < n_rows ? float(data_a[src_offset + row_offset + i * BLOCK_SIZE + tid]) * scale : 0.0;
         }
 
 #ifdef FWHT_SHMEM
diff --git src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
index 3af09fb4..dba9d8ca 100644
--- src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
+++ src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
@@ -1110,8 +1110,10 @@ void process_shaders() {
     string_to_spv("sum_rows_f32", "sum_rows.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
     string_to_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
     string_to_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
-    string_to_spv("fwht_f32", "fwht.comp", {});
-    string_to_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
+    string_to_spv("fwht_f32", "fwht.comp", {{"A_TYPE", "float"}});
+    string_to_spv("fwht_shmem_f32", "fwht.comp", {{"A_TYPE", "float"}, {"FWHT_SHMEM", "1"}});
+    string_to_spv("fwht_f16", "fwht.comp", {{"A_TYPE", "float16_t"}});
+    string_to_spv("fwht_shmem_f16", "fwht.comp", {{"A_TYPE", "float16_t"}, {"FWHT_SHMEM", "1"}});
     string_to_spv("count_equal_i32", "count_equal.comp", merge_maps(base_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
     string_to_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
     string_to_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
diff --git src/ggml-webgpu/ggml-webgpu-shader-lib.hpp src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
index d1cd0818..85d51a01 100644
--- src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
+++ src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
@@ -1228,7 +1228,7 @@ class ggml_webgpu_shader_lib {
     wgpu::Device           device;
     pre_wgsl::Preprocessor preprocessor;
 
-    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;       // key is fixed, no variants yet
+    std::unordered_map<int, webgpu_pipeline> sum_rows_pipelines;       // key is whether to compute the mean
     std::unordered_map<int, webgpu_pipeline> argmax_pipelines;         // key is vec4
     std::unordered_map<int, webgpu_pipeline> argsort_pipelines;        // key is order
     std::unordered_map<int, webgpu_pipeline> argsort_merge_pipelines;  // key is order
@@ -1319,16 +1319,20 @@ class ggml_webgpu_shader_lib {
     ggml_webgpu_shader_lib(wgpu::Device device) { this->device = device; }
 
     webgpu_pipeline get_sum_rows_pipeline(const ggml_webgpu_shader_lib_context & context) {
-        auto it = sum_rows_pipelines.find(1);
+        const bool mean = context.dst->op == GGML_OP_MEAN;
+        auto       it   = sum_rows_pipelines.find(mean);
         if (it != sum_rows_pipelines.end()) {
             return it->second;
         }
         std::vector<std::string> defines;
         defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
+        if (mean) {
+            defines.push_back("MEAN");
+        }
 
-        auto processed        = preprocessor.preprocess(wgsl_sum_rows, defines);
-        sum_rows_pipelines[1] = ggml_webgpu_create_pipeline(device, processed, "sum_rows");
-        return sum_rows_pipelines[1];
+        auto processed           = preprocessor.preprocess(wgsl_sum_rows, defines);
+        sum_rows_pipelines[mean] = ggml_webgpu_create_pipeline(device, processed, mean ? "mean" : "sum_rows");
+        return sum_rows_pipelines[mean];
     }
 
     webgpu_pipeline get_row_norm_pipeline(const ggml_webgpu_shader_lib_context & context) {
diff --git src/ggml-webgpu/ggml-webgpu.cpp src/ggml-webgpu/ggml-webgpu.cpp
index bb2deb9d..768eb96f 100644
--- src/ggml-webgpu/ggml-webgpu.cpp
+++ src/ggml-webgpu/ggml-webgpu.cpp
@@ -182,7 +182,6 @@ struct webgpu_capabilities {
     uint32_t subgroup_size     = 0;
     uint32_t min_subgroup_size = 0;
     uint32_t max_subgroup_size = 0;
-    size_t   memset_bytes_per_thread;
 };
 
 // Stores global webgpu members
@@ -203,6 +202,11 @@ struct webgpu_global_context_struct {
     wgpu::Buffer    memset_params_buf;
     webgpu_pipeline memset_pipeline;
 
+    // Packed source data and write descriptors for set_tensor_n
+    wgpu::Buffer    set_tensor_n_src_buf;
+    wgpu::Buffer    set_tensor_n_descs_buf;
+    webgpu_pipeline set_tensor_n_pipeline;
+
     std::string vendor;
 
     // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
@@ -227,6 +231,14 @@ struct webgpu_global_context_struct {
             this->memset_params_buf.Destroy();
             this->memset_params_buf = nullptr;
         }
+        if (this->set_tensor_n_src_buf) {
+            this->set_tensor_n_src_buf.Destroy();
+            this->set_tensor_n_src_buf = nullptr;
+        }
+        if (this->set_tensor_n_descs_buf) {
+            this->set_tensor_n_descs_buf.Destroy();
+            this->set_tensor_n_descs_buf = nullptr;
+        }
 #ifdef GGML_WEBGPU_DEBUG
         if (this->debug_host_buf) {
             this->debug_host_buf.Destroy();
@@ -256,8 +268,6 @@ struct webgpu_context_struct {
     wgpu::ComputePassEncoder active_compute_pass;
     bool                     batch_compute_passes = true;
 
-    size_t memset_bytes_per_thread;
-
 #ifdef GGML_WEBGPU_GPU_PROFILE
     // Profiling: per-shader GPU time in ms
     std::unordered_map<std::string, double> shader_gpu_time_ms;
@@ -672,38 +682,54 @@ static void ggml_backend_webgpu_buffer_memset(webgpu_global_context & ctx,
                                               size_t                  size) {
     std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
 
-    std::vector<uint32_t>             params  = { (uint32_t) offset, (uint32_t) size, value };
-    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, 0, buf.GetSize()) };
-    size_t                            bytes_per_wg =
-        ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup * ctx->capabilities.memset_bytes_per_thread;
-    uint32_t wg_x = CEIL_DIV(size + 3, bytes_per_wg);
-
-    ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * sizeof(uint32_t));
-
-    wgpu::BindGroupEntry params_entry = {};
-    params_entry.binding              = 1;
-    params_entry.buffer               = ctx->memset_params_buf;
-    params_entry.offset               = 0;
-    params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
-    entries.push_back(params_entry);
-
-    wgpu::BindGroupDescriptor bind_group_desc;
-    bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
-    bind_group_desc.entryCount = entries.size();
-    bind_group_desc.entries    = entries.data();
-    bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
-    wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);
-
-    wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
-    wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
-    pass.SetPipeline(ctx->memset_pipeline.pipeline);
-    pass.SetBindGroup(0, bind_group);
-    pass.DispatchWorkgroups(wg_x, 1, 1);
-    pass.End();
-
-    wgpu::CommandBuffer              command  = encoder.Finish();
-    std::vector<wgpu::CommandBuffer> commands = { command };
-    ctx->queue.Submit(commands.size(), commands.data());
+    const size_t max_bind_size = ctx->capabilities.limits.maxStorageBufferBindingSize;
+    const size_t max_wg_size   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+
+    while (size > 0) {
+        const size_t align_offset = offset & ~(ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
+        const size_t lead         = offset - align_offset;
+        const size_t chunk_size   = std::min(size, max_bind_size - lead);
+        const size_t bind_size    = ROUNDUP_POW2(chunk_size + lead, WEBGPU_STORAGE_BUF_BINDING_MULT);
+
+        std::vector<uint32_t>             params  = { (uint32_t) lead, (uint32_t) chunk_size, value };
+        std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_bind_group_entry(0, buf, align_offset,
+                                                                                        bind_size) };
+        uint32_t                          wg_x;
+        uint32_t                          wg_y;
+        uint32_t                          n_values = CEIL_DIV(chunk_size + lead, sizeof(uint32_t));
+        uint32_t                          total_wg = CEIL_DIV(n_values, max_wg_size);
+        compute_2d_workgroups(total_wg, ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, wg_x, wg_y);
+
+        ctx->queue.WriteBuffer(ctx->memset_params_buf, 0, params.data(), params.size() * sizeof(uint32_t));
+
+        wgpu::BindGroupEntry params_entry = {};
+        params_entry.binding              = 1;
+        params_entry.buffer               = ctx->memset_params_buf;
+        params_entry.offset               = 0;
+        params_entry.size                 = WEBGPU_PARAMS_BUF_SIZE_BYTES;
+        entries.push_back(params_entry);
+
+        wgpu::BindGroupDescriptor bind_group_desc;
+        bind_group_desc.layout     = ctx->memset_pipeline.pipeline.GetBindGroupLayout(0);
+        bind_group_desc.entryCount = entries.size();
+        bind_group_desc.entries    = entries.data();
+        bind_group_desc.label      = ctx->memset_pipeline.name.c_str();
+        wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);
+
+        wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
+        wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
+        pass.SetPipeline(ctx->memset_pipeline.pipeline);
+        pass.SetBindGroup(0, bind_group);
+        pass.DispatchWorkgroups(wg_x, wg_y, 1);
+        pass.End();
+
+        wgpu::CommandBuffer              command  = encoder.Finish();
+        std::vector<wgpu::CommandBuffer> commands = { command };
+        ctx->queue.Submit(commands.size(), commands.data());
+
+        offset += chunk_size;
+        size -= chunk_size;
+    }
 }
 
 /** End WebGPU Actions */
@@ -2813,7 +2839,7 @@ static webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
     float theta_scale = powf(freq_base, -2.0f / n_dims);
 
     float corr_dims[2];
-    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
+    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, true, corr_dims);
 
     std::vector<uint32_t> params = {
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
@@ -3252,7 +3278,8 @@ static webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor
                                      total_sum ? 0 : (uint32_t) (src->nb[3] / ggml_type_size(src->type)),
                                      total_sum ? static_cast<uint32_t>(ggml_nelements(src)) : (uint32_t) src->ne[0],
                                      total_sum ? 1 : (uint32_t) src->ne[1],
-                                     total_sum ? 1 : (uint32_t) src->ne[2] };
+                                     total_sum ? 1 : (uint32_t) src->ne[2],
+                                     total_sum ? 1 : (uint32_t) ggml_nrows(dst) };
 
     std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src),
                                                   ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst) };
@@ -3520,6 +3547,7 @@ static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
             return ggml_webgpu_cumsum(ctx, src0, node);
         case GGML_OP_SUM:
         case GGML_OP_SUM_ROWS:
+        case GGML_OP_MEAN:
             return ggml_webgpu_sum_rows(ctx, src0, node);
         case GGML_OP_CONV_2D:
             return ggml_webgpu_conv_2d(ctx, src0, src1, node);
@@ -3887,6 +3915,112 @@ static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
     WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor, buf_ctx->global_ctx);
 }
 
+static void ggml_backend_webgpu_buffer_set_tensor_n(ggml_backend_buffer_t        buffer,
+                                                    struct ggml_tensor * const * tensors,
+                                                    const void * const *         data,
+                                                    const size_t *               offsets,
+                                                    const size_t *               sizes,
+                                                    size_t                       n_tensors) {
+    WEBGPU_CPU_PROFILE_TOTAL_START(set_tensor_n);
+    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
+    webgpu_global_context &              ctx     = buf_ctx->global_ctx;
+
+    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor_n(" << buf_ctx->label << ", " << (const void *) tensors
+                                                                << ", " << (const void *) data << ", " << offsets
+                                                                << ", " << sizes << ", " << n_tensors << ")");
+
+    const size_t max_binding_size = ctx->capabilities.limits.maxStorageBufferBindingSize;
+    const size_t max_wg_per_dim   = ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
+
+    std::lock_guard<std::recursive_mutex> lock(ctx->mutex);
+
+    std::vector<uint8_t>  src;    // we use src to copy the data of many tensors
+    std::vector<uint32_t> descs;  // [dst offset in bytes, src offset in words, size in bytes, unused]
+    wgpu::Buffer          dst_buf;
+
+    auto grow_buffer = [&ctx](wgpu::Buffer & buf, size_t size, const char * label) {
+        if (buf == nullptr || buf.GetSize() < size) {
+            if (buf) {
+                buf.Destroy();
+            }
+            ggml_webgpu_create_buffer(ctx->device, buf, size, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage,
+                                      label);
+        }
+    };
+
+    auto flush_tensors = [&]() {
+        if (descs.empty()) {
+            return;
+        }
+
+        src.resize(ROUNDUP_POW2(src.size(), 4));
+        const size_t descs_size = descs.size() * sizeof(uint32_t);
+        const size_t n_dsts     = descs.size() / 4;
+
+        grow_buffer(ctx->set_tensor_n_src_buf, src.size(), "set_tensor_n_src_buf");
+        grow_buffer(ctx->set_tensor_n_descs_buf, descs_size, "set_tensor_n_descs_buf");
+        ctx->queue.WriteBuffer(ctx->set_tensor_n_src_buf, 0, src.data(), src.size());
+        ctx->queue.WriteBuffer(ctx->set_tensor_n_descs_buf, 0, descs.data(), descs_size);
+
+        std::vector<wgpu::BindGroupEntry> entries = {
+            ggml_webgpu_make_bind_group_entry(0, dst_buf, 0, dst_buf.GetSize()),
+            ggml_webgpu_make_bind_group_entry(1, ctx->set_tensor_n_src_buf, 0, src.size()),
+            ggml_webgpu_make_bind_group_entry(2, ctx->set_tensor_n_descs_buf, 0, descs_size),
+        };
+
+        wgpu::BindGroupDescriptor bind_group_desc;
+        bind_group_desc.layout     = ctx->set_tensor_n_pipeline.pipeline.GetBindGroupLayout(0);
+        bind_group_desc.entryCount = entries.size();
+        bind_group_desc.entries    = entries.data();
+        bind_group_desc.label      = ctx->set_tensor_n_pipeline.name.c_str();
+        wgpu::BindGroup bind_group = ctx->device.CreateBindGroup(&bind_group_desc);
+
+        uint32_t wg_x;
+        uint32_t wg_y;
+        uint32_t total_wg = n_dsts;
+        compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
+
+        wgpu::CommandEncoder     encoder = ctx->device.CreateCommandEncoder();
+        wgpu::ComputePassEncoder pass    = encoder.BeginComputePass();
+        pass.SetPipeline(ctx->set_tensor_n_pipeline.pipeline);
+        pass.SetBindGroup(0, bind_group);
+        pass.DispatchWorkgroups(wg_x, wg_y, 1);
+        pass.End();
+
+        wgpu::CommandBuffer commands = encoder.Finish();
+        ctx->queue.Submit(1, &commands);
+
+        src.clear();
+        descs.clear();
+    };
+
+    if (n_tensors > 0) {
+        dst_buf = ggml_webgpu_tensor_buf(tensors[0]);
+    }
+
+    for (size_t i = 0; i < n_tensors; i++) {
+        if (sizes[i] == 0) {
+            continue;
+        }
+        wgpu::Buffer buf    = ggml_webgpu_tensor_buf(tensors[i]);
+        size_t       offset = ggml_webgpu_tensor_offset(tensors[i]) + offsets[i];
+        if (!descs.empty() && (dst_buf.Get() != buf.Get() || src.size() + 4 + sizes[i] > max_binding_size ||
+                               (descs.size() + 4) * sizeof(uint32_t) > max_binding_size)) {
+            flush_tensors();
+        }
+        dst_buf = buf;
+
+        // Place the data at the same in-word byte lane as its destination
+        size_t src_offset = ROUNDUP_POW2(src.size(), 4) + offset % 4;
+        src.resize(src_offset + sizes[i]);
+        memcpy(src.data() + src_offset, data[i], sizes[i]);
+        descs.insert(descs.end(), { (uint32_t) offset, (uint32_t) (src_offset / 4), (uint32_t) sizes[i], 0 });
+    }
+    flush_tensors();
+
+    WEBGPU_CPU_PROFILE_TOTAL_END(set_tensor_n, ctx);
+}
+
 static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
@@ -3947,6 +4081,117 @@ static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
     WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor, buf_ctx->global_ctx);
 }
 
+static void ggml_backend_webgpu_buffer_get_tensor_n(ggml_backend_buffer_t              buffer,
+                                                    const struct ggml_tensor * const * tensors,
+                                                    void * const *                     data,
+                                                    const size_t *                     offsets,
+                                                    const size_t *                     sizes,
+                                                    size_t                             n_tensors) {
+    WEBGPU_CPU_PROFILE_TOTAL_START(get_tensor_n);
+    ggml_backend_webgpu_buffer_context * buf_ctx = (ggml_backend_webgpu_buffer_context *) buffer->context;
+    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor_n(" << buf_ctx->label << ", " << (const void *) tensors
+                                                                << ", " << (const void *) data << ", " << offsets
+                                                                << ", " << sizes << ", " << n_tensors << ")");
+    webgpu_global_context global_ctx = buf_ctx->global_ctx;
+    wgpu::Device          device     = global_ctx->device;
+
+    typedef struct {
+        size_t staging_size;
+        size_t align_offset;
+        size_t local_offset;
+    } tensor_desc;
+
+    std::vector<tensor_desc> t_descs;
+    std::vector<std::pair<size_t, size_t>>
+        staging_chunk_descs;  // pair (boundary id, sum of the tensor size to the bounary id)
+    t_descs.reserve(n_tensors);
+
+    size_t sum_tensor_size = 0;
+    for (size_t i = 0; i < n_tensors; i++) {
+        const ggml_tensor * tensor = tensors[i];
+        tensor_desc         td;
+
+        td.align_offset = ggml_webgpu_tensor_offset(tensor) + offsets[i];
+
+        td.local_offset = td.align_offset % 4;
+        if (td.local_offset != 0) {
+            // If offset is not a multiple of 4, we need to round it down to the previous
+            // multiple of 4
+            td.align_offset -= td.local_offset;
+        }
+
+        td.staging_size = sizes[i] + td.local_offset;
+        if (td.staging_size % 4 != 0) {
+            // If size is not a multiple of 4, we need to round it up to the next
+            // multiple of 4
+            td.staging_size += 4 - (td.staging_size % 4);
+        }
+
+        t_descs.push_back(td);
+
+        if (sum_tensor_size + td.staging_size > global_ctx->capabilities.limits.maxBufferSize) {
+            if (i == 0) {
+                GGML_ABORT("it is not expected that one tensor size exceed the one wgpu::Buffer limit size.");
+            }
+            staging_chunk_descs.push_back({ i - 1, sum_tensor_size });
+            sum_tensor_size = td.staging_size;
+        } else {
+            sum_tensor_size += td.staging_size;
+        }
+    }
+    staging_chunk_descs.push_back({ n_tensors - 1, sum_tensor_size });
+
+    std::lock_guard<std::recursive_mutex> lock(global_ctx->mutex);
+
+    size_t tsi = 0;
+    for (auto & [tei, sum_chunk_size] : staging_chunk_descs) {
+        if (global_ctx->get_tensor_staging_buf == nullptr ||
+            global_ctx->get_tensor_staging_buf.GetSize() < sum_chunk_size) {
+            // Create a new staging buffer if it doesn't exist or is too small
+            if (global_ctx->get_tensor_staging_buf) {
+                global_ctx->get_tensor_staging_buf.Destroy();
+            }
+            ggml_webgpu_create_buffer(device, global_ctx->get_tensor_staging_buf, sum_chunk_size,
+                                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
+                                      "get_tensor_staging_buf");
+        }
+
+        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
+
+        // Copy the data of the each tensor from the buffer to the staging buffer
+        size_t i_offset = 0;
+        for (size_t i = tsi; i <= tei; i++) {
+            encoder.CopyBufferToBuffer(ggml_webgpu_tensor_buf(tensors[i]), t_descs[i].align_offset,
+                                       global_ctx->get_tensor_staging_buf, i_offset, t_descs[i].staging_size);
+            i_offset += t_descs[i].staging_size;
+        }
+
+        wgpu::CommandBuffer commands = encoder.Finish();
+
+        // Submit the command buffer to the queue
+        global_ctx->queue.Submit(1, &commands);
+
+        // Map the staging buffer to read the data
+        ggml_backend_webgpu_map_buffer(global_ctx, global_ctx->get_tensor_staging_buf, wgpu::MapMode::Read, 0,
+                                       sum_chunk_size);
+        // Must specify size here since the staging buffer might be larger than the tensor size
+        const void * mapped_range = global_ctx->get_tensor_staging_buf.GetConstMappedRange(0, sum_chunk_size);
+
+        i_offset = 0;
+        for (size_t i = tsi; i <= tei; i++) {
+            // Copy the data from the mapped range to the output buffer
+            std::memcpy(data[i], (const void *) ((const char *) mapped_range + i_offset + t_descs[i].local_offset),
+                        sizes[i]);
+            i_offset += t_descs[i].staging_size;
+        }
+
+        tsi = tei + 1;
+        global_ctx->get_tensor_staging_buf.Unmap();
+    }
+
+    WEBGPU_CPU_PROFILE_TOTAL_END(get_tensor_n, buf_ctx->global_ctx);
+}
+
 static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
     WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_clear(" << buffer << ", " << (uint32_t) value << ")");
     WEBGPU_CPU_PROFILE_TOTAL_START(clear);
@@ -3967,8 +4212,8 @@ static ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
     /* .cpy_tensor      = */ NULL,  // TODO: optional, implement this
     /* .clear           = */ ggml_backend_webgpu_buffer_clear,
     /* .reset           = */ NULL,  // TODO: optional, think it coordinates with
-    /* .set_tensor_n    = */ NULL,
-    /* .get_tensor_n    = */ NULL,
+    /* .set_tensor_n    = */ ggml_backend_webgpu_buffer_set_tensor_n,
+    /* .get_tensor_n    = */ ggml_backend_webgpu_buffer_get_tensor_n,
     /* .cpy_tensor_n    = */ NULL,
 };
 
@@ -4006,12 +4251,10 @@ static size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_
     return dev_ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
 }
 
-// maxBufferSize might be larger, but you can't bind more than
-// maxStorageBufferBindingSize to a single binding.
 static size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
     ggml_backend_webgpu_device_context * dev_ctx =
         static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
-    return dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;
+    return dev_ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
 }
 
 static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
@@ -4176,20 +4419,19 @@ static ggml_guid_t ggml_backend_webgpu_guid(void) {
 }
 
 static void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
-    // we use the maximum workgroup size for the memset pipeline
-    size_t max_threads = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup *
-                         ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
-    // Size the bytes_per_thread so that the largest buffer size can be handled
-    ctx->capabilities.memset_bytes_per_thread =
-        CEIL_DIV(ctx->capabilities.limits.maxStorageBufferBindingSize, max_threads);
-    std::vector<wgpu::ConstantEntry> constants(2);
+    std::vector<wgpu::ConstantEntry> constants(1);
     constants[0].key     = "wg_size";
     constants[0].value   = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    constants[1].key     = "bytes_per_thread";
-    constants[1].value   = ctx->capabilities.memset_bytes_per_thread;
     ctx->memset_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_memset, "memset", constants);
 }
 
+static void ggml_webgpu_init_scatter_write_pipeline(webgpu_global_context & ctx) {
+    std::vector<wgpu::ConstantEntry> constants(1);
+    constants[0].key           = "wg_size";
+    constants[0].value         = ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+    ctx->set_tensor_n_pipeline = ggml_webgpu_create_pipeline(ctx->device, wgsl_set_tensor_n, "set_tensor_n", constants);
+}
+
 static void ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
     wgpu::RequestAdapterOptions options = {};
 
@@ -4333,6 +4575,7 @@ static void create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
     GGML_ASSERT(ctx->webgpu_global_ctx->device != nullptr);
 
     ggml_webgpu_init_memset_pipeline(ctx->webgpu_global_ctx);
+    ggml_webgpu_init_scatter_write_pipeline(ctx->webgpu_global_ctx);
     ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, ctx->webgpu_global_ctx->memset_params_buf,
                               WEBGPU_PARAMS_BUF_SIZE_BYTES, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                               "memset_params_buf");
@@ -4691,7 +4934,7 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
             supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
             break;
         case GGML_OP_ROPE:
-            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
+            supports_op = ggml_rope_get_truncate(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
             break;
         case GGML_OP_GLU:
             switch (ggml_get_glu_op(op)) {
@@ -4834,6 +5077,7 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
             supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguously_allocated(src0);
             break;
         case GGML_OP_SUM_ROWS:
+        case GGML_OP_MEAN:
             supports_op = op->type == GGML_TYPE_F32 && src0->type == op->type && ggml_is_contiguous_rows(src0);
             break;
         case GGML_OP_UPSCALE:
diff --git src/ggml-webgpu/wgsl-shaders/memset.wgsl src/ggml-webgpu/wgsl-shaders/memset.wgsl
index 194d2d6f..af5a23bc 100644
--- src/ggml-webgpu/wgsl-shaders/memset.wgsl
+++ src/ggml-webgpu/wgsl-shaders/memset.wgsl
@@ -11,30 +11,33 @@ struct Params {
 var<uniform> params: Params;
 
 override wg_size: u32;
-override bytes_per_thread: u32;
 
 @compute @workgroup_size(wg_size)
-fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
-    let i = gid.x * bytes_per_thread;
+fn main(@builtin(workgroup_id) wid: vec3<u32>,
+        @builtin(num_workgroups) num_wg: vec3<u32>,
+        @builtin(local_invocation_id) lid: vec3<u32>) {
+
+    let i = (wid.x + wid.y * num_wg.x) * wg_size + lid.x;
     let start = params.offset;
     let end = params.offset + params.size;
+    let id = (start >> 2u) + i;
+    let id_u32 = id * 4;
+
+    if (id_u32 >= end) {
+        return;
+    }
+
+    let lo = max(start, id_u32) - id_u32;
+    let hi = min(end, id_u32 + 4u) - id_u32;
+    var mask = 0xFFFFFFFFu << (8u * lo);
+    if (hi < 4u) {
+        mask &= (1u << (8u * hi)) - 1u;
+    }
 
-    for (var j: u32 = 0u; j < bytes_per_thread; j += 4) {
-        let byte_index = start + i + j;
-        if (byte_index + 4 <= end) {
-            output_buffer[byte_index >> 2] = params.value;
-        } else {
-            // Handle tail (unaligned)
-            for (var k: u32 = 0; k < 4; k++) {
-                let idx = byte_index + k;
-                if (idx < end) {
-                    let word_idx = idx >> 2;
-                    let bit_offset = (idx & 3) * 8u;
-                    let mask = ~(0xffu << bit_offset);
-                    let existing = output_buffer[word_idx];
-                    output_buffer[word_idx] = (existing & mask) | (params.value & (0xffu << bit_offset));
-                }
-            }
-        }
+    if (mask == 0xFFFFFFFFu) {
+        output_buffer[id] = params.value;
+    } else {
+        let existing = output_buffer[id];
+        output_buffer[id] = (existing & ~(mask)) | (params.value & mask);
     }
 }
diff --git src/ggml-webgpu/wgsl-shaders/set_tensor_n.wgsl src/ggml-webgpu/wgsl-shaders/set_tensor_n.wgsl
new file mode 100644
index 00000000..f9ba2641
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/set_tensor_n.wgsl
@@ -0,0 +1,42 @@
+@group(0) @binding(0)
+var<storage, read_write> output_buffer: array<atomic<u32>>;
+
+@group(0) @binding(1)
+var<storage, read> src_buffer: array<u32>;
+
+@group(0) @binding(2)
+var<storage, read> descs: array<vec4<u32>>;
+
+override wg_size: u32;
+
+@compute @workgroup_size(wg_size)
+fn main(@builtin(workgroup_id) wid: vec3<u32>,
+        @builtin(num_workgroups) num_wg: vec3<u32>,
+        @builtin(local_invocation_id) lid: vec3<u32>) {
+
+    let w = wid.y * num_wg.x + wid.x;
+
+    if (w >= arrayLength(&descs)) {
+        return;
+    }
+
+    let dst_i     = descs[w].x / 4u;
+    let src_i     = descs[w].y;
+    let lane    = descs[w].x & 3u;
+    let end     = lane + descs[w].z;
+    let n_words = (end + 3u) / 4u;
+
+    for (var i = lid.x; i < n_words; i += wg_size) {
+        // [lo, hi) is the byte range of this word that belongs to the descs
+        let lo    = select(0u, lane, i == 0u);
+        let hi    = min(end - i * 4u, 4u);
+        let mask  = select((1u << (hi * 8u)) - 1u, 0xffffffffu, hi == 4u) & (0xffffffffu << (lo * 8u));
+        let value = src_buffer[src_i + i];
+        if (mask == 0xffffffffu) {
+            atomicStore(&output_buffer[dst_i + i], value);
+        } else {
+            atomicAnd(&output_buffer[dst_i + i], ~mask);
+            atomicOr(&output_buffer[dst_i + i], value & mask);
+        }
+    }
+}
diff --git src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl
index 1ead3ac3..e4f17526 100644
--- src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl
+++ src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl
@@ -15,7 +15,8 @@ struct Params {
 
     ne0: u32,
     ne1: u32,
-    ne2: u32
+    ne2: u32,
+    nrows: u32
 };
 
 @group(0) @binding(2)
@@ -28,7 +29,11 @@ fn main(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(num_workgroups) num_wg: vec3<u32>,
         @builtin(local_invocation_id) lid: vec3<u32>) {
 
-    var i = wid.x + wid.y * num_wg.x;
+    let row = wid.x + wid.y * num_wg.x;
+    if (row >= params.nrows) {
+        return;
+    }
+    var i = row;
     let i3 = i / (params.ne2 * params.ne1);
     i = i % (params.ne2 * params.ne1);
     let i2 = i / params.ne1;
@@ -51,6 +56,10 @@ fn main(@builtin(workgroup_id) wid: vec3<u32>,
     }
 
     if (lid.x == 0) {
-        dst[params.offset_dst + wid.x] = shared_sum[0];
+#ifdef MEAN
+        dst[params.offset_dst + row] = shared_sum[0] / f32(params.ne0);
+#else
+        dst[params.offset_dst + row] = shared_sum[0];
+#endif
     }
 }
diff --git src/ggml.c src/ggml.c
index e8114810..f8cf75b3 100644
--- src/ggml.c
+++ src/ggml.c
@@ -4359,7 +4359,7 @@ static struct ggml_tensor * ggml_rope_impl(
 
     struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
 
-    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
+    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*no_trunc*/ 0, n_ctx_orig };
     memcpy(params +  5, &freq_base,    sizeof(float));
     memcpy(params +  6, &freq_scale,   sizeof(float));
     memcpy(params +  7, &ext_factor,   sizeof(float));
@@ -4532,11 +4532,15 @@ static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, fl
 }
 
 void ggml_rope_yarn_corr_dims(
-    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
+    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, bool truncate, float dims[2]
 ) {
     // start and end correction dims
-    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
-    float end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
+    float start = ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
+    float end   = ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);
+    if (truncate) {
+        start = floorf(start);
+        end   = ceilf(end);
+    }
     dims[0] = MAX(0, start);
     dims[1] = MIN(n_dims - 1, end);
 }
@@ -4597,6 +4601,19 @@ struct ggml_tensor * ggml_rope_set_offset(
     return a;
 }
 
+struct ggml_tensor * ggml_rope_set_truncate(
+        struct ggml_tensor * a,
+        bool                 truncate) {
+    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
+    ggml_set_op_params_i32(a, 3, !truncate);
+    return a;
+}
+
+bool ggml_rope_get_truncate(const struct ggml_tensor * a) {
+    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
+    return ggml_get_op_params_i32(a, 3) == 0;
+}
+
 static int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
     return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
 }
@@ -7237,7 +7254,6 @@ static void ggml_compute_backward(
                 //const int n_past = ((int32_t *) tensor->op_params)[0];
                 const int n_dims     = ((const int32_t *) tensor->op_params)[1];
                 const int mode       = ((const int32_t *) tensor->op_params)[2];
-                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                 const int n_ctx_orig = ((const int32_t *) tensor->op_params)[4];
                 float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                 int sections[4] = {0, 0, 0, 0};
@@ -7255,6 +7271,7 @@ static void ggml_compute_backward(
                         mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                     ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                         mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
+                ggml_rope_set_truncate(rope_back, ggml_rope_get_truncate(tensor));
                 ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
             }
             GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
