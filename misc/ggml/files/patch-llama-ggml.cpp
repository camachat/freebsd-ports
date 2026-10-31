diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index aba21524..72e961d5 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -1962,7 +1962,7 @@ static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggm
     const ggml_tensor * x_up = up->src[0];
     const ggml_tensor * y    = up->src[1];
 
-    if (up->op != GGML_OP_MUL_MAT || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
+    if ((up->op != GGML_OP_MUL_MAT && up->op != GGML_OP_MUL_MAT_ID) || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
         return false;
     }
 
@@ -2004,12 +2004,18 @@ static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_t
     const ggml_tensor * x_up = up->src[0];
     const ggml_tensor * y    = up->src[1];
     const int           cc   = ggml_cuda_info().devices[device].cc;
-    if (ggml_cuda_should_use_mmvq(x_up->type, cc, y->ne[1]) || !ggml_cuda_should_use_mmq(x_up->type, cc, y->ne[1], 0)) {
+
+    // For MUL_MAT_ID: same kernel choice as ggml_cuda_mul_mat_id, with the nominal tiles of all experts (the per-expert token counts are only known on the GPU).
+    const bool    is_id     = up->op == GGML_OP_MUL_MAT_ID;
+    const int64_t ncols     = is_id ? y->ne[2]    : y->ne[1];
+    const int64_t nchannels = is_id ? x_up->ne[2] : y->ne[2]*y->ne[3];
+    const bool    use_mmvq  = is_id ? ncols <= get_mmvq_mmid_max_batch(x_up->type, cc) : ggml_cuda_should_use_mmvq(x_up->type, cc, ncols);
+    if (use_mmvq || !ggml_cuda_should_use_mmq(x_up->type, cc, ncols, /*n_experts =*/ is_id ? x_up->ne[2] : 0)) {
         return false;
     }
 
     // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
-    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, x_up->ne[1], y->ne[1], y->ne[2]*y->ne[3], device);
+    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, x_up->ne[1], ncols, nchannels, device);
 }
 
 static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
diff --git src/ggml-cuda/mmq.cu src/ggml-cuda/mmq.cu
index 27558ec7..4af4f36e 100644
--- src/ggml-cuda/mmq.cu
+++ src/ggml-cuda/mmq.cu
@@ -170,10 +170,13 @@ void ggml_cuda_mul_mat_q(
 
     // Fused gate/up/GLU: src0 is the up weight, dst the GLU output.
     const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
+    ggml_cuda_mm_fusion_args_device fusion_device = {};
     if (fusion) {
-        // TODO MoE support
-        GGML_ASSERT(gate && !ids && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
+        GGML_ASSERT(gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
         GGML_ASSERT(gate->type == src0->type && ggml_are_same_shape(gate, src0) && ggml_are_same_stride(gate, src0));
+        fusion_device.gate      = gate->data;
+        fusion_device.glu_op    = fusion->glu_op;
+        fusion_device.glu_limit = fusion->glu_limit;
     }
 
     GGML_TENSOR_BINARY_OP_LOCALS;
@@ -286,13 +289,6 @@ void ggml_cuda_mul_mat_q(
                                 ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
         const int64_t s13 = ne12*s12;
 
-        ggml_cuda_mm_fusion_args_device fusion_device = {};
-        if (gate) {
-            fusion_device.gate      = gate->data;
-            fusion_device.glu_op    = fusion->glu_op;
-            fusion_device.glu_limit = fusion->glu_limit;
-        }
-
         const mmq_args args = {
             src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
             src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
@@ -378,7 +374,7 @@ void ggml_cuda_mul_mat_q(
         ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
         ne02, ne02, s02, s12, s2,
         ne03, ne13, s03, s13, s3,
-        ne12, J_best};
+        ne12, J_best, fusion_device};
 
     ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
 }
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index 97ad492e..807736e4 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -3010,6 +3010,35 @@ static void top_k_f32_sycl(
         return;
     }
 
+    // Short rows: one work-item per column and k work-group argmax rounds.
+    // Ties take the lower column.
+    const int wg = std::max(WARP_SIZE, next_power_of_2((int) ncols));
+    if (wg <= ggml_sycl_info().max_work_group_sizes[ctx.device]) {
+        main_stream->parallel_for(
+            sycl::nd_range<1>(sycl::range<1>(nrows * wg), sycl::range<1>(wg)),
+            [=](sycl::nd_item<1> item_ct1) {
+                const int64_t row = item_ct1.get_group(0);
+                const int     tid = item_ct1.get_local_id(0);
+                const auto    grp = item_ct1.get_group();
+
+                // NaN is never preferred, and every round still has a winner since k <= ncols
+                const float x     = tid < ncols ? src[row * ncols + tid] : -INFINITY;
+                const float val   = sycl::isnan(x) ? -INFINITY : x;
+                bool        taken = tid >= ncols;
+                for (int i = 0; i < k; i++) {
+                    const float cand   = taken ? -INFINITY : val;
+                    const float best   = sycl::reduce_over_group(grp, cand, sycl::maximum<float>());
+                    const int   winner = sycl::reduce_over_group(grp, !taken && cand == best ? tid : INT_MAX,
+                                                                 sycl::minimum<int>());
+                    if (tid == winner) {
+                        dst_indices[row * k + i] = tid;
+                        taken = true;
+                    }
+                }
+            });
+        return;
+    }
+
     const int block_size = 128;
 
     const sycl::range<1> block_dims(block_size);
