diff --git CMakeLists.txt CMakeLists.txt
index 2effd587..b9f7deb1 100644
--- CMakeLists.txt
+++ CMakeLists.txt
@@ -213,7 +213,7 @@ set   (GGML_CUDA_COMPRESSION_MODE "size" CACHE STRING
 set_property(CACHE GGML_CUDA_COMPRESSION_MODE PROPERTY STRINGS "none;speed;balance;size")
 
 option(GGML_HIP                             "ggml: use HIP"                                   OFF)
-option(GGML_HIP_GRAPHS                      "ggml: use HIP graph, experimental, slow"         OFF)
+option(GGML_HIP_GRAPHS                      "ggml: use HIP graph"                              ON)
 option(GGML_HIP_RCCL                        "ggml: use ROCm Collective Comm. Library"         OFF)
 option(GGML_HIP_NO_VMM                      "ggml: do not try to use HIP VMM"                 ON)
 option(GGML_HIP_ROCWMMA_FATTN               "ggml: enable rocWMMA for FlashAttention"         OFF)
diff --git src/CMakeLists.txt src/CMakeLists.txt
index 48fbe208..3e48860b 100644
--- src/CMakeLists.txt
+++ src/CMakeLists.txt
@@ -470,11 +470,10 @@ endforeach()
 
 target_link_libraries(ggml-base PRIVATE Threads::Threads)
 
-find_library(MATH_LIBRARY m)
-if (MATH_LIBRARY)
-    if (NOT WIN32 OR NOT DEFINED ENV{ONEAPI_ROOT})
-        target_link_libraries(ggml-base PRIVATE m)
-    endif()
+if (DEFINED MATH_LIBRARY)
+    target_link_libraries(ggml-base PRIVATE ${MATH_LIBRARY})
+elseif (NOT WIN32 AND NOT DEFINED ENV{ONEAPI_ROOT})
+    target_link_libraries(ggml-base PRIVATE m)
 endif()
 
 if (CMAKE_SYSTEM_NAME MATCHES "Android")
diff --git src/ggml-backend-meta.cpp src/ggml-backend-meta.cpp
index 6d22f342..41a61775 100644
--- src/ggml-backend-meta.cpp
+++ src/ggml-backend-meta.cpp
@@ -1205,40 +1205,57 @@ static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, gg
 
     if (split_state.n_segments != 1) {
         GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
-        GGML_ASSERT(offset == 0);
-        GGML_ASSERT(size == ggml_nbytes(tensor));
         GGML_ASSERT(tensor->ne[3] == 1);
+
         size_t offset_data = 0;
         std::vector<size_t> simple_offsets(n_bufs, 0);
         if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
             GGML_ASSERT(tensor->ne[2] == 1);
+
+            const size_t row_stride = tensor->nb[1];
+            GGML_ASSERT(offset % row_stride == 0);
+            GGML_ASSERT(size   % row_stride == 0);
+            const int64_t r_start = offset / row_stride;
+            const int64_t r_count = size   / row_stride;
+            GGML_ASSERT(r_start + r_count <= tensor->ne[1]);
+
             const int64_t blck_size = ggml_blck_size(tensor->type);
             for (size_t s = 0; s < split_state.n_segments; s++) {
                 for (size_t j = 0; j < n_bufs; j++) {
                     ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                     GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                     const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
-                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data, simple_offsets[j], nbytes,
-                        tensor->ne[1], simple_tensor->nb[1], tensor->nb[1]);
+                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
+                        simple_offsets[j] + r_start * simple_tensor->nb[1], nbytes,
+                        r_count, simple_tensor->nb[1], tensor->nb[1]);
                     offset_data       += nbytes;
                     simple_offsets[j] += nbytes;
                 }
             }
-            GGML_ASSERT(offset_data*tensor->ne[1] == size);
+            GGML_ASSERT(offset_data*r_count == size);
             return;
         }
         GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
+
+        const size_t row_stride = tensor->nb[2];
+        GGML_ASSERT(offset % row_stride == 0);
+        GGML_ASSERT(size   % row_stride == 0);
+        const int64_t r_start = offset / row_stride;
+        const int64_t r_count = size   / row_stride;
+        GGML_ASSERT(r_start + r_count <= tensor->ne[2]);
+
         for (size_t s = 0; s < split_state.n_segments; s++) {
             for (size_t j = 0; j < n_bufs; j++) {
                 ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                 const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
-                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data, simple_offsets[j], nbytes,
-                    tensor->ne[2], simple_tensor->nb[2], tensor->nb[2]);
+                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
+                    simple_offsets[j] + r_start * simple_tensor->nb[2], nbytes,
+                    r_count, simple_tensor->nb[2], tensor->nb[2]);
                 offset_data       += nbytes;
                 simple_offsets[j] += nbytes;
             }
         }
-        GGML_ASSERT(offset_data*tensor->ne[2] == size);
+        GGML_ASSERT(offset_data*r_count == size);
         return;
     }
 
@@ -1295,40 +1312,57 @@ static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, co
 
     if (split_state.n_segments != 1) {
         GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
-        GGML_ASSERT(offset == 0);
-        GGML_ASSERT(size == ggml_nbytes(tensor));
         GGML_ASSERT(tensor->ne[3] == 1);
+
         size_t offset_data = 0;
         std::vector<size_t> simple_offsets(n_bufs, 0);
         if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
             GGML_ASSERT(tensor->ne[2] == 1);
+
+            const size_t row_stride = tensor->nb[1];
+            GGML_ASSERT(offset % row_stride == 0);
+            GGML_ASSERT(size   % row_stride == 0);
+            const int64_t r_start = offset / row_stride;
+            const int64_t r_count = size   / row_stride;
+            GGML_ASSERT(r_start + r_count <= tensor->ne[1]);
+
             const int64_t blck_size = ggml_blck_size(tensor->type);
             for (size_t s = 0; s < split_state.n_segments; s++) {
                 for (size_t j = 0; j < n_bufs; j++) {
                     const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                     GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                     const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
-                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data, simple_offsets[j], nbytes,
-                        tensor->ne[1], simple_tensor->nb[1], tensor->nb[1]);
+                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
+                        simple_offsets[j] + r_start * simple_tensor->nb[1], nbytes,
+                        r_count, simple_tensor->nb[1], tensor->nb[1]);
                     offset_data       += nbytes;
                     simple_offsets[j] += nbytes;
                 }
             }
-            GGML_ASSERT(offset_data*tensor->ne[1] == size);
+            GGML_ASSERT(offset_data*r_count == size);
             return;
         }
         GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
+
+        const size_t row_stride = tensor->nb[2];
+        GGML_ASSERT(offset % row_stride == 0);
+        GGML_ASSERT(size   % row_stride == 0);
+        const int64_t r_start = offset / row_stride;
+        const int64_t r_count = size   / row_stride;
+        GGML_ASSERT(r_start + r_count <= tensor->ne[2]);
+
         for (size_t s = 0; s < split_state.n_segments; s++) {
             for (size_t j = 0; j < n_bufs; j++) {
                 const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                 const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
-                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data, simple_offsets[j], nbytes,
-                    tensor->ne[2], simple_tensor->nb[2], tensor->nb[2]);
+                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
+                    simple_offsets[j] + r_start * simple_tensor->nb[2], nbytes,
+                    r_count, simple_tensor->nb[2], tensor->nb[2]);
                 offset_data       += nbytes;
                 simple_offsets[j] += nbytes;
             }
         }
-        GGML_ASSERT(offset_data*tensor->ne[2] == size);
+        GGML_ASSERT(offset_data*r_count == size);
         return;
     }
 
diff --git src/ggml-backend-reg.cpp src/ggml-backend-reg.cpp
index 05871092..8165ae2c 100644
--- src/ggml-backend-reg.cpp
+++ src/ggml-backend-reg.cpp
@@ -181,6 +181,12 @@ struct ggml_backend_registry {
             return;
         }
 
+        for (auto & entry : backends) {
+            if (entry.reg == reg) {
+                return;
+            }
+        }
+
 #ifndef NDEBUG
         GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
             __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
@@ -192,6 +198,12 @@ struct ggml_backend_registry {
     }
 
     void register_device(ggml_backend_dev_t device) {
+        for (auto & dev : devices) {
+            if (dev == device) {
+                return;
+            }
+        }
+
 #ifndef NDEBUG
         GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
 #endif
diff --git src/ggml-cann/aclnn_ops.cpp src/ggml-cann/aclnn_ops.cpp
index a950475f..2dc0f409 100644
--- src/ggml-cann/aclnn_ops.cpp
+++ src/ggml-cann/aclnn_ops.cpp
@@ -25,6 +25,7 @@
 #include "ggml-impl.h"
 #include "ggml.h"
 
+
 #include <aclnnop/aclnn_add.h>
 #include <aclnnop/aclnn_add_rms_norm.h>
 #include <aclnnop/aclnn_addcdiv.h>
@@ -45,7 +46,9 @@
 #include <aclnnop/aclnn_fused_infer_attention_score_v2.h>
 #include <aclnnop/aclnn_ger.h>
 #include <aclnnop/aclnn_group_norm.h>
+#include <aclnnop/aclnn_gather_v2.h>
 #include <aclnnop/aclnn_grouped_matmul_v3.h>
+#include <aclnnop/aclnn_scatter.h>
 #include <aclnnop/aclnn_gt_scalar.h>
 #include <aclnnop/aclnn_im2col.h>
 #include <aclnnop/aclnn_index_copy.h>
@@ -62,6 +65,7 @@
 #include <aclnnop/aclnn_permute.h>
 #include <aclnnop/aclnn_pow.h>
 #include <aclnnop/aclnn_pow_tensor_tensor.h>
+#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>
 #include <aclnnop/aclnn_reduce_sum.h>
 #include <aclnnop/aclnn_reflection_pad1d.h>
 #include <aclnnop/aclnn_repeat.h>
@@ -69,11 +73,15 @@
 #include <aclnnop/aclnn_rms_norm.h>
 #include <aclnnop/aclnn_roll.h>
 #include <aclnnop/aclnn_softmax.h>
+#include <aclnnop/aclnn_softmax_cross_entropy_with_logits.h>
 #include <aclnnop/aclnn_sub.h>
 #include <aclnnop/aclnn_sum.h>
 #include <aclnnop/aclnn_threshold.h>
 #include <aclnnop/aclnn_tril.h>
+#include <aclnnop/aclnn_triangular_solve.h>
 #include <aclnnop/aclnn_triu.h>
+#include <aclnnop/aclnn_logical_not.h>
+#include <aclnnop/aclnn_masked_fill_scalar.h>
 #include <aclnnop/aclnn_upsample_nearest_2d.h>
 #include <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
 #include <aclnnop/aclnn_zero.h>
@@ -151,6 +159,107 @@ void ggml_cann_op_unary_gated(std::function<void(ggml_backend_cann_context &, ac
     GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_src1.get());
 }
 
+// Fused SwiGLU using aclnnSwiGlu: splits input along innermost dim, applies
+// SiLU to left half, multiplies by right half.
+//
+// Falls back to the generic two-kernel path when src[1] != nullptr (two
+// independent halves) or swapped != 0 (reversed activation order), as
+// aclnnSwiGlu only handles the single interleaved tensor in standard order.
+//
+// CANN tiling for SwiGlu requires (storageShapeDim + viewDims) to be even.
+// aclCreateTensor always uses storageShapeDim=1, so viewDims must be odd.
+// We use a 3D view (1+3=4, even) to satisfy this constraint while preserving
+// correct split semantics along the innermost (ne[0]) dimension.
+void ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    auto silu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
+        GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_src, acl_dst);
+    };
+
+    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
+    if (dst->src[1] != nullptr || swapped != 0) {
+        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
+        return;
+    }
+
+    // aclnnSwiGlu requires the split dim (src->ne[0]) to be even; fall back otherwise.
+    if (dst->src[0]->ne[0] % 2 != 0) {
+        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
+        return;
+    }
+
+    ggml_tensor * src0 = dst->src[0];
+    size_t elem_size = ggml_element_size(src0);
+
+    // src0 GGML: [2*ne0, ne1, ne2, ne3] → 3D view [2*ne0, ne1, ne2*ne3]
+    // CANN reversed: [ne2*ne3, ne1, 2*ne0], split along CANN dim 2 (last).
+    int64_t ne0_x2   = src0->ne[0];
+    int64_t ne1      = src0->ne[1];
+    int64_t ne23     = src0->ne[2] * src0->ne[3];
+    int64_t src3d_ne[] = { ne0_x2, ne1, ne23 };
+    size_t  src3d_nb[] = { (size_t)src0->nb[0], (size_t)src0->nb[1], (size_t)src0->nb[2] };
+    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->data, ggml_cann_type_mapping(src0->type),
+                                                     elem_size, src3d_ne, src3d_nb, 3);
+
+    // dst GGML: [ne0, ne1, ne2, ne3] → 3D view [ne0, ne1, ne2*ne3]
+    int64_t ne0      = dst->ne[0];
+    int64_t dst3d_ne[] = { ne0, ne1, ne23 };
+    size_t  dst3d_nb[] = { (size_t)dst->nb[0], (size_t)dst->nb[1], (size_t)dst->nb[2] };
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->data, ggml_cann_type_mapping(dst->type),
+                                                     elem_size, dst3d_ne, dst3d_nb, 3);
+
+    // CANN tensor [ne23, ne1, 2*ne0]: split along CANN dim 2 (last) = 2*ne0.
+    GGML_CANN_CALL_ACLNN_OP(ctx, SwiGlu, acl_src.get(), (int64_t)2, acl_dst.get());
+}
+
+// Fused GeGLU using aclnnGeGluV3: splits input along ne[0] (CANN last dim),
+// activates the LEFT half with GELU, multiplies by right half.
+// approximate: 0=tanh, 1=none(erf). activateLeft=true matches GGML convention.
+// outGelu is a required-but-discard output buffer.
+//
+// Falls back to the generic two-kernel path when src[1] != nullptr (two
+// independent halves) or swapped != 0 (reversed activation order), as
+// aclnnGeGluV3 only handles the single interleaved tensor in standard order.
+void ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate) {
+    auto gelu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
+        GGML_CANN_CALL_ACLNN_OP(ctx, Gelu, acl_src, acl_dst);
+    };
+
+    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
+    if (dst->src[1] != nullptr || swapped != 0) {
+        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
+        return;
+    }
+
+    // aclnnGeGluV3 requires the split dim (src->ne[0]) to be even; fall back otherwise.
+    if (dst->src[0]->ne[0] % 2 != 0) {
+        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
+        return;
+    }
+
+    ggml_tensor * src0 = dst->src[0];
+    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
+
+    // Allocate a temporary buffer for the required outGelu output (same shape as dst).
+    // Build contiguous strides since the pool allocation is a fresh buffer.
+    size_t  elem_size    = ggml_element_size(dst);
+    int64_t ne[GGML_MAX_DIMS] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
+    size_t  nb[GGML_MAX_DIMS];
+    nb[0] = elem_size;
+    for (int i = 1; i < GGML_MAX_DIMS; i++) {
+        nb[i] = nb[i - 1] * ne[i - 1];
+    }
+    size_t gelu_out_size = nb[GGML_MAX_DIMS - 1] * ne[GGML_MAX_DIMS - 1];
+    ggml_cann_pool_alloc gelu_out_alloc(ctx.pool(), gelu_out_size);
+
+    acl_tensor_ptr acl_gelu_out = ggml_cann_create_tensor(
+        gelu_out_alloc.get(), ggml_cann_type_mapping(dst->type), elem_size, ne, nb, GGML_MAX_DIMS);
+    // V3 adds activateLeft param; true → Gelu(left)*right, matching GGML convention.
+    // GGML dim 0 → CANN last dim (index GGML_MAX_DIMS-1 = 3 for 4D tensor).
+    GGML_CANN_CALL_ACLNN_OP(ctx, GeGluV3, acl_src.get(), (int64_t)(GGML_MAX_DIMS - 1), approximate, true,
+                             acl_dst.get(), acl_gelu_out.get());
+}
+
 /**
  * @brief Repeats elements of a tensor along each dimension according to the
  * specified repeat array.
@@ -445,28 +554,33 @@ void ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes);
     void *               buffer = temp_buffer_allocator.get();
 
-    int64_t div_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
-    size_t  div_nb[GGML_MAX_DIMS];
-    div_nb[0] = sizeof(float);
+    int64_t norm_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
+    size_t  norm_nb[GGML_MAX_DIMS];
+    norm_nb[0] = sizeof(float);
     for (int i = 1; i < GGML_MAX_DIMS; ++i) {
-        div_nb[i] = div_nb[i - 1] * div_ne[i - 1];
+        norm_nb[i] = norm_nb[i - 1] * norm_ne[i - 1];
     }
-    acl_tensor_ptr acl_div = ggml_cann_create_tensor(buffer, ACL_FLOAT, type_size, div_ne, div_nb, GGML_MAX_DIMS);
+    acl_tensor_ptr acl_norm = ggml_cann_create_tensor(buffer, ACL_FLOAT, sizeof(float), norm_ne, norm_nb, GGML_MAX_DIMS);
 
     std::vector<int64_t> norm_dims  = { 3 };
     acl_int_array_ptr    dims_array = ggml_cann_create_int_array(norm_dims.data(), norm_dims.size());
 
     float          p_value  = 2.0f;
     acl_scalar_ptr p_scalar = ggml_cann_create_scalar(&p_value, aclDataType::ACL_FLOAT);
-    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), true, acl_div.get());
+    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), true, acl_norm.get());
+
+    ggml_cann_pool_alloc clamp_buffer_allocator(ctx.pool());
+    acl_tensor_ptr       acl_clamped;
 
-    // Clamp norm to at least eps: scale = 1/fmaxf(norm, eps)
-    acl_scalar_ptr acl_min = ggml_cann_create_scalar(&eps, aclDataType::ACL_FLOAT);
-    float          flt_max = FLT_MAX;
-    acl_scalar_ptr acl_max = ggml_cann_create_scalar(&flt_max, aclDataType::ACL_FLOAT);
-    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_div.get(), acl_min.get(), acl_max.get(), acl_div.get());
+    if (eps > 0.0f) {
+        void *         clamp_buf  = clamp_buffer_allocator.alloc(n_bytes);
+        acl_clamped               = ggml_cann_create_tensor(clamp_buf, ACL_FLOAT, sizeof(float), norm_ne, norm_nb, GGML_MAX_DIMS);
+        acl_scalar_ptr eps_scalar = ggml_cann_create_scalar(&eps, aclDataType::ACL_FLOAT);
+        GGML_CANN_CALL_ACLNN_OP(ctx, ClampMin, acl_norm.get(), eps_scalar.get(), acl_clamped.get());
+    }
 
-    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div.get(), acl_dst.get());
+    aclTensor * acl_div_input = acl_clamped ? acl_clamped.get() : acl_norm.get();
+    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div_input, acl_dst.get());
 }
 
 void ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
@@ -482,56 +596,30 @@ void ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor *
     logits_nb[1]              = logits_nb[0] * logits_ne[0];
     acl_tensor_ptr acl_logits = ggml_cann_create_tensor(src0->data, ACL_FLOAT, sizeof(float), logits_ne, logits_nb, 2);
 
-    size_t               log_softmax_type_size = sizeof(float);
-    int64_t              log_softmax_n_bytes   = nr * nc * log_softmax_type_size;
-    ggml_cann_pool_alloc log_softmax_allocator(ctx.pool(), log_softmax_n_bytes);
-    void *               log_softmax_buffer = log_softmax_allocator.get();
-
-    int64_t log_softmax_ne[] = { nc, nr };
-    size_t  log_softmax_nb[2];
-    log_softmax_nb[0]              = log_softmax_type_size;
-    log_softmax_nb[1]              = log_softmax_nb[0] * log_softmax_ne[0];
-    acl_tensor_ptr acl_log_softmax = ggml_cann_create_tensor(log_softmax_buffer, ACL_FLOAT, log_softmax_type_size,
-                                                             log_softmax_ne, log_softmax_nb, 2);
-
-    GGML_CANN_CALL_ACLNN_OP(ctx, LogSoftmax, acl_logits.get(), 1, acl_log_softmax.get());
-
     int64_t labels_ne[] = { nc, nr };
     size_t  labels_nb[2];
     labels_nb[0]              = ggml_type_size(src1->type);
     labels_nb[1]              = labels_nb[0] * labels_ne[0];
     acl_tensor_ptr acl_labels = ggml_cann_create_tensor(src1->data, ACL_FLOAT, sizeof(float), labels_ne, labels_nb, 2);
 
-    size_t               mul_type_size = sizeof(float);
-    int64_t              mul_n_bytes   = nr * nc * mul_type_size;
-    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_n_bytes);
-    void *               mul_buffer = mul_allocator.get();
-
-    int64_t mul_ne[] = { nc, nr };
-    size_t  mul_nb[2];
-    mul_nb[0]                     = mul_type_size;
-    mul_nb[1]                     = mul_nb[0] * mul_ne[0];
-    acl_tensor_ptr acl_mul_result = ggml_cann_create_tensor(mul_buffer, ACL_FLOAT, mul_type_size, mul_ne, mul_nb, 2);
-
-    GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_log_softmax.get(), acl_labels.get(), acl_mul_result.get());
+    size_t               loss_per_sample_type_size = sizeof(float);
+    int64_t              loss_per_sample_n_bytes   = nr * loss_per_sample_type_size;
+    ggml_cann_pool_alloc loss_per_sample_allocator(ctx.pool(), loss_per_sample_n_bytes);
+    void *               loss_per_sample_buffer = loss_per_sample_allocator.get();
 
-    size_t               sum_per_sample_type_size = sizeof(float);
-    int64_t              sum_per_sample_n_bytes   = nr * sum_per_sample_type_size;
-    ggml_cann_pool_alloc sum_per_sample_allocator(ctx.pool(), sum_per_sample_n_bytes);
-    void *               sum_per_sample_buffer = sum_per_sample_allocator.get();
+    int64_t loss_per_sample_ne[] = { nr };
+    size_t  loss_per_sample_nb[1];
+    loss_per_sample_nb[0] = loss_per_sample_type_size;
+    acl_tensor_ptr acl_loss_per_sample = ggml_cann_create_tensor(
+        loss_per_sample_buffer, ACL_FLOAT, loss_per_sample_type_size, loss_per_sample_ne, loss_per_sample_nb, 1);
 
-    int64_t sum_per_sample_ne[] = { nr };
-    size_t  sum_per_sample_nb[1];
-    sum_per_sample_nb[0]              = sum_per_sample_type_size;
-    acl_tensor_ptr acl_sum_per_sample = ggml_cann_create_tensor(
-        sum_per_sample_buffer, ACL_FLOAT, sum_per_sample_type_size, sum_per_sample_ne, sum_per_sample_nb, 1);
+    size_t               backprop_n_bytes = nr * nc * sizeof(float);
+    ggml_cann_pool_alloc backprop_allocator(ctx.pool(), backprop_n_bytes);
+    void *               backprop_buffer = backprop_allocator.get();
+    acl_tensor_ptr acl_backprop = ggml_cann_create_tensor(backprop_buffer, ACL_FLOAT, sizeof(float), logits_ne, logits_nb, 2);
 
-    std::vector<int64_t> sum_dims   = { 1 };
-    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(sum_dims.data(), sum_dims.size());
-    bool                 keep_dims  = false;
-
-    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_mul_result.get(), dims_array.get(), keep_dims, ACL_FLOAT,
-                            acl_sum_per_sample.get());
+    GGML_CANN_CALL_ACLNN_OP(ctx, SoftmaxCrossEntropyWithLogits, acl_logits.get(), acl_labels.get(),
+                            acl_loss_per_sample.get(), acl_backprop.get());
 
     size_t               total_sum_type_size = sizeof(float);
     int64_t              total_sum_n_bytes   = 1 * total_sum_type_size;
@@ -547,11 +635,12 @@ void ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor *
 
     std::vector<int64_t> total_sum_dims    = { 0 };
     acl_int_array_ptr total_sum_dims_array = ggml_cann_create_int_array(total_sum_dims.data(), total_sum_dims.size());
+    bool              keep_dims            = false;
 
-    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_sum_per_sample.get(), total_sum_dims_array.get(), keep_dims, ACL_FLOAT,
+    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_loss_per_sample.get(), total_sum_dims_array.get(), keep_dims, ACL_FLOAT,
                             acl_total_sum.get());
 
-    float          value        = -1.0f / static_cast<float>(nr);
+    float          value        = 1.0f / static_cast<float>(nr);
     acl_scalar_ptr scale_factor = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
     acl_tensor_ptr acl_dst =
         ggml_cann_create_tensor(dst->data, ACL_FLOAT, sizeof(float), total_sum_ne, total_sum_nb, 1);
@@ -589,6 +678,33 @@ void ggml_cann_group_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
                             acl_mean_out.get(), acl_rstd_out.get());
 }
 
+void ggml_cann_set(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    ggml_tensor * src0 = dst->src[0];
+    ggml_tensor * src1 = dst->src[1];
+
+    size_t nb1     = ((int32_t *) dst->op_params)[0];
+    size_t nb2     = ((int32_t *) dst->op_params)[1];
+    size_t nb3     = ((int32_t *) dst->op_params)[2];
+    size_t offset  = ((int32_t *) dst->op_params)[3];
+    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];
+
+    size_t param_nb[] = { ggml_element_size(src0), nb1, nb2, nb3 };
+
+    // Create a view of dst at the target offset with src1's dimensions
+    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
+    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);
+
+    if (!inplace) {
+        // First copy src0 to dst entirely
+        size_t cpy_size = ggml_nbytes(dst);
+        ACL_CHECK(
+            aclrtMemcpyAsync(dst->data, cpy_size, src0->data, cpy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
+    }
+
+    // Copy src1 into the target region of dst
+    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src1.get());
+}
+
 void ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     ggml_tensor * src0 = dst->src[0];
     ggml_tensor * src1 = dst->src[1];
@@ -652,6 +768,113 @@ void ggml_cann_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     aclnn_reduce_sum(ctx, dst, reduce_dims, 4);
 }
 
+void ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    ggml_tensor * src = dst->src[0];
+    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
+    // GGML cumsum operates along dim 0 (innermost / ne[0]).
+    // ggml_cann_create_tensor reverses dimensions to [ne3,ne2,ne1,ne0],
+    // so GGML dim 0 maps to CANN dim 3 (the last dim of the 4-D tensor).
+    GGML_CANN_CALL_ACLNN_OP(ctx, Cumsum, acl_src.get(), (int64_t)3,
+                            ggml_cann_type_mapping(dst->type), acl_dst.get());
+}
+
+void ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    ggml_tensor * src0 = dst->src[0];  // A: [N, N, B2, B3] lower triangular
+    ggml_tensor * src1 = dst->src[1];  // B: [K, N, B2, B3]
+
+    acl_tensor_ptr acl_a = ggml_cann_create_tensor(src0);
+    acl_tensor_ptr acl_b = ggml_cann_create_tensor(src1);
+    acl_tensor_ptr acl_x = ggml_cann_create_tensor(dst);
+
+    // mOut: triangular copy of A (required output), same shape as A.
+    const size_t a_bytes = ggml_nbytes(src0);
+    ggml_cann_pool_alloc m_alloc(ctx.pool(), a_bytes);
+    acl_tensor_ptr acl_m = ggml_cann_create_tensor(
+        m_alloc.get(), ggml_cann_type_mapping(src0->type),
+        ggml_type_size(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);
+
+    // Solve AX = B: upper=false (lower tri), transpose=false, unitriangular=false.
+    GGML_CANN_CALL_ACLNN_OP(ctx, TriangularSolve,
+        acl_b.get(), acl_a.get(), false, false, false,
+        acl_x.get(), acl_m.get());
+}
+
+void ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    ggml_tensor * src = dst->src[0];
+
+    GGML_ASSERT(src->ne[1] == 1);
+
+    const int64_t N       = src->ne[0];
+    const int64_t n_batch = src->ne[2] * src->ne[3];
+    const size_t  nb_f32  = sizeof(float);
+
+    // Fill dst with zeros.
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
+    {
+        float          zero = 0.0f;
+        acl_scalar_ptr acl_zero = ggml_cann_create_scalar(&zero, ACL_FLOAT);
+        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_zero.get());
+    }
+
+    // Copy src vector onto the diagonal of dst via strided views.
+    // src viewed as [N, n_batch], contiguous strides.
+    int64_t ne_vec[2]      = { N, n_batch };
+    size_t  nb_src_vec[2]  = { nb_f32, N * nb_f32 };
+    // dst diagonal view: stride (N+1)*4 steps along the diagonal.
+    size_t  nb_dst_diag[2] = { (N + 1) * nb_f32, N * N * nb_f32 };
+
+    acl_tensor_ptr acl_src_vec  = ggml_cann_create_tensor(src->data, ACL_FLOAT, nb_f32, ne_vec, nb_src_vec, 2);
+    acl_tensor_ptr acl_dst_diag = ggml_cann_create_tensor(dst->data, ACL_FLOAT, nb_f32, ne_vec, nb_dst_diag, 2);
+
+    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst_diag.get(), acl_src_vec.get());
+}
+
+void ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    float c = ggml_get_op_params_f32(dst, 0);
+
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
+    acl_scalar_ptr acl_c   = ggml_cann_create_scalar(&c, ACL_FLOAT);
+    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_c.get());
+}
+
+void ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    ggml_tensor * src = dst->src[0];
+
+    const int64_t S       = src->ne[0];
+    const int64_t n_batch = src->ne[2] * src->ne[3];
+    const size_t  nb_f32  = sizeof(float);
+
+    int64_t ne3d[3] = { S, S, n_batch };
+    size_t  nb3d[3] = { nb_f32, S * nb_f32, S * S * nb_f32 };
+
+    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);
+
+    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src->data, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->data, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);
+
+    switch (ttype) {
+        case GGML_TRI_TYPE_LOWER:
+            // Tril(-1): preserve row > col (strict lower), zero upper + diagonal.
+            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)-1, acl_dst.get());
+            break;
+        case GGML_TRI_TYPE_UPPER_DIAG:
+            // Triu(0): preserve row <= col (upper + diagonal), zero strict lower.
+            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)0, acl_dst.get());
+            break;
+        case GGML_TRI_TYPE_UPPER:
+            // Triu(1): preserve row < col (strict upper), zero lower + diagonal.
+            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)1, acl_dst.get());
+            break;
+        case GGML_TRI_TYPE_LOWER_DIAG:
+            // Tril(0): preserve row >= col (lower + diagonal), zero strict upper.
+            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)0, acl_dst.get());
+            break;
+        default:
+            GGML_ABORT("unsupported tri type");
+    }
+}
+
 void ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     ggml_tensor *  src     = dst->src[0];
     acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
@@ -1695,152 +1918,90 @@ void ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     aclnn_softmax(ctx, softmax_tensor.get(), 3, acl_dst.get());
 }
 
-/**
- * @brief Performs index select operation on a 4D tensor using the CANN backend.
- *
- * This function applies the `IndexSelect` operation along a specific dimension
- * of the source tensor (`src_buffer`) using the indices from the index tensor (`index`).
- * It iterates over the last two dimensions of the source tensor, creates the corresponding
- * CANN tensors for the source, index, and output slices, and executes the `IndexSelect`
- * operation for each slice.
- *
- * @param ctx The context for CANN backend operations.
- * @param src_buffer The source buffer containing the 4D input tensor data.
- * @param src_ne The dimensions of the source tensor.
- * @param src_nb The strides (byte offsets) of the source tensor.
- * @param dst_buffer The destination buffer where the output tensor data will be written.
- * @param dst_ne The dimensions of the destination tensor.
- * @param dst_nb The strides (byte offsets) of the destination tensor.
- * @param index The index tensor specifying the indices to select from the source tensor.
- * @param type The data type of the source and destination tensors.
- */
-static void aclnn_index_select_4d(ggml_backend_cann_context & ctx,
-                                  void *                      src_buffer,
-                                  int64_t *                   src_ne,
-                                  size_t *                    src_nb,
-                                  void *                      dst_buffer,
-                                  int64_t *                   dst_ne,
-                                  size_t *                    dst_nb,
-                                  ggml_tensor *               index,
-                                  ggml_type                   type) {
-    for (int64_t i = 0; i < src_ne[3]; i++) {
-        for (int64_t j = 0; j < src_ne[2]; j++) {
-            // src
-            acl_tensor_ptr acl_src_tensor =
-                ggml_cann_create_tensor((char *) src_buffer + i * src_nb[3] + j * src_nb[2],
-                                        ggml_cann_type_mapping(type), ggml_type_size(type), src_ne, src_nb, 2);
-
-            // index
-            acl_tensor_ptr acl_index = ggml_cann_create_tensor(
-                (char *) index->data + (i % index->ne[2]) * index->nb[2] + (j % index->ne[1]) * index->nb[1],
-                ggml_cann_type_mapping(index->type), ggml_element_size(index), index->ne, index->nb, 1);
-
-            // out
-            acl_tensor_ptr acl_out =
-                ggml_cann_create_tensor((char *) dst_buffer + i * dst_nb[3] + j * dst_nb[2],
-                                        ggml_cann_type_mapping(type), ggml_type_size(type), dst_ne, dst_nb, 2);
-            GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, acl_src_tensor.get(), 0, acl_index.get(), acl_out.get());
-        }
-    }
-}
-
-/**
- * @brief Performs inplace index copy operation on a 4D tensor using the CANN backend.
- *
- * This function applies the `IndexCopy` operation along a specific dimension of the
- * destination tensor (`dst_buffer`) by copying elements from the source tensor (`src_buffer`)
- * to positions specified by the index tensor (`index`).
- * It iterates over the last two dimensions of the tensors, creates the corresponding
- * CANN tensors for source, index, and destination slices, and performs the index copy
- * operation for each slice.
- *
- * @param ctx The context for CANN backend operations.
- * @param src_buffer The source buffer containing the 4D input tensor data to be copied.
- * @param src_ne The dimensions of the source tensor.
- * @param src_nb The strides (byte offsets) of the source tensor.
- * @param dst_buffer The destination buffer where values will be copied to.
- * @param dst_ne The dimensions of the destination tensor.
- * @param dst_nb The strides (byte offsets) of the destination tensor.
- * @param index The index tensor specifying target positions in the destination tensor.
- * @param type The data type of the source and destination tensors.
- */
-static void aclnn_index_copy_4d(ggml_backend_cann_context & ctx,
-                                void *                      src_buffer,
-                                int64_t *                   src_ne,
-                                size_t *                    src_nb,
-                                void *                      dst_buffer,
-                                int64_t *                   dst_ne,
-                                size_t *                    dst_nb,
-                                ggml_tensor *               index,
-                                ggml_type                   type) {
-    for (int64_t i = 0; i < src_ne[3]; i++) {
-        for (int64_t j = 0; j < src_ne[2]; j++) {
-            // src
-            acl_tensor_ptr acl_src_tensor =
-                ggml_cann_create_tensor((char *) src_buffer + i * src_nb[3] + j * src_nb[2],
-                                        ggml_cann_type_mapping(type), ggml_type_size(type), src_ne, src_nb, 2);
-
-            // index
-            acl_tensor_ptr acl_index = ggml_cann_create_tensor(
-                (char *) index->data + (i % index->ne[2]) * index->nb[2] + (j % index->ne[1]) * index->nb[1],
-                ggml_cann_type_mapping(index->type), ggml_element_size(index), index->ne, index->nb, 1);
-
-            // out
-            acl_tensor_ptr acl_out =
-                ggml_cann_create_tensor((char *) dst_buffer + i * dst_nb[3] + j * dst_nb[2],
-                                        ggml_cann_type_mapping(type), ggml_type_size(type), dst_ne, dst_nb, 2);
-            GGML_CANN_CALL_ACLNN_OP(ctx, InplaceIndexCopy, acl_out.get(), 0, acl_index.get(), acl_src_tensor.get());
-        }
-    }
-}
 
 void ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
-    ggml_tensor * src0 = dst->src[0];  // src
+    ggml_tensor * src0 = dst->src[0];  // weight
     ggml_tensor * src1 = dst->src[1];  // index
 
     GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16
                 || dst->type == GGML_TYPE_BF16);
 
+    // n_idx: number of row indices per (i2, i3) batch slice.
+    // ggml guarantees: src0->ne[2] == src1->ne[1], src0->ne[3] == src1->ne[2], src1->ne[3] == 1.
+    const int64_t n_idx = src1->ne[0];
+
+    // Gather all (i2, i3) batch slices from src into dst.
+    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0].
+    // GatherV2 with dim=0 gathers along ACL dim-0 == ggml ne[1] (the vocabulary / row axis).
+    // nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
+    //     nb[2..3] for computing per-batch-slice base pointer offsets).
+    auto gather_batched = [&](void * src_base, aclDataType acl_type, size_t type_size,
+                              const size_t * nb) {
+        int64_t src_ne[2]  = { src0->ne[0], src0->ne[1] };
+        size_t  src_nb_2d[2] = { nb[0], nb[1] };
+        int64_t dst_ne[2]  = { src0->ne[0], n_idx };
+        size_t  dst_nb_2d[2] = { dst->nb[0], dst->nb[1] };
+        int64_t idx_ne[1]  = { n_idx };
+        size_t  idx_nb[1]  = { (size_t)ggml_element_size(src1) };
+
+        for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
+            for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
+                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
+                    (char *)src_base + i3 * nb[3] + i2 * nb[2],
+                    acl_type, type_size, src_ne, src_nb_2d, 2);
+                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
+                    (char *)src1->data + i3 * src1->nb[2] + i2 * src1->nb[1],
+                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_size(src1),
+                    idx_ne, idx_nb, 1);
+                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
+                    (char *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2],
+                    acl_type, type_size, dst_ne, dst_nb_2d, 2);
+                GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_src.get(), 0, acl_idx.get(), acl_dst.get());
+            }
+        }
+    };
+
     switch (src0->type) {
         case GGML_TYPE_BF16:
         case GGML_TYPE_F16:
         case GGML_TYPE_F32:
             if (src0->type == dst->type) {
-                aclnn_index_select_4d(ctx, src0->data, src0->ne, src0->nb, dst->data, dst->ne, dst->nb, src1,
-                                      dst->type);
+                gather_batched(src0->data,
+                               ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
+                               src0->nb);
             } else {
-                acl_tensor_ptr       acl_src0 = ggml_cann_create_tensor(src0);
-                ggml_cann_pool_alloc src_buffer_allocator(ctx.pool(), ggml_nelements(src0) * ggml_element_size(dst));
-                void *               src_trans_buffer = src_buffer_allocator.get();
-                size_t               src_trans_nb[GGML_MAX_DIMS];
-                src_trans_nb[0] = dst->nb[0];
+                // Cast src0 to dst type, then gather.
+                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
+                                                        ggml_nelements(src0) * ggml_element_size(dst));
+                size_t src_cast_nb[GGML_MAX_DIMS];
+                src_cast_nb[0] = ggml_type_size(dst->type);
                 for (int i = 1; i < GGML_MAX_DIMS; i++) {
-                    src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
+                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                 }
-                acl_tensor_ptr src_trans_tensor =
-                    ggml_cann_create_tensor(src_trans_buffer, ggml_cann_type_mapping(dst->type),
-                                            ggml_type_size(dst->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
-                aclnn_cast(ctx, acl_src0.get(), src_trans_tensor.get(), ggml_cann_type_mapping(dst->type));
-                aclnn_index_select_4d(ctx, src_trans_buffer, src0->ne, src_trans_nb, dst->data, dst->ne, dst->nb, src1,
-                                      dst->type);
+                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
+                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
+                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
+                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
+                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));
+
+                gather_batched(src_cast_allocator.get(),
+                               ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
+                               src_cast_nb);
             }
             break;
         case GGML_TYPE_Q8_0:
             {
-                // add 1 dim for bcast mul.
+                // Dequantize Q8_0 to dst type, then gather.
                 size_t  weight_nb[GGML_MAX_DIMS + 1], scale_nb[GGML_MAX_DIMS + 1], dequant_nb[GGML_MAX_DIMS + 1];
                 int64_t weight_ne[GGML_MAX_DIMS + 1], scale_ne[GGML_MAX_DIMS + 1], *dequant_ne;
-                int64_t scale_offset = 0;
-                // [3,4,5,64] -> [3,4,5,2,32]
-                weight_ne[0]         = QK8_0;
-                weight_ne[1]         = src0->ne[0] / QK8_0;
-                weight_nb[0]         = sizeof(int8_t);
-                weight_nb[1]         = weight_nb[0] * weight_ne[0];
+                weight_ne[0] = QK8_0;
+                weight_ne[1] = src0->ne[0] / QK8_0;
+                weight_nb[0] = sizeof(int8_t);
+                weight_nb[1] = weight_nb[0] * weight_ne[0];
                 for (int i = 2; i < GGML_MAX_DIMS + 1; i++) {
                     weight_ne[i] = src0->ne[i - 1];
                     weight_nb[i] = weight_nb[i - 1] * weight_ne[i - 1];
                 }
-                // [3,4,5,64] -> [3,4,5,2,1]
                 scale_ne[0] = 1;
                 scale_ne[1] = src0->ne[0] / QK8_0;
                 scale_nb[0] = sizeof(uint16_t);
@@ -1849,31 +2010,33 @@ void ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
                     scale_ne[i] = src0->ne[i - 1];
                     scale_nb[i] = scale_nb[i - 1] * scale_ne[i - 1];
                 }
-                // [3,4,5,64] -> [3,4,5,2,32]
                 dequant_ne    = weight_ne;
                 dequant_nb[0] = ggml_type_size(dst->type);
                 for (int i = 1; i < GGML_MAX_DIMS + 1; i++) {
                     dequant_nb[i] = dequant_nb[i - 1] * dequant_ne[i - 1];
                 }
-                scale_offset = ggml_nelements(src0) * sizeof(int8_t);
-                ggml_cann_pool_alloc dequant_buffer_allocator(ctx.pool(),
-                                                              ggml_nelements(src0) * ggml_type_size(dst->type));
-                acl_tensor_ptr       acl_weight_tensor = ggml_cann_create_tensor(src0->data, ACL_INT8, sizeof(int8_t),
-                                                                                 weight_ne, weight_nb, GGML_MAX_DIMS + 1);
-                acl_tensor_ptr       acl_scale_tensor =
-                    ggml_cann_create_tensor(src0->data, ACL_FLOAT16, sizeof(uint16_t), scale_ne, scale_nb,
-                                            GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_offset);
-                acl_tensor_ptr dequant_tensor =
-                    ggml_cann_create_tensor(dequant_buffer_allocator.get(), ggml_cann_type_mapping(dst->type),
-                                            ggml_type_size(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
-                aclnn_mul(ctx, acl_weight_tensor.get(), acl_scale_tensor.get(), dequant_tensor.get());
-                dequant_nb[0] = ggml_type_size(dst->type);
+                const int64_t scale_offset = ggml_nelements(src0) * sizeof(int8_t);
+                ggml_cann_pool_alloc dequant_allocator(ctx.pool(),
+                                                       ggml_nelements(src0) * ggml_type_size(dst->type));
+                acl_tensor_ptr acl_weight = ggml_cann_create_tensor(src0->data, ACL_INT8, sizeof(int8_t),
+                                                                     weight_ne, weight_nb, GGML_MAX_DIMS + 1);
+                acl_tensor_ptr acl_scale  = ggml_cann_create_tensor(
+                    src0->data, ACL_FLOAT16, sizeof(uint16_t), scale_ne, scale_nb,
+                    GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_offset);
+                acl_tensor_ptr acl_dequant = ggml_cann_create_tensor(
+                    dequant_allocator.get(), ggml_cann_type_mapping(dst->type),
+                    ggml_type_size(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
+                aclnn_mul(ctx, acl_weight.get(), acl_scale.get(), acl_dequant.get());
+
+                // Reinterpret dequant buffer as 4D [src0->ne] with contiguous strides.
                 dequant_ne    = src0->ne;
+                dequant_nb[0] = ggml_type_size(dst->type);
                 for (int i = 1; i < GGML_MAX_DIMS; i++) {
                     dequant_nb[i] = dequant_nb[i - 1] * src0->ne[i - 1];
                 }
-                aclnn_index_select_4d(ctx, dequant_buffer_allocator.get(), dequant_ne, dequant_nb, dst->data, dst->ne,
-                                      dst->nb, src1, dst->type);
+                gather_batched(dequant_allocator.get(),
+                               ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
+                               dequant_nb);
                 break;
             }
         default:
@@ -1883,31 +2046,70 @@ void ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
 }
 
 void ggml_cann_set_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
-    ggml_tensor * src0 = dst->src[0];  // src
-    ggml_tensor * src1 = dst->src[1];  // index
+    ggml_tensor * src0 = dst->src[0];  // source values
+    ggml_tensor * src1 = dst->src[1];  // row indices
+
+    // n_idx: number of source rows to scatter per batch slice.
+    // ggml guarantees: src0->ne[1] == src1->ne[0].
+    const int64_t n_idx = src1->ne[0];
+
+    // Copy n_idx rows of src [ne0, n_idx] into dst [ne0, ne1] at positions given by a 1D index.
+    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0] for dst.
+    // InplaceIndexCopy with dim=0 copies along ACL dim-0 == ggml ne[1] (the row axis).
+    // src_nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
+    //         nb[2..3] for computing per-batch-slice base pointer offsets).
+    auto scatter_batched = [&](void * src_base, aclDataType acl_type, size_t type_size,
+                               const size_t * src_nb) {
+        int64_t d_ne[2]    = { dst->ne[0], dst->ne[1] };
+        size_t  d_nb[2]    = { dst->nb[0], dst->nb[1] };
+        int64_t s_ne[2]    = { dst->ne[0], n_idx };
+        size_t  s_nb_2d[2] = { src_nb[0], src_nb[1] };
+        int64_t i_ne[1]    = { n_idx };
+        size_t  i_nb[1]    = { (size_t)ggml_element_size(src1) };
+
+        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
+            for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
+                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
+                    (char *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2],
+                    acl_type, type_size, d_ne, d_nb, 2);
+                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
+                    (char *)src1->data + (i3 % src1->ne[2]) * src1->nb[2] + (i2 % src1->ne[1]) * src1->nb[1],
+                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_size(src1),
+                    i_ne, i_nb, 1);
+                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
+                    (char *)src_base + i3 * src_nb[3] + i2 * src_nb[2],
+                    acl_type, type_size, s_ne, s_nb_2d, 2);
+                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceIndexCopy, acl_dst.get(), 0, acl_idx.get(), acl_src.get());
+            }
+        }
+    };
 
     switch (dst->type) {
         case GGML_TYPE_F32:
-            {
-                aclnn_index_copy_4d(ctx, src0->data, src0->ne, src0->nb, dst->data, dst->ne, dst->nb, src1, dst->type);
-                break;
-            }
+            scatter_batched(src0->data,
+                            ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
+                            src0->nb);
+            break;
         case GGML_TYPE_F16:
         case GGML_TYPE_BF16:
             {
-                acl_tensor_ptr       acl_src0 = ggml_cann_create_tensor(src0);
-                ggml_cann_pool_alloc src_buffer_allocator(ctx.pool(), ggml_nelements(src0) * sizeof(uint16_t));
-                void *               src_trans_buffer = src_buffer_allocator.get();
-                size_t               src_trans_nb[GGML_MAX_DIMS];
-                src_trans_nb[0] = sizeof(uint16_t);
+                // Cast src0 (F32) to dst type first.
+                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
+                                                        ggml_nelements(src0) * ggml_type_size(dst->type));
+                size_t src_cast_nb[GGML_MAX_DIMS];
+                src_cast_nb[0] = ggml_type_size(dst->type);
                 for (int i = 1; i < GGML_MAX_DIMS; i++) {
-                    src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
+                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                 }
-                acl_tensor_ptr src_trans_tensor = ggml_cann_create_tensor(
-                    src_trans_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
-                aclnn_cast(ctx, acl_src0.get(), src_trans_tensor.get(), ggml_cann_type_mapping(dst->type));
-                aclnn_index_copy_4d(ctx, src_trans_buffer, src0->ne, src_trans_nb, dst->data, dst->ne, dst->nb, src1,
-                                    dst->type);
+                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
+                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
+                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
+                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
+                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));
+
+                scatter_batched(src_cast_allocator.get(),
+                                ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
+                                src_cast_nb);
                 break;
             }
         default:
@@ -3268,29 +3470,50 @@ void ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst
     int64_t           paddingsArray[2] = { opts[0], opts[1] };
     acl_int_array_ptr paddings         = ggml_cann_create_int_array(paddingsArray, 2);
 
-    for (int64_t i = 0; i < src0->ne[3]; i++) {
-        acl_tensor_ptr acl_src =
-            ggml_cann_create_tensor((char *) src0->data + i * src0->ne[3], ggml_cann_type_mapping(src0->type),
-                                    ggml_element_size(src0), src0->ne, src0->nb, 3);
+    // Collapsing ne[2]*ne[3] into a single batch dimension requires that dim3
+    // is contiguous with respect to dim2 in both src and dst.
+    GGML_ASSERT(src0->nb[3] == src0->nb[2] * src0->ne[2]);
+    GGML_ASSERT(dst->nb[3]  == dst->nb[2]  * dst->ne[2]);
 
-        acl_tensor_ptr acl_dst =
-            ggml_cann_create_tensor((char *) dst->data + i * src0->ne[3], ggml_cann_type_mapping(dst->type),
-                                    ggml_element_size(dst), dst->ne, dst->nb, 3);
+    int64_t src_ne_3d[3] = { src0->ne[0], src0->ne[1], src0->ne[2] * src0->ne[3] };
+    int64_t dst_ne_3d[3] = { dst->ne[0],  dst->ne[1],  dst->ne[2]  * dst->ne[3]  };
 
-        GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
-    }
+    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->data, ggml_cann_type_mapping(src0->type),
+                                                     ggml_element_size(src0), src_ne_3d, src0->nb, 3);
+
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->data, ggml_cann_type_mapping(dst->type),
+                                                     ggml_element_size(dst), dst_ne_3d, dst->nb, 3);
+
+    GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
 }
 
 void ggml_cann_count_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     ggml_tensor * src0 = dst->src[0];
     ggml_tensor * src1 = dst->src[1];
 
+    // Write element-wise equality (0 or 1) into a temporary buffer to avoid
+    // modifying src0 in-place.  Use the same type as src0 so ReduceSum can
+    // consume it directly without a type cast.
+    ggml_cann_pool_alloc eq_alloc(ctx.pool(), ggml_nelements(src0) * ggml_element_size(src0));
+    size_t eq_nb[GGML_MAX_DIMS];
+    eq_nb[0] = ggml_element_size(src0);
+    for (int i = 1; i < GGML_MAX_DIMS; i++) {
+        eq_nb[i] = eq_nb[i - 1] * src0->ne[i - 1];
+    }
+    acl_tensor_ptr acl_eq = ggml_cann_create_tensor(
+        eq_alloc.get(), ggml_cann_type_mapping(src0->type), ggml_element_size(src0),
+        src0->ne, eq_nb, GGML_MAX_DIMS);
+
     acl_tensor_ptr acl_self  = ggml_cann_create_tensor(src0);
     acl_tensor_ptr acl_other = ggml_cann_create_tensor(src1);
-
-    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceEqTensor, acl_self.get(), acl_other.get());
-
-    ggml_cann_sum(ctx, dst);
+    GGML_CANN_CALL_ACLNN_OP(ctx, EqTensor, acl_self.get(), acl_other.get(), acl_eq.get());
+
+    // Sum the 0/1 values into dst.
+    acl_tensor_ptr    acl_dst    = ggml_cann_create_tensor(dst);
+    int64_t           dims[4]    = { 0, 1, 2, 3 };
+    acl_int_array_ptr dims_arr   = ggml_cann_create_int_array(dims, 4);
+    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_eq.get(), dims_arr.get(), true,
+                            ggml_cann_type_mapping(dst->type), acl_dst.get());
 }
 
 void ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
@@ -3306,6 +3529,27 @@ void ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
     GGML_CANN_CALL_ACLNN_OP(ctx, GtScalar, acl_src.get(), alpha.get(), acl_dst.get());
 }
 
+void ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    ggml_tensor * src0 = dst->src[0];
+
+    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
+    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
+
+    float          beta_val      = 1.0f;
+    float          threshold_val = 20.0f;
+    acl_scalar_ptr beta          = ggml_cann_create_scalar(&beta_val,      ACL_FLOAT);
+    acl_scalar_ptr threshold     = ggml_cann_create_scalar(&threshold_val, ACL_FLOAT);
+
+    GGML_CANN_CALL_ACLNN_OP(ctx, Softplus, acl_src.get(), beta.get(), threshold.get(), acl_dst.get());
+}
+
+void ggml_cann_geglu_quick(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
+    auto gelu_quick_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
+        GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
+    };
+    ggml_cann_op_unary_gated(gelu_quick_fn, ctx, dst);
+}
+
 /**
  * @brief Performs expert-specific matrix multiplication (MoE) with
  * floating-point precision using the CANN backend.
@@ -3892,46 +4136,65 @@ void ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst
 }
 
 static void ggml_cann_out_prod_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
-    ggml_tensor * src0 = dst->src[0];  // weight
-    ggml_tensor * src1 = dst->src[1];  // input
+    ggml_tensor * src0 = dst->src[0];  // weight  [ne00=m, ne01=K, ne02, ne03]
+    ggml_tensor * src1 = dst->src[1];  // input   [ne10=n, ne11=K, ne12, ne13]
     GGML_TENSOR_BINARY_OP_LOCALS
 
-    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
-    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_dst.get());
+    // dst[i,j] = sum_k src0[i,k] * src1[j,k]  i.e. dst = src0 @ src1^T.
+    //
+    // ggml_cann_create_tensor reverses dimension order, so ACL sees:
+    //   acl_src0 slice:   ggml[m,K]  ->  ACL[K,m]
+    //   acl_src1 slice:   ggml[n,K]  ->  ACL[K,n]
+    //   acl_dst  slice:   ggml[m,n]  ->  ACL[n,m]
+    //
+    // Build a transposed view of src1 by swapping ne[0]/ne[1]:
+    //   src1_t:  ggml[K,n] (swapped strides)  ->  ACL[n,K]
+    //
+    // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst  ✓
+    //
+    // The outer batch loop is kept because src0 may have fewer batch slices than
+    // dst (ne02 <= ne2, ne03 <= ne3): this is a strided-broadcast not supported
+    // by standard CANN Matmul broadcasting.
+
+    const aclDataType src0_acl_type = ggml_cann_type_mapping(src0->type);
+    const aclDataType src1_acl_type = ggml_cann_type_mapping(src1->type);
+    const aclDataType dst_acl_type  = ggml_cann_type_mapping(dst->type);
+    const size_t      src0_type_sz  = ggml_type_size(src0->type);
+    const size_t      src1_type_sz  = ggml_type_size(src1->type);
+    const size_t      dst_type_sz   = ggml_type_size(dst->type);
 
     const int64_t dps2 = ne2 / ne02;
     const int64_t dps3 = ne3 / ne03;
+
     for (int64_t i3 = 0; i3 < ne3; i3++) {
         for (int64_t i2 = 0; i2 < ne2; i2++) {
             const int64_t i02 = i2 / dps2;
             const int64_t i03 = i3 / dps3;
 
-            const int64_t  i12 = i2;
-            const int64_t  i13 = i3;
-            acl_tensor_ptr accumulator =
-                ggml_cann_create_tensor((char *) dst->data + i2 * nb2 + i3 * nb3, ggml_cann_type_mapping(dst->type),
-                                        ggml_type_size(dst->type), dst->ne, dst->nb, 2);
-
-            // The outer product needs to be accumulated in this dimension.
-            for (int64_t i1 = 0; i1 < ne11; i1++) {
-                acl_tensor_ptr acl_input = ggml_cann_create_tensor(
-                    (char *) src1->data + i1 * nb11 + i12 * nb12 + i13 * nb13, ggml_cann_type_mapping(src0->type),
-                    ggml_type_size(src0->type), src1->ne, src1->nb, 1);
-
-                acl_tensor_ptr acl_weight = ggml_cann_create_tensor(
-                    (char *) src0->data + i1 * nb01 + i02 * nb02 + i03 * nb03, ggml_cann_type_mapping(src0->type),
-                    ggml_type_size(src0->type), src0->ne, src0->nb, 1);
-
-                ggml_cann_pool_alloc output_allocator(ctx.pool());
-                void *               output_buffer = output_allocator.alloc(ggml_nbytes(dst));
-                acl_tensor_ptr       acl_out = ggml_cann_create_tensor(output_buffer, ggml_cann_type_mapping(dst->type),
-                                                                       ggml_type_size(dst->type), dst->ne, dst->nb, 2);
-
-                GGML_CANN_CALL_ACLNN_OP(ctx, Ger, acl_input.get(), acl_weight.get(), acl_out.get());
-                float       alpha_value = 1.0f;
-                aclScalar * alpha       = aclCreateScalar(&alpha_value, ACL_FLOAT);
-                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, accumulator.get(), acl_out.get(), alpha);
-            }
+            // src0 2D slice at [i02, i03]: ggml [m, K] -> ACL [K, m]
+            int64_t src0_ne[2] = { ne00, ne01 };
+            size_t  src0_nb[2] = { nb00, nb01 };
+            acl_tensor_ptr acl_src0_s = ggml_cann_create_tensor(
+                (char *) src0->data + i02 * nb02 + i03 * nb03,
+                src0_acl_type, src0_type_sz, src0_ne, src0_nb, 2);
+
+            // src1 transposed 2D slice at [i2, i3]: swap ne/nb -> ggml[K,n] -> ACL[n,K]
+            int64_t src1_t_ne[2] = { ne11, ne10 };
+            size_t  src1_t_nb[2] = { nb11, nb10 };
+            acl_tensor_ptr acl_src1_t = ggml_cann_create_tensor(
+                (char *) src1->data + i2 * nb12 + i3 * nb13,
+                src1_acl_type, src1_type_sz, src1_t_ne, src1_t_nb, 2);
+
+            // dst 2D slice at [i2, i3]: ggml [m, n] -> ACL [n, m]
+            int64_t dst_ne[2] = { ne0, ne1 };
+            size_t  dst_nb[2] = { nb0, nb1 };
+            acl_tensor_ptr acl_dst_s = ggml_cann_create_tensor(
+                (char *) dst->data + i2 * nb2 + i3 * nb3,
+                dst_acl_type, dst_type_sz, dst_ne, dst_nb, 2);
+
+            // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst_s  ✓
+            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul,
+                acl_src1_t.get(), acl_src0_s.get(), acl_dst_s.get(), (int8_t) 1);
         }
     }
 }
@@ -4170,3 +4433,4 @@ void ggml_cann_gated_linear_attn(ggml_backend_cann_context & ctx, ggml_tensor *
         }
     }
 }
+
diff --git src/ggml-cann/aclnn_ops.h src/ggml-cann/aclnn_ops.h
index 7f5ba4d3..cdbf9260 100644
--- src/ggml-cann/aclnn_ops.h
+++ src/ggml-cann/aclnn_ops.h
@@ -32,6 +32,9 @@
 #include <aclnnop/aclnn_cat.h>
 #include <aclnnop/aclnn_clamp.h>
 #include <aclnnop/aclnn_cos.h>
+#include <aclnnop/aclnn_cumsum.h>
+#include <aclnnop/aclnn_tril.h>
+#include <aclnnop/aclnn_triu.h>
 #include <aclnnop/aclnn_exp.h>
 #include <aclnnop/aclnn_gelu.h>
 #include <aclnnop/aclnn_gelu_v2.h>
@@ -47,6 +50,9 @@
 #include <aclnnop/aclnn_sign.h>
 #include <aclnnop/aclnn_silu.h>
 #include <aclnnop/aclnn_sin.h>
+#include <aclnnop/aclnn_softplus.h>
+#include <aclnnop/aclnn_swi_glu.h>
+#include <aclnnop/aclnn_geglu.h>
 #include <aclnnop/aclnn_slice.h>
 #include <aclnnop/aclnn_sqrt.h>
 #include <aclnnop/aclnn_tanh.h>
@@ -69,6 +75,9 @@
  */
 void ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst);
 
+void ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+void ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate);
+
 /**
  * @brief   Applies the Leaky ReLU activation function to a tensor using the CANN
  *          backend.
@@ -325,6 +334,48 @@ void ggml_cann_sum_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);
 
 void ggml_cann_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst);
 
+/**
+ * @brief   Computes the cumulative sum of a ggml tensor along dim 0 using the
+ *          CANN backend.
+ *
+ * @param ctx The CANN context used for operations.
+ * @param dst The destination tensor. dst->op is `GGML_OP_CUMSUM`.
+ */
+void ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+
+/**
+ * @brief   Computes a triangular mask (tril/triu) of a square ggml tensor
+ *          using the CANN backend.
+ *
+ * @param ctx The CANN context used for operations.
+ * @param dst The destination tensor. dst->op is `GGML_OP_TRI`.
+ */
+void ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+
+/**
+ * @brief   Solves a triangular linear system AX=B using the CANN backend.
+ *
+ * @param ctx The CANN context used for operations.
+ * @param dst The destination tensor. dst->op is `GGML_OP_SOLVE_TRI`.
+ */
+void ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+
+/**
+ * @brief   Creates a diagonal matrix from a vector using the CANN backend.
+ *
+ * @param ctx The CANN context used for operations.
+ * @param dst The destination tensor. dst->op is `GGML_OP_DIAG`.
+ */
+void ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+
+/**
+ * @brief   Fills a tensor with a constant scalar value using the CANN backend.
+ *
+ * @param ctx The CANN context used for operations.
+ * @param dst The destination tensor. dst->op is `GGML_OP_FILL`.
+ */
+void ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+
 /**
  * @brief   Upsamples a ggml tensor using nearest neighbor interpolation using
  *          the CANN backend.
@@ -461,6 +512,9 @@ void ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor *
 // @see ggml_cann_dup.
 void ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst);
 
+// @see ggml_cann_acc, but copies src1 into dst instead of adding.
+void ggml_cann_set(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+
 /**
  * @brief   Computes the softmax activation with optional masking.
  *
@@ -813,6 +867,8 @@ void ggml_cann_count_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst);
  *            dst->op is expected to be `GGML_OP_STEP`.
  */
 void ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+void ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst);
+void ggml_cann_geglu_quick(ggml_backend_cann_context & ctx, ggml_tensor * dst);
 
 /**
  * @brief   Performs the Flash Attention extended operator using the CANN backend.
diff --git src/ggml-cann/ggml-cann.cpp src/ggml-cann/ggml-cann.cpp
index 5fc484b3..3618ba7f 100644
--- src/ggml-cann/ggml-cann.cpp
+++ src/ggml-cann/ggml-cann.cpp
@@ -1428,6 +1428,22 @@ static bool ggml_backend_cann_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
     return false;
 }
 
+/**
+ * @brief Set a region of a tensor's device memory to a specified value.
+ *
+ * @param buffer The CANN buffer containing the tensor.
+ * @param tensor Pointer to the tensor whose memory will be set.
+ * @param value The value to which each byte in the region will be set.
+ * @param offset Byte offset within the tensor's data to start setting.
+ * @param size Number of bytes to set.
+ */
+static void ggml_backend_cann_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
+    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;
+
+    ggml_cann_set_device(ctx->device);
+    ACL_CHECK(aclrtMemset((char *) tensor->data + offset, size, value, size));
+}
+
 /**
  * @brief Clear a CANN buffer by setting all its memory to a specified value.
  *
@@ -1454,7 +1470,7 @@ static const ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
     /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
     /* .get_base        = */ ggml_backend_cann_buffer_get_base,
     /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
-    /* .memset_tensor   = */ NULL,
+    /* .memset_tensor   = */ ggml_backend_cann_buffer_memset_tensor,
     /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
     /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
     /* .set_tensor_2d   = */ NULL,
@@ -1835,6 +1851,9 @@ static bool ggml_cann_compute_forward(ggml_backend_cann_context & ctx, struct gg
                 case GGML_UNARY_OP_STEP:
                     ggml_cann_step(ctx, dst);
                     break;
+                case GGML_UNARY_OP_SOFTPLUS:
+                    ggml_cann_softplus(ctx, dst);
+                    break;
                 default:
                     return false;
             }
@@ -1845,20 +1864,16 @@ static bool ggml_cann_compute_forward(ggml_backend_cann_context & ctx, struct gg
                     GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                     break;
                 case GGML_GLU_OP_GEGLU:
+                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
+                    break;
                 case GGML_GLU_OP_GEGLU_ERF:
-                    // aclnnGelu internally uses the erf-based approximation.
-                    GGML_CANN_CALL_OP_UNARY_GATED(Gelu);
+                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                     break;
                 case GGML_GLU_OP_SWIGLU:
-                    GGML_CANN_CALL_OP_UNARY_GATED(Silu);
+                    ggml_cann_swiglu(ctx, dst);
                     break;
                 case GGML_GLU_OP_GEGLU_QUICK:
-                    {
-                        auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
-                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
-                        };
-                        ggml_cann_op_unary_gated(lambda, ctx, dst);
-                    }
+                    ggml_cann_geglu_quick(ctx, dst);
                     break;
                 default:
                     return false;
@@ -1920,6 +1935,9 @@ static bool ggml_cann_compute_forward(ggml_backend_cann_context & ctx, struct gg
         case GGML_OP_CPY:
             ggml_cann_cpy(ctx, dst);
             break;
+        case GGML_OP_SET:
+            ggml_cann_set(ctx, dst);
+            break;
         case GGML_OP_CONT:
             ggml_cann_dup(ctx, dst);
             break;
@@ -1989,6 +2007,21 @@ static bool ggml_cann_compute_forward(ggml_backend_cann_context & ctx, struct gg
         case GGML_OP_SSM_CONV:
             ggml_cann_ssm_conv(ctx, dst);
             break;
+        case GGML_OP_CUMSUM:
+            ggml_cann_cumsum(ctx, dst);
+            break;
+        case GGML_OP_TRI:
+            ggml_cann_tri(ctx, dst);
+            break;
+        case GGML_OP_FILL:
+            ggml_cann_fill(ctx, dst);
+            break;
+        case GGML_OP_DIAG:
+            ggml_cann_diag(ctx, dst);
+            break;
+        case GGML_OP_SOLVE_TRI:
+            ggml_cann_solve_tri(ctx, dst);
+            break;
         default:
             return false;
     }
@@ -2324,6 +2357,7 @@ static enum ggml_status ggml_backend_cann_graph_compute(ggml_backend_t backend,
     if (use_cann_graph) {
         // If no matching graph is found, the graph needs to be recaptured.
         graph_capture_required = !cann_ctx->graph_lru_cache.find_and_move_to_front(cgraph);
+
         if (graph_capture_required) {
             // If no matching graph is found, add a new ACL graph.
             ggml_cann_graph * new_graph = ggml_cann_graph::create_from_cgraph(cgraph);
@@ -2382,6 +2416,7 @@ static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, const ggml_ten
                 case GGML_UNARY_OP_SGN:
                 case GGML_UNARY_OP_STEP:
                 case GGML_UNARY_OP_GELU_ERF:
+                case GGML_UNARY_OP_SOFTPLUS:
                     return true;
                 default:
                     return false;
@@ -2572,6 +2607,7 @@ static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, const ggml_ten
         case GGML_OP_SUM_ROWS:
         case GGML_OP_ARGSORT:
         case GGML_OP_ACC:
+        case GGML_OP_SET:
         case GGML_OP_GROUP_NORM:
             return true;
         case GGML_OP_PAD:
@@ -2649,6 +2685,16 @@ static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, const ggml_ten
             }
         case GGML_OP_SSM_CONV:
             return true;
+        case GGML_OP_CUMSUM:
+            return op->src[0]->type == GGML_TYPE_F32;
+        case GGML_OP_TRI:
+            return op->src[0]->type == GGML_TYPE_F32;
+        case GGML_OP_FILL:
+            return op->src[0]->type == GGML_TYPE_F32;
+        case GGML_OP_DIAG:
+            return op->src[0]->type == GGML_TYPE_F32;
+        case GGML_OP_SOLVE_TRI:
+            return op->src[0]->type == GGML_TYPE_F32;
         default:
             return false;
     }
diff --git src/ggml-cpu/amx/mmq.cpp src/ggml-cpu/amx/mmq.cpp
index 93a6d397..d9383a04 100644
--- src/ggml-cpu/amx/mmq.cpp
+++ src/ggml-cpu/amx/mmq.cpp
@@ -2005,12 +2005,12 @@ void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const v
     const int lda = KB * sizeof(TA);
     //const int ldb = KB * sizeof(TB);
 
-    static thread_local packed_B_t Tile0[TILE_N * TILE_K];
-    static thread_local packed_B_t Tile1[TILE_N * TILE_K];
-    static thread_local int8_t Tile23[TILE_M * TILE_K];
+    alignas(64) static thread_local packed_B_t Tile0[TILE_N * TILE_K];
+    alignas(64) static thread_local packed_B_t Tile1[TILE_N * TILE_K];
+    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];
 
-    static thread_local int32_t TileC0[TILE_M * TILE_N * 4];
-    static thread_local int32_t TileC1[TILE_M * TILE_N * 4];
+    alignas(64) static thread_local int32_t TileC0[TILE_M * TILE_N * 4];
+    alignas(64) static thread_local int32_t TileC1[TILE_M * TILE_N * 4];
 
     // double buffering C to interleave avx512 and amx
     int32_t * C_cur = TileC0;
@@ -2187,21 +2187,21 @@ void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const v
     const int m1 = std::max(M - TILE_M, 0);
     //const int lda = KB * sizeof(TA);
 
-    static thread_local int8_t Tile0[TILE_N * TILE_K];
-    static thread_local int8_t Tile1[TILE_N * TILE_K];
-    static thread_local int8_t Tile23[TILE_M * TILE_K];
+    alignas(64) static thread_local int8_t Tile0[TILE_N * TILE_K];
+    alignas(64) static thread_local int8_t Tile1[TILE_N * TILE_K];
+    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];
 
     // mat mul result for each group
-    static thread_local int32_t Tile4[TILE_M * TILE_N];
-    static thread_local int32_t Tile5[TILE_M * TILE_N];
-    static thread_local int32_t Tile6[TILE_M * TILE_N];
-    static thread_local int32_t Tile7[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Tile4[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Tile5[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Tile6[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Tile7[TILE_M * TILE_N];
 
     // sum of each QK_K block, contains 8 groups, int32
-    static thread_local int32_t Sumi4[TILE_M * TILE_N];
-    static thread_local int32_t Sumi5[TILE_M * TILE_N];
-    static thread_local int32_t Sumi6[TILE_M * TILE_N];
-    static thread_local int32_t Sumi7[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Sumi4[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Sumi5[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Sumi6[TILE_M * TILE_N];
+    alignas(64) static thread_local int32_t Sumi7[TILE_M * TILE_N];
 
     const int k_group_size = std::is_same<TB, block_q6_K>::value ? 16 : 32;
     for (int i = 0; i < KB; ++i) {
diff --git src/ggml-cpu/arch/x86/quants.c src/ggml-cpu/arch/x86/quants.c
index 0a3e071e..94b19b82 100644
--- src/ggml-cpu/arch/x86/quants.c
+++ src/ggml-cpu/arch/x86/quants.c
@@ -2300,9 +2300,8 @@ void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const voi
 
 #if defined __AVX2__
 
-    const __m256i m4 = _mm256_set1_epi8(0xF);
-    const __m256i m2 = _mm256_set1_epi8(3);
-    const __m256i m32s = _mm256_set1_epi8(32);
+    const __m256i m3 = _mm256_set1_epi8(3);
+    const __m256i m15 = _mm256_set1_epi8(15);
 
     __m256 acc = _mm256_setzero_ps();
 
@@ -2314,53 +2313,45 @@ void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const voi
         const uint8_t * GGML_RESTRICT qh = x[i].qh;
         const int8_t  * GGML_RESTRICT q8 = y[i].qs;
 
+        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
         const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
+        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
+        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
 
         __m256i sumi = _mm256_setzero_si256();
 
         int is = 0;
 
         for (int j = 0; j < QK_K/128; ++j) {
-
-            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
-            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
-            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
-            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
-            is += 4;
-
             const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
             const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
             const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;
 
-            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
-            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
-            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
-            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);
+            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
+            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
+            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
+            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);
 
-            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
-            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
-            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
-            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);
+            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
+            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
+            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
+            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
 
             const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
             const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
             const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
             const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
 
-            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
-            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
-            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
-            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);
-
             __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
             __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
             __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
             __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);
 
-            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
-            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
-            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
-            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);
+            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
+            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
+            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
+            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
+            is += 4;
 
             p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
             p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
@@ -2372,6 +2363,7 @@ void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const voi
 
         }
 
+        sumi = _mm256_sub_epi32(sumi, q8sclsub);
         acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
     }
 
diff --git src/ggml-cpu/vec.h src/ggml-cpu/vec.h
index a0375a28..bcd68da9 100644
--- src/ggml-cpu/vec.h
+++ src/ggml-cpu/vec.h
@@ -1036,12 +1036,12 @@ inline static float ggml_gelu_quick_f32(float x) {
     return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
 }
 
-//inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
-//    const uint16_t * i16 = (const uint16_t *) x;
-//    for (int i = 0; i < n; ++i) {
-//        y[i] = ggml_table_gelu_quick_f16[i16[i]];
-//    }
-//}
+inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
+    const uint16_t * i16 = (const uint16_t *) x;
+    for (int i = 0; i < n; ++i) {
+        y[i] = ggml_table_gelu_quick_f16[i16[i]];
+    }
+}
 
 #ifdef GGML_GELU_QUICK_FP16
 inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
@@ -1060,13 +1060,6 @@ inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float *
 }
 #endif
 
-inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
-    for (int i = 0; i < n; ++i) {
-        float v = GGML_CPU_FP16_TO_FP32(x[i]);
-        y[i] = GGML_CPU_FP32_TO_FP16(v*(1.0f/(1.0f+expf(GELU_QUICK_COEF*v))));
-    }
-}
-
 // Sigmoid Linear Unit (SiLU) function
 inline static float ggml_silu_f32(float x) {
     return x/(1.0f + expf(-x));
diff --git src/ggml-cuda/concat.cu src/ggml-cuda/concat.cu
index e9ffd274..102f944f 100644
--- src/ggml-cuda/concat.cu
+++ src/ggml-cuda/concat.cu
@@ -1,96 +1,79 @@
 #include "concat.cuh"
 
 // contiguous kernels
-static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int ne0, const int ne00) {
-    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
-    if (nidx >= ne0) {
-        return;
-    }
-
-    int offset_dst =
-        nidx +
-        blockIdx.y * ne0 +
-        blockIdx.z * ne0 * gridDim.y;
-
-    if (nidx < ne00) { // src0
-        int offset_src =
-            nidx +
-            blockIdx.y * ne00 +
-            blockIdx.z * ne00 * gridDim.y;
-        dst[offset_dst] = x[offset_src];
-    } else {
-        int offset_src =
-            (nidx - ne00) +
-            blockIdx.y * (ne0 - ne00) +
-            blockIdx.z * (ne0 - ne00) * gridDim.y;
-        dst[offset_dst] = y[offset_src];
-    }
-}
-
-static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int ne0, const int ne01) {
-    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
-    if (nidx >= ne0) {
-        return;
-    }
+template <int dim>
+static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_f32_cont(const float * x,
+                                                                                 const float * y,
+                                                                                 float *       dst,
+                                                                                 int64_t       ne00,
+                                                                                 int64_t       ne01,
+                                                                                 int64_t       ne02,
+                                                                                 int64_t       ne0,
+                                                                                 int64_t       ne1,
+                                                                                 int64_t       ne2) {
+    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");
+
+    const int64_t n = ne0 * ne1 * ne2;
+
+    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
+        if constexpr (dim == 0) {
+            const int64_t row = i / ne0;
+            const int64_t i0  = i - row * ne0;
+
+            if (i0 < ne00) {
+                dst[i] = x[row * ne00 + i0];
+            } else {
+                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
+            }
+        } else if constexpr (dim == 1) {
+            const int64_t dst_plane  = ne0 * ne1;
+            const int64_t src0_plane = ne0 * ne01;
+            const int64_t src1_plane = dst_plane - src0_plane;
+            const int64_t i2         = i / dst_plane;
+            const int64_t i01        = i - i2 * dst_plane;
+
+            if (i01 < src0_plane) {
+                dst[i] = x[i2 * src0_plane + i01];
+            } else {
+                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
+            }
+        } else {
+            const int64_t src0_size = ne0 * ne1 * ne02;
 
-    int offset_dst =
-        nidx +
-        blockIdx.y * ne0 +
-        blockIdx.z * ne0 * gridDim.y;
-
-    if (blockIdx.y < (unsigned)ne01) { // src0
-        int offset_src =
-            nidx +
-            blockIdx.y * ne0 +
-            blockIdx.z * ne0 * ne01;
-        dst[offset_dst] = x[offset_src];
-    } else {
-        int offset_src =
-            nidx +
-            (blockIdx.y - ne01) * ne0 +
-            blockIdx.z * ne0 * (gridDim.y - ne01);
-        dst[offset_dst] = y[offset_src];
+            if (i < src0_size) {
+                dst[i] = x[i];
+            } else {
+                dst[i] = y[i - src0_size];
+            }
+        }
     }
 }
 
-static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int ne0, const int ne02) {
-    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
-    if (nidx >= ne0) {
-        return;
-    }
-
-    int offset_dst =
-        nidx +
-        blockIdx.y * ne0 +
-        blockIdx.z * ne0 * gridDim.y;
-
-    if (blockIdx.z < (unsigned)ne02) { // src0
-        int offset_src =
-            nidx +
-            blockIdx.y * ne0 +
-            blockIdx.z * ne0 * gridDim.y;
-        dst[offset_dst] = x[offset_src];
-    } else {
-        int offset_src =
-            nidx +
-            blockIdx.y * ne0 +
-            (blockIdx.z - ne02) * ne0 *  gridDim.y;
-        dst[offset_dst] = y[offset_src];
-    }
-}
+static void concat_f32_cuda(const float * x,
+                            const float * y,
+                            float *       dst,
+                            int64_t       ne00,
+                            int64_t       ne01,
+                            int64_t       ne02,
+                            int64_t       ne0,
+                            int64_t       ne1,
+                            int64_t       ne2,
+                            int           dim,
+                            cudaStream_t  stream) {
+    const int64_t n          = ne0 * ne1 * ne2;
+    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
 
-static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
-    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
-    dim3 gridDim(num_blocks, ne1, ne2);
     if (dim == 0) {
-        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
+        concat_f32_cont<0>
+            <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
         return;
     }
     if (dim == 1) {
-        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
+        concat_f32_cont<1>
+            <<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
         return;
     }
-    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
+    concat_f32_cont<2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
 }
 
 // non-contiguous kernel (slow)
diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index 18595631..1c2c3b4a 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -3592,6 +3592,30 @@ static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
         return true;
     }
 
+    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
+     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
+        const ggml_tensor * unary = cgraph->nodes[node_idx];
+        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];
+
+        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
+            return false;
+        }
+
+        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
+            return false;
+        }
+
+        if (unary->type != sqr->type) {
+            return false;
+        }
+
+        if (!ggml_is_contiguous(unary->src[0])) {
+            return false;
+        }
+
+        return true;
+    }
+
     if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
      && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
         const ggml_tensor *scale  = cgraph->nodes[node_idx];
@@ -4100,6 +4124,12 @@ static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cud
                         continue;
                     }
 
+                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
+                        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i+1]);
+                        i++;
+                        continue;
+                    }
+
                     if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
                         i += 2;
                         ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i], node);
diff --git src/ggml-cuda/mmq.cuh src/ggml-cuda/mmq.cuh
index b1a319de..91a1b737 100644
--- src/ggml-cuda/mmq.cuh
+++ src/ggml-cuda/mmq.cuh
@@ -3478,10 +3478,10 @@ template <ggml_type type, int mmq_x, bool need_check>
 static __global__ void mul_mat_q(
         const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
         const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
-        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
-        const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
-        const int sample_ratio, const int nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
-        const int ncols_max) {
+        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
+        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
+        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
+        const uint3 ntx) {
 
     // Skip unused template specializations for faster compilation:
     if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
@@ -3495,8 +3495,7 @@ static __global__ void mul_mat_q(
     constexpr int qk    = ggml_cuda_type_traits<type>::qk;
     constexpr int mmq_y = get_mmq_y_device();
 
-    const int ntx = (ncols_max + mmq_x - 1) / mmq_x; // Number of tiles x
-    const int nty = (nrows_x   + mmq_y - 1) / mmq_y; // Number of tiles y
+    const uint32_t nty = (nrows_x + mmq_y - 1) / mmq_y; // Number of tiles y
 
     // Initialize the ids for writing back data with just the index.
     // For regular matrix multiplications this is never changed.
@@ -3517,8 +3516,9 @@ static __global__ void mul_mat_q(
     // On non-CDNA AMD or old CUDA the performance with stream-k was worse, use conventional tiling instead:
 #if (defined(GGML_USE_HIP) && !defined(CDNA)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
     {
-        const int wt = blockIdx.z / nchannels_y;
-        const int zt = blockIdx.z - wt*nchannels_y;
+        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
+        const int wt = tmp2.x;
+        const int zt = tmp2.y;
         const int jt = blockIdx.y;
         const int it = blockIdx.x;
 
@@ -3561,40 +3561,40 @@ static __global__ void mul_mat_q(
         const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
         const int tile_y_max_j = col_diff - jt*mmq_x - 1;
 
-        const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;
+        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;
 
         constexpr bool fixup = false;
         mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
             (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
-             tile_x_max_i, tile_y_max_j, 0, ncols_x/qk);
+             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
         return;
     }
 #endif // (defined(GGML_USE_HIP) && !defined(CDNA4) && !defined(CDNA3)) || __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
 
-    constexpr int ITER_K = get_iter_k(type);
-
-    const     int64_t blocks_per_ne00 = ncols_x / qk;
-    constexpr int     blocks_per_iter = ITER_K / qk;
+    constexpr int ITER_K          = get_iter_k(type);
+    constexpr int blocks_per_iter = ITER_K / qk;
 
     // kbc == k block continuous, current index in continuous ijk space.
-    int64_t kbc      = (int64_t) blockIdx.x     *nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
-    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
+    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
+    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
 
-    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
-    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;
+    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
+    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;
 
     // kb0 == k index when doing the matrix multiplication for an output tile.
-    int kb0_start = kbc % blocks_per_ne00;
-    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
-    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
-        int tmp = kbc;
-        const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
-        tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
-        const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
-        tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
-        const int zt = tmp / (ntx*blocks_per_ne00);
-        tmp -= zt * (ntx*blocks_per_ne00);
-        const int jt = tmp / blocks_per_ne00;
+    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
+    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
+    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
+        int tmp = fastdiv(kbc, blocks_per_ne00);
+        uint2 tmp2 = fast_div_modulo(tmp, ntx);
+        const int jt = tmp2.y;
+        tmp = tmp2.x;
+        tmp2 = fast_div_modulo(tmp, nchannels_y);
+        const int zt = tmp2.y;
+        tmp = tmp2.x;
+        tmp2 = fast_div_modulo(tmp, nsamples_y);
+        const int wt = tmp2.y;
+        const int it = tmp2.x;
 
         // Defaults for regular matrix multiplication:
         int col_low    = 0;
@@ -3612,11 +3612,11 @@ static __global__ void mul_mat_q(
             offset_dst = 0;
 
             if (jt*mmq_x >= col_diff) {
-                kbc += blocks_per_ne00;
-                kbc -= kbc % blocks_per_ne00;
+                kbc += blocks_per_ne00.z;
+                kbc -= fastmodulo(kbc, blocks_per_ne00);
 
                 kb0_start = 0;
-                kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
+                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
 
                 continue;
             }
@@ -3641,32 +3641,34 @@ static __global__ void mul_mat_q(
         const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
         const int tile_y_max_j = col_diff - jt*mmq_x - 1;
 
-        const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;
+        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;
 
         constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
         mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
             (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
              tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
 
-        kbc += blocks_per_ne00;
-        kbc -= kbc % blocks_per_ne00;
+        kbc += blocks_per_ne00.z;
+        kbc -= fastmodulo(kbc, blocks_per_ne00);
 
         kb0_start = 0;
-        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
+        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
     }
 
     if (kbc >= kbc_stop) {
         return;
     }
 
-    int tmp = kbc;
-    const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
-    tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
-    const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
-    tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
-    const int zt = tmp / (ntx*blocks_per_ne00);
-    tmp -= zt * (ntx*blocks_per_ne00);
-    const int jt = tmp / blocks_per_ne00;
+    int tmp = fastdiv(kbc, blocks_per_ne00);
+    uint2 tmp2 = fast_div_modulo(tmp, ntx);
+    const int jt = tmp2.y;
+    tmp = tmp2.x;
+    tmp2 = fast_div_modulo(tmp, nchannels_y);
+    const int zt = tmp2.y;
+    tmp = tmp2.x;
+    tmp2 = fast_div_modulo(tmp, nsamples_y);
+    const int wt = tmp2.y;
+    const int it = tmp2.x;
 
     // Defaults for regular matrix multiplication:
     int col_low    = 0;
@@ -3708,7 +3710,7 @@ static __global__ void mul_mat_q(
     const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
     const int tile_y_max_j = col_diff - jt*mmq_x - 1;
 
-    const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;
+    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;
 
     constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
     mul_mat_q_process_tile<type, mmq_x, need_check, fixup>
@@ -3717,46 +3719,37 @@ static __global__ void mul_mat_q(
 }
 
 template <ggml_type type, int mmq_x, bool need_check>
-static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
-                                                const int32_t * expert_bounds,
-                                                float * __restrict__ dst,
-                                                const float * __restrict__ tmp_last_tile,
-                                                const int    ncols_x,
-                                                const int    nrows_x,
-                                                const int    ncols_dst,
-                                                const size_t stride_col_dst,
-                                                const int    nchannels_y,
-                                                const size_t stride_channel_dst,
-                                                const int    nsamples_y,
-                                                const size_t stride_sample_dst,
-                                                const int    ncols_max) {
-    constexpr int     mmq_y           = get_mmq_y_device();
-    constexpr int     qk              = ggml_cuda_type_traits<type>::qk;
-    constexpr int     ITER_K          = get_iter_k(type);
-
-    constexpr int     blocks_per_iter = ITER_K / qk;
-    const     int64_t blocks_per_ne00 = ncols_x / qk;
+__launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device()/2, 1)
+static __global__ void mul_mat_q_stream_k_fixup(
+        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
+        float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
+        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
+        const int stride_sample_dst, const uint3 ntx) {
+    constexpr int mmq_y           = get_mmq_y_device();
+    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
+    constexpr int ITER_K          = get_iter_k(type);
+    constexpr int blocks_per_iter = ITER_K / qk;
 
-    constexpr int nwarps = mmq_get_nwarps_device();
+    constexpr int nwarps = mmq_get_nwarps_device()/2;
     constexpr int warp_size = ggml_cuda_get_physical_warp_size();
 
-    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};
+    float sum[mmq_x / nwarps] = {0.0f};
+    const int i = blockIdx.y*warp_size + threadIdx.x;
 
-    const int ntx  = (ncols_max + mmq_x - 1) / mmq_x;
-    const int nty  = (nrows_x   + mmq_y - 1) / mmq_y;
+    const int nty = (nrows_x + mmq_y - 1) / mmq_y;
 
     const int bidx0 = blockIdx.x;
 
     // kbc == k block continuous, current index in continuous ijk space.
-    int64_t kbc0      = (int64_t) bidx0     *nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
-    int64_t kbc0_stop = (int64_t)(bidx0 + 1)*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
+    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
+    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
 
-    kbc0      -= (kbc0      % blocks_per_ne00) % blocks_per_iter;
-    kbc0_stop -= (kbc0_stop % blocks_per_ne00) % blocks_per_iter;
+    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
+    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;
 
     const bool did_not_have_any_data   = kbc0 == kbc0_stop;
-    const bool wrote_beginning_of_tile = kbc0 % blocks_per_ne00 == 0;
-    const bool did_not_write_last      = kbc0/blocks_per_ne00 == kbc0_stop/blocks_per_ne00 && kbc0_stop % blocks_per_ne00 != 0;
+    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
+    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
     if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
         return;
     }
@@ -3765,11 +3758,11 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
 
     // Iterate over previous blocks and sum up partial sums written to fixup buffer.
     // All CUDA blocks that get here must have a previous block that needs a fixup.
-    int64_t bidx = bidx0 - 1;
-    int64_t kbc_stop = kbc0;
+    int bidx = bidx0 - 1;
+    int kbc_stop = kbc0;
     while(true) {
-        int64_t kbc = bidx*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
-        kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;
+        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
+        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;
 
         if (kbc == kbc_stop) { // Did not have any data.
             bidx--;
@@ -3779,20 +3772,16 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
 
         any_fixup = true;
 
+
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
             const int j = j0 + threadIdx.y;
 
-#pragma unroll
-            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
-                const int i = i0 + threadIdx.x;
-
-                sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
-            }
+            sum[j0/nwarps] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
         }
 
         // If this block started in a previous tile we are done and don't need to combine additional partial results.
-        if (kbc % blocks_per_ne00 == 0 || kbc/blocks_per_ne00 < kbc0/blocks_per_ne00) {
+        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
             break;
         }
         bidx--;
@@ -3803,14 +3792,16 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
         return;
     }
 
-    int tmp = kbc0;
-    const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
-    tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
-    const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
-    tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
-    const int zt = tmp / (ntx*blocks_per_ne00);
-    tmp -= zt * (ntx*blocks_per_ne00);
-    const int jt = tmp / blocks_per_ne00;
+    int tmp = fastdiv(kbc0, blocks_per_ne00);
+    uint2 tmp2 = fast_div_modulo(tmp, ntx);
+    const int jt = tmp2.y;
+    tmp = tmp2.x;
+    tmp2 = fast_div_modulo(tmp, nchannels_y);
+    const int zt = tmp2.y;
+    tmp = tmp2.x;
+    tmp2 = fast_div_modulo(tmp, nsamples_y);
+    const int wt = tmp2.y;
+    const int it = tmp2.x;
 
     if (!ids_dst) {
         const int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst + it*mmq_y;
@@ -3818,6 +3809,9 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
 
         const int i_max = nrows_x   - it*mmq_y - 1;
         const int j_max = ncols_dst - jt*mmq_x - 1;
+        if (need_check && i > i_max) {
+            return;
+        }
 
 #pragma unroll
         for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
@@ -3827,16 +3821,7 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
                 return;
             }
 
-#pragma unroll
-            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
-                const int i = i0 + threadIdx.x;
-
-                if (need_check && i > i_max) {
-                    continue;
-                }
-
-                dst[j*stride_col_dst + i] += sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size];
-            }
+            dst[j*stride_col_dst + i] += sum[j0/nwarps];
         }
         return;
     }
@@ -3856,6 +3841,9 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
 
     const int i_max = nrows_x  - it*mmq_y - 1;
     const int j_max = col_diff - jt*mmq_x - 1;
+    if (need_check && i > i_max) {
+        return;
+    }
 
 #pragma unroll
     for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
@@ -3865,16 +3853,7 @@ static __global__ void mul_mat_q_stream_k_fixup(const int32_t * ids_dst,
             return;
         }
 
-#pragma unroll
-        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
-            const int i = i0 + threadIdx.x;
-
-            if (need_check && i > i_max) {
-                continue;
-            }
-
-            dst[ids_dst_shared[j]*stride_col_dst + i] += sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size];
-        }
+        dst[ids_dst_shared[j]*stride_col_dst + i] += sum[j0/nwarps];
     }
 }
 
@@ -3922,29 +3901,44 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
     const int channel_ratio = args.nchannels_y / args.nchannels_x;
     const int sample_ratio  = args.nsamples_y  / args.nsamples_x;
 
+    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
+    const uint3 ntx_fd             = init_fastdiv_values(ntx);
+    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
+    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
+    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
+    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);
+
     if (!args.use_stream_k) {
         if (args.nrows_x % mmq_y == 0) {
             constexpr bool need_check = false;
             mul_mat_q<type, mmq_x, need_check><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                 (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
-                 args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
-                 channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
-                 sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
-                 args.ncols_max);
+                 blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
+                 channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
+                 sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
+                 ntx_fd);
         } else {
             constexpr bool need_check = true;
             mul_mat_q<type, mmq_x, need_check><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
                 (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
-                 args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
-                 channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
-                 sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
-                 args.ncols_max);
+                 blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
+                 channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
+                 sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
+                 ntx_fd);
         }
         return;
     }
 
-    const dim3 block_nums_stream_k(nsm, 1, 1);
-    const bool fixup_needed = ntx*nty*ntzw % nsm != 0;
+    // For the stream-k kernel it is possible to run it with tiling by setting the number of CUDA blocks equal to the number of tiles.
+    // This is worthwhile if the efficiency of tiling is high and skipping the fixup kernel is more important.
+    const int ntiles_dst = ntx * nty * ntzw;
+    const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
+    const int tiles_efficiency_percent = 100 * ntiles_dst / (nsm*tiles_nwaves);
+    const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);
+
+    GGML_ASSERT(ntiles_dst * blocks_per_ne00_fd.z < (1 << 30)); // Assert that variable kbc will not overflow.
+
+    const bool fixup_needed = ntiles_dst % block_nums_stream_k.x != 0;
 
     ggml_cuda_pool & pool = ctx.pool(id);
     ggml_cuda_pool_alloc<float> tmp_fixup(pool);
@@ -3952,40 +3946,45 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
         tmp_fixup.alloc(block_nums_stream_k.x * mmq_x*mmq_y);
     }
 
+    const dim3 block_nums_fixup(block_nums_stream_k.x, mmq_y/warp_size, 1);
+    const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);
+
     if (args.nrows_x % mmq_y == 0) {
         constexpr bool need_check = false;
         mul_mat_q<type, mmq_x, need_check><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
             (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
-             args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
-             channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
-             sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
-             args.ncols_max);
+             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
+             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
+             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
+             ntx_fd);
 
         if (!fixup_needed) {
             return;
         }
 
-        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<block_nums_stream_k, block_dims, 0, stream>>>
-            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.ncols_x, args.nrows_x, args.ncols_dst,
-             args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y, args.stride_sample_dst,
-             args.ncols_max);
+        CUDA_CHECK(cudaGetLastError());
+        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
+            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
+             args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
+             ntx_fd);
     } else {
         constexpr bool need_check = true;
         mul_mat_q<type, mmq_x, need_check><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
             (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr,
-             args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
-             channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
-             sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
-             args.ncols_max);
+             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
+             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
+             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
+             ntx_fd);
 
         if (!fixup_needed) {
             return;
         }
 
-        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<block_nums_stream_k, block_dims, 0, stream>>>
-            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.ncols_x, args.nrows_x, args.ncols_dst,
-             args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y, args.stride_sample_dst,
-             args.ncols_max);
+        CUDA_CHECK(cudaGetLastError());
+        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
+            (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
+             args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
+             ntx_fd);
     }
 }
 
diff --git src/ggml-cuda/unary.cu src/ggml-cuda/unary.cu
index 4ad30fa1..2aeba26f 100644
--- src/ggml-cuda/unary.cu
+++ src/ggml-cuda/unary.cu
@@ -65,6 +65,11 @@ static __device__ __forceinline__ float op_sqr(float x) {
     return x * x;
 }
 
+static __device__ __forceinline__ float op_relu_sqr(float x) {
+    const float r = fmaxf(x, 0.0f);
+    return r * r;
+}
+
 static __device__ __forceinline__ float op_sqrt(float x) {
     return sqrtf(x);
 }
@@ -615,3 +620,21 @@ void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary
             GGML_ABORT("Unsupported unary op for fused unary+mul");
     }
 }
+
+/* fused relu + sqr */
+
+void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
+    const ggml_tensor * src = relu_node->src[0];
+    cudaStream_t stream = ctx.stream();
+
+    GGML_ASSERT(ggml_is_contiguous(src));
+    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
+    GGML_ASSERT(src->type == sqr_node->type);
+
+    const int k = ggml_nelements(src);
+    if (src->type == GGML_TYPE_F16) {
+        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
+    } else {
+        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
+    }
+}
diff --git src/ggml-cuda/unary.cuh src/ggml-cuda/unary.cuh
index f1dd2183..81ed873e 100644
--- src/ggml-cuda/unary.cuh
+++ src/ggml-cuda/unary.cuh
@@ -91,6 +91,8 @@ void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
 
 void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);
 
+void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);
+
 __device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
     return x / (1.0f + expf(-x));
 }
diff --git src/ggml-hexagon/ggml-hexagon.cpp src/ggml-hexagon/ggml-hexagon.cpp
index 3d68b800..0d9b5e28 100644
--- src/ggml-hexagon/ggml-hexagon.cpp
+++ src/ggml-hexagon/ggml-hexagon.cpp
@@ -12,9 +12,12 @@
 #include <cstddef>
 #include <stdexcept>
 #include <string>
+#include <sstream>
+#include <iomanip>
 #include <unordered_set>
 #include <unordered_map>
 #include <regex>
+#include <queue>
 
 #ifdef _WIN32
 #    include <sal.h>
@@ -41,18 +44,26 @@
 #include "htp_iface.h"
 #include "htp-drv.h"
 
+using intvec  = std::vector<int>;
+using uintvec = std::vector<unsigned int>;
+using u32vec  = std::vector<uint32_t>;
+
 static size_t opt_ndev         = 1;
 static size_t opt_nhvx         = 0; // use all
 static int    opt_arch         = 0; // autodetect
 static int    opt_etm          = 0;
 static int    opt_verbose      = 0;
-static int    opt_profile      = 0;
+static int    opt_profile      = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
 static int    opt_hostbuf      = 1; // hostbuf ON by default
 static int    opt_use_hmx      = 1; // when set, enable HMX; when 0, use HVX only
 
+// Default PMU events, if profiling with PMU (mode=2) is enabled
+// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
+//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
+static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };
+
 // Enable all stages by default
-static int opt_opmask   = HTP_OPMASK_QUEUE | HTP_OPMASK_COMPUTE;
-static int opt_opsync   = 0;  // synchronous ops
+static int opt_opstage  = HTP_OPSTAGE_QUEUE | HTP_OPSTAGE_COMPUTE;
 static int opt_opbatch  = 1024; // max number of ops in a batch
 static int opt_opqueue  = 16;   // max number of pending batches
 static std::regex* opt_opfilter = NULL; // regex of ops to not claim
@@ -104,19 +115,26 @@ static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct
 }
 
 static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const ggml_tensor * op,
-                                      uint32_t op_usec, uint32_t op_cycles, uint32_t op_pkts, uint64_t call_usec) {
+                                      uint32_t op_usec, uint32_t op_cycles, const uint32_t pmu[]) {
     if (!opt_profile) return;
 
     op_desc desc(op);
-    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s: %s : %s : %s : %s : %s : op-usec %u op-cycles %u op-pkts %u (%f) call-usec %llu\n", sess_name.c_str(),
-                ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, desc.buffs,
-                op_usec, op_cycles, op_pkts, (float) op_cycles / op_pkts, (unsigned long long) call_usec);
+
+    char pmu_str[256] = "";
+    if (opt_profile > 1) {
+        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
+        sprintf(pmu_str, " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
+                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
+    }
+
+    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s: %s : %s : %s : %s : usec %u cycles %u%s\n", sess_name.c_str(),
+            ggml_op_desc(op), desc.names, desc.dims, desc.types, desc.strides, op_usec, op_cycles, pmu_str);
 }
 
 // ** backend sessions
 
 struct ggml_hexagon_opbatch;
-struct ggml_hexagon_opshm;
+struct ggml_hexagon_opqueue;
 
 struct ggml_hexagon_session {
     std::string      name;
@@ -132,8 +150,8 @@ struct ggml_hexagon_session {
     bool             valid_iface;
 
     std::atomic<int>      op_pending;
-    ggml_hexagon_opbatch *op_batch;
-    ggml_hexagon_opshm   *op_shm;
+    ggml_hexagon_opbatch* op_batch;
+    ggml_hexagon_opqueue* op_queue;
 
     ggml_backend_buffer_type buffer_type        = {};
     ggml_backend_buffer_type repack_buffer_type = {};
@@ -1521,65 +1539,14 @@ static ggml_backend_buffer_type_i ggml_backend_hexagon_repack_buffer_type_interf
 
 // Backend session implementation
 
-struct ggml_hexagon_opshm {
-    ggml_hexagon_shared_buffer *sbuf;
-
-    std::vector<bool> block_mask;
-    size_t            block_size;
-
-    uint8_t * base()     const { return this->sbuf->base; }
-    int       fd()       const { return this->sbuf->fd;   }
-    size_t    n_blocks() const { return this->block_mask.size(); }
-
-    ggml_hexagon_opshm(ggml_hexagon_session *sess, size_t max_batch, size_t max_pending) {
-        size_t n_bufs    = HTP_OP_MAX_BUFS;
-        size_t n_ops     = max_batch;
-        size_t n_tensors = n_ops + n_ops * HTP_OP_MAX_INPUTS;
-
-        block_mask.resize(max_pending, true);
-
-        block_size = sizeof(htp_buf_desc) * n_bufs    +
-                     sizeof(htp_tensor)   * n_tensors +
-                     sizeof(htp_op_desc)  * n_ops;
-
-        sbuf = new ggml_hexagon_shared_buffer(sess, block_size * block_mask.size(), true /* pinned */);
-
-        if (opt_verbose) {
-            GGML_LOG_INFO("ggml-hex: %s allocated shared buf %zu : block-size %zu max-batch %zu max-pending %zu\n",
-                    sess->c_name(), (size_t) sbuf->size, block_size, max_batch, max_pending);
-        }
-    }
-
-    ~ggml_hexagon_opshm() {
-        delete sbuf;
-    }
-
-    uint8_t * allocate() {
-        auto it = std::find(block_mask.begin(), block_mask.end(), true);
-        if (it == block_mask.end())
-            return nullptr;
-
-        unsigned int i = std::distance(block_mask.begin(), it);
-        uint8_t*  addr = sbuf->base + (i * block_size);
-        block_mask[i]  = false;
-
-        HEX_VERBOSE("ggml-hex: %s allocated op shm #%u %p\n", sbuf->sess->c_name(), i, (void*) addr);
-        return addr;
-    }
-
-    void release(uint8_t * addr) {
-        int i = (addr - sbuf->base) / block_size;
-        block_mask[i] = true;
-        HEX_VERBOSE("ggml-hex: %s released op shm #%u %p\n", sbuf->sess->c_name(), i, (void*) addr);
-    }
-};
-
 struct ggml_hexagon_opbatch {
-    const char* name;
+    ggml_hexagon_session*            sess;
 
-    std::vector<htp_buf_desc> buffers;
-    std::vector<htp_tensor>   tensors;
-    std::vector<htp_op_desc>  ops;
+    std::vector<const ggml_tensor*>  ops;       // pointers to original ops
+
+    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
+    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
+    std::vector<htp_op_desc>         h_ops;     // htp op descriptors
 
     std::unordered_map<int, int>                b_map; // buffer fd   to index
     std::unordered_map<const ggml_tensor*, int> t_map; // tensor ptr  to index
@@ -1606,19 +1573,21 @@ struct ggml_hexagon_opbatch {
         d_map.clear();
     }
 
-    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t max_batch) {
-        name = sess->c_name();
+    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size) {
+        this->sess = sess;
 
         n_bufs_max = HTP_OP_MAX_BUFS;
-        n_ops_max  = max_batch;
+        n_ops_max  = batch_size;
         n_tens_max = n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS;
 
         b_vmem_max = HTP_OP_MAX_VMEM;
 
-        buffers.resize(n_bufs_max);
-        tensors.resize(n_tens_max);
         ops.resize(n_ops_max);
 
+        h_bufs.resize(n_bufs_max);
+        h_tens.resize(n_tens_max);
+        h_ops.resize(n_ops_max);
+
         b_map.reserve(n_bufs_max);
         t_map.reserve(n_tens_max);
         d_map.reserve(n_tens_max);
@@ -1640,7 +1609,7 @@ struct ggml_hexagon_opbatch {
 
         b_map.insert({sbuf->fd, bi});
 
-        htp_buf_desc &b = buffers[bi];
+        htp_buf_desc &b = h_bufs[bi];
         b.base = (uint64_t) sbuf->base;
         b.fd   = sbuf->fd;
         b.size = sbuf->size;
@@ -1664,7 +1633,7 @@ struct ggml_hexagon_opbatch {
         // First lookup by tensor data
         auto range = d_map.equal_range(t->data);
         for (auto it = range.first; it != range.second; ++it) {
-            htp_tensor * h = &tensors[it->second];
+            htp_tensor * h = &h_tens[it->second];
             if (same_shape(h, t)) { return it->second; }
         }
 
@@ -1682,7 +1651,7 @@ struct ggml_hexagon_opbatch {
         uint64_t t_offset = (uint8_t *) t->data - sbuf->base;
         size_t   t_size   = ggml_nbytes(t);
 
-        htp_tensor &h = tensors[ti];
+        htp_tensor &h = h_tens[ti];
         h.bi    = add_buffer(sbuf);
         h.data  = t_offset;
         h.size  = t_size;
@@ -1737,65 +1706,170 @@ struct ggml_hexagon_opbatch {
     // assumes that fit_op() was called first and returned true
     void add_op(htp_op_code opcode, const struct ggml_tensor * t) {
         // Add new op
-        htp_op_desc &o = ops[n_ops++];
+
+        unsigned int n = n_ops++;
         GGML_ASSERT(n_ops <= n_ops_max);
 
+        ops[n] = t;
+
+        htp_op_desc &o = h_ops[n];
         memcpy(&o.params, &t->op_params, sizeof(t->op_params));
         o.opcode = opcode;
         o.flags  = 0;
 
-        if (!(opt_opmask & HTP_OPMASK_COMPUTE)) {
+        if (!(opt_opstage & HTP_OPSTAGE_COMPUTE)) {
             o.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
         }
 
-        ggml_hexagon_dump_op_exec(name, t, o.flags);
+        ggml_hexagon_dump_op_exec(sess->c_name(), t, o.flags);
 
         for (unsigned int i=0; i < HTP_OP_MAX_INPUTS; i++) {
             o.src[i] = t->src[i] ? add_tensor(t->src[i]) : 0xffff;
         }
         o.dst = add_tensor(t);
     }
+};
+
+struct ggml_hexagon_opqueue {
+    // Shared buffer for storing batches
+    ggml_hexagon_shared_buffer *shm_buf;
+    size_t                      shm_blk_size;
+
+    using opvec = std::vector<const ggml_tensor*>;
+
+    std::queue<unsigned int>    done;       // completed batch ids
+    std::vector<opvec>          op_cache;   // per batch op cache
+    std::vector<uint64_t>       start_usec; // per batch start time
+
+    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) {
+        size_t n_bufs    = HTP_OP_MAX_BUFS;
+        size_t n_ops     = batch_size;
+        size_t n_tensors = n_ops + n_ops * HTP_OP_MAX_INPUTS;
+
+        shm_blk_size = sizeof(htp_buf_desc)  * n_bufs    +
+                       sizeof(htp_tensor)    * n_tensors +
+                       sizeof(htp_op_desc)   * n_ops     +
+                       sizeof(htp_prof_desc) * n_ops;
+
+        shm_buf = new ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, true /* pinned */);
+
+        op_cache.resize(depth);
+        start_usec.resize(depth, 0);
+
+        // init done queue
+        for (unsigned int i = 0; i < depth; i++) { done.push(i); }
+
+        if (opt_verbose) {
+            GGML_LOG_INFO("ggml-hex: %s allocated op-queue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
+                    sess->c_name(), batch_size, depth, shm_buf->size, shm_blk_size);
+        }
+    }
 
-    size_t flush(uint8_t * mem_addr, size_t mem_size) {
-        static_assert(sizeof(htp_buf_desc) % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
-        static_assert(sizeof(htp_tensor)   % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
-        static_assert(sizeof(htp_op_desc)  % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
+    ~ggml_hexagon_opqueue() {
+        delete shm_buf;
+    }
 
-        const size_t b_size = sizeof(htp_buf_desc) * n_bufs;
-        const size_t t_size = sizeof(htp_tensor)   * n_tens;
-        const size_t o_size = sizeof(htp_op_desc)  * n_ops;
+    // push new batch
+    bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, ggml_hexagon_opbatch* op_batch) {
+        static_assert(sizeof(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
+        static_assert(sizeof(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
+        static_assert(sizeof(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
+        static_assert(sizeof(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
+        static_assert(sizeof(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
+        static_assert(sizeof(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");
 
-        const size_t m_size = b_size + t_size + o_size;
-        GGML_ASSERT(m_size <= mem_size);
+        if (done.empty()) { return false; }
 
-        uint8_t * b_ptr = (uint8_t *) mem_addr;
-        uint8_t * t_ptr = (uint8_t *) b_ptr + b_size;
-        uint8_t * o_ptr = (uint8_t *) t_ptr + t_size;
+        req.id        = done.front(); done.pop(); // batch id
+        req.n_bufs    = op_batch->n_bufs;
+        req.n_tensors = op_batch->n_tens;
+        req.n_ops     = op_batch->n_ops;
 
-        memcpy(b_ptr, (void *) buffers.data(), b_size);
-        memcpy(t_ptr, (void *) tensors.data(), t_size);
-        memcpy(o_ptr, (void *) ops.data(),     o_size);
+        op_cache[req.id]   = op_batch->ops;
+        start_usec[req.id] = ggml_time_us();
 
-        HEX_VERBOSE("ggml-hex: %s flush-opbatch : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu\n",
-                name, n_bufs, n_tens, n_ops, b_vmem, b_size, t_size, o_size);
+        const size_t b_size = sizeof(htp_buf_desc)  * req.n_bufs;
+        const size_t t_size = sizeof(htp_tensor)    * req.n_tensors;
+        const size_t o_size = sizeof(htp_op_desc)   * req.n_ops;
+        const size_t p_size = sizeof(htp_prof_desc) * req.n_ops;
+
+        dbuf.ptr      = shm_buf->base + (req.id * shm_blk_size);
+        dbuf.fd       = shm_buf->fd;
+        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
+        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base;
+        dbuf.size     = b_size + t_size + o_size + p_size;
+
+        GGML_ASSERT(dbuf.size <= shm_blk_size);
+
+        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
+        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
+        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
+        uint8_t * o_ptr = m_ptr;
+
+        memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);
+        memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);
+        memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);
+
+        HEX_VERBOSE("ggml-hex: %s op-queue push batch #%u : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
+                shm_buf->sess->c_name(), req.id, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
+                b_size, t_size, o_size, (size_t) dbuf.size);
+
+        op_batch->reset();
 
         if (opt_verbose > 1) {
             htp_buf_desc *b = (htp_buf_desc*) b_ptr;
-            for (unsigned int i=0; i < n_bufs; i++) {
-                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", name, i,
+            for (unsigned int i=0; i < req.n_bufs; i++) {
+                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                             b[i].fd, (void *) b[i].base, (size_t) b[i].size);
             }
             htp_tensor *t = (htp_tensor*) t_ptr;
-            for (unsigned int i=0; i < n_tens; i++) {
+            for (unsigned int i=0; i < req.n_tensors; i++) {
                 GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %u size %u : %zu:%zu:%zu:%zu\n",
-                            name, i, t[i].bi, t[i].data, t[i].size,
+                            shm_buf->sess->c_name(), i, t[i].bi, t[i].data, t[i].size,
                             (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
             }
         }
 
-        reset();
+        return true;
+    }
 
-        return m_size;
+    void pop(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
+        GGML_ASSERT(rsp.id < op_cache.size());
+
+        done.push(rsp.id);
+
+        const size_t b_size = sizeof(htp_buf_desc)  * rsp.n_bufs;
+        const size_t t_size = sizeof(htp_tensor)    * rsp.n_tensors;
+        const size_t o_size = sizeof(htp_op_desc)   * rsp.n_ops;
+        const size_t p_size = sizeof(htp_prof_desc) * rsp.n_ops;
+
+        const size_t m_size = b_size + t_size + o_size + p_size;
+        GGML_ASSERT(m_size <= shm_blk_size);
+
+        HEX_VERBOSE("ggml-hex: %s op-queue pop batch #%u : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
+                shm_buf->sess->c_name(), rsp.id, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
+                (size_t) dbuf.size, b_size, t_size, o_size);
+
+        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
+        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);
+
+        if (opt_profile && rsp.n_ops > 0) {
+            auto & ops = op_cache[rsp.id];
+
+            uint64_t batch_usec = ggml_time_us() - start_usec[rsp.id];
+            uint32_t htp_usec   = 0;
+
+            GGML_ASSERT(rsp.n_ops <= ops.size());
+
+            const htp_prof_desc * pd = (const htp_prof_desc *) p_ptr;
+            for (uint32_t i = 0; i < rsp.n_ops; i++) {
+                htp_usec += pd[i].usecs;
+                ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i].usecs, pd[i].cycles, pd[i].pmu);
+            }
+
+            GGML_LOG_DEBUG("ggml-hex: %s profile-batch n-ops %u batch-dur-usec %lld htp-ops-usec %u\n",
+                           shm_buf->sess->c_name(), rsp.n_ops, (long long) batch_usec, htp_usec);
+        }
     }
 };
 
@@ -1824,17 +1898,12 @@ void ggml_hexagon_session::flush_pending(bool all) {
             GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", this->c_name(), rsp_size, n_dbufs);
         }
 
-        op_shm->release((uint8_t*) dbuf.ptr);
-
         if (rsp.status != HTP_STATUS_OK) {
             GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp: %s\n", this->c_name(), status_to_str(rsp.status));
             // TODO: handle errors
         }
 
-        // FIXME: profile will be per opreq
-        // this->prof_usecs  = rsp.prof_usecs;
-        // this->prof_cycles = rsp.prof_cycles;
-        // this->prof_pkts   = rsp.prof_pkts;
+        op_queue->pop(rsp, dbuf);
 
         this->op_pending--;  // atomic dec
 
@@ -1845,28 +1914,17 @@ void ggml_hexagon_session::flush_pending(bool all) {
 void ggml_hexagon_session::flush_batch() {
     if (op_batch->empty()) { return; }
 
-    htp_opbatch_req req;
-    req.n_bufs    = op_batch->n_bufs;
-    req.n_tensors = op_batch->n_tens;
-    req.n_ops     = op_batch->n_ops;
+    htp_opbatch_req req {};
+    dspqueue_buffer dbuf{};
 
-    dspqueue_buffer dbuf;
-    dbuf.fd     = op_shm->fd();
-    dbuf.flags  = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
-    dbuf.ptr    = op_shm->allocate();
-    if (!dbuf.ptr) {
+    if (!op_queue->push(req, dbuf, op_batch)) {
         flush_pending(false);
-        dbuf.ptr = op_shm->allocate();
+        op_queue->push(req, dbuf, op_batch);
     }
 
-    dbuf.offset = (uint8_t*) dbuf.ptr - (uint8_t*) op_shm->base();
-    dbuf.size   = op_batch->flush((uint8_t*) dbuf.ptr, op_shm->block_size);
-
     // Bump pending flag (cleared in the session::flush once we get the response)
     this->op_pending++;  // atomic inc
 
-    HEX_VERBOSE("ggml-hex: %s: queue-opbatch : %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);
-
     int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
     if (err != 0) {
         GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
@@ -2016,25 +2074,33 @@ void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
     }
 
     if (opt_etm) {
-        err = htp_iface_enable_etm(this->handle);
+        err = htp_iface_etm(this->handle, 1);
         if (err != 0) {
             GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
         }
     }
 
-    // Start the DSP-side service. We need to pass the queue ID to the
-    // DSP in a FastRPC call; the DSP side will import the queue and start
-    // listening for packets in a callback.
+    if (opt_profile) {
+        htp_iface_pmu_conf pmu_conf{};
+        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);
+
+        err = htp_iface_profiler(this->handle, opt_profile, &pmu_conf);
+        if (err != 0) {
+            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
+        }
+    }
+
+    // Allocate buffers and state for op batching
+    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch);
+    this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);
+
+    // Start processing op batch requests
     err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx, opt_use_hmx);
     if (err != 0) {
         GGML_LOG_ERROR("ggml-hex: failed to start session: 0x%08x\n", (unsigned) err);
         throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
     }
     this->valid_iface = true;
-
-    // Allocate buffers and state for op batching
-    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch);
-    this->op_shm   = new ggml_hexagon_opshm(this, opt_opbatch, opt_opqueue);
 }
 
 void ggml_hexagon_session::release() noexcept(true) {
@@ -2043,7 +2109,7 @@ void ggml_hexagon_session::release() noexcept(true) {
     int err;
 
     delete this->op_batch;
-    delete this->op_shm;
+    delete this->op_queue;
 
     // Stop the DSP-side service and close the queue
     if (this->valid_iface) {
@@ -2054,12 +2120,20 @@ void ggml_hexagon_session::release() noexcept(true) {
     }
 
     if (opt_etm) {
-        err = htp_iface_disable_etm(this->handle);
+        err = htp_iface_etm(this->handle, 0);
         if (err != 0) {
             GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
         }
     }
 
+    if (opt_profile) {
+        htp_iface_pmu_conf pmu_conf{};
+        err = htp_iface_profiler(this->handle, 0, &pmu_conf);
+        if (err != 0) {
+            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
+        }
+    }
+
     if (this->valid_queue) {
         err = dspqueue_close(queue);
         if (err != 0) {
@@ -2077,7 +2151,7 @@ ggml_hexagon_session::ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) n
     repack_buffer_type.device = dev;
 
     op_batch = nullptr;
-    op_shm   = nullptr;
+    op_queue = nullptr;
 
     try {
         allocate(dev_id);
@@ -2596,6 +2670,62 @@ static bool ggml_hexagon_supported_cumsum(const struct ggml_hexagon_session * se
     return true;
 }
 
+static bool ggml_hexagon_supported_diag(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
+    const struct ggml_tensor * src0 = op->src[0];
+    const struct ggml_tensor * dst  = op;
+
+    // diag only supports F32 currently
+    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
+        return false;
+    }
+
+    // Input must have ne[1] == 1 (vector input)
+    if (src0->ne[1] != 1) {
+        return false;
+    }
+
+    // Output must be square in first two dimensions
+    if (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
+        return false;
+    }
+
+    GGML_UNUSED(sess);
+    return true;
+}
+
+static bool ggml_hexagon_supported_solve_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
+    const struct ggml_tensor * src0 = op->src[0]; // A
+    const struct ggml_tensor * src1 = op->src[1]; // B
+    const struct ggml_tensor * dst  = op;         // X
+
+    if (!src0 || !src1) {
+        return false;
+    }
+
+    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
+        return false;
+    }
+
+    if (src0->ne[0] != src0->ne[1]) {
+        return false;
+    }
+
+    if (src0->ne[1] != src1->ne[1]) {
+        return false;
+    }
+
+    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
+        return false;
+    }
+
+    if (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
+        return false;
+    }
+
+    GGML_UNUSED(sess);
+    return true;
+}
+
 static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
     auto sess = static_cast<ggml_hexagon_session *>(backend->context);
     return sess->c_name();
@@ -2632,7 +2762,9 @@ static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
         case GGML_OP_ROPE:           return HTP_OP_ROPE;
         case GGML_OP_REPEAT:         return HTP_OP_REPEAT;
         case GGML_OP_CUMSUM:         return HTP_OP_CUMSUM;
-
+        case GGML_OP_FILL:           return HTP_OP_FILL;
+        case GGML_OP_DIAG:           return HTP_OP_DIAG;
+        case GGML_OP_SOLVE_TRI:      return HTP_OP_SOLVE_TRI;
         case GGML_OP_UNARY:
             switch (ggml_get_unary_op(t)) {
                 case GGML_UNARY_OP_SILU:     return HTP_OP_UNARY_SILU;
@@ -2673,7 +2805,7 @@ static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, gg
 
     for (int i = 0; i < graph->n_nodes; ++i) {
         ggml_tensor * n = graph->nodes[i];
-        if (op_is_compute(n)) {
+        if (op_is_compute(n) && (opt_opstage & HTP_OPSTAGE_QUEUE)) {
             sess->enqueue_op(op_remap_to_htp(n), n);
         }
     }
@@ -3029,6 +3161,17 @@ static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * se
     return true;
 }
 
+static bool ggml_hexagon_supported_fill(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
+    const struct ggml_tensor * dst = op;
+
+    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
+        return false;
+    }
+
+    GGML_UNUSED(sess);
+    return true;
+}
+
 static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
     auto sess = static_cast<ggml_hexagon_session *>(dev->context);
 
@@ -3159,6 +3302,18 @@ static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, cons
             supp = ggml_hexagon_supported_cumsum(sess, op);
             break;
 
+        case GGML_OP_FILL:
+            supp = ggml_hexagon_supported_fill(sess, op);
+            break;
+
+        case GGML_OP_DIAG:
+            supp = ggml_hexagon_supported_diag(sess, op);
+            break;
+
+        case GGML_OP_SOLVE_TRI:
+            supp = ggml_hexagon_supported_solve_tri(sess, op);
+            break;
+
         default:
             break;
     }
@@ -3294,6 +3449,26 @@ static void * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, cons
     return NULL;
 }
 
+template<typename T> std::vector<T> str_to_vec(const char* str) {
+    std::stringstream ss(str);
+    std::vector<T> v;
+    std::string    t;
+
+    while (std::getline(ss, t, ',')) {
+        v.push_back(std::stoul(t, nullptr, 0));
+    }
+
+    return v;
+}
+
+template<typename T, int BASE=10> std::string vec_to_str(std::vector<T> v) {
+    std::stringstream ss;
+    ss << std::setbase(BASE) << std::showbase;
+    for (auto i : v) { ss << i << ','; }
+    auto str = ss.str(); str.pop_back(); // drop last comma
+    return str;
+}
+
 static void ggml_hexagon_init(ggml_backend_reg * reg) {
     // Basic sanity checks to make sure definitions match
     static_assert((unsigned int) HTP_TYPE_Q4_0 == (unsigned int) GGML_TYPE_Q4_0,
@@ -3307,8 +3482,7 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
 
     const char * str_verbose = getenv("GGML_HEXAGON_VERBOSE");
     const char * str_hostbuf = getenv("GGML_HEXAGON_HOSTBUF");
-    const char * str_opmask  = getenv("GGML_HEXAGON_OPMASK");
-    const char * str_opsync  = getenv("GGML_HEXAGON_OPSYNC");
+    const char * str_opstage = getenv("GGML_HEXAGON_OPSTAGE");
     const char * str_opbatch = getenv("GGML_HEXAGON_OPBATCH");
     const char * str_opqueue = getenv("GGML_HEXAGON_OPQUEUE");
     const char * str_opfilter= getenv("GGML_HEXAGON_OPFILTER");
@@ -3321,19 +3495,30 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
 
     auto RE_ICASE = std::regex_constants::icase;
 
-    opt_opfilter     = str_opfilter     ? new std::regex(str_opfilter, RE_ICASE) : NULL;
-    opt_verbose      = str_verbose ? atoi(str_verbose) : 0;
-    opt_hostbuf      = str_hostbuf ? atoi(str_hostbuf) : opt_hostbuf;
-    opt_opmask       = str_opmask  ? strtoul(str_opmask, NULL, 0)  : opt_opmask;
-    opt_opsync       = str_opsync  ? atoi(str_opsync)              : opt_opsync;
-    opt_opbatch      = str_opbatch ? strtoul(str_opbatch, NULL, 0) : opt_opbatch;
-    opt_opqueue      = str_opqueue ? strtoul(str_opqueue, NULL, 0) : opt_opqueue;
-    opt_profile      = str_profile ? atoi(str_profile) : 0;
-    opt_etm          = str_etm     ? atoi(str_etm)     : 0;
-    opt_nhvx         = str_nhvx    ? strtoul(str_nhvx, NULL, 0) : opt_nhvx;
-    opt_use_hmx      = str_use_hmx ? atoi(str_use_hmx) : opt_use_hmx;
-    opt_ndev         = str_ndev    ? strtoul(str_ndev, NULL, 0) : opt_ndev;
-    opt_hostbuf      = str_hostbuf ? atoi(str_hostbuf) : opt_hostbuf;
+    opt_opfilter     = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
+    opt_verbose      = str_verbose  ? atoi(str_verbose)             : 0;
+    opt_hostbuf      = str_hostbuf  ? atoi(str_hostbuf)             : opt_hostbuf;
+    opt_opstage      = str_opstage  ? strtoul(str_opstage, NULL, 0) : opt_opstage;
+    opt_opbatch      = str_opbatch  ? strtoul(str_opbatch, NULL, 0) : opt_opbatch;
+    opt_opqueue      = str_opqueue  ? strtoul(str_opqueue, NULL, 0) : opt_opqueue;
+    opt_etm          = str_etm      ? atoi(str_etm)                 : 0;
+    opt_nhvx         = str_nhvx     ? strtoul(str_nhvx, NULL, 0)    : opt_nhvx;
+    opt_use_hmx      = str_use_hmx  ? atoi(str_use_hmx)             : opt_use_hmx;
+    opt_ndev         = str_ndev     ? strtoul(str_ndev, NULL, 0)    : opt_ndev;
+    opt_hostbuf      = str_hostbuf  ? atoi(str_hostbuf)             : opt_hostbuf;
+
+    if (str_profile) {
+        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
+            auto v  = str_to_vec<uint32_t>(str_profile);
+            switch (v.size()) {
+                case 1:  opt_profile = v[0]; return opt_pmu_evt; // mode with default pmu events
+                case 8:  opt_profile = 2;    return v;           // mode with custom  pmu events
+                default: opt_profile = 0;    return {};          // garbage input
+            }}();
+        if (opt_profile == 1) opt_pmu_evt = {};
+        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
+                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
+    }
 
     if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
         opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
diff --git src/ggml-hexagon/htp/CMakeLists.txt src/ggml-hexagon/htp/CMakeLists.txt
index 9ca75945..8bd52847 100644
--- src/ggml-hexagon/htp/CMakeLists.txt
+++ src/ggml-hexagon/htp/CMakeLists.txt
@@ -34,6 +34,9 @@ add_library(${HTP_LIB} SHARED
     argsort-ops.c
     ssm-conv.c
     cumsum-ops.c
+    fill-ops.c
+    diag-ops.c
+    solve-tri-ops.c
 )
 
 target_compile_definitions(${HTP_LIB} PRIVATE
diff --git src/ggml-hexagon/htp/diag-ops.c src/ggml-hexagon/htp/diag-ops.c
new file mode 100644
index 00000000..9b3194d9
--- /dev/null
+++ src/ggml-hexagon/htp/diag-ops.c
@@ -0,0 +1,216 @@
+#pragma clang diagnostic ignored "-Wunused-but-set-variable"
+
+#include <HAP_farf.h>
+#include <HAP_perf.h>
+
+#define GGML_COMMON_DECL_C
+#include "ggml-common.h"
+#include "htp-ctx.h"
+#include "htp-ops.h"
+#include "hvx-types.h"
+#include "hex-utils.h"
+#include "hvx-copy.h"
+#include "hex-dma.h"
+
+#define htp_diag_tensors_preamble                           \
+    const struct htp_tensor * restrict src0 = octx->src[0]; \
+    const struct htp_tensor * restrict dst  = octx->dst;    \
+                                                     \
+    const uint32_t ne02 = src0->ne[2];               \
+                                                     \
+    const uint32_t ne0 = dst->ne[0];                 \
+    const uint32_t ne1 = dst->ne[1];                 \
+                                                     \
+    const uint32_t nb02 = src0->nb[2];               \
+    const uint32_t nb03 = src0->nb[3];               \
+                                                     \
+    const uint32_t nb1 = dst->nb[1];                 \
+    const uint32_t nb2 = dst->nb[2];                 \
+    const uint32_t nb3 = dst->nb[3];
+
+struct htp_diag_context {
+    struct htp_ops_context * octx;
+    size_t          src_batch_size;
+    size_t          dst_row_size;
+    size_t          src_batch_size_aligned;
+    size_t          dst_row_size_aligned;
+    uint32_t        batches_per_thread;
+    uint32_t        total_batches;
+};
+
+#define htp_diag_preamble                                              \
+    struct htp_diag_context * dctx = (struct htp_diag_context *) data; \
+    struct htp_ops_context *  octx = dctx->octx;                       \
+    htp_diag_tensors_preamble;
+
+static inline void hvx_diag_row_f32(const float * restrict src, float * restrict dst,
+                                    uint32_t row_idx, uint32_t n) {
+    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
+    dst[row_idx] = src[row_idx];
+}
+
+// ---------------------------------------------------------------------------
+// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
+// ---------------------------------------------------------------------------
+
+static void diag_thread_f32_dma(unsigned int nth, unsigned int ith, void * data) {
+    htp_diag_preamble;
+    dma_queue * dma_queue = octx->ctx->dma[ith];
+
+    uint64_t t1, t2;
+    t1 = HAP_perf_get_qtimer_count();
+
+    const uint32_t ib0 = dctx->batches_per_thread * ith;
+    const uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->total_batches);
+
+    if (ib0 >= ib1) {
+        return;
+    }
+
+    const size_t src_batch_size         = dctx->src_batch_size;
+    const size_t dst_row_size           = dctx->dst_row_size;
+    const size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
+    const size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;
+
+    const uint8_t * src_data = (const uint8_t *) src0->data;
+    uint8_t *       dst_data = (uint8_t *) dst->data;
+
+    // 1 src buffer + 1 dst row buffer per thread in VTCM
+    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
+    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);
+
+    for (uint32_t ib = ib0; ib < ib1; ib++) {
+        const uint32_t i3 = ib / ne02;
+        const uint32_t i2 = ib % ne02;
+
+        const uint8_t * src_batch = src_data + i3 * nb03 + i2 * nb02;
+
+        // Fetch source vector into VTCM
+        dma_queue_push_ddr_to_vtcm(dma_queue,
+                                   dma_make_ptr(src_spad, src_batch),
+                                   src_batch_size_aligned, src_batch_size, 1);
+        dma_queue_flush(dma_queue);
+
+        const float * src_spad_f32 = (const float *) src_spad;
+        float       * dst_spad_f32 = (float *) dst_spad;
+
+        for (uint32_t i1 = 0; i1 < ne1; i1++) {
+            // Compute row in VTCM
+            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
+
+            // Write completed row back to DDR
+            uint8_t * dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
+            dma_queue_push_vtcm_to_ddr(dma_queue,
+                                       dma_make_ptr(dst_row, dst_spad),
+                                       dst_row_size, dst_row_size_aligned, 1);
+            dma_queue_flush(dma_queue);
+        }
+    }
+
+    t2 = HAP_perf_get_qtimer_count();
+
+    FARF(HIGH, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n",
+         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
+         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
+         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
+}
+
+// ---------------------------------------------------------------------------
+// Per thread worker: Direct HVX (no DMA)
+// ---------------------------------------------------------------------------
+
+static void diag_thread_f32(unsigned int nth, unsigned int ith, void * data) {
+    htp_diag_preamble;
+
+    uint64_t t1, t2;
+    t1 = HAP_perf_get_qtimer_count();
+
+    const uint8_t * src_data = (const uint8_t *) src0->data;
+    uint8_t *       dst_data = (uint8_t *) dst->data;
+
+    const uint32_t ib0 = dctx->batches_per_thread * ith;
+    const uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->total_batches);
+
+    for (uint32_t ib = ib0; ib < ib1; ib++) {
+        const uint32_t i3 = ib / ne02;
+        const uint32_t i2 = ib % ne02;
+
+        const float * restrict src_batch = (const float *)(src_data + i3 * nb03 + i2 * nb02);
+
+        for (uint32_t i1 = 0; i1 < ne1; i1++) {
+            float * restrict dst_row = (float *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
+            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
+        }
+    }
+
+    t2 = HAP_perf_get_qtimer_count();
+
+    FARF(HIGH, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n",
+         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
+         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
+         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
+}
+
+int op_diag_f32(struct htp_ops_context * octx) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst  = octx->dst;
+
+    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
+        return HTP_STATUS_OK;
+    }
+
+    const uint32_t total_batches = src0->ne[2] * src0->ne[3];
+    const uint32_t n_threads     = MIN(octx->n_threads, total_batches);
+
+    const size_t src_batch_size         = src0->ne[0] * sizeof(float);
+    const size_t dst_row_size           = dst->ne[0] * sizeof(float);
+    const size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
+    const size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);
+
+    // 1 src buffer + 1 dst row buffer per thread
+    const size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;
+
+    octx->src0_spad.size_per_thread = src_batch_size_aligned;
+    octx->dst_spad.size_per_thread  = dst_row_size_aligned;
+
+    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
+    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;
+
+    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = NULL;
+    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = NULL;
+
+    struct htp_diag_context dctx = {
+        .octx                   = octx,
+        .src_batch_size         = src_batch_size,
+        .dst_row_size           = dst_row_size,
+        .src_batch_size_aligned = src_batch_size_aligned,
+        .dst_row_size_aligned   = dst_row_size_aligned,
+        .batches_per_thread     = (total_batches + n_threads - 1) / n_threads,
+        .total_batches          = total_batches,
+    };
+
+    if (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
+        worker_pool_run_func(octx->ctx->worker_pool, diag_thread_f32, &dctx, n_threads);
+    } else {
+        worker_pool_run_func(octx->ctx->worker_pool, diag_thread_f32_dma, &dctx, n_threads);
+    }
+
+    return HTP_STATUS_OK;
+}
+
+int op_diag(struct htp_ops_context * octx) {
+    const struct htp_tensor * dst = octx->dst;
+
+    int err = HTP_STATUS_OK;
+
+    switch (dst->type) {
+        case HTP_TYPE_F32:
+            err = op_diag_f32(octx);
+            break;
+        default:
+            err = HTP_STATUS_NO_SUPPORT;
+            break;
+    }
+
+    return err;
+}
diff --git src/ggml-hexagon/htp/fill-ops.c src/ggml-hexagon/htp/fill-ops.c
new file mode 100644
index 00000000..3ccfbe74
--- /dev/null
+++ src/ggml-hexagon/htp/fill-ops.c
@@ -0,0 +1,123 @@
+#pragma clang diagnostic ignored "-Wunused-variable"
+#pragma clang diagnostic ignored "-Wunused-function"
+#pragma clang diagnostic ignored "-Wunused-but-set-variable"
+
+#include <HAP_farf.h>
+#include <HAP_perf.h>
+
+#include <string.h>
+
+#include "hvx-copy.h"
+#include "hvx-utils.h"
+
+#define GGML_COMMON_DECL_C
+#include "ggml-common.h"
+#include "htp-ctx.h"
+#include "htp-ops.h"
+
+// ggml op_params layout for FILL:
+//   op_params[0] (as float) - the scalar fill value
+
+#define fill_preamble \
+    const struct htp_tensor * dst = octx->dst; \
+    \
+    const uint32_t ne0 = dst->ne[0]; \
+    const uint32_t ne1 = dst->ne[1]; \
+    const uint32_t ne2 = dst->ne[2]; \
+    const uint32_t ne3 = dst->ne[3]; \
+    \
+    const uint32_t nb1 = dst->nb[1]; \
+    const uint32_t nb2 = dst->nb[2]; \
+    const uint32_t nb3 = dst->nb[3]; \
+    \
+    const uint32_t nr = ne1 * ne2 * ne3;
+
+struct htp_fill_context {
+    struct htp_ops_context * octx;
+    uint32_t nrows_per_thread;
+    uint32_t total_rows;  // ne1 * ne2 * ne3
+    bool     opt_path;
+    HVX_Vector splat_vec;
+    uint32_t   elem_size;
+};
+
+static void fill_thread(unsigned int nth, unsigned int ith, void * data) {
+    const struct htp_fill_context * fctx = (const struct htp_fill_context *) data;
+    struct htp_ops_context        * octx = fctx->octx;
+    fill_preamble;
+
+    // Parallelise over the flat row index spanning ne1*ne2*ne3
+    const uint32_t ir0 = fctx->nrows_per_thread * ith;
+    const uint32_t ir1 = MIN(ir0 + fctx->nrows_per_thread, fctx->total_rows);
+
+    uint64_t t1 = HAP_perf_get_qtimer_count();
+
+    if (fctx->opt_path) {
+        // Opt path: tensor is fully contiguous, treat as flat array
+        const uint32_t elem_start = ir0 * ne0;
+        const uint32_t elem_end = ir1 * ne0;
+        uint8_t * dst_ptr = (uint8_t *) dst->data + elem_start * fctx->elem_size;
+        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_end - elem_start, fctx->elem_size);
+    } else {
+        // Non-contiguous path: must respect strides
+        for (uint32_t ir = ir0; ir < ir1; ++ir) {
+            const uint32_t i1 = ir % ne1;
+            const uint32_t i2 = (ir / ne1) % ne2;
+            const uint32_t i3 = ir / (ne1 * ne2);
+            uint8_t * dst_ptr = (uint8_t *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
+            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_size);
+        }
+    }
+
+    uint64_t t2 = HAP_perf_get_qtimer_count();
+    FARF(HIGH, "fill %u/%u: rows %u:%u usec %u\n",
+         ith, nth, ir0, ir1, (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
+}
+
+int op_fill(struct htp_ops_context * octx) {
+    fill_preamble;
+
+    if (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
+        return HTP_STATUS_NO_SUPPORT;
+    }
+
+    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
+        return HTP_STATUS_OK;
+    }
+
+    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
+    const uint32_t n_threads = MIN(nr, octx->n_threads);
+
+    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
+    const bool opt_path = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);
+
+    FARF(HIGH, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
+         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (int) opt_path);
+
+    float val_f32 = 0.f;
+    memcpy(&val_f32, &octx->op_params[0], sizeof(float));
+
+    struct htp_fill_context fctx = {
+        .octx             = octx,
+        .nrows_per_thread = (nr + n_threads - 1) / n_threads,
+        .total_rows       = nr,
+        .opt_path         = opt_path,
+    };
+
+    switch (dst->type) {
+    case HTP_TYPE_F32:
+        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
+        fctx.elem_size = sizeof(float);
+        break;
+    case HTP_TYPE_F16:
+        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
+        fctx.elem_size = sizeof(_Float16);
+        break;
+    default:
+        return HTP_STATUS_NO_SUPPORT;
+    }
+
+    worker_pool_run_func(octx->ctx->worker_pool, fill_thread, &fctx, n_threads);
+
+    return HTP_STATUS_OK;
+}
diff --git src/ggml-hexagon/htp/hex-utils.h src/ggml-hexagon/htp/hex-utils.h
index f6713c5c..329249e1 100644
--- src/ggml-hexagon/htp/hex-utils.h
+++ src/ggml-hexagon/htp/hex-utils.h
@@ -4,6 +4,7 @@
 #include <stdbool.h>
 #include <stdint.h>
 #include <qurt_memory.h>
+#include <qurt.h>
 
 #include "hexagon_types.h"
 #include "hexagon_protos.h"
@@ -100,4 +101,31 @@ static inline void hex_pause() {
     asm volatile(" pause(#255)\n");
 }
 
+#ifndef HEX_NUM_PMU_COUNTERS
+#define HEX_NUM_PMU_COUNTERS 8
+#endif
+
+static inline void hex_get_pmu(uint32_t counters[]) {
+#if __HVX_ARCH__ >= 79
+    asm volatile("%0 = upmucnt0" : "=r"(counters[0]));
+    asm volatile("%0 = upmucnt1" : "=r"(counters[1]));
+    asm volatile("%0 = upmucnt2" : "=r"(counters[2]));
+    asm volatile("%0 = upmucnt3" : "=r"(counters[3]));
+    asm volatile("%0 = upmucnt4" : "=r"(counters[4]));
+    asm volatile("%0 = upmucnt5" : "=r"(counters[5]));
+    asm volatile("%0 = upmucnt6" : "=r"(counters[6]));
+    asm volatile("%0 = upmucnt7" : "=r"(counters[7]));
+#else
+    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
+    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
+    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
+    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
+    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
+    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
+    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
+    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
+    // qurt_pmu_get_pmucnt(counters);
+#endif
+}
+
 #endif /* HEX_UTILS_H */
diff --git src/ggml-hexagon/htp/hmx-matmul-ops.c src/ggml-hexagon/htp/hmx-matmul-ops.c
index dbca8220..05e3c6c2 100644
--- src/ggml-hexagon/htp/hmx-matmul-ops.c
+++ src/ggml-hexagon/htp/hmx-matmul-ops.c
@@ -1683,7 +1683,7 @@ int mat_mul_qk_0_d16a32_out_stationary(struct htp_context *ctx, float *restrict
     __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);
     assert((size_t)(vtcm_ptr - (uint8_t *)ctx->vtcm_base) <= vtcm_budget);
 
-    FARF(HIGH, "hmx-mm: m=%d k=%d n=%d wtype=%d block M=%zu N=%zu K=%zu vtcm=%zu/%zu", __func__, m, k, n, weight_type,
+    FARF(HIGH, "hmx-mm: m=%d k=%d n=%d wtype=%d block M=%zu N=%zu K=%zu vtcm=%zu/%zu", m, k, n, weight_type,
          M_BLOCK_SIZE, N_BLOCK_SIZE, K_BLOCK_SIZE, (size_t) (vtcm_ptr - (uint8_t *) ctx->vtcm_base), vtcm_budget);
 
     // initialize eye tile (32x32 identity matrix)
diff --git src/ggml-hexagon/htp/htp-ctx.h src/ggml-hexagon/htp/htp-ctx.h
index 8b5e47ad..d704fede 100644
--- src/ggml-hexagon/htp/htp-ctx.h
+++ src/ggml-hexagon/htp/htp-ctx.h
@@ -10,6 +10,7 @@
 #include <dspqueue.h>
 #include <stdatomic.h>
 #include <stdint.h>
+#include <stdbool.h>
 
 #define HTP_MAX_NTHREADS 10
 #define HTP_MAX_MMAPS    16
@@ -66,7 +67,9 @@ struct htp_context {
     int                    thread_id;
     int                    thread_prio;
 
-    int                    hmx_enabled;
+    bool                   hmx_enabled;
+    bool                   etm;
+    uint32_t               profiler;
 
     uint8_t *              vtcm_base;
     size_t                 vtcm_size;
@@ -98,5 +101,8 @@ int op_repeat(struct htp_ops_context * octx);
 int op_argsort(struct htp_ops_context * octx);
 int op_ssm_conv(struct htp_ops_context * octx);
 int op_cumsum(struct htp_ops_context * octx);
+int op_fill(struct htp_ops_context * octx);
+int op_diag(struct htp_ops_context * octx);
+int op_solve_tri(struct htp_ops_context * octx);
 
 #endif /* HTP_CTX_H */
diff --git src/ggml-hexagon/htp/htp-ops.h src/ggml-hexagon/htp/htp-ops.h
index 79b5ecd2..4397245c 100644
--- src/ggml-hexagon/htp/htp-ops.h
+++ src/ggml-hexagon/htp/htp-ops.h
@@ -42,9 +42,9 @@ enum htp_data_type {
 
 // Mask to enable various stages of the Ops.
 // Used for debugging and profiling.
-enum htp_op_mask {
-    HTP_OPMASK_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into the DSP)
-    HTP_OPMASK_COMPUTE  = (1 << 1),  // Enable Compute
+enum htp_op_stage {
+    HTP_OPSTAGE_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into NPU)
+    HTP_OPSTAGE_COMPUTE  = (1 << 1),  // Enable Compute
 };
 
 // Do not reorder first 4 (used as an index)
@@ -80,7 +80,9 @@ enum htp_op_code {
     HTP_OP_SSM_CONV,
     HTP_OP_REPEAT,
     HTP_OP_CUMSUM,
-
+    HTP_OP_FILL,
+    HTP_OP_DIAG,
+    HTP_OP_SOLVE_TRI,
     HTP_OP_INVALID
 };
 
@@ -135,27 +137,45 @@ struct htp_op_desc {
     int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
     uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
     uint16_t dst;                       // Output tensor index
+};
+
+enum htp_profiler_mode {
+    HTP_PROF_DISABLED = 0,
+    HTP_PROF_BASIC    = 1,
+    HTP_PROF_PMU      = 2,
+};
+
+#define HTP_PROF_PMU_NCNT 8
 
-    // the rest is filled in-place by the NPU
-    uint32_t prof_usecs;                // Number of usec per request
-    uint32_t prof_cycles;               // Number of cycles per request
-    uint32_t prof_pkts;                 // Number of instruction packets per request
-    uint32_t unused;
+// Profile descriptor
+struct htp_prof_desc {
+    uint32_t opcode;                 // GGML/HTP Op
+    uint32_t usecs;                  // Number of usec
+    uint32_t cycles;                 // Number of cycles
+    uint32_t pad;                    // Unused
+    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
 };
 
 struct htp_opbatch_req {
+    uint32_t id;          // Batch id
     uint32_t n_bufs;      // Number of buffers
     uint32_t n_tensors;   // Number of tensors
     uint32_t n_ops;       // Number of ops
     uint32_t flags;       // unused
+    uint32_t pad;         // unused
     // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
     // struct htp_tensor    tensors[]; -- dspqueue buf 0
     // struct htp_op_desc   ops[];     -- dspqueue buf 0
 };
 
 struct htp_opbatch_rsp {
+    uint32_t id;         // Batch id
     uint32_t status;     // HTP_STATUS_...
-    // struct htp_op_req ops[];     -- dspqueue buf 0
+    uint32_t n_bufs;     // Number of buffers
+    uint32_t n_tensors;  // Number of tensors
+    uint32_t n_ops;      // Number of op profile descriptors
+    uint32_t pad;        // unused
+    // struct htp_prof_desc profs[];  -- dspqueue buf 0
 };
 
 #endif /* HTP_OPS_H */
diff --git src/ggml-hexagon/htp/htp_iface.idl src/ggml-hexagon/htp/htp_iface.idl
index 3eb5d5a6..dbcafd1d 100644
--- src/ggml-hexagon/htp/htp_iface.idl
+++ src/ggml-hexagon/htp/htp_iface.idl
@@ -6,13 +6,17 @@
 #include "AEEStdDef.idl"
 #include "remote.idl"
 
+struct htp_iface_pmu_conf {
+    uint32 events[8];
+};
+
 interface htp_iface : remote_handle64 {
     AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx, in uint32 use_hmx);
     AEEResult stop();
     AEEResult mmap(in uint32 fd, in uint32 size, in uint32 pinned);
     AEEResult munmap(in uint32 fd);
-    AEEResult enable_etm();
-    AEEResult disable_etm();
+    AEEResult profiler(in uint32 mode, in htp_iface_pmu_conf pmu);
+    AEEResult etm(in uint32 enable);
 };
 
 #endif /* HTP_IDL */
diff --git src/ggml-hexagon/htp/hvx-base.h src/ggml-hexagon/htp/hvx-base.h
index ed6026e7..d0926ded 100644
--- src/ggml-hexagon/htp/hvx-base.h
+++ src/ggml-hexagon/htp/hvx-base.h
@@ -256,6 +256,18 @@ static inline HVX_Vector hvx_vec_mul_f16_f16(HVX_Vector a, HVX_Vector b)
     return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b));
 }
 
+static inline HVX_Vector hvx_vec_add_f32_f32(HVX_Vector a, HVX_Vector b) {
+    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
+}
+
+static inline HVX_Vector hvx_vec_sub_f32_f32(HVX_Vector a, HVX_Vector b) {
+    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
+}
+
+static inline HVX_Vector hvx_vec_mul_f32_f32(HVX_Vector a, HVX_Vector b) {
+    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
+}
+
 #else
 
 static inline HVX_Vector hvx_vec_add_f16_f16(HVX_Vector a, HVX_Vector b)
@@ -273,6 +285,18 @@ static inline HVX_Vector hvx_vec_mul_f16_f16(HVX_Vector a, HVX_Vector b)
     return Q6_Vhf_vmpy_VhfVhf(a, b);
 }
 
+static inline HVX_Vector hvx_vec_add_f32_f32(HVX_Vector a, HVX_Vector b) {
+    return Q6_Vsf_vadd_VsfVsf(a, b);
+}
+
+static inline HVX_Vector hvx_vec_sub_f32_f32(HVX_Vector a, HVX_Vector b) {
+    return Q6_Vsf_vsub_VsfVsf(a, b);
+}
+
+static inline HVX_Vector hvx_vec_mul_f32_f32(HVX_Vector a, HVX_Vector b) {
+    return Q6_Vsf_vmpy_VsfVsf(a, b);
+}
+
 #endif // __HVX_ARCH__ < 79
 
 #endif /* HVX_BASE_H */
diff --git src/ggml-hexagon/htp/main.c src/ggml-hexagon/htp/main.c
index 5091623a..f5834730 100644
--- src/ggml-hexagon/htp/main.c
+++ src/ggml-hexagon/htp/main.c
@@ -27,6 +27,7 @@
 #include "htp-ctx.h"
 #include "htp-ops.h"
 #include "htp-ops.h"
+#include "htp_iface.h"
 #include "worker-pool.h"
 
 AEEResult htp_iface_open(const char * uri, remote_handle64 * handle) {
@@ -100,6 +101,74 @@ AEEResult htp_iface_open(const char * uri, remote_handle64 * handle) {
         }
     }
 
+#if __HVX_ARCH__ >= 75
+    {
+        // Set HMX clock
+        HAP_power_request_t request;
+        memset(&request, 0, sizeof(HAP_power_request_t));
+        request.type = HAP_power_set_HMX_v2;
+        request.hmx_v2.set_clock = TRUE;
+        request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
+        request.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
+        request.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
+        request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
+        FARF(ALWAYS, "Setting HMX clock\n");
+        err = HAP_power_set((void *) &ctx, &request);
+        if (err != AEE_SUCCESS) {
+            FARF(ERROR, "Error setting HMX clock.");
+            return err;
+        }
+    }
+#endif
+
+    return AEE_SUCCESS;
+}
+
+AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
+    int err = enable ? HAP_user_etm_enable() : HAP_user_etm_disable();
+    if (err) {
+        if (err == AEE_EVERSIONNOTSUPPORT) {
+            FARF(ERROR, "API HAP_user_etm_enable/disable is not supported\n");
+        } else {
+            FARF(ERROR, "Error executing HAP_user_etm_enable/disable with error code : 0x%x\n", err);
+        }
+    }
+    return err;
+}
+
+AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, const htp_iface_pmu_conf* pmu_conf) {
+    struct htp_context * ctx = (struct htp_context *) handle;
+    if (!ctx) {
+        return AEE_EBADPARM;
+    }
+
+    if (mode == HTP_PROF_PMU) {
+        const uint32_t* events = pmu_conf->events;
+
+        // Pack 4 event IDs (low 8 bits) into each 32-bit config register
+        uint32_t evtcfg = 0, evtcfg1 = 0, cfg = 0, i = 0;
+        for (; i < HEX_NUM_PMU_COUNTERS/2; i++) {
+            evtcfg  |= ((events[i + 0] & 0xFF) << (i * 8));
+            evtcfg1 |= ((events[i + 4] & 0xFF) << (i * 8));
+        }
+
+        // For events >255 pack high 2 bits of all 8 event IDs into cfg register
+        // 2 bits per counter: bits [1:0] for counter 0, [3:2] for counter 1, etc.
+        for (i = 0; i < HEX_NUM_PMU_COUNTERS; i++) {
+            cfg |= (((events[i] >> 8) & 3) << (i * 2));
+        }
+
+        FARF(ALWAYS, "Configuring PMU registers: evtcfg = 0x%x, evtcfg1 = 0x%x, pmucfg = 0x%x", evtcfg, evtcfg1, cfg);
+
+        // Configure PMU registers
+        qurt_pmu_set(QURT_PMUCFG,     cfg);
+        qurt_pmu_set(QURT_PMUEVTCFG,  evtcfg);
+        qurt_pmu_set(QURT_PMUEVTCFG1, evtcfg1);
+        qurt_pmu_enable(1);
+    }
+
+    ctx->profiler = mode;
+
     return AEE_SUCCESS;
 }
 
@@ -129,35 +198,19 @@ AEEResult htp_iface_close(remote_handle64 handle) {
         }
     }
 
-    free(ctx);
-    return AEE_SUCCESS;
-}
-
-AEEResult htp_iface_enable_etm(remote_handle64 handle) {
-    int err = HAP_user_etm_enable();
-    if (err) {
-        if (err == AEE_EVERSIONNOTSUPPORT) {
-            FARF(ERROR, "API HAP_user_etm_enable is not supported\n");
-        } else {
-            FARF(ERROR, "Error executing HAP_user_etm_enable with error code : 0x%x\n", err);
-        }
+    if (ctx->profiler) {
+        qurt_pmu_enable(1);
     }
-    return err;
-}
 
-AEEResult htp_iface_disable_etm(remote_handle64 handle) {
-    int err = HAP_user_etm_disable();
-    if (err) {
-        if (err == AEE_EVERSIONNOTSUPPORT) {
-            FARF(ERROR, "API HAP_user_etm_disable is not supported\n");
-        } else {
-            FARF(ERROR, "Error executing HAP_user_etm_disable with error code : 0x%x\n", err);
-        }
+    if (ctx->etm) {
+        HAP_user_etm_disable();
     }
-    return err;
+
+    free(ctx);
+    return AEE_SUCCESS;
 }
 
-AEEResult htp_iface_mmap(remote_handle64 handle, int fd, uint32_t size, uint32_t pinned) {
+AEEResult htp_iface_mmap(remote_handle64 handle, uint32 fd, uint32 size, uint32 pinned) {
     struct htp_context * ctx = (struct htp_context *) handle;
     if (!ctx) {
         return AEE_EBADPARM;
@@ -204,7 +257,7 @@ AEEResult htp_iface_mmap(remote_handle64 handle, int fd, uint32_t size, uint32_t
     return AEE_ENOMEMORY;
 }
 
-AEEResult htp_iface_munmap(remote_handle64 handle, int fd) {
+AEEResult htp_iface_munmap(remote_handle64 handle, uint32 fd) {
     struct htp_context * ctx = (struct htp_context *) handle;
     if (!ctx) {
         return AEE_EBADPARM;
@@ -434,19 +487,39 @@ static void htp_error_callback(dspqueue_t queue, int error, void * context) {
 struct profile_data {
     uint64_t usecs;
     uint64_t cycles;
-    uint64_t pkts;
+    uint32_t pmu_counters[HEX_NUM_PMU_COUNTERS];
 };
 
-static inline void profile_start(struct profile_data * d) {
-    d->usecs  = HAP_perf_get_qtimer_count();
-    d->cycles = hex_get_cycles();
-    d->pkts   = hex_get_pktcnt();
+static inline void profile_start(uint32_t mode, struct profile_data * d) {
+    switch (mode) {
+        case HTP_PROF_PMU:
+            hex_get_pmu(d->pmu_counters);
+            // fallthrough
+        case HTP_PROF_BASIC:
+            d->usecs  = HAP_perf_get_qtimer_count();
+            d->cycles = hex_get_cycles();
+            break;
+        default:
+            break;
+    }
 }
 
-static inline void profile_stop(struct profile_data * d) {
-    d->usecs  = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - d->usecs);
-    d->cycles = hex_get_cycles() - d->cycles;
-    d->pkts   = hex_get_pktcnt() - d->pkts;
+static inline void profile_stop(uint32_t mode, struct profile_data * d) {
+    uint32_t pmu_counters[HEX_NUM_PMU_COUNTERS];
+    switch (mode) {
+        case HTP_PROF_PMU:
+            hex_get_pmu(pmu_counters);
+            for (int i = 0; i < HEX_NUM_PMU_COUNTERS; i++) {
+                d->pmu_counters[i] = pmu_counters[i] - d->pmu_counters[i];
+            }
+            // fallthrough
+        case HTP_PROF_BASIC:
+            d->usecs  = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - d->usecs);
+            d->cycles = hex_get_cycles() - d->cycles;
+            break;
+        default:
+            break;
+    }
 }
 
 static int execute_op(struct htp_ops_context * octx) {
@@ -514,6 +587,15 @@ static int execute_op(struct htp_ops_context * octx) {
         case HTP_OP_CUMSUM:
             return op_cumsum(octx);
 
+        case HTP_OP_FILL:
+            return op_fill(octx);
+
+        case HTP_OP_DIAG:
+            return op_diag(octx);
+
+        case HTP_OP_SOLVE_TRI:
+            return op_solve_tri(octx);
+
         case HTP_OP_INVALID:
             break;
 
@@ -720,29 +802,32 @@ static void htp_packet_callback(dspqueue_t queue, int error, void * context) {
             continue;
         }
 
+        // Reset poll count for valid requests
+        poll_count = DSPQUEUE_POLL_COUNT;
+
         const uint32_t n_bufs = req.n_bufs;
         const uint32_t n_tens = req.n_tensors;
         const uint32_t n_ops  = req.n_ops;
 
-        const uint32_t b_size = sizeof(struct htp_buf_desc) * n_bufs;
-        const uint32_t t_size = sizeof(struct htp_tensor)   * n_tens;
-        const uint32_t o_size = sizeof(struct htp_op_desc)  * n_ops;
+        const uint32_t b_size = sizeof(struct htp_buf_desc)  * n_bufs;
+        const uint32_t t_size = sizeof(struct htp_tensor)    * n_tens;
+        const uint32_t o_size = sizeof(struct htp_op_desc)   * n_ops;
+        const uint32_t p_size = sizeof(struct htp_prof_desc) * n_ops;
 
-        if (dbuf.size < b_size + t_size + o_size) {
+        if (dbuf.size < b_size + t_size + o_size + p_size) {
             FARF(ERROR, "invalid opbatch memory block size %u", dbuf.size);
             break;
         }
 
-        // Reset poll count for valid requests
-        poll_count = DSPQUEUE_POLL_COUNT;
+        FARF(HIGH, "processing opbatch #%u: n-bufs %u n-tensors %u n-ops %u : m-size %u b-size %u t-size %u o-size %u", req.id,
+                n_bufs, n_tens, n_ops, dbuf.size, b_size, t_size, o_size);
 
+        // Setup descriptor pointers
         uint8_t * m_ptr = dbuf.ptr;
-        struct htp_buf_desc* bufs = (struct htp_buf_desc*) m_ptr; m_ptr += b_size;
-        struct htp_tensor*   tens = (struct htp_tensor*)   m_ptr; m_ptr += t_size;
-        struct htp_op_desc*   ops = (struct htp_op_desc*)  m_ptr;
-
-        FARF(HIGH, "processing opbatch: n-bufs %u n-tensors %u n-ops %u : m-size %u b-size %u t-size %u o-size %u",
-                n_bufs, n_tens, n_ops, dbuf.size, b_size, t_size, o_size);
+        struct htp_buf_desc* bufs = (struct htp_buf_desc*)  m_ptr; m_ptr += b_size;
+        struct htp_tensor*   tens = (struct htp_tensor*)    m_ptr; m_ptr += t_size;
+        struct htp_op_desc*   ops = (struct htp_op_desc*)   m_ptr; m_ptr += o_size;
+        struct htp_prof_desc* pds = (struct htp_prof_desc*) m_ptr;
 
         prep_op_bufs(ctx, bufs, n_bufs);
         prep_tensors(ctx, bufs, tens, n_tens);
@@ -754,22 +839,34 @@ static void htp_packet_callback(dspqueue_t queue, int error, void * context) {
 
         for (uint32_t i=0; i < n_ops; i++) {
             struct profile_data prof;
-            profile_start(&prof);
+
+            profile_start(ctx->profiler, &prof);
 
             proc_op_req(octx, tens, i, &ops[i]);
 
-            profile_stop(&prof);
-            ops[i].prof_usecs  = prof.usecs;
-            ops[i].prof_cycles = prof.cycles;
-            ops[i].prof_pkts   = prof.pkts;
+            profile_stop(ctx->profiler, &prof);
+
+            if (ctx->profiler) {
+                pds[i].opcode = ops[i].opcode;
+                pds[i].usecs  = prof.usecs;
+                pds[i].cycles = prof.cycles;
+                for (int j = 0; j < HEX_NUM_PMU_COUNTERS; j++) {
+                    pds[i].pmu[j] = prof.pmu_counters[j];
+                }
+            }
         }
 
         // dspqueue_write_early_wakeup_noblock(ctx->queue, 10, 0);
 
         struct htp_opbatch_rsp rsp;
-        rsp.status = HTP_STATUS_OK; // FIXME
+        rsp.id        = req.id;
+        rsp.status    = HTP_STATUS_OK;
+        rsp.n_bufs    = n_bufs;
+        rsp.n_tensors = n_tens;
+        rsp.n_ops     = n_ops;
 
         dbuf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
+
         err = dspqueue_write(queue, 0, 1, &dbuf, sizeof(rsp), (const uint8_t *) &rsp, DSPQUEUE_TIMEOUT_NONE);
         if (err != 0) {
             FARF(ERROR, "dspqueue_write failed: 0x%08x", (unsigned) err);
diff --git src/ggml-hexagon/htp/matmul-ops.c src/ggml-hexagon/htp/matmul-ops.c
index bac06693..a0c26513 100644
--- src/ggml-hexagon/htp/matmul-ops.c
+++ src/ggml-hexagon/htp/matmul-ops.c
@@ -3017,6 +3017,10 @@ int op_matmul(struct htp_ops_context * octx) {
     const int act_stride = (int)(src1->nb[1] / sizeof(float));
     const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));
 
+    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
+        return HTP_STATUS_OK;
+    }
+
     if (src0->type == HTP_TYPE_F16) {
         if (is_batched) {
             hmx_matmul_w16a32_batched_params_t batch_params = {
diff --git src/ggml-hexagon/htp/solve-tri-ops.c src/ggml-hexagon/htp/solve-tri-ops.c
new file mode 100644
index 00000000..ae8e1a50
--- /dev/null
+++ src/ggml-hexagon/htp/solve-tri-ops.c
@@ -0,0 +1,267 @@
+#pragma clang diagnostic ignored "-Wunused-but-set-variable"
+
+#include <HAP_farf.h>
+#include <HAP_perf.h>
+#include <string.h>
+
+#define GGML_COMMON_DECL_C
+#include "ggml-common.h"
+#include "htp-ctx.h"
+#include "htp-ops.h"
+#include "hvx-types.h"
+#include "hvx-utils.h"
+
+struct htp_solve_tri_context {
+    struct htp_ops_context * octx;
+    uint32_t                 jobs_per_thread;
+    uint32_t                 total_jobs;
+    uint32_t                 k_chunks;
+    uint32_t                 col_block;
+};
+
+static inline void solve_tri_row_scalar(const float * A_row,
+                                        const float * B_row,
+                                        float *       X,
+                                        uint32_t      row,
+                                        uint32_t      k,
+                                        uint32_t      col0,
+                                        uint32_t      coln,
+                                        float         inv_diag) {
+    for (uint32_t col = col0; col < col0 + coln; ++col) {
+        float sum = 0.0f;
+        for (uint32_t t = 0; t < row; ++t) {
+            sum += A_row[t] * X[t * k + col];
+        }
+        X[row * k + col] = (B_row[col] - sum) * inv_diag;
+    }
+}
+
+static inline HVX_Vector hvx_load_partial_f32(const float * src, uint32_t n) {
+    HVX_Vector v = *((const HVX_UVector *) src);
+    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * sizeof(float));
+    return Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
+}
+
+static inline void solve_tri_row_hvx(const float * A_row,
+                                     const float * B_row,
+                                     float *       X,
+                                     uint32_t      row,
+                                     uint32_t      k,
+                                     uint32_t      col0,
+                                     uint32_t      coln,
+                                     float         inv_diag) {
+    const bool full = (coln == VLEN_FP32);
+
+    HVX_Vector sum_v = Q6_V_vzero();
+    for (uint32_t t = 0; t < row; ++t) {
+        const float   a         = A_row[t];
+        const float * x_row_col = X + t * k + col0;
+
+        HVX_Vector x_v = full ? *((const HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
+        HVX_Vector a_v = hvx_vec_splat_f32(a);
+        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
+    }
+
+    const float * b_row_col = B_row + col0;
+    float *       x_out_col = X + row * k + col0;
+
+    HVX_Vector b_v        = full ? *((const HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
+    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);
+
+    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
+    hvx_vec_store_u((void *) x_out_col, coln * sizeof(float), out_v);
+}
+
+// Batch-level thread: each job is one full batch.
+static void solve_tri_batch_thread_f32(unsigned int nth, unsigned int ith, void * data) {
+    struct htp_solve_tri_context * sctx = (struct htp_solve_tri_context *) data;
+    struct htp_ops_context *       octx = sctx->octx;
+
+    const struct htp_tensor * src0 = octx->src[0];  // A
+    const struct htp_tensor * src1 = octx->src[1];  // B
+    const struct htp_tensor * dst  = octx->dst;     // X
+
+    const uint32_t n = src0->ne[0];
+    const uint32_t k = src1->ne[0];
+
+    const uint32_t ne02 = src0->ne[2];
+
+    const uint32_t col_block = VLEN_FP32;
+    const uint32_t k_full    = (k / col_block) * col_block;
+
+    const uint32_t start_batch = sctx->jobs_per_thread * ith;
+    const uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->total_jobs);
+
+    uint64_t t1, t2;
+    t1 = HAP_perf_get_qtimer_count();
+
+    for (uint32_t batch = start_batch; batch < end_batch; ++batch) {
+        const uint32_t i03 = batch / ne02;
+        const uint32_t i02 = batch - i03 * ne02;
+
+        const float * A_batch =
+            (const float *) ((const uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
+        const float * B_batch =
+            (const float *) ((const uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
+        float * X_batch = (float *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);
+
+        for (uint32_t row = 0; row < n; ++row) {
+            const float   diag     = A_batch[row * n + row];
+            const float   inv_diag = 1.0f / diag;
+            const float * A_row    = A_batch + row * n;
+            const float * B_row    = B_batch + row * k;
+
+            uint32_t col0 = 0;
+            for (; col0 < k_full; col0 += col_block) {
+                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
+            }
+
+            if (col0 < k) {
+                const uint32_t coln = k - col0;
+                if (coln >= 8) {
+                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
+                } else {
+                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
+                }
+            }
+        }
+    }
+
+    t2 = HAP_perf_get_qtimer_count();
+
+    FARF(HIGH, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u usec %u\n",
+         ith, nth, n, n, k, n, start_batch, end_batch,
+         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
+}
+
+// Chunk-level thread: each job is one (batch, col_chunk) pair.
+static void solve_tri_chunk_thread_f32(unsigned int nth, unsigned int ith, void * data) {
+    struct htp_solve_tri_context * sctx = (struct htp_solve_tri_context *) data;
+    struct htp_ops_context *       octx = sctx->octx;
+
+    const struct htp_tensor * src0 = octx->src[0];  // A
+    const struct htp_tensor * src1 = octx->src[1];  // B
+    const struct htp_tensor * dst  = octx->dst;     // X
+
+    const uint32_t n = src0->ne[0];
+    const uint32_t k = src1->ne[0];
+
+    const uint32_t ne02 = src0->ne[2];
+
+    const uint32_t start_job = sctx->jobs_per_thread * ith;
+    const uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->total_jobs);
+
+    uint64_t t1, t2;
+    t1 = HAP_perf_get_qtimer_count();
+
+    for (uint32_t job = start_job; job < end_job; ++job) {
+        const uint32_t batch = job / sctx->k_chunks;
+        const uint32_t chunk = job - batch * sctx->k_chunks;
+
+        const uint32_t i03 = batch / ne02;
+        const uint32_t i02 = batch - i03 * ne02;
+
+        const uint32_t col0 = chunk * sctx->col_block;
+        const uint32_t coln = MIN(sctx->col_block, k - col0);
+
+        const float * A_batch =
+            (const float *) ((const uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
+        const float * B_batch =
+            (const float *) ((const uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
+        float * X_batch = (float *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);
+
+        const bool use_hvx = (coln >= 8);
+
+        for (uint32_t row = 0; row < n; ++row) {
+            const float diag     = A_batch[row * n + row];
+            const float inv_diag = 1.0f / diag;
+
+            const float * A_row = A_batch + row * n;
+            const float * B_row = B_batch + row * k;
+
+            if (use_hvx) {
+                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
+            } else {
+                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
+            }
+        }
+    }
+
+    t2 = HAP_perf_get_qtimer_count();
+
+    FARF(HIGH, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) job %u:%u usec %u\n",
+         ith, nth, n, n, k, n, start_job, end_job,
+         (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
+}
+
+int op_solve_tri(struct htp_ops_context * octx) {
+    const struct htp_tensor * src0 = octx->src[0];  // A
+    const struct htp_tensor * src1 = octx->src[1];  // B
+    const struct htp_tensor * dst  = octx->dst;     // X
+
+    if (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
+        return HTP_STATUS_NO_SUPPORT;
+    }
+
+    // left=true, lower=true, uni=false only
+    if (src0->ne[0] != src0->ne[1]) {
+        return HTP_STATUS_INVAL_PARAMS;
+    }
+    if (src0->ne[1] != src1->ne[1]) {
+        return HTP_STATUS_INVAL_PARAMS;
+    }
+    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
+        return HTP_STATUS_INVAL_PARAMS;
+    }
+    if (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
+        dst->ne[3] != src1->ne[3]) {
+        return HTP_STATUS_INVAL_PARAMS;
+    }
+
+    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
+        return HTP_STATUS_OK;
+    }
+
+    const uint32_t k = src1->ne[0];
+
+    const uint32_t col_block     = VLEN_FP32;
+    const uint32_t k_chunks      = (k + col_block - 1) / col_block;
+    const uint32_t total_batches = src0->ne[2] * src0->ne[3];
+    const bool     batched       = total_batches >= (uint32_t) octx->n_threads;
+
+    FARF(HIGH, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
+         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
+         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
+         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);
+
+    if (batched) {
+        // Batch-level parallelism
+        const uint32_t n_threads = MIN((uint32_t) octx->n_threads, total_batches);
+
+        struct htp_solve_tri_context sctx = {
+            .octx            = octx,
+            .jobs_per_thread = (total_batches + n_threads - 1) / n_threads,
+            .total_jobs      = total_batches,
+            .k_chunks        = k_chunks,
+            .col_block       = col_block,
+        };
+
+        worker_pool_run_func(octx->ctx->worker_pool, solve_tri_batch_thread_f32, &sctx, n_threads);
+    } else {
+        // Chunk-level parallelism
+        const uint32_t total_jobs = total_batches * k_chunks;
+        const uint32_t n_threads  = MIN((uint32_t) octx->n_threads, MAX(total_jobs, 1));
+
+        struct htp_solve_tri_context sctx = {
+            .octx            = octx,
+            .jobs_per_thread = (total_jobs + n_threads - 1) / n_threads,
+            .total_jobs      = total_jobs,
+            .k_chunks        = k_chunks,
+            .col_block       = col_block,
+        };
+
+        worker_pool_run_func(octx->ctx->worker_pool, solve_tri_chunk_thread_f32, &sctx, n_threads);
+    }
+
+    return HTP_STATUS_OK;
+}
diff --git src/ggml-hexagon/libggml-htp.inf src/ggml-hexagon/libggml-htp.inf
index 656d2d9a..39cefcdd 100644
--- src/ggml-hexagon/libggml-htp.inf
+++ src/ggml-hexagon/libggml-htp.inf
@@ -8,7 +8,7 @@ CatalogFile = libggml-htp.cat
 PnpLockDown = 1
 
 [DestinationDirs]
-Drivers_Dir = 6
+Drivers_Dir = 13
 
 [SourceDisksNames]
 1 = %DiskId%
@@ -18,6 +18,7 @@ libggml-htp-v68.so = 1
 libggml-htp-v69.so = 1
 libggml-htp-v73.so = 1
 libggml-htp-v75.so = 1
+libggml-htp-v79.so = 1
 libggml-htp-v81.so = 1
 
 [ControlFlags]
@@ -31,6 +32,7 @@ libggml-htp-v68.so,,,0x10 ;COPYFLG_NO_OVERWRITE
 libggml-htp-v69.so,,,0x10 ;COPYFLG_NO_OVERWRITE
 libggml-htp-v73.so,,,0x10 ;COPYFLG_NO_OVERWRITE
 libggml-htp-v75.so,,,0x10 ;COPYFLG_NO_OVERWRITE
+libggml-htp-v79.so,,,0x10 ;COPYFLG_NO_OVERWRITE
 libggml-htp-v81.so,,,0x10 ;COPYFLG_NO_OVERWRITE
 
 [Strings]
diff --git src/ggml-metal/ggml-metal-device.cpp src/ggml-metal/ggml-metal-device.cpp
index 07d016d2..d211bf79 100644
--- src/ggml-metal/ggml-metal-device.cpp
+++ src/ggml-metal/ggml-metal-device.cpp
@@ -677,7 +677,15 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_meta
     const ggml_type tsrc1 = op->src[1]->type;
 
     const bool bc_inp = op->src[0]->ne[0] % 32 != 0;
-    const bool bc_out = op->ne[0] % 64 != 0 || op->ne[1] % 32 != 0;
+
+    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
+    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
+
+    const bool has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;
+
+    const bool bc_out = has_tensor
+        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
+        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);
 
     snprintf(base, 256, "kernel_mul_mm_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
     snprintf(name, 256, "%s_bci=%d_bco=%d", base, bc_inp, bc_out);
@@ -694,8 +702,20 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_meta
         ggml_metal_cv_free(cv);
     }
 
-    // when the output size is not multiple of 64x32, we need extra smem to prevent out-of-bounds writes
-    res.smem = bc_out ? 8192 : 4096 + 2048;
+    if (has_tensor) {
+        res.nr0 = NRA;
+        res.nr1 = NRB;
+
+        const size_t smem_a = NRA * N_MM_NK_TOTAL * sizeof(ggml_fp16_t);
+        res.smem = smem_a;
+    } else {
+        res.nr0 = 64;
+        res.nr1 = 32;
+
+        res.smem = bc_out ? 8192 : (4096 + 2048);
+    }
+
+    res.nsg = N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;
 
     return res;
 }
diff --git src/ggml-metal/ggml-metal-device.h src/ggml-metal/ggml-metal-device.h
index b4235013..a6c1dab5 100644
--- src/ggml-metal/ggml-metal-device.h
+++ src/ggml-metal/ggml-metal-device.h
@@ -102,6 +102,8 @@ ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev
 
 void ggml_metal_library_free(ggml_metal_library_t lib);
 
+ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);
+
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);
 struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);
 
diff --git src/ggml-metal/ggml-metal-device.m src/ggml-metal/ggml-metal-device.m
index 27cb1683..fe90aafe 100644
--- src/ggml-metal/ggml-metal-device.m
+++ src/ggml-metal/ggml-metal-device.m
@@ -95,8 +95,8 @@ int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_wi
 
 struct ggml_metal_library {
     id<MTLLibrary> obj;
-    id<MTLDevice> device;
 
+    ggml_metal_device_t dev;
     ggml_metal_pipelines_t pipelines; // cache of compiled pipelines
 
     NSLock * lock;
@@ -251,7 +251,7 @@ ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
     ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
 
     res->obj       = library;
-    res->device    = device;
+    res->dev       = dev;
     res->pipelines = ggml_metal_pipelines_init();
     res->lock      = [NSLock new];
 
@@ -318,7 +318,7 @@ ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev
     }
 
     res->obj       = library;
-    res->device    = device;
+    res->dev       = dev;
     res->pipelines = ggml_metal_pipelines_init();
     res->lock      = [NSLock new];
 
@@ -341,6 +341,10 @@ void ggml_metal_library_free(ggml_metal_library_t lib) {
     free(lib);
 }
 
+ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib) {
+    return lib->dev;
+}
+
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
     [lib->lock lock];
 
@@ -405,7 +409,8 @@ struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_
             return res;
         }
 
-        id<MTLComputePipelineState> obj = [lib->device newComputePipelineStateWithFunction:mtl_function error:&error];
+        id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
+        id<MTLComputePipelineState> obj = [device newComputePipelineStateWithFunction:mtl_function error:&error];
 
         [mtl_function release];
 
@@ -699,7 +704,7 @@ ggml_metal_device_t ggml_metal_device_init(int device) {
                     "    auto sB = tB.slice(0, 0); \n"
                     "    mm.run(sB, sA, cT); \n"
                     " \n"
-                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
+                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                     " \n"
                     "    cT.store(tC); \n"
                     "}";
@@ -749,7 +754,7 @@ ggml_metal_device_t ggml_metal_device_init(int device) {
                     "    auto sB = tB.slice(0, 0); \n"
                     "    mm.run(sB, sA, cT); \n"
                     " \n"
-                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
+                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                     " \n"
                     "    cT.store(tC); \n"
                     "}";
@@ -814,7 +819,7 @@ ggml_metal_device_t ggml_metal_device_init(int device) {
             }
 
             // print MTL GPU family:
-            GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, dev->props.name);
+            GGML_LOG_INFO("%s: GPU name:   %s (%s)\n", __func__, dev->props.name, dev->props.desc);
 
             // determine max supported GPU family
             // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
@@ -931,13 +936,13 @@ void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
 }
 
 struct ggml_metal_event {
-    void * obj; // id<MTLEvent>
+    void * obj; // id<MTLSharedEvent>
 
     atomic_int value;
 };
 
 void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
-    id<MTLEvent> event = (id<MTLEvent>)ev->obj;
+    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;
 
     id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;
 
@@ -945,7 +950,7 @@ void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t
 }
 
 void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
-    id<MTLEvent> event = (id<MTLEvent>)ev->obj;
+    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;
 
     id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;
 
@@ -953,7 +958,7 @@ void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cm
 }
 
 ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
-    id<MTLEvent> event = [dev->mtl_device newEvent];
+    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];
 
     ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));
 
@@ -964,7 +969,7 @@ ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
 }
 
 void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
-    id<MTLEvent> event = ev->obj;
+    id<MTLSharedEvent> event = ev->obj;
     [event release];
 
     free(ev);
@@ -973,14 +978,13 @@ void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev
 }
 
 void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
-    @autoreleasepool {
-        id<MTLEvent> event = ev->obj;
-
-        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];
-        [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
-        [cmd_buf commit];
-        [cmd_buf waitUntilCompleted];
+    id<MTLSharedEvent> event = ev->obj;
+    const bool res = [event waitUntilSignaledValue:atomic_load_explicit(&ev->value, memory_order_relaxed) timeoutMS:60000];
+    if (!res) {
+        GGML_ABORT("%s: failed to wait for event\n", __func__);
     }
+
+    GGML_UNUSED(dev);
 }
 
 void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
diff --git src/ggml-metal/ggml-metal-impl.h src/ggml-metal/ggml-metal-impl.h
index 379a8b33..ff74cafb 100644
--- src/ggml-metal/ggml-metal-impl.h
+++ src/ggml-metal/ggml-metal-impl.h
@@ -1,6 +1,19 @@
 #ifndef GGML_METAL_IMPL
 #define GGML_METAL_IMPL
 
+// kernel parameters for mat-mat threadgroups
+//
+// TODO: become function constants
+
+#define SZ_SIMDGROUP 16
+#define N_MM_NK 2
+#define N_MM_NK_TOTAL (SZ_SIMDGROUP * N_MM_NK)
+
+#define N_MM_BLOCK_X 4
+#define N_MM_BLOCK_Y 2
+#define N_MM_SIMD_GROUP_X 2
+#define N_MM_SIMD_GROUP_Y 2
+
 // kernel parameters for mat-vec threadgroups
 //
 // N_R0: number of src0 rows to process per simdgroup
diff --git src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal-ops.cpp
index e1735279..5fa162c8 100644
--- src/ggml-metal/ggml-metal-ops.cpp
+++ src/ggml-metal/ggml-metal-ops.cpp
@@ -2195,7 +2195,12 @@ int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
         const size_t smem = pipeline.smem;
 
         ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
-        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
+
+        const int nr0 = pipeline.nr0;
+        const int nr1 = pipeline.nr1;
+        const int nsg = pipeline.nsg;
+
+        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, 32, nsg, 1);
     } else {
         auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);
 
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
diff --git src/ggml-metal/ggml-metal.metal src/ggml-metal/ggml-metal.metal
index 9f38c9d2..c372eaed 100644
--- src/ggml-metal/ggml-metal.metal
+++ src/ggml-metal/ggml-metal.metal
@@ -9306,7 +9306,137 @@ constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
 constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
 
 // each block_q contains 16*nl weights
-template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
+#ifdef GGML_METAL_HAS_TENSOR
+template<
+    typename SA, typename SA_4x4, typename SA_8x8,
+    typename SB, typename SB_2x4, typename SB_8x8,
+    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
+    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
+kernel void kernel_mul_mm(
+        constant ggml_metal_kargs_mul_mm & args,
+        device const char * srcA,
+        device const char * srcB,
+        device       char * dst,
+        threadgroup  char * shmem [[threadgroup(0)]],
+        uint3  tgpig [[threadgroup_position_in_grid]],
+        ushort tiitg [[thread_index_in_threadgroup]],
+        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
+    (void) sgitg;
+
+    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
+    const int K = args.ne00;
+    const int M = args.ne0;
+    const int N = args.ne1;
+
+    // Batch dimension handling
+    const int im = tgpig.z;
+    const int i12 = im % args.ne12;
+    const int i13 = im / args.ne12;
+
+    // Batch offsets for srcA and srcB
+    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
+
+    // Tile dimensions
+    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
+    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
+
+    // Tile offsets in output matrix
+    const int ra = tgpig.y * NRA;
+    const int rb = tgpig.x * NRB;
+
+    // Threadgroup memory for dequantized A tile only
+    threadgroup SA * sa = (threadgroup SA *)(shmem);
+
+    // Work-item count for A loading
+    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
+    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;
+
+    // tA wraps threadgroup memory
+    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));
+
+    // tB wraps device memory directly
+    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
+    const int strideB = args.nb11 / sizeof(T1);
+    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));
+
+    // Configure matmul operation
+    mpp::tensor_ops::matmul2d<
+        mpp::tensor_ops::matmul2d_descriptor(
+            NRB, NRA, N_MM_NK_TOTAL, false, true, true,
+            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
+        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;
+
+    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
+
+    // Accumulate partial results over K dimension
+    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
+        // === PHASE 1: Dequantization of A into threadgroup memory ===
+        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
+            const int row = work / N_MM_NK;
+            const int k_chunk = work % N_MM_NK;
+            const int k_pos = loop_k + k_chunk * 16;
+            const short k_base = k_chunk * 16;
+
+            // Bounds check: skip device read if row is out of matrix bounds
+            if (ra + row < M) {
+                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
+                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
+                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
+                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
+                    // Mirrors the legacy kernel's existing guard.
+                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);
+
+                    FOR_UNROLL (short i = 0; i < 16; i++) {
+                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
+                    }
+                } else {
+                    const int block_idx = k_pos / (16 * nl);
+                    const short il = (k_pos / 16) % nl;
+
+                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);
+
+                    SA_4x4 temp_a;
+                    dequantize_func(row_ptr + block_idx, il, temp_a);
+
+                    FOR_UNROLL (short i = 0; i < 16; i++) {
+                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
+                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
+                    }
+                }
+            } else {
+                // Zero-pad rows beyond matrix bounds
+                FOR_UNROLL (short i = 0; i < 16; i++) {
+                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
+                }
+            }
+        }
+
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+
+        // === PHASE 2: Tensor matmul ===
+        auto mA = tA.slice(0, 0);
+        auto mB = tB.slice(loop_k, rb);
+
+        mm.run(mB, mA, cT);
+
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+
+    // Store result tile to output matrix (with batch offset)
+    // cT.store handles bounds checking via tD's extents (M, N)
+    device float * dstBatch = (device float *)dst + im * N * M;
+
+    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
+    cT.store(tD.slice(ra, rb));
+}
+
+#else
+
+template<
+    typename S0, typename S0_4x4, typename S0_8x8,
+    typename S1, typename S1_2x4, typename S1_8x8,
+    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
+    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
 kernel void kernel_mul_mm(
         constant ggml_metal_kargs_mul_mm & args,
         device const char * src0,
@@ -9320,10 +9450,6 @@ kernel void kernel_mul_mm(
     threadgroup S0 * sa = (threadgroup S0 *)(shmem);
     threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
 
-#ifdef GGML_METAL_HAS_TENSOR
-    threadgroup float * sc = (threadgroup float *)(shmem);
-#endif
-
     constexpr int NR0 = 64;
     constexpr int NR1 = 32;
 
@@ -9363,7 +9489,6 @@ kernel void kernel_mul_mm(
         + args.nb11*(r1 + lr1)
         + args.nb10*iy);
 
-#ifndef GGML_METAL_HAS_TENSOR
     S0_8x8 ma[4];
     S1_8x8 mb[2];
 
@@ -9372,19 +9497,8 @@ kernel void kernel_mul_mm(
     for (short i = 0; i < 8; i++){
         mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
     }
-#else
-    auto tA = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
-    auto tB = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));
-
-    mpp::tensor_ops::matmul2d<
-        mpp::tensor_ops::matmul2d_descriptor(NR1, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
-        execution_simdgroups<4>> mm;
-
-    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
-#endif
 
     for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
-#ifndef GGML_METAL_HAS_TENSOR
         // load data and store to threadgroup memory
         if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
             threadgroup_barrier(mem_flags::mem_threadgroup);
@@ -9454,66 +9568,6 @@ kernel void kernel_mul_mm(
 
             *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
         }
-#else
-        // load data and store to threadgroup memory
-        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
-            threadgroup_barrier(mem_flags::mem_threadgroup);
-
-            // no need for dequantization
-            for (short i = 0; i < 16; i++) {
-                const short sx = 2*il0 + i/8;
-                const short sy = (tiitg/NL0)/8;
-
-                const short lx = i%8;
-                const short ly = (tiitg/NL0)%8;
-                //const short lx = (tiitg/NL0)%8;
-                //const short ly = i%8;
-
-                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
-            }
-        } else {
-            S0_4x4 temp_a;
-            dequantize_func(x, il, temp_a);
-
-            threadgroup_barrier(mem_flags::mem_threadgroup);
-
-            FOR_UNROLL (short i = 0; i < 16; i++) {
-                const short sx = 2*il0 + i/8;
-                const short sy = (tiitg/NL0)/8;
-
-                const short lx = i%8;
-                const short ly = (tiitg/NL0)%8;
-                //const short lx = (tiitg/NL0)%8;
-                //const short ly = i%8;
-
-                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
-            }
-        }
-
-        if (FC_mul_mm_bc_inp) {
-            for (short i = 0; i < 8; ++i) {
-                const short sx = (tiitg%NL1);
-                const short sy = (tiitg/NL1)/8;
-
-                const short lx = i;
-                const short ly = (tiitg/NL1)%8;
-                //const short lx = (tiitg/NL1)%8;
-                //const short ly = i;
-
-                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
-            }
-        } else {
-            const short sx = (tiitg%NL1);
-            const short sy = (tiitg/NL1)/8;
-
-            //const short lx = i;
-            const short ly = (tiitg/NL1)%8;
-            //const short lx = (tiitg/NL1)%8;
-            //const short ly = i;
-
-            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
-        }
-#endif
 
         il = (il + 2 < nl) ? il + 2 : il % 2;
         x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
@@ -9522,7 +9576,6 @@ kernel void kernel_mul_mm(
 
         threadgroup_barrier(mem_flags::mem_threadgroup);
 
-#ifndef GGML_METAL_HAS_TENSOR
         // load matrices from threadgroup memory and conduct outer products
         threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
         threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));
@@ -9549,24 +9602,10 @@ kernel void kernel_mul_mm(
             lsma += 8*64;
             lsmb += 4*64;
         }
-#else
-        auto sA = tA.slice(0, 0);
-        auto sB = tB.slice(0, 0);
-
-        mm.run(sB, sA, cT);
-#endif
     }
 
     if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
         // if no bounds checks on the output are needed, we can directly write to device memory
-#ifdef GGML_METAL_HAS_TENSOR
-        device float * C = (device float *) dst +
-            r0 + \
-            r1 * args.ne0 + im*args.ne1*args.ne0;
-
-        auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(args.ne0, NR1));
-        cT.store(tC);
-#else
         device float * C = (device float *) dst +
             (r0 + 32*(sgitg &  1)) + \
             (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;
@@ -9574,21 +9613,15 @@ kernel void kernel_mul_mm(
         for (short i = 0; i < 8; i++) {
             simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
         }
-#endif
     } else {
         // block is smaller than 64x32, we should avoid writing data outside of the matrix
         threadgroup_barrier(mem_flags::mem_threadgroup);
 
         threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
 
-#ifdef GGML_METAL_HAS_TENSOR
-        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
-        cT.store(tC);
-#else
         for (short i = 0; i < 8; i++) {
             simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
         }
-#endif
 
         threadgroup_barrier(mem_flags::mem_threadgroup);
 
@@ -9614,6 +9647,8 @@ kernel void kernel_mul_mm(
     }
 }
 
+#endif // GGML_METAL_HAS_TENSOR
+
 template<short ne20> // n_expert_used
 kernel void kernel_mul_mm_id_map0(
         constant ggml_metal_kargs_mul_mm_id_map0 & args,
@@ -9789,7 +9824,7 @@ kernel void kernel_mul_mm_id(
 
                 const short ib = 8*sx + sy;
 
-                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
+                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
             }
         } else {
             S0_4x4 temp_a;
diff --git src/ggml-opencl/CMakeLists.txt src/ggml-opencl/CMakeLists.txt
index 772fc537..5ed83eeb 100644
--- src/ggml-opencl/CMakeLists.txt
+++ src/ggml-opencl/CMakeLists.txt
@@ -96,6 +96,8 @@ set(GGML_OPENCL_KERNELS
     mul_mv_q6_k_f32_flat
     mul_mv_q8_0_f32
     mul_mv_q8_0_f32_flat
+    mul_mv_iq4_nl_f32
+    mul_mv_iq4_nl_f32_flat
     mul_mv_mxfp4_f32
     mul_mv_mxfp4_f32_flat
     mul_mv_id_q4_0_f32_8x_flat
@@ -110,12 +112,15 @@ set(GGML_OPENCL_KERNELS
     mul_mm_q4_0_f32_l4_lm
     mul_mm_q4_1_f32_l4_lm
     mul_mm_q8_0_f32_l4_lm
+    mul_mm_iq4_nl_f32_l4_lm
     mul_mm_q4_k_f32_l4_lm
     mul_mm_q5_k_f32_l4_lm
     mul_mm_q6_k_f32_l4_lm
     mul_mm_q8_0_f32_8x4
     gemv_noshuffle_q4_1_f32
     gemm_noshuffle_q4_1_f32
+    gemv_noshuffle_iq4_nl_f32
+    gemm_noshuffle_iq4_nl_f32
     gemv_noshuffle_general_q8_0_f32
     gemv_noshuffle_q4_k_f32
     gemm_noshuffle_q4_k_f32
diff --git src/ggml-opencl/ggml-opencl.cpp src/ggml-opencl/ggml-opencl.cpp
index 8bc7ae65..4d31591a 100644
--- src/ggml-opencl/ggml-opencl.cpp
+++ src/ggml-opencl/ggml-opencl.cpp
@@ -545,6 +545,9 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_convert_block_q5_K_noshuffle;
     cl_kernel kernel_restore_block_q5_K_noshuffle;
     cl_kernel kernel_convert_block_q6_K, kernel_restore_block_q6_K;
+    cl_kernel kernel_convert_block_iq4_nl, kernel_restore_block_iq4_nl;
+    cl_kernel kernel_convert_block_iq4_nl_noshuffle;
+    cl_kernel kernel_restore_block_iq4_nl_noshuffle;
     cl_kernel kernel_mul_mat_q4_0_f32_1d_8x_flat, kernel_mul_mat_q4_0_f32_1d_16x_flat;
     cl_kernel kernel_mul_mv_q4_1_f32;
     cl_kernel kernel_mul_mv_q4_1_f32_flat;
@@ -556,6 +559,8 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_mul_mv_q6_K_f32_flat;
     cl_kernel kernel_mul_mv_mxfp4_f32, kernel_mul_mv_mxfp4_f32_flat;
     cl_kernel kernel_mul_mv_q8_0_f32, kernel_mul_mv_q8_0_f32_flat;
+    cl_kernel kernel_mul_mv_iq4_nl_f32;
+    cl_kernel kernel_mul_mv_iq4_nl_f32_flat;
     cl_kernel kernel_solve_tri_f32;
     cl_kernel kernel_im2col_f32, kernel_im2col_f16;
     cl_kernel kernel_argsort_f32_i32;
@@ -594,6 +599,7 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_mul_mm_q4_k_f32_l4_lm;
     cl_kernel kernel_mul_mm_q5_k_f32_l4_lm;
     cl_kernel kernel_mul_mm_q6_k_f32_l4_lm;
+    cl_kernel kernel_mul_mm_iq4_nl_f32_l4_lm;
 
     std::vector<ProfilingInfo> profiling_info;
 
@@ -734,6 +740,8 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_gemm_noshuffle_q6_K_f32;
     cl_kernel kernel_gemv_noshuffle_q5_k_f32;
     cl_kernel kernel_gemm_noshuffle_q5_k_f32;
+    cl_kernel kernel_gemv_noshuffle_iq4_nl_f32;
+    cl_kernel kernel_gemm_noshuffle_iq4_nl_f32;
 #endif // GGML_OPENCL_USE_ADRENO_KERNELS
 
     void free() {
@@ -954,6 +962,10 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx, ggml_cl_ve
         CL_CHECK((backend_ctx->kernel_restore_block_q6_K  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q6_K", &err), err));
         CL_CHECK((backend_ctx->kernel_convert_block_q6_K_noshuffle  = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q6_K_noshuffle", &err), err));
         CL_CHECK((backend_ctx->kernel_restore_block_q6_K_noshuffle  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q6_K_noshuffle", &err), err));
+        CL_CHECK((backend_ctx->kernel_convert_block_iq4_nl = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_iq4_nl", &err), err));
+        CL_CHECK((backend_ctx->kernel_restore_block_iq4_nl = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_iq4_nl", &err), err));
+        CL_CHECK((backend_ctx->kernel_convert_block_iq4_nl_noshuffle = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_iq4_nl_noshuffle", &err), err));
+        CL_CHECK((backend_ctx->kernel_restore_block_iq4_nl_noshuffle = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_iq4_nl_noshuffle", &err), err));
         GGML_LOG_CONT(".");
     }
 
@@ -1359,6 +1371,40 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx, ggml_cl_ve
         GGML_LOG_CONT(".");
     }
 
+    // mul_mv_iq4_nl_f32
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "mul_mv_iq4_nl_f32.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("mul_mv_iq4_nl_f32.cl");
+#endif
+        cl_program prog =
+            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
+
+        CL_CHECK((backend_ctx->kernel_mul_mv_iq4_nl_f32 = clCreateKernel(prog, "kernel_mul_mv_iq4_nl_f32", &err), err));
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
+    // mul_mv_iq4_nl_f32_flat
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "mul_mv_iq4_nl_f32_flat.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("mul_mv_iq4_nl_f32_flat.cl");
+#endif
+        cl_program prog =
+            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
+
+        CL_CHECK((backend_ctx->kernel_mul_mv_iq4_nl_f32_flat = clCreateKernel(prog, "kernel_mul_mv_iq4_nl_f32_flat", &err), err));
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
     // mul_mv_mxfp4_f32
     {
 #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -1567,6 +1613,23 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx, ggml_cl_ve
         GGML_LOG_CONT(".");
     }
 
+    // mul_mm_iq4_nl_f32_l4_lm
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "mul_mm_iq4_nl_f32_l4_lm.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("mul_mm_iq4_nl_f32_l4_lm.cl");
+#endif
+        cl_program prog =
+            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
+
+        CL_CHECK((backend_ctx->kernel_mul_mm_iq4_nl_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_iq4_nl_f32_l4_lm", &err), err));
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
     // mul_mm_q4_k_f32_l4_lm
     {
 #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -2647,6 +2710,45 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx, ggml_cl_ve
         GGML_LOG_CONT(".");
     }
 
+    // gemm_noshuffle_iq4_nl_f32
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "gemm_noshuffle_iq4_nl_f32.cl.h"
+       };
+#else
+        const std::string kernel_src = read_file("gemm_noshuffle_iq4_nl_f32.cl");
+#endif
+        cl_program prog = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
+        CL_CHECK((backend_ctx->kernel_gemm_noshuffle_iq4_nl_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_iq4_nl_f32", &err), err));
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
+    // gemv_noshuffle_iq4_nl_f32
+    {
+        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
+                                       " -cl-mad-enable ";
+        if (backend_ctx->has_vector_subgroup_broadcast) {
+            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
+        }
+
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "gemv_noshuffle_iq4_nl_f32.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("gemv_noshuffle_iq4_nl_f32.cl");
+#endif
+
+        cl_program prog = build_program_from_source(
+            backend_ctx->context, backend_ctx->device, kernel_src.c_str(), CL_gemv_compile_opts);
+
+        CL_CHECK((backend_ctx->kernel_gemv_noshuffle_iq4_nl_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_iq4_nl_f32", &err), err));
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
     // mul_mm_q8_0_f32_8x4
     {
 #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -3597,6 +3699,30 @@ struct ggml_tensor_extra_cl_q8_0 {
     }
 };
 
+struct ggml_tensor_extra_cl_iq4_nl {
+    cl_mem q = nullptr;
+    cl_mem q_img = nullptr;
+
+    cl_mem d = nullptr;
+    cl_mem d_img = nullptr;
+
+    size_t size_q = 0;
+    size_t size_d = 0;
+
+    ~ggml_tensor_extra_cl_iq4_nl() {
+        reset();
+    }
+
+    void reset() {
+        if (q != nullptr) { CL_CHECK(clReleaseMemObject(q)); q = nullptr; }
+        if (d != nullptr) { CL_CHECK(clReleaseMemObject(d)); d = nullptr; }
+        q_img = nullptr;
+        d_img = nullptr;
+        size_q = 0;
+        size_d = 0;
+    }
+};
+
 struct ggml_tensor_extra_cl_q4_K {
     // Quantized values
     cl_mem q = nullptr;
@@ -4097,6 +4223,7 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
                 return op->src[1]->type == GGML_TYPE_F32;
             } else if (op->src[0]->type == GGML_TYPE_Q4_0  || op->src[0]->type == GGML_TYPE_Q4_1 ||
                        op->src[0]->type == GGML_TYPE_MXFP4 ||
+                       op->src[0]->type == GGML_TYPE_IQ4_NL ||
                        op->src[0]->type == GGML_TYPE_Q4_K  ||
                        op->src[0]->type == GGML_TYPE_Q5_K  ||
                        op->src[0]->type == GGML_TYPE_Q6_K) {
@@ -4295,6 +4422,12 @@ struct ggml_backend_opencl_buffer_context {
         for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
             delete e;
         }
+        for (ggml_tensor_extra_cl_iq4_nl * e : temp_tensor_extras_iq4_nl) {
+            delete e;
+        }
+        for (ggml_tensor_extra_cl_iq4_nl * e : temp_tensor_extras_iq4_nl_in_use) {
+            delete e;
+        }
         for (ggml_tensor_extra_cl_q4_K * e : temp_tensor_extras_q4_K) {
             delete e;
         }
@@ -4390,6 +4523,21 @@ struct ggml_backend_opencl_buffer_context {
         return extra;
     }
 
+    ggml_tensor_extra_cl_iq4_nl * ggml_opencl_alloc_temp_tensor_extra_iq4_nl() {
+        ggml_tensor_extra_cl_iq4_nl * extra;
+        if (temp_tensor_extras_iq4_nl.empty()) {
+            extra = new ggml_tensor_extra_cl_iq4_nl();
+        } else {
+            extra = temp_tensor_extras_iq4_nl.back();
+            temp_tensor_extras_iq4_nl.pop_back();
+        }
+
+        temp_tensor_extras_iq4_nl_in_use.push_back(extra);
+
+        extra->reset();
+        return extra;
+    }
+
     ggml_tensor_extra_cl_q4_K * ggml_opencl_alloc_temp_tensor_extra_q4_K() {
         ggml_tensor_extra_cl_q4_K * extra;
         if (temp_tensor_extras_q4_K.empty()) {
@@ -4461,6 +4609,11 @@ struct ggml_backend_opencl_buffer_context {
         }
         temp_tensor_extras_q8_0_in_use.clear();
 
+        for (ggml_tensor_extra_cl_iq4_nl * e : temp_tensor_extras_iq4_nl_in_use) {
+            temp_tensor_extras_iq4_nl.push_back(e);
+        }
+        temp_tensor_extras_iq4_nl_in_use.clear();
+
         for (ggml_tensor_extra_cl_q4_K * e : temp_tensor_extras_q4_K_in_use) {
             temp_tensor_extras_q4_K.push_back(e);
         }
@@ -4492,6 +4645,8 @@ struct ggml_backend_opencl_buffer_context {
     std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4_in_use;
     std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0;
     std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0_in_use;
+    std::vector<ggml_tensor_extra_cl_iq4_nl *> temp_tensor_extras_iq4_nl;
+    std::vector<ggml_tensor_extra_cl_iq4_nl *> temp_tensor_extras_iq4_nl_in_use;
     std::vector<ggml_tensor_extra_cl_q4_K *> temp_tensor_extras_q4_K;
     std::vector<ggml_tensor_extra_cl_q4_K *> temp_tensor_extras_q4_K_in_use;
     std::vector<ggml_tensor_extra_cl_q5_K *> temp_tensor_extras_q5_K;
@@ -5123,6 +5278,87 @@ static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer,
 
         return;
     }
+    if (tensor->type == GGML_TYPE_IQ4_NL) {
+        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
+        GGML_ASSERT(extra_orig && "Tensors in OpenCL backend should have been allocated and initialized");
+
+        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
+        ggml_tensor_extra_cl_iq4_nl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_iq4_nl();
+
+        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
+        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)/2);
+        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");
+
+        cl_int err;
+        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
+            ggml_nbytes(tensor), NULL, &err);
+        CL_CHECK(err);
+        CL_CHECK(clEnqueueWriteBuffer(
+            queue, data_device, CL_TRUE, 0,
+            ggml_nbytes(tensor), data, 0, NULL, NULL));
+
+        cl_buffer_region region;
+
+        // Create subbuffer for scales.
+        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
+        region.size = size_d;
+        extra->d = clCreateSubBuffer(
+            extra_orig->data_device, CL_MEM_READ_WRITE,
+            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
+        CL_CHECK(err);
+        auto previous_origin = region.origin;
+
+        // Create subbuffer for quants.
+        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
+        region.size = size_q;
+        extra->q = clCreateSubBuffer(
+            extra_orig->data_device, CL_MEM_READ_WRITE,
+            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
+        CL_CHECK(err);
+
+    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+        cl_kernel kernel = backend_ctx->kernel_convert_block_iq4_nl;
+        if (use_adreno_kernels(backend_ctx, tensor)) {
+            kernel = backend_ctx->kernel_convert_block_iq4_nl_noshuffle;
+        }
+    #else
+        cl_kernel kernel = backend_ctx->kernel_convert_block_iq4_nl;
+    #endif
+        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
+        cl_uchar mask_0F = 0x0F;
+        cl_uchar mask_F0 = 0xF0;
+
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &n_blk));
+
+        size_t global_work_size[] = {(size_t)CEIL_DIV(n_blk, 64)*64, 1, 1};
+        size_t local_work_size[] = {64, 1, 1};
+
+        cl_event evt;
+        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
+        CL_CHECK(clWaitForEvents(1, &evt));
+        CL_CHECK(clReleaseMemObject(data_device));
+
+        tensor->extra = extra;
+
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+        if (use_adreno_kernels(backend_ctx, tensor)) {
+            int M = tensor->ne[1];
+            int K = tensor->ne[0];
+            GGML_ASSERT(K % 32 == 0);
+
+            // Transpose q as ushort
+            transpose_2d_as_16b(backend_ctx, extra->q, extra->q, size_q, K/4, M);
+            // Transpose d as ushort
+            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);
+        }
+#endif
+        return;
+    }
     if (tensor->type == GGML_TYPE_Q4_K) {
         ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
         GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");
@@ -5775,6 +6011,78 @@ static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer,
         CL_CHECK(clReleaseMemObject(data_device));
         return;
     }
+    if (tensor->type == GGML_TYPE_IQ4_NL) {
+        ggml_tensor_extra_cl_iq4_nl * extra = (ggml_tensor_extra_cl_iq4_nl *)tensor->extra;
+
+        cl_int err;
+        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
+            ggml_nbytes(tensor), NULL, &err);
+        CL_CHECK(err);
+
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+        if (use_adreno_kernels(backend_ctx, tensor)) {
+            static ggml_cl_buffer buf_trans_q;
+            static ggml_cl_buffer buf_trans_d;
+            static ggml_cl_buffer buf_unpacked;
+
+            cl_int M = tensor->ne[1];
+            cl_int K = tensor->ne[0];
+            GGML_ASSERT(K % 32 == 0);
+
+            size_t size_q = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*(ggml_blck_size(tensor->type)/2);
+            size_t size_d = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
+            GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");
+
+            buf_trans_q.allocate(backend_ctx->context, size_q);
+            buf_trans_d.allocate(backend_ctx->context, size_d);
+            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));
+
+            // transpose q, d back
+            transpose_2d_as_16b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/4);
+            transpose_2d_as_16b(backend_ctx, extra->d, buf_trans_d.buffer, size_d, M, K/32);
+
+            cl_uchar mask_0F = 0x0F;
+            cl_uchar mask_F0 = 0xF0;
+
+            cl_kernel kernel = backend_ctx->kernel_restore_block_iq4_nl_noshuffle;
+            cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
+
+            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q.buffer));
+            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_d.buffer));
+            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_unpacked.buffer));
+            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
+            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));
+            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &n_blk));
+
+            size_t global_work_size[] = {(size_t)n_blk, 1, 1};
+            size_t local_work_size[] = {1, 1, 1};
+
+            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
+            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
+            return;
+        }
+#endif
+        cl_kernel kernel = backend_ctx->kernel_restore_block_iq4_nl;
+        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
+
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_blk));
+
+        size_t global_work_size[] = {(size_t)n_blk, 1, 1};
+        size_t local_work_size[] = {1, 1, 1};
+
+        cl_event evt;
+        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
+            global_work_size, local_work_size, 0, NULL, &evt));
+        CL_CHECK(clWaitForEvents(1, &evt));
+        CL_CHECK(clEnqueueReadBuffer(
+            queue, data_device, CL_TRUE, offset,
+            size, data, 0, NULL, NULL));
+        CL_CHECK(clReleaseMemObject(data_device));
+        return;
+    }
     if (tensor->type == GGML_TYPE_Q4_K) {
         ggml_tensor_extra_cl_q4_K * extra = (ggml_tensor_extra_cl_q4_K *)tensor->extra;
 
@@ -9840,6 +10148,178 @@ static void ggml_cl_mul_mat_q4_1_f32_adreno(ggml_backend_t backend, const ggml_t
 #endif
 }
 
+static void ggml_cl_mul_mat_iq4_nl_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+    GGML_ASSERT(src0);
+    GGML_ASSERT(src0->extra);
+    GGML_ASSERT(src1);
+    GGML_ASSERT(src1->extra);
+    GGML_ASSERT(dst);
+    GGML_ASSERT(dst->extra);
+
+    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
+
+    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
+    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
+    ggml_tensor_extra_cl_iq4_nl * extra0_iq4_nl = (ggml_tensor_extra_cl_iq4_nl *)src0->extra;
+
+    cl_ulong offset1 = extra1->offset + src1->view_offs;
+    cl_ulong offsetd = extrad->offset + dst->view_offs;
+
+    const int  ne00 = src0->ne[0];
+    const int  ne01 = src0->ne[1];
+
+    const int  ne1 = dst->ne[1];
+
+    GGML_ASSERT(ne00 % 32 == 0);
+
+    cl_context context = backend_ctx->context;
+    cl_kernel kernel;
+
+    cl_int              err;
+    cl_image_format     img_fmt;
+    cl_image_desc       img_desc;
+    cl_buffer_region    region;
+
+    int M = ne01;
+    int N = ne1;
+    int K = ne00;
+
+    if (ne1 == 1) {
+        cl_mem q_img = nullptr;
+        cl_mem b_sub_buf = nullptr;
+        cl_mem b_img = nullptr;
+
+        // image for q
+        img_fmt = { CL_R, CL_UNSIGNED_INT32};
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = M * K / 2 / 4;
+        img_desc.buffer = extra0_iq4_nl->q;
+        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+        // subbuffer for activations
+        region.origin = offset1;
+        region.size = K * N * sizeof(float);
+        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+
+        // image for activations
+        img_fmt = {CL_RGBA, CL_FLOAT};
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = K * N / 4;
+        img_desc.buffer = b_sub_buf;
+        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+        kernel = backend_ctx->kernel_gemv_noshuffle_iq4_nl_f32;
+
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &q_img));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_iq4_nl->d));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extrad->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offsetd));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &ne00));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne01));
+
+        size_t local_work_size[3] = {64, 4, 1};
+        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, 4, 1};
+
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+
+        CL_CHECK(clReleaseMemObject(q_img));
+        CL_CHECK(clReleaseMemObject(b_sub_buf));
+        CL_CHECK(clReleaseMemObject(b_img));
+    } else {
+        cl_mem b_sub_buf = nullptr;
+        cl_mem b_sub_buf_trans = nullptr;
+        cl_mem b_img = nullptr;
+        cl_mem b_img_trans = nullptr;
+
+        // subbuffer for activations
+        region.origin = offset1;
+        region.size = K * N * sizeof(float);
+        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+
+        // image for activations
+        img_fmt = {CL_RGBA, CL_FLOAT};
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = K * N / 4;
+        img_desc.buffer = b_sub_buf;
+        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+        // pad N to multiple of 8
+        int extra_elements = N % 8;
+        int padding = 0;
+        if (extra_elements > 0){
+            padding = 8 - extra_elements;
+        }
+
+        // subbuffer for transposed activations
+        region.origin = 0;
+        region.size = K * (N + padding) * sizeof(float)/2;
+        backend_ctx->prealloc_act_trans.allocate(context, region.size);
+        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+
+        // image for transposed activations
+        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = K * (N + padding) / 4;
+        img_desc.buffer = b_sub_buf_trans;
+        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));
+
+        // transpose activations
+        int height_B = N/4;
+        if (height_B == 0) {
+            height_B = 1;
+        }
+        int width_B = K/4;
+        int padded_height_B = (N + padding)/4;
+
+        kernel = backend_ctx->kernel_transpose_32_16;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));
+
+        size_t local_work_size_t[2] = { 1, 16 };
+        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
+        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);
+
+        // gemm
+        kernel = backend_ctx->kernel_gemm_noshuffle_iq4_nl_f32;
+        int padded_N = N + padding;
+
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_iq4_nl->q));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_iq4_nl->d));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img_trans));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extrad->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offsetd));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &ne01));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &padded_N));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne1));
+
+        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
+        size_t local_work_size[3] = {1, 128, 1};
+
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+
+        CL_CHECK(clReleaseMemObject(b_sub_buf));
+        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
+        CL_CHECK(clReleaseMemObject(b_img));
+        CL_CHECK(clReleaseMemObject(b_img_trans));
+    }
+#else
+    GGML_UNUSED(backend);
+    GGML_UNUSED(src0);
+    GGML_UNUSED(src1);
+    GGML_UNUSED(dst);
+#endif
+}
+
 static void ggml_cl_mul_mat_q8_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
 #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
     GGML_ASSERT(src0);
@@ -10634,6 +11114,7 @@ static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, co
     ggml_tensor_extra_cl_q4_1 * extra0_q4_1 = (ggml_tensor_extra_cl_q4_1 *)src0->extra;
     ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *)src0->extra;
     ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)src0->extra;
+    ggml_tensor_extra_cl_iq4_nl * extra0_iq4_nl = (ggml_tensor_extra_cl_iq4_nl *)src0->extra;
     ggml_tensor_extra_cl_q4_K * extra0_q4_K = (ggml_tensor_extra_cl_q4_K *)src0->extra;
     ggml_tensor_extra_cl_q5_K * extra0_q5_K = (ggml_tensor_extra_cl_q5_K *)src0->extra;
     ggml_tensor_extra_cl_q6_K * extra0_q6_K = (ggml_tensor_extra_cl_q6_K *)src0->extra;
@@ -10738,6 +11219,12 @@ static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, co
             return;
     }
 
+    // iq4_nl x fp32
+    if (src0t == GGML_TYPE_IQ4_NL && src1t == GGML_TYPE_F32) {
+        ggml_cl_mul_mat_iq4_nl_f32_adreno(backend, src0, src1, dst);
+        return;
+    }
+
     // q8_0 x fp32
     if (src0t == GGML_TYPE_Q8_0 && src1t == GGML_TYPE_F32 &&
         enable_adreno_trans_weight(backend_ctx, src0)) {
@@ -11302,6 +11789,48 @@ static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, co
                 backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                 return;
             }
+            case GGML_TYPE_IQ4_NL: {
+                if (ne11 < 32) {
+                    break;
+                }
+                if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
+                    break;
+                }
+
+                kernel = backend_ctx->kernel_mul_mm_iq4_nl_f32_l4_lm;
+                nth0 = 128; // calculated as (BM*BN)/(TM*TN)
+
+                int batch_stride_a = ne00*ne01;
+                int batch_stride_b = ne10*ne11;
+                int batch_stride_d = ne0*ne1;
+
+                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_iq4_nl->q));
+                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_iq4_nl->d));
+                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
+                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
+                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
+                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
+                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
+                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
+                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
+                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
+                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
+                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
+                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
+                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
+                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
+                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
+                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
+                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
+                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
+
+                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
+                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
+                size_t local_work_size[] = {(size_t)nth0, 1, 1};
+
+                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+                return;
+            }
             case GGML_TYPE_Q4_K: {
                 if (ne11 < 32) {
                     break;
@@ -11829,6 +12358,70 @@ static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, co
             CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
             CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
             CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
+#endif // GGML_OPENCL_SOA_Q
+            break;
+        }
+        case GGML_TYPE_IQ4_NL: {
+#ifdef GGML_OPENCL_SOA_Q
+            kernel = backend_ctx->kernel_mul_mv_iq4_nl_f32_flat;
+
+            if (backend_ctx->gpu_family == INTEL) {
+                nth0 = 16;
+                nth1 = 1;
+                ndst = 8;
+            } else if (backend_ctx->gpu_family == ADRENO) {
+                nth0 = 64;
+                nth1 = 1;
+                ndst = 8;
+            } else {
+                GGML_ASSERT(false && "TODO: Unknown GPU");
+            }
+
+            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_iq4_nl->q));
+            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_iq4_nl->d));
+            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
+            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
+            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
+            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
+            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
+            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
+            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
+            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
+            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
+            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
+            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
+            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
+            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
+#else
+            kernel = backend_ctx->kernel_mul_mv_iq4_nl_f32;
+
+            if (backend_ctx->gpu_family == INTEL) {
+                nth0 = 16;
+                nth1 = 1;
+                ndst = 4;
+            } else if (backend_ctx->gpu_family == ADRENO) {
+                nth0 = 64;
+                nth1 = 1;
+                ndst = 4;
+            } else {
+                GGML_ASSERT(false && "TODO: Unknown GPU");
+            }
+
+            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
+            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
+            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
+            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
+            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
+            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
+            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
+            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
+            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
+            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
+            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
+            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
+            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
+            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
+            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
 #endif // GGML_OPENCL_SOA_Q
             break;
         }
@@ -12131,6 +12724,7 @@ static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, co
     if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_MXFP4 ||
         src0t == GGML_TYPE_Q4_1 ||
         src0t == GGML_TYPE_Q8_0 ||
+        src0t == GGML_TYPE_IQ4_NL ||
         src0t == GGML_TYPE_Q2_K) {
         // Each SIMD group produces N_DST values in the result. Assuming each
         // workgroup has N_SIMDGROUP SIMD groups, then each workgroup will
diff --git src/ggml-opencl/kernels/cvt.cl src/ggml-opencl/kernels/cvt.cl
index 39af32d2..f3937d83 100644
--- src/ggml-opencl/kernels/cvt.cl
+++ src/ggml-opencl/kernels/cvt.cl
@@ -87,6 +87,17 @@ struct block_q6_K {
     half d;                  // super-block scale
 };
 
+//------------------------------------------------------------------------------
+// block_iq4_nl
+//------------------------------------------------------------------------------
+#define QK4_NL 32
+
+struct block_iq4_nl
+{
+    half d;
+    uint8_t qs[QK4_NL / 2];
+};
+
 //------------------------------------------------------------------------------
 // kernel_convert_block_q4_0
 // Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
@@ -895,3 +906,99 @@ kernel void kernel_restore_block_q6_K_noshuffle(
         b->scales[i] = s[i];
     }
 }
+
+//------------------------------------------------------------------------------
+// kernel_convert_block_iq4_nl
+// Convert the block_iq4_nl format to 2 separate arrays (AOS -> SOA).
+//------------------------------------------------------------------------------
+kernel void kernel_convert_block_iq4_nl(
+    global struct block_iq4_nl * src0,
+    global uchar * dst_q,
+    global half  * dst_d,
+    uchar          mask_0F,
+    uchar          mask_F0,
+    ulong          n_blk
+) {
+    if (get_global_id(0) >= n_blk) {
+        return;
+    }
+    global struct block_iq4_nl * b = (global struct block_iq4_nl *) src0 + get_global_id(0);
+    global uchar * q = (global uchar *) dst_q + QK4_NL/2*get_global_id(0);
+    global half  * d = (global half *) dst_d + get_global_id(0);
+
+    *d = b->d;
+
+    for (int i = 0; i < QK4_NL/2; ++i) {
+        q[i] = b->qs[i];
+    }
+}
+
+kernel void kernel_restore_block_iq4_nl(
+    global uchar * src_q,
+    global half  * src_d,
+    global struct block_iq4_nl * dst,
+    ulong          n_blk
+) {
+    if (get_global_id(0) >= n_blk) {
+        return;
+    }
+    global struct block_iq4_nl * b = (global struct block_iq4_nl *) dst + get_global_id(0);
+    global uchar * q = (global uchar *) src_q + QK4_NL/2*get_global_id(0);
+    global half  * d = (global half *) src_d + get_global_id(0);
+
+    b->d = *d;
+
+    for (int i = 0; i < QK4_NL/2; ++i) {
+        b->qs[i] = q[i];
+    }
+}
+
+kernel void kernel_convert_block_iq4_nl_noshuffle(
+    global struct block_iq4_nl * src0,
+    global uchar * dst_q,
+    global half  * dst_d,
+    uchar          mask_0F,
+    uchar          mask_F0,
+    ulong          n_blk
+) {
+    if (get_global_id(0) >= n_blk) {
+        return;
+    }
+    global struct block_iq4_nl * b = (global struct block_iq4_nl *) src0 + get_global_id(0);
+    global uchar * q = (global uchar *) dst_q + QK4_NL/2*get_global_id(0);
+    global half  * d = (global half *) dst_d + get_global_id(0);
+
+    *d = b->d;
+    for (int i = 0; i < QK4_NL/4; ++i) {
+        uchar x0 = b->qs[2*i + 0];
+        uchar x1 = b->qs[2*i + 1];
+
+        q[i + 0       ] = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
+        q[i + QK4_NL/4] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
+    }
+}
+
+kernel void kernel_restore_block_iq4_nl_noshuffle(
+    global uchar * src_q,
+    global half  * src_d,
+    global struct block_iq4_nl * dst,
+    uchar mask_0F,
+    uchar mask_F0,
+    ulong n_blk
+) {
+    if (get_global_id(0) >= n_blk) {
+        return;
+    }
+    global struct block_iq4_nl * b = (global struct block_iq4_nl *) dst + get_global_id(0);
+    global uchar * q = (global uchar *) src_q + QK4_NL/2*get_global_id(0);
+    global half  * d = (global half *) src_d + get_global_id(0);
+
+    b->d = *d;
+    for (int i = 0; i < QK4_NL/4; ++i) {
+        uchar x0 = q[i + 0       ];
+        uchar x1 = q[i + QK4_NL/4];
+
+        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
+        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
+    }
+}
diff --git src/ggml-opencl/kernels/gemm_noshuffle_iq4_nl_f32.cl src/ggml-opencl/kernels/gemm_noshuffle_iq4_nl_f32.cl
new file mode 100644
index 00000000..6869d822
--- /dev/null
+++ src/ggml-opencl/kernels/gemm_noshuffle_iq4_nl_f32.cl
@@ -0,0 +1,150 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
+
+#ifdef cl_qcom_reqd_sub_group_size
+#define ADRENO_GPU 1
+#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
+#endif
+
+constant half kvalues_iq4nl[16] = {
+    (half)-127.f, (half)-104.f, (half)-83.f, (half)-65.f,
+    (half) -49.f, (half) -35.f, (half)-22.f, (half)-10.f,
+    (half)   1.f, (half)  13.f, (half) 25.f, (half) 38.f,
+    (half)  53.f, (half)  69.f, (half) 89.f, (half)113.f
+};
+
+// Packed LUT: 2 FP16 values per uint, 8 unique constant loads instead of 16
+constant uint iq4nl_packed[8] = {
+    0xD680D7F0u,  // idx 0,1: -127, -104
+    0xD410D530u,  // idx 2,3: -83, -65
+    0xD060D220u,  // idx 4,5: -49, -35
+    0xC900CD80u,  // idx 6,7: -22, -10
+    0x4A803C00u,  // idx 8,9: 1, 13
+    0x50C04E40u,  // idx 10,11: 25, 38
+    0x545052A0u,  // idx 12,13: 53, 69
+    0x57105590u   // idx 14,15: 89, 113
+};
+
+// Packed dequant: 1 uint constant load (8-way divergence) + shift + as_half
+#define IQ4_NL_DEQUANT(nibble) as_half((ushort)(iq4nl_packed[(nibble) >> 1] >> (((nibble) & 1u) << 4)))
+
+#ifdef ADRENO_GPU
+REQD_SUBGROUP_SIZE_128
+#endif
+
+kernel void kernel_gemm_noshuffle_iq4_nl_f32(
+        global const ushort * src0_q,
+        global const half  * src0_d,
+        read_only image1d_buffer_t src1,
+        global float * dst,
+        ulong offsetd,
+        int m,
+        int n,
+        int k,
+        int n_no_padding
+) {
+    dst = (global float *)((global char *)dst + offsetd);
+
+    int m_4 = m >> 2;
+    int n_4 = n >> 2;
+
+    int gy = get_global_id(0);
+    int gx = get_global_id(1);
+    int gx_2 = gx << 2;
+
+    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
+    half8 B;
+    half4 dequantized_weights;
+
+    global const ushort * weight_ptr = src0_q + gx_2;
+    global const half * scale_ptr = src0_d + gx_2;
+
+    for (int i = 0; i < k; i += 4) {
+        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
+        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
+
+        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
+
+        half4 scale = vload4(0, scale_ptr + (i/32)*(m));
+
+        // j=0
+        dequantized_weights.s0 = IQ4_NL_DEQUANT(bits4.s0 & 0x000Fu) * scale.s0;
+        dequantized_weights.s1 = IQ4_NL_DEQUANT(bits4.s1 & 0x000Fu) * scale.s1;
+        dequantized_weights.s2 = IQ4_NL_DEQUANT(bits4.s2 & 0x000Fu) * scale.s2;
+        dequantized_weights.s3 = IQ4_NL_DEQUANT(bits4.s3 & 0x000Fu) * scale.s3;
+        c0 += B * dequantized_weights.s0;
+        c1 += B * dequantized_weights.s1;
+        c2 += B * dequantized_weights.s2;
+        c3 += B * dequantized_weights.s3;
+
+        // j=1
+        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
+        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
+        dequantized_weights.s0 = IQ4_NL_DEQUANT((bits4.s0 >> 4) & 0x000Fu) * scale.s0;
+        dequantized_weights.s1 = IQ4_NL_DEQUANT((bits4.s1 >> 4) & 0x000Fu) * scale.s1;
+        dequantized_weights.s2 = IQ4_NL_DEQUANT((bits4.s2 >> 4) & 0x000Fu) * scale.s2;
+        dequantized_weights.s3 = IQ4_NL_DEQUANT((bits4.s3 >> 4) & 0x000Fu) * scale.s3;
+        c0 += B * dequantized_weights.s0;
+        c1 += B * dequantized_weights.s1;
+        c2 += B * dequantized_weights.s2;
+        c3 += B * dequantized_weights.s3;
+
+        // j=2
+        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
+        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
+        dequantized_weights.s0 = IQ4_NL_DEQUANT((bits4.s0 >> 8) & 0x000Fu) * scale.s0;
+        dequantized_weights.s1 = IQ4_NL_DEQUANT((bits4.s1 >> 8) & 0x000Fu) * scale.s1;
+        dequantized_weights.s2 = IQ4_NL_DEQUANT((bits4.s2 >> 8) & 0x000Fu) * scale.s2;
+        dequantized_weights.s3 = IQ4_NL_DEQUANT((bits4.s3 >> 8) & 0x000Fu) * scale.s3;
+        c0 += B * dequantized_weights.s0;
+        c1 += B * dequantized_weights.s1;
+        c2 += B * dequantized_weights.s2;
+        c3 += B * dequantized_weights.s3;
+
+        // j=3
+        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
+        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
+        dequantized_weights.s0 = IQ4_NL_DEQUANT((bits4.s0 >> 12) & 0x000Fu) * scale.s0;
+        dequantized_weights.s1 = IQ4_NL_DEQUANT((bits4.s1 >> 12) & 0x000Fu) * scale.s1;
+        dequantized_weights.s2 = IQ4_NL_DEQUANT((bits4.s2 >> 12) & 0x000Fu) * scale.s2;
+        dequantized_weights.s3 = IQ4_NL_DEQUANT((bits4.s3 >> 12) & 0x000Fu) * scale.s3;
+        c0 += B * dequantized_weights.s0;
+        c1 += B * dequantized_weights.s1;
+        c2 += B * dequantized_weights.s2;
+        c3 += B * dequantized_weights.s3;
+    }
+
+    int idx = (gy<<3)*m + (gx<<2);
+
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
+        idx += m;
+    }
+    if(idx+3 < m*n_no_padding){
+        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
+    }
+}
diff --git src/ggml-opencl/kernels/gemv_noshuffle_iq4_nl_f32.cl src/ggml-opencl/kernels/gemv_noshuffle_iq4_nl_f32.cl
new file mode 100644
index 00000000..9386bf25
--- /dev/null
+++ src/ggml-opencl/kernels/gemv_noshuffle_iq4_nl_f32.cl
@@ -0,0 +1,302 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+#pragma OPENCL EXTENSION cl_khr_subgroups : enable
+
+#ifdef cl_qcom_reqd_sub_group_size
+#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
+#define ADRENO_GPU 1
+#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
+#endif
+
+#define QK4_NL 32
+#define NSUBGROUPS 4
+#define SUBGROUP_SIZE 64
+
+constant half kvalues_iq4nl[16] = {
+    (half)-127.f, (half)-104.f, (half)-83.f, (half)-65.f,
+    (half) -49.f, (half) -35.f, (half)-22.f, (half)-10.f,
+    (half)   1.f, (half)  13.f, (half) 25.f, (half) 38.f,
+    (half)  53.f, (half)  69.f, (half) 89.f, (half)113.f
+};
+
+// Packed LUT: 2 FP16 values per uint, 8 unique constant loads instead of 16
+constant uint iq4nl_packed[8] = {
+    0xD680D7F0u,  // idx 0,1: -127, -104
+    0xD410D530u,  // idx 2,3: -83, -65
+    0xD060D220u,  // idx 4,5: -49, -35
+    0xC900CD80u,  // idx 6,7: -22, -10
+    0x4A803C00u,  // idx 8,9: 1, 13
+    0x50C04E40u,  // idx 10,11: 25, 38
+    0x545052A0u,  // idx 12,13: 53, 69
+    0x57105590u   // idx 14,15: 89, 113
+};
+
+// Packed dequant: 1 uint constant load (8-way divergence) + shift + as_half
+#define IQ4_NL_DEQUANT(nibble) as_half((ushort)(iq4nl_packed[(nibble) >> 1] >> (((nibble) & 1u) << 4)))
+
+#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, y) \
+    float shared_y; \
+    shared_y = sub_group_broadcast(y.s0, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s0 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s1 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s1, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s2, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s3, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s4, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s2 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s3 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s5, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s6, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s7, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s0, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s4 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s5 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s1, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s2, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s3, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s4, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s6 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s7 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s5, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s6, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s7, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+
+
+#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, y) \
+    shared_y = sub_group_broadcast(y.s0, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s0 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s1 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s1, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s2, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s3, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s4, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s2 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s3 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s5, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s6, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s7, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s0, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s4 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s5 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s1, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s2, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s3, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s4, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s6 & 0x000F)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s7 & 0x000F)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s5, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x00F0) >> 4)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x00F0) >> 4)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s6, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x0F00) >> 8)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x0F00) >> 8)) * scale.s1 * shared_y; \
+    shared_y = sub_group_broadcast(y.s7, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0xF000) >> 12)) * scale.s0 * shared_y; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0xF000) >> 12)) * scale.s1 * shared_y; \
+
+
+#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, y) \
+    float8 shared_y; \
+    shared_y = sub_group_broadcast(y, 0); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s0 & 0x000F))         * scale.s0 * shared_y.s0; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x00F0) >> 4))  * scale.s0 * shared_y.s1; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x0F00) >> 8))  * scale.s0 * shared_y.s2; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0xF000) >> 12)) * scale.s0 * shared_y.s3; \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s2 & 0x000F))         * scale.s0 * shared_y.s4; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x00F0) >> 4))  * scale.s0 * shared_y.s5; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x0F00) >> 8))  * scale.s0 * shared_y.s6; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0xF000) >> 12)) * scale.s0 * shared_y.s7; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s1 & 0x000F))         * scale.s1 * shared_y.s0; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x00F0) >> 4))  * scale.s1 * shared_y.s1; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x0F00) >> 8))  * scale.s1 * shared_y.s2; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0xF000) >> 12)) * scale.s1 * shared_y.s3; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s3 & 0x000F))         * scale.s1 * shared_y.s4; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x00F0) >> 4))  * scale.s1 * shared_y.s5; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x0F00) >> 8))  * scale.s1 * shared_y.s6; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0xF000) >> 12)) * scale.s1 * shared_y.s7; \
+    shared_y = sub_group_broadcast(y, 1); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s4 & 0x000F))         * scale.s0 * shared_y.s0; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x00F0) >> 4))  * scale.s0 * shared_y.s1; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x0F00) >> 8))  * scale.s0 * shared_y.s2; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0xF000) >> 12)) * scale.s0 * shared_y.s3; \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s6 & 0x000F))         * scale.s0 * shared_y.s4; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x00F0) >> 4))  * scale.s0 * shared_y.s5; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x0F00) >> 8))  * scale.s0 * shared_y.s6; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0xF000) >> 12)) * scale.s0 * shared_y.s7; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s5 & 0x000F))         * scale.s1 * shared_y.s0; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x00F0) >> 4))  * scale.s1 * shared_y.s1; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x0F00) >> 8))  * scale.s1 * shared_y.s2; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0xF000) >> 12)) * scale.s1 * shared_y.s3; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s7 & 0x000F))         * scale.s1 * shared_y.s4; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x00F0) >> 4))  * scale.s1 * shared_y.s5; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x0F00) >> 8))  * scale.s1 * shared_y.s6; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0xF000) >> 12)) * scale.s1 * shared_y.s7; \
+
+
+#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, y) \
+    shared_y = sub_group_broadcast(y, 2); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s0 & 0x000F))         * scale.s0 * shared_y.s0; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x00F0) >> 4))  * scale.s0 * shared_y.s1; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0x0F00) >> 8))  * scale.s0 * shared_y.s2; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s0 & 0xF000) >> 12)) * scale.s0 * shared_y.s3; \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s2 & 0x000F))         * scale.s0 * shared_y.s4; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x00F0) >> 4))  * scale.s0 * shared_y.s5; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0x0F00) >> 8))  * scale.s0 * shared_y.s6; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s2 & 0xF000) >> 12)) * scale.s0 * shared_y.s7; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s1 & 0x000F))         * scale.s1 * shared_y.s0; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x00F0) >> 4))  * scale.s1 * shared_y.s1; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0x0F00) >> 8))  * scale.s1 * shared_y.s2; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s1 & 0xF000) >> 12)) * scale.s1 * shared_y.s3; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s3 & 0x000F))         * scale.s1 * shared_y.s4; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x00F0) >> 4))  * scale.s1 * shared_y.s5; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0x0F00) >> 8))  * scale.s1 * shared_y.s6; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s3 & 0xF000) >> 12)) * scale.s1 * shared_y.s7; \
+    shared_y = sub_group_broadcast(y, 3); \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s4 & 0x000F))         * scale.s0 * shared_y.s0; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x00F0) >> 4))  * scale.s0 * shared_y.s1; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0x0F00) >> 8))  * scale.s0 * shared_y.s2; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s4 & 0xF000) >> 12)) * scale.s0 * shared_y.s3; \
+    total_sums.s0 += IQ4_NL_DEQUANT((bits4.s6 & 0x000F))         * scale.s0 * shared_y.s4; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x00F0) >> 4))  * scale.s0 * shared_y.s5; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0x0F00) >> 8))  * scale.s0 * shared_y.s6; \
+    total_sums.s0 += IQ4_NL_DEQUANT(((bits4.s6 & 0xF000) >> 12)) * scale.s0 * shared_y.s7; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s5 & 0x000F))         * scale.s1 * shared_y.s0; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x00F0) >> 4))  * scale.s1 * shared_y.s1; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0x0F00) >> 8))  * scale.s1 * shared_y.s2; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s5 & 0xF000) >> 12)) * scale.s1 * shared_y.s3; \
+    total_sums.s1 += IQ4_NL_DEQUANT((bits4.s7 & 0x000F))         * scale.s1 * shared_y.s4; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x00F0) >> 4))  * scale.s1 * shared_y.s5; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0x0F00) >> 8))  * scale.s1 * shared_y.s6; \
+    total_sums.s1 += IQ4_NL_DEQUANT(((bits4.s7 & 0xF000) >> 12)) * scale.s1 * shared_y.s7; \
+
+#ifdef ADRENO_GPU
+REQD_SUBGROUP_SIZE_64
+#endif
+kernel void kernel_gemv_noshuffle_iq4_nl_f32(
+        read_only  image1d_buffer_t src0_q,
+        global half2  * src0_d,
+        read_only  image1d_buffer_t src1,
+        global float * dst,
+        ulong offsetd,
+        int ne00,
+        int ne01)
+{
+    uint groupId = get_local_id(1);
+    uint gid     = get_global_id(0);
+    ushort slid    = get_sub_group_local_id();
+
+    uint K = ne00;
+    uint M = ne01;
+
+    uint LINE_STRIDE_A = M / 2;
+    uint BLOCK_STRIDE_A = NSUBGROUPS * M;
+
+    private uint4     regA;
+    private half2     regS;
+    private float8    regB;
+
+    private float2 totalSum = (float2)(0.0f);
+
+    // loop along K in block granularity, skip 4 blocks every iter
+    for (uint k = groupId; k < (K / QK4_NL); k += NSUBGROUPS) {
+        regS = src0_d[gid + k * LINE_STRIDE_A]; // each fiber loads scale of two rows
+        // first 4 fibers in each wave load 8 B values to its private scope
+        if (slid < 4) {
+            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
+            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
+        }
+
+        // load half weights for two blocks in consecutive rows
+        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
+        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
+        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
+        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
+#ifdef VECTOR_SUB_GROUP_BROADCAST
+        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regB);
+#else
+        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regB);
+#endif // VECTOR_SUB_GROUP_BROADCAST
+
+        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
+        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
+        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
+        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
+#ifdef VECTOR_SUB_GROUP_BROADCAST
+        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regB);
+#else
+        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regB);
+#endif // VECTOR_SUB_GROUP_BROADCAST
+    }
+
+    // reduction in local memory, assumes #wave=4
+    local float2 reduceLM[SUBGROUP_SIZE * 3];
+    if (groupId == 1) {
+        reduceLM[SUBGROUP_SIZE * 0 + slid] = totalSum;
+    }
+    if (groupId == 2) {
+        reduceLM[SUBGROUP_SIZE * 1 + slid] = totalSum;
+    }
+    if (groupId == 3) {
+        reduceLM[SUBGROUP_SIZE * 2 + slid] = totalSum;
+    }
+
+    barrier(CLK_LOCAL_MEM_FENCE);
+
+    if (groupId == 0) {
+        totalSum += reduceLM[SUBGROUP_SIZE * 0 + slid];
+    }
+    if (groupId == 0) {
+        totalSum += reduceLM[SUBGROUP_SIZE * 1 + slid];
+    }
+    if (groupId == 0) {
+        totalSum += reduceLM[SUBGROUP_SIZE * 2 + slid];
+    }
+
+    // 2 outputs per fiber in wave 0
+    if (groupId == 0) {
+        dst = (global float*)((global char*)dst + offsetd);
+        vstore2(totalSum, 0, &(dst[gid * 2]));
+    }
+
+}
diff --git src/ggml-opencl/kernels/mul_mm_iq4_nl_f32_l4_lm.cl src/ggml-opencl/kernels/mul_mm_iq4_nl_f32_l4_lm.cl
new file mode 100644
index 00000000..11ff7f8d
--- /dev/null
+++ src/ggml-opencl/kernels/mul_mm_iq4_nl_f32_l4_lm.cl
@@ -0,0 +1,171 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+
+#define LOAD_VEC_A 8
+#define LOAD_VEC_B 4
+
+#define BM 64
+#define BN 64
+#define BK 32
+#define TM 4
+#define TN 8
+
+constant float kvalues_iq4nl[16] = {
+    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
+      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
+};
+
+kernel void kernel_mul_mm_iq4_nl_f32_l4_lm(
+    global uchar4 * src0_q,
+    global half   * src0_d,
+    global float4 * src1,
+    ulong offset1,
+    global float  * dst,
+    ulong offsetd,
+
+    int ne00,
+    int ne01,
+    int ne02,
+    int ne11,
+    int ne12,
+
+    int stride_a,
+    int stride_b,
+    int stride_d,
+
+    int batch_stride_a,
+    int batch_stride_b,
+    int batch_stride_d,
+
+    int r2,
+    int r3
+) {
+    src1 = (global float4*)((global char*)src1 + offset1);
+    dst  = (global float *)((global char*)dst  + offsetd);
+
+    local float buf_a[BM * BK];
+    local float buf_b[BN * BK];
+
+    const int batch_idx = get_global_id(2);
+
+    const int i13 = batch_idx / ne12;
+    const int i12 = batch_idx % ne12;
+
+    const int i03 = i13 / r3;
+    const int i02 = i12 / r2;
+
+    const int batch_idx_a = i03 * ne02 + i02;
+
+    const int ir = get_group_id(0);
+    const int ic = get_group_id(1);
+
+    const int tid = get_local_id(0);
+    const int th_r  = tid % (BM / TM);
+    const int th_c  = tid / (BM / TM);
+
+    const int loadr_a = get_local_id(0) % (BK / LOAD_VEC_A);
+    const int loadc_a = get_local_id(0) / (BK / LOAD_VEC_A);
+    const int loadr_b = get_local_id(0) % (BK / LOAD_VEC_B);
+    const int loadc_b = get_local_id(0) / (BK / LOAD_VEC_B);
+
+    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;
+    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;
+
+    int pos_a = (batch_idx_a * batch_stride_a + ir * BM * stride_a) / LOAD_VEC_A;
+    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;
+
+    float sums[TM * TN];
+    float cache_a[TM];
+    float cache_b[TN];
+
+    for (int i = 0; i < TM * TN; i++) {
+        sums[i] = 0.0f;
+    }
+
+    for (int block = 0; block < ne00; block += BK) {
+        for (int l = 0; l < BM; l += loadstride_a) {
+            if (ir*BM + loadc_a + l < ne01) {
+                int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
+                int ib  = idx / 4;
+                int iqs = idx % 4;
+
+                float d = (float)src0_d[ib];
+                global uchar4 * qs = src0_q + ib*4 + iqs;
+                uchar4 q = *qs;
+                // IQ4_NL: use lookup table instead of linear (nibble - 8)
+                float4 v1 = (float4)(kvalues_iq4nl[(q.s0   )&0x0F], kvalues_iq4nl[(q.s1   )&0x0F],
+                                     kvalues_iq4nl[(q.s2   )&0x0F], kvalues_iq4nl[(q.s3   )&0x0F])*d;
+                float4 v2 = (float4)(kvalues_iq4nl[(q.s0>>4)&0x0F], kvalues_iq4nl[(q.s1>>4)&0x0F],
+                                     kvalues_iq4nl[(q.s2>>4)&0x0F], kvalues_iq4nl[(q.s3>>4)&0x0F])*d;
+
+                buf_a[(loadr_a * 4 +  0) * BM + loadc_a + l] = v1.s0;
+                buf_a[(loadr_a * 4 +  1) * BM + loadc_a + l] = v1.s1;
+                buf_a[(loadr_a * 4 +  2) * BM + loadc_a + l] = v1.s2;
+                buf_a[(loadr_a * 4 +  3) * BM + loadc_a + l] = v1.s3;
+                buf_a[(loadr_a * 4 + 16) * BM + loadc_a + l] = v2.s0;
+                buf_a[(loadr_a * 4 + 17) * BM + loadc_a + l] = v2.s1;
+                buf_a[(loadr_a * 4 + 18) * BM + loadc_a + l] = v2.s2;
+                buf_a[(loadr_a * 4 + 19) * BM + loadc_a + l] = v2.s3;
+            } else {
+                buf_a[(loadr_a * 4 +  0) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 +  1) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 +  2) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 +  3) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 + 16) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 + 17) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 + 18) * BM + loadc_a + l] = 0.0f;
+                buf_a[(loadr_a * 4 + 19) * BM + loadc_a + l] = 0.0f;
+            }
+        }
+
+        for (int l = 0; l < BN; l += loadstride_b) {
+            if (ic*BN + loadc_b + l < ne11) {
+                int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
+                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
+                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
+                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
+                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
+            } else {
+                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0f;
+                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0f;
+                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0f;
+                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0f;
+            }
+        }
+
+        barrier(CLK_LOCAL_MEM_FENCE);
+
+        pos_a += BK / LOAD_VEC_A;
+        pos_b += BK / LOAD_VEC_B;
+
+        for (int i = 0; i < BK; i++) {
+            for (int j = 0; j < TM; j++) {
+                cache_a[j] = buf_a[(i) * BM + th_r * TM + j];
+            }
+
+            for (int j = 0; j < TN; j++) {
+                cache_b[j] = buf_b[(i) * BN + th_c * TN + j];
+            }
+
+            for (int cc = 0; cc < TN; cc++) {
+                for (int cr = 0; cr < TM; cr++) {
+                    const int sums_idx = cc*TM + cr;
+                    sums[sums_idx] = mad(cache_a[cr], cache_b[cc], sums[sums_idx]);
+                }
+            }
+        }
+        barrier(CLK_LOCAL_MEM_FENCE);
+    }
+
+    const int dr = ir * BM + th_r * TM;
+    const int dc = ic * BN + th_c * TN;
+
+    const int offsets = batch_idx * batch_stride_d;
+
+    for (int cc = 0; cc < TN; cc++) {
+        for (int cr = 0; cr < TM; cr++) {
+            if (dr + cr < ne01 && dc + cc < ne11) {
+                dst[offsets + (dc + cc) * stride_d + dr + cr] = sums[cc * TM + cr];
+            }
+        }
+    }
+}
diff --git src/ggml-opencl/kernels/mul_mv_iq4_nl_f32.cl src/ggml-opencl/kernels/mul_mv_iq4_nl_f32.cl
new file mode 100644
index 00000000..a6a325cd
--- /dev/null
+++ src/ggml-opencl/kernels/mul_mv_iq4_nl_f32.cl
@@ -0,0 +1,164 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+
+#ifdef cl_intel_subgroups
+#pragma OPENCL EXTENSION cl_intel_subgroups : enable
+#else
+#pragma OPENCL EXTENSION cl_khr_subgroups : enable
+#endif
+
+#ifdef cl_intel_required_subgroup_size
+#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
+#define INTEL_GPU 1
+#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
+#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
+#elif defined(cl_qcom_reqd_sub_group_size)
+#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
+#define ADRENO_GPU 1
+#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
+#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
+#endif
+
+#define QK4_NL 32
+
+typedef char int8_t;
+typedef uchar uint8_t;
+typedef short int16_t;
+typedef ushort uint16_t;
+typedef int int32_t;
+typedef uint uint32_t;
+
+constant float kvalues_iq4nl[16] = {
+    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
+      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
+};
+
+//------------------------------------------------------------------------------
+// block_iq4_nl
+//------------------------------------------------------------------------------
+struct block_iq4_nl
+{
+    half d;
+    uint8_t qs[QK4_NL / 2];
+};
+
+//------------------------------------------------------------------------------
+// mul_vec_q_n_f32
+//------------------------------------------------------------------------------
+// Compute inner product between half a block of iq4_nl and 16 floats (yl).
+// il indicates where the quants begin (0 or 8).
+inline float block_iq4_nl_dot_y(
+        global struct block_iq4_nl * qb_curr,
+        private float * yl,
+        int il
+) {
+    float d = qb_curr->d;
+    float acc = 0.f;
+    global uchar * qs = qb_curr->qs + il;
+    for (int i = 0; i < 8; ++i) {
+        acc += yl[i]   * kvalues_iq4nl[qs[i] & 0x0F];
+        acc += yl[i+8] * kvalues_iq4nl[qs[i] >> 4];
+    }
+    return d * acc;
+}
+
+#ifdef INTEL_GPU
+#define N_DST 4 // each subgroup group works on 4 rows
+#define N_SUBGROUP 1 // number of subgroups in a thread group
+#define N_SUBGROUP_SIZE 16 // assuming subgroup size is 16
+#elif defined (ADRENO_GPU)
+#define N_DST 4
+#define N_SUBGROUP 1
+#define N_SUBGROUP_SIZE 64
+#endif
+
+inline void mul_vec_q_n_f32(
+        global void * src0,
+        global float * src1,
+        global float * dst,
+        int ne00,
+        int ne01,
+        int ne02,
+        int ne10,
+        int ne12,
+        int ne0,
+        int ne1,
+        int r2,
+        int r3
+) {
+
+    const ulong nb = ne00/QK4_NL;
+
+    int r0 = get_group_id(0);
+    int r1 = get_group_id(1);
+    int im = get_group_id(2);
+
+    int first_row = (r0 * N_SUBGROUP + get_sub_group_id()) * N_DST;
+
+    int i12 = im%ne12;
+    int i13 = im/ne12;
+
+    ulong offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
+
+    global struct block_iq4_nl * x = (global struct block_iq4_nl *) src0 + offset0;
+    global float               * y = (global float               *) src1 + r1*ne10 + im*ne00*ne1;
+
+    float yl[16];       // src1 vector cache
+    float sumf[N_DST]={0.f};
+
+    int ix = get_sub_group_local_id()/2;
+    int il = 8*(get_sub_group_local_id()%2);
+
+    global float * yb = y + ix * QK4_NL + il;
+
+    // each thread in a SIMD group deals with half a block.
+    for (int ib = ix; ib < nb; ib += N_SUBGROUP_SIZE/2) {
+        for (int i = 0; i < 8; ++i) {
+            yl[i]   = yb[i];
+            yl[i+8] = yb[i+16];
+        }
+
+        for (int row = 0; row < N_DST; row++) {
+            sumf[row] += block_iq4_nl_dot_y(x+ib+row*nb, yl, il);
+        }
+
+        yb += QK4_NL * (N_SUBGROUP_SIZE/2);
+    }
+
+    float tot[N_DST] = {
+        sub_group_reduce_add(sumf[0]), sub_group_reduce_add(sumf[1]),
+        sub_group_reduce_add(sumf[2]), sub_group_reduce_add(sumf[3])};
+    for (int row = 0; row < N_DST; ++row) {
+        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot[row];
+        }
+    }
+}
+
+#ifdef INTEL_GPU
+REQD_SUBGROUP_SIZE_16
+#elif defined (ADRENO_GPU)
+REQD_SUBGROUP_SIZE_64
+#endif
+kernel void kernel_mul_mv_iq4_nl_f32(
+        global void * src0,
+        ulong offset0,
+        global float * src1,
+        ulong offset1,
+        global float * dst,
+        ulong offsetd,
+        int ne00,
+        int ne01,
+        int ne02,
+        int ne10,
+        int ne12,
+        int ne0,
+        int ne1,
+        int r2,
+        int r3
+) {
+    src0 = (global void*)((global char*)src0 + offset0);
+    src1 = (global float*)((global char*)src1 + offset1);
+    dst = (global float*)((global char*)dst + offsetd);
+
+    mul_vec_q_n_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
+}
diff --git src/ggml-opencl/kernels/mul_mv_iq4_nl_f32_flat.cl src/ggml-opencl/kernels/mul_mv_iq4_nl_f32_flat.cl
new file mode 100644
index 00000000..8c5b3f52
--- /dev/null
+++ src/ggml-opencl/kernels/mul_mv_iq4_nl_f32_flat.cl
@@ -0,0 +1,202 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+
+#ifdef cl_intel_subgroups
+#pragma OPENCL EXTENSION cl_intel_subgroups : enable
+#else
+#pragma OPENCL EXTENSION cl_khr_subgroups : enable
+#endif
+
+#ifdef cl_intel_required_subgroup_size
+#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
+#define INTEL_GPU 1
+#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
+#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
+#elif defined(cl_qcom_reqd_sub_group_size)
+#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
+#define ADRENO_GPU 1
+#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
+#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
+#endif
+
+#define QK4_NL 32
+
+typedef char int8_t;
+typedef uchar uint8_t;
+typedef short int16_t;
+typedef ushort uint16_t;
+typedef int int32_t;
+typedef uint uint32_t;
+
+constant float kvalues_iq4nl[16] = {
+    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
+      1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
+};
+
+//------------------------------------------------------------------------------
+// block_iq4_nl
+//------------------------------------------------------------------------------
+struct block_iq4_nl
+{
+    half d;
+    uint8_t qs[QK4_NL / 2];
+};
+
+// Compute dot product between half a block of iq4_nl quants and activations.
+// x points to the quant bytes, dh points to the scale.
+// yl has 16 activation values: [0..7] for low nibbles, [8..15] for high nibbles.
+// il indicates offset into the quant bytes (0 or 8).
+inline float block_iq4_nl_dot_y_flat(
+        global uchar * x,
+        global half  * dh,
+        private float * yl,
+        int il
+) {
+    float d = *dh;
+    global uchar * qs = x + il;
+    float acc = 0.f;
+    for (int i = 0; i < 8; ++i) {
+        acc += yl[i]   * kvalues_iq4nl[qs[i] & 0x0F];
+        acc += yl[i+8] * kvalues_iq4nl[qs[i] >> 4];
+    }
+    return d * acc;
+}
+
+#undef N_DST
+#undef N_SIMDGROUP
+#undef N_SIMDWIDTH
+
+#ifdef INTEL_GPU
+#define N_DST 8 // each subgroup works on 8 rows
+#define N_SUBGROUP 1 // number of subgroups in a thread group
+#define N_SUBGROUP_SIZE 16 // assuming subgroup size is 16
+#elif defined (ADRENO_GPU)
+#define N_DST 8
+#define N_SUBGROUP 1
+#define N_SUBGROUP_SIZE 64
+#endif
+
+inline void mul_vec_q_n_f32_8x_flat(
+        global uchar * src0_q,
+        global half  * src0_d,
+        global float * src1,
+        global float * dst,
+        int ne00,
+        int ne01,
+        int ne02,
+        int ne10,
+        int ne12,
+        int ne0,
+        int ne1,
+        int r2,
+        int r3
+) {
+    const ulong nb = ne00/QK4_NL;
+
+    int r0 = get_group_id(0);
+    int r1 = get_group_id(1);
+    int im = get_group_id(2);
+
+    int first_row = (r0 * N_SUBGROUP + get_sub_group_id()) * N_DST;
+
+    int i12 = im%ne12;
+    int i13 = im/ne12;
+
+    // The number of scales is the same as the number of blocks.
+    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
+    // Each block contains QK4_NL/2 uchars, hence offset for qs is as follows.
+    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_NL/2;
+
+    global uchar * x = (global uchar *) src0_q + offset0_q;
+    global half  * d = (global half  *) src0_d + offset0_d;
+    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;
+
+    float yl[16];
+    float8 sumf = 0.f;
+
+    int ix = get_sub_group_local_id()/2;
+    int il = 8*(get_sub_group_local_id()%2);
+
+    global float * yb = y + ix*QK4_NL + il;
+
+    for (int ib = ix; ib < nb; ib += N_SUBGROUP_SIZE/2) {
+        for (int i = 0; i < 8; ++i) {
+            yl[i]   = yb[i];
+            yl[i+8] = yb[i+16];
+        }
+
+        sumf.s0 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 0*nb*QK4_NL/2, d + ib + 0*nb, yl, il);
+        sumf.s1 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 1*nb*QK4_NL/2, d + ib + 1*nb, yl, il);
+        sumf.s2 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 2*nb*QK4_NL/2, d + ib + 2*nb, yl, il);
+        sumf.s3 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 3*nb*QK4_NL/2, d + ib + 3*nb, yl, il);
+
+        sumf.s4 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 4*nb*QK4_NL/2, d + ib + 4*nb, yl, il);
+        sumf.s5 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 5*nb*QK4_NL/2, d + ib + 5*nb, yl, il);
+        sumf.s6 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 6*nb*QK4_NL/2, d + ib + 6*nb, yl, il);
+        sumf.s7 += block_iq4_nl_dot_y_flat(x + ib*QK4_NL/2 + 7*nb*QK4_NL/2, d + ib + 7*nb, yl, il);
+
+        yb += QK4_NL * (N_SUBGROUP_SIZE/2);
+    }
+
+    float8 tot = (float8)(
+        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
+        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
+        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
+        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
+    );
+
+    if (get_sub_group_local_id() == 0) {
+        if (first_row + 0 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
+        }
+        if (first_row + 1 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
+        }
+        if (first_row + 2 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
+        }
+        if (first_row + 3 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
+        }
+
+        if (first_row + 4 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
+        }
+        if (first_row + 5 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
+        }
+        if (first_row + 6 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
+        }
+        if (first_row + 7 < ne01) {
+            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
+        }
+    }
+}
+
+#ifdef INTEL_GPU
+REQD_SUBGROUP_SIZE_16
+#elif defined (ADRENO_GPU)
+REQD_SUBGROUP_SIZE_64
+#endif
+kernel void kernel_mul_mv_iq4_nl_f32_flat(
+        global uchar * src0_q,
+        global half  * src0_d,
+        global float * src1,
+        ulong offset1,
+        global float * dst,
+        ulong offsetd,
+        int ne00,
+        int ne01,
+        int ne02,
+        int ne10,
+        int ne12,
+        int ne0,
+        int ne1,
+        int r2,
+        int r3
+) {
+    src1 = (global float*)((global char*)src1 + offset1);
+    dst = (global float*)((global char*)dst + offsetd);
+
+    mul_vec_q_n_f32_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
+}
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
diff --git src/ggml-rpc/ggml-rpc.cpp src/ggml-rpc/ggml-rpc.cpp
index 2ded7397..505bec73 100644
--- src/ggml-rpc/ggml-rpc.cpp
+++ src/ggml-rpc/ggml-rpc.cpp
@@ -1101,7 +1101,7 @@ bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
         fs::path cache_file = fs::path(cache_dir) / hash_str;
         std::ofstream ofs(cache_file, std::ios::binary);
         ofs.write((const char *)data, size);
-        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.c_str());
+        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
     }
     ggml_backend_tensor_set(tensor, data, offset, size);
     return true;
diff --git src/ggml-sycl/common.hpp src/ggml-sycl/common.hpp
index fd84c917..5abf2290 100644
--- src/ggml-sycl/common.hpp
+++ src/ggml-sycl/common.hpp
@@ -28,6 +28,13 @@
 
 namespace syclexp = sycl::ext::oneapi::experimental;
 
+#if defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
+    #include <sycl/ext/oneapi/bfloat16.hpp>
+    #ifndef GGML_SYCL_HAS_BF16
+        #define GGML_SYCL_HAS_BF16
+    #endif
+#endif
+
 #if GGML_SYCL_DNNL
 #include "dnnl.hpp"
 #include "dnnl_sycl.hpp"
@@ -217,7 +224,7 @@ struct sycl_device_info {
                        // cudaOccupancyMaxActiveBlocksPerMultiprocessor
     bool    vmm;                // virtual memory support
     size_t  total_vram;
-    //sycl_hw_info hw_info;     \\ device id and aarch, currently not used
+    sycl_hw_info hw_info;
     optimize_feature opt_feature;
 };
 
diff --git src/ggml-sycl/convert.cpp src/ggml-sycl/convert.cpp
index f3c521b4..67b9c06f 100644
--- src/ggml-sycl/convert.cpp
+++ src/ggml-sycl/convert.cpp
@@ -2,13 +2,6 @@
 #include "dequantize.hpp"
 #include "presets.hpp"
 
-#if defined(__INTEL_LLVM_COMPILER)
-    #if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
-        #include <sycl/ext/oneapi/bfloat16.hpp>
-        #define GGML_SYCL_HAS_BF16
-    #endif
-#endif
-
 template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
 static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k,
                              const sycl::nd_item<3> &item_ct1) {
@@ -767,6 +760,22 @@ to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
 }
 
 
+#ifdef GGML_SYCL_HAS_BF16
+to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
+    switch (type) {
+        case GGML_TYPE_F32:
+            return convert_unary_sycl<float>;
+        case GGML_TYPE_F16:
+            return convert_unary_sycl<sycl::half>;
+        case GGML_TYPE_BF16:
+            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
+        default:
+            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
+            return nullptr;
+    }
+}
+#endif
+
 to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
     switch (type) {
         case GGML_TYPE_F32:
diff --git src/ggml-sycl/convert.hpp src/ggml-sycl/convert.hpp
index 6e621f21..8de79d10 100644
--- src/ggml-sycl/convert.hpp
+++ src/ggml-sycl/convert.hpp
@@ -23,6 +23,11 @@ typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;
 to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
 to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);
 
+#ifdef GGML_SYCL_HAS_BF16
+typedef to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
+to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
+#endif
+
 // Nc = Non-contiguous
 template <typename T>
 using to_t_nc_sycl_t = void (*)(const void * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
@@ -35,15 +40,19 @@ template<typename dst_t, typename src_t>
  inline dst_t ggml_sycl_cast(src_t x) {
     if constexpr (std::is_same_v<dst_t, src_t>) {
         return x;
+#ifdef GGML_SYCL_HAS_BF16
     } else if constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
         return sycl::ext::oneapi::bfloat16(float(x));
     } else if constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
         return static_cast<float>(x);
+#endif
     } else if constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
         return x.template convert<sycl::half, sycl::rounding_mode::rte>();
+#ifdef GGML_SYCL_HAS_BF16
     } else if constexpr (std::is_same_v<src_t, sycl::float2> &&
                          std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
         return {x.x, x.y};
+#endif
     } else if constexpr(std::is_same_v<dst_t, int32_t>) {
         return int32_t(x);
     } else {
diff --git src/ggml-sycl/gemm.hpp src/ggml-sycl/gemm.hpp
index dcf6c7ae..c202da11 100644
--- src/ggml-sycl/gemm.hpp
+++ src/ggml-sycl/gemm.hpp
@@ -29,6 +29,9 @@ public:
     static constexpr dt to_dt() {
         if constexpr (std::is_same_v<T, float>) return dt::f32;
         else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
+#ifdef GGML_SYCL_HAS_BF16
+        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return dt::bf16;
+#endif
         else static_assert(0);
     }
 
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index c02a41ad..1eead625 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -104,6 +104,7 @@ static ggml_sycl_device_info ggml_sycl_init() {
 
         info.max_work_group_sizes[i] = prop.get_max_work_group_size();
         info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
+        info.devices[i].hw_info = get_device_hw_info(&device);
 
     }
 
@@ -2176,6 +2177,31 @@ inline void ggml_sycl_op_mul_mat_sycl(
 #else
     bool use_fp16 = false;
 #endif
+
+#if GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
+    // Fast path for bf16 src0
+    if (src0->type == GGML_TYPE_BF16 && !g_ggml_sycl_disable_dnn && ggml_is_contiguous(src0) &&
+        row_diff == src0->ne[1]) {
+        using bf16_t = sycl::ext::oneapi::bfloat16;
+        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
+        if (src1->type != GGML_TYPE_BF16) {
+            const to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
+            GGML_ASSERT(to_bf16_sycl != nullptr);
+            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
+        } else {
+            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*sizeof(bf16_t));
+        }
+        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
+                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
+                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
+                                  dst_dd_i, DnnlGemmWrapper::to_dt<float>(), stream);
+        GGML_UNUSED(dst);
+        GGML_UNUSED(src1_ddq_i);
+        GGML_UNUSED(src1_padded_row_size);
+        return;
+    }
+#endif
+
     if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
         row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
         ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
@@ -3678,9 +3704,16 @@ static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor
     // Dispatch becomes obscure with the reorder, MMVQ when the reorder optimization
     // is enabled takes precedence over DMMV, the current if-else implementation
     // requires disabling DMMV if both conditions are met
+
     if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                           ggml_sycl_supports_reorder_mmvq(src0->type)))) {
-        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
+      // Arc770 get benefit with Q4_0 by skipping it.
+      if (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
+                gpu_arch::intel_gpu_acm_g10 &&
+            src0->type == GGML_TYPE_Q4_0)) {
+        use_dequantize_mul_mat_vec =
+            use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
+      }
     }
 
     if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
@@ -3783,6 +3816,51 @@ __dpct_inline__ static void k_copy_dst_from_contiguous(
     }
 }
 
+// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
+static bool ggml_sycl_mul_mat_id_mmvq_fused(
+    ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
+    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst)
+{
+    const int64_t ne10 = src1->ne[0];
+    const int64_t ne11 = src1->ne[1];
+    const int64_t ne12 = src1->ne[2];
+    if (ne12 != 1) return false;
+    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
+    if (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) return false;
+    if (!ggml_is_contiguous(src1)) return false;
+
+    // Reorder layout not supported; fall back.
+    const ggml_tensor_extra_gpu * src0_extra =
+        static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
+    if (src0_extra && src0_extra->optimized_feature.reorder) return false;
+
+    const int64_t n_ids_per_group = ids->ne[0];
+    if (ids->ne[1] != 1) return false;
+    if (ne11 != 1 && ne11 != n_ids_per_group) return false;
+
+    const queue_ptr stream           = ctx.stream();
+    const int       src1_padded_cols = GGML_PAD((int) ne10, MATRIX_ROW_PADDING);
+    const int       n_experts_used   = (int) n_ids_per_group;
+    const int       nrows            = (int) src0->ne[1];
+
+    ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(),
+        (size_t) ne11 * src1_padded_cols * sizeof(block_q8_1) / QK8_1);
+    char * src1_ddq = src1_q8_alloc.get();
+    quantize_row_q8_1_sycl<quantize_q8_1>(
+        (const float *) src1->data, src1_ddq, (int) ne10, (int) ne11,
+        src1_padded_cols, stream);
+
+    const size_t bytes_per_qrow = (size_t) src1_padded_cols * sizeof(block_q8_1) / QK8_1;
+    const size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;
+
+    return ggml_sycl_mul_mat_vec_q_id(
+        src0->type, src0->data, src1_ddq, (const int32_t *) ids->data,
+        (float *) dst->data, (int) ne10, nrows, n_experts_used,
+        /*expert_weight_stride=*/ src0->nb[2],
+        /*dst_row_stride=*/ dst->nb[1],
+        src1_row_stride, stream);
+}
+
 static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                  ggml_tensor *dst) try {
     scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
@@ -3798,6 +3876,12 @@ static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
     const int64_t n_as = ne02;
     const int64_t n_ids = ids->ne[0];
 
+    if (ne12 == 1) {
+        if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
+            return;
+        }
+    }
+
     std::vector<char> ids_host(ggml_nbytes(ids));
     const char * ids_dev = (const char *) ids->data;
 
@@ -3848,8 +3932,9 @@ static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
             }
         }
     } else {
-        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
-        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
+        const int64_t n_routed_rows = ids->ne[1] * n_ids;
+        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne10);
+        ggml_sycl_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*n_routed_rows*ne0);
 
         src1_row.data = src1_contiguous.get();
         dst_row.data  =  dst_contiguous.get();
diff --git src/ggml-sycl/mmvq.cpp src/ggml-sycl/mmvq.cpp
index 3a4577ec..8fa2198f 100644
--- src/ggml-sycl/mmvq.cpp
+++ src/ggml-sycl/mmvq.cpp
@@ -1199,3 +1199,154 @@ void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tens
     GGML_UNUSED(src1_ddf_i);
     GGML_UNUSED(ctx);
 }
+
+// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
+template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
+static void mul_mat_vec_q_moe(
+    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
+    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
+    const int ncols, const int nrows,
+    const size_t expert_weight_stride, const size_t dst_row_stride,
+    const size_t src1_row_stride,
+    const sycl::nd_item<3> & item_ct1) {
+
+    const int expert_idx = item_ct1.get_group(1);
+    const int i02        = ids_dev[expert_idx];
+
+    const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
+    const char * vy = (const char *) vy_base + (size_t) expert_idx * src1_row_stride;
+    float *      dst = (float *) ((char *) dst_base + (size_t) expert_idx * dst_row_stride);
+
+    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
+
+    if (row >= nrows) {
+        return;
+    }
+
+    const int     blocks_per_row  = ncols / qk;
+    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
+
+    float tmp = 0.0f;
+
+    const block_q_t *  x = (const block_q_t *) vx;
+    const block_q8_1 * y = (const block_q8_1 *) vy;
+
+    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
+        const int ibx = row * blocks_per_row + i;
+        const int iby = i * (qk / QK8_1);
+
+        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
+            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
+            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
+        }
+    }
+
+#pragma unroll
+    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
+        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
+    }
+
+    if (item_ct1.get_local_id(2) == 0) {
+        dst[row] = tmp;
+    }
+}
+
+template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
+static void launch_mul_mat_vec_q_moe(
+    const void * vx_base, const void * vy, const int32_t * ids_dev,
+    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
+    const size_t expert_weight_stride, const size_t dst_row_stride,
+    const size_t src1_row_stride,
+    dpct::queue_ptr stream) {
+    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
+    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
+    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
+    stream->submit([&](sycl::handler & cgh) {
+        cgh.parallel_for(
+            sycl::nd_range<3>(block_nums * block_dims, block_dims),
+            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
+                mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
+                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
+                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
+            });
+    });
+}
+
+bool ggml_sycl_mul_mat_vec_q_id(
+    enum ggml_type     src0_type,
+    const void *       vx_base,
+    const void *       vy,
+    const int32_t *    ids_dev,
+    float *            dst_base,
+    int                ncols,
+    int                nrows,
+    int                n_experts_used,
+    size_t             expert_weight_stride,
+    size_t             dst_row_stride,
+    size_t             src1_row_stride,
+    dpct::queue_ptr    stream) {
+    switch (src0_type) {
+        case GGML_TYPE_Q4_0:
+            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q4_1:
+            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q5_0:
+            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q5_1:
+            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q8_0:
+            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q2_K:
+            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q3_K:
+            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q4_K:
+            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q5_K:
+            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_Q6_K:
+            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_MXFP4:
+            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        case GGML_TYPE_NVFP4:
+            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
+                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
+                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
+            return true;
+        default:
+            return false;
+    }
+}
diff --git src/ggml-sycl/mmvq.hpp src/ggml-sycl/mmvq.hpp
index 049b43d4..d674dc1d 100644
--- src/ggml-sycl/mmvq.hpp
+++ src/ggml-sycl/mmvq.hpp
@@ -24,4 +24,20 @@ void ggml_sycl_op_mul_mat_vec_q(
     const int64_t src1_ncols, const int64_t src1_padded_row_size,
     const dpct::queue_ptr &stream);
 
+// Requires standard (non-reorder) block layout for src0.
+// Returns false if src0_type isn't handled; caller should fall back.
+bool ggml_sycl_mul_mat_vec_q_id(
+    enum ggml_type     src0_type,
+    const void *       vx_base,             // start of stacked expert weights
+    const void *       vy,                  // pre-quantized src1 (Q8_1)
+    const int32_t *    ids_dev,             // device-side int32, length n_experts_used
+    float *            dst_base,
+    int                ncols,
+    int                nrows,
+    int                n_experts_used,
+    size_t             expert_weight_stride, // bytes between experts in vx_base
+    size_t             dst_row_stride,       // bytes between dst rows
+    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
+    dpct::queue_ptr    stream);
+
 #endif // GGML_SYCL_MMVQ_HPP
diff --git src/ggml-sycl/set_rows.cpp src/ggml-sycl/set_rows.cpp
index a641c100..8fb41943 100644
--- src/ggml-sycl/set_rows.cpp
+++ src/ggml-sycl/set_rows.cpp
@@ -4,7 +4,11 @@
 namespace utils {
 template<typename T>
 static constexpr bool is_arithmetic_v() {
-    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half> || std::is_same_v<T, sycl::ext::oneapi::bfloat16>;
+    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
+#ifdef GGML_SYCL_HAS_BF16
+        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
+#endif
+        ;
 }
 }
 
@@ -181,6 +185,7 @@ static void set_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * s
                 stream
             );
             break;
+#ifdef GGML_SYCL_HAS_BF16
         case GGML_TYPE_BF16:
             set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                 src0_d, src1_d, (char *)dst->data,
@@ -193,6 +198,7 @@ static void set_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor * s
                 stream
             );
             break;
+#endif
         case GGML_TYPE_Q8_0:
             set_rows_sycl_q<TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(src0_d, src1_d, (block_q8_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
             break;
diff --git src/ggml-sycl/sycl_hw.cpp src/ggml-sycl/sycl_hw.cpp
index 70411400..03b0c37a 100644
--- src/ggml-sycl/sycl_hw.cpp
+++ src/ggml-sycl/sycl_hw.cpp
@@ -1,15 +1,67 @@
 #include "sycl_hw.hpp"
 
-// TODO: currently not used
-/*
-sycl_hw_info get_device_hw_info(sycl::device *device_ptr) {
-  sycl_hw_info res;
-  int32_t id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
-  res.device_id = id;
+using namespace std;
 
-  syclex::architecture arch = device_ptr->get_info<syclex::info::device::architecture>();
-  res.arch = arch;
+/*defined in
+* /opt/intel/oneapi/compiler/latest/include/sycl/ext/oneapi/experimental/device_architecture.def
+*/
+static map<gpu_arch, std::pair<const char*, sycl_intel_gpu_family>> arch2name = {
+    {gpu_arch::intel_gpu_bdw,     {"intel_gpu_bdw",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_skl,     {"intel_gpu_skl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_kbl,     {"intel_gpu_kbl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_cfl,     {"intel_gpu_cfl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_apl,     {"intel_gpu_apl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_glk,     {"intel_gpu_glk",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_whl,     {"intel_gpu_whl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_aml,     {"intel_gpu_aml",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_cml,     {"intel_gpu_cml",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_icllp,   {"intel_gpu_icllp",   GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_ehl,     {"intel_gpu_ehl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_tgllp,   {"intel_gpu_tgllp",   GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_rkl,     {"intel_gpu_rkl",     GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_adl_s,   {"intel_gpu_adl_s",   GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_adl_p,   {"intel_gpu_adl_p",   GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_adl_n,   {"intel_gpu_adl_n",   GPU_FAMILY_IGPU_NON_XE}},
+    {gpu_arch::intel_gpu_dg1,     {"intel_gpu_dg1",     GPU_FAMILY_DGPU_CLIENT_GAME}},
+    {gpu_arch::intel_gpu_acm_g10, {"intel_gpu_acm_g10", GPU_FAMILY_DGPU_CLIENT_GAME}},
+    {gpu_arch::intel_gpu_acm_g11, {"intel_gpu_acm_g11", GPU_FAMILY_DGPU_CLIENT_GAME}},
+    {gpu_arch::intel_gpu_acm_g12, {"intel_gpu_acm_g12", GPU_FAMILY_DGPU_CLIENT_GAME}},
+    {gpu_arch::intel_gpu_pvc,     {"intel_gpu_pvc",     GPU_FAMILY_DGPU_CLOUD}},
+    {gpu_arch::intel_gpu_pvc_vg,  {"intel_gpu_pvc_vg",  GPU_FAMILY_DGPU_CLOUD}},
+    {gpu_arch::intel_gpu_mtl_u,   {"intel_gpu_mtl_u",   GPU_FAMILY_IGPU_XE}},
+    {gpu_arch::intel_gpu_mtl_h,   {"intel_gpu_mtl_h",   GPU_FAMILY_IGPU_XE}},
+    {gpu_arch::intel_gpu_arl_h,   {"intel_gpu_arl_h",   GPU_FAMILY_IGPU_XE}},
+    {gpu_arch::intel_gpu_bmg_g21, {"intel_gpu_bmg_g21", GPU_FAMILY_DGPU_CLIENT_GAME}},
+    {gpu_arch::intel_gpu_bmg_g31, {"intel_gpu_bmg_g31", GPU_FAMILY_DGPU_CLIENT_GAME}},
+    {gpu_arch::intel_gpu_lnl_m,   {"intel_gpu_lnl_m",   GPU_FAMILY_IGPU_XE}},
+    {gpu_arch::intel_gpu_ptl_h,   {"intel_gpu_ptl_h",   GPU_FAMILY_IGPU_XE}},
+    {gpu_arch::intel_gpu_ptl_u,   {"intel_gpu_ptl_u",   GPU_FAMILY_IGPU_XE}},
+    {gpu_arch::intel_gpu_wcl,     {"intel_gpu_wcl",     GPU_FAMILY_IGPU_XE}}
+};
+
+
+sycl_hw_info get_device_hw_info(sycl::device* device_ptr) {
+    sycl_hw_info res;
+    int32_t id =
+        device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
+    res.device_id = id;
+
+    res.name = device_ptr->get_info<sycl::info::device::name>();
 
-  return res;
+    syclex::architecture arch =
+        device_ptr->get_info<syclex::info::device::architecture>();
+    res.arch = arch;
+
+    map<syclex::architecture,
+        std::pair<const char*, sycl_intel_gpu_family>>::iterator it =
+        arch2name.find(res.arch);
+    if (it != arch2name.end()) {
+        res.arch_name = it->second.first;
+        res.gpu_family = it->second.second;
+    } else {
+        res.arch_name = "unknown";
+        res.gpu_family = GPU_FAMILY_UKNOWN;
+    }
+
+    return res;
 }
-*/
diff --git src/ggml-sycl/sycl_hw.hpp src/ggml-sycl/sycl_hw.hpp
index 36b140bf..a5d20462 100644
--- src/ggml-sycl/sycl_hw.hpp
+++ src/ggml-sycl/sycl_hw.hpp
@@ -9,18 +9,30 @@
 #include <sycl/sycl.hpp>
 
 namespace syclex = sycl::ext::oneapi::experimental;
+using gpu_arch = sycl::ext::oneapi::experimental::architecture;
+
+// It's used to mark the GPU computing capacity
+// The value must flow the order of performance.
+enum sycl_intel_gpu_family {
+  GPU_FAMILY_UKNOWN = -1,
+  // iGPU without Xe core, before Meteor Lake iGPU(Xe)
+  GPU_FAMILY_IGPU_NON_XE = 0,
+  // iGPU with Xe core, Meteor Lake iGPU or newer.
+  GPU_FAMILY_IGPU_XE = 1,
+  // dGPU for gaming in client/data center (DG1/FLex 140 or newer).
+  GPU_FAMILY_DGPU_CLIENT_GAME = 2,
+  // dGPU for AI in cloud, PVC or newer.
+  GPU_FAMILY_DGPU_CLOUD = 3
+};
 
-// TODO: currently not used
-/*
 struct sycl_hw_info {
   syclex::architecture arch;
+  const char* arch_name;
   int32_t device_id;
+  std::string name;
+  sycl_intel_gpu_family gpu_family;
 };
 
-bool is_in_vector(std::vector<int> &vec, int item);
-
 sycl_hw_info get_device_hw_info(sycl::device *device_ptr);
-*/
-
 
 #endif // SYCL_HW_HPP
diff --git src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan.cpp
index 702a249d..69c24bb5 100644
--- src/ggml-vulkan/ggml-vulkan.cpp
+++ src/ggml-vulkan/ggml-vulkan.cpp
@@ -20,12 +20,19 @@ DispatchLoaderDynamic & ggml_vk_default_dispatcher();
 #define VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()
 
 #include <vulkan/vulkan.hpp>
-// SPIRV-Headers: LunarG Windows SDK uses Include/spirv-headers/spirv.hpp (not spirv/unified1/). MinGW/MSYS2 and
-// Linux packages use Khronos layout spirv/unified1/spirv.hpp. See docs/build.md#vulkan.
-#if defined(_WIN32) && !defined(__MINGW32__)
-#include <spirv-headers/spirv.hpp>
+
+// SPIR-V Headers: different SDK installations expose different include paths.
+// LunarG Vulkan SDK on Windows typically provides <spirv-headers/spirv.hpp>.
+// Linux packages, MSYS2 and MinGW often use the Khronos layout <spirv/unified1/spirv.hpp>.
+#if __has_include(<spirv/unified1/spirv.hpp>)
+#    include <spirv/unified1/spirv.hpp>
+#elif __has_include(<spirv-headers/spirv.hpp>)
+#    include <spirv-headers/spirv.hpp>
+#elif __has_include(<spirv.hpp>)
+#    include <spirv.hpp>
 #else
-#include <spirv/unified1/spirv.hpp>
+     // Fallback to let the compiler throw a standard "file not found" error
+#    include <spirv/unified1/spirv.hpp>
 #endif
 
 #include <algorithm>
@@ -792,6 +799,7 @@ struct vk_device_struct {
     vk_pipeline pipeline_arange_f32;
 
     vk_pipeline pipeline_fill_f32;
+    vk_pipeline pipeline_fill_f16;
 
     vk_pipeline pipeline_geglu[2];
     vk_pipeline pipeline_reglu[2];
@@ -4577,6 +4585,7 @@ static void ggml_vk_load_shaders(vk_device& device) {
     ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
 
     ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
 
 #define CREATE_GLU(name)  \
     ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
@@ -9844,6 +9853,9 @@ static vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, const
         if (dst->type == GGML_TYPE_F32) {
             return ctx->device->pipeline_fill_f32;
         }
+        if (dst->type == GGML_TYPE_F16) {
+            return ctx->device->pipeline_fill_f16;
+        }
         return nullptr;
     default:
         return nullptr;
@@ -13002,6 +13014,7 @@ static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgr
             if (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                 ctx->query_node_idx[ctx->query_idx] = node_idx;
                 compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
+                ggml_vk_sync_buffers(ctx, compute_ctx);
             }
         }
         // Add all fused nodes to the unsynchronized lists.
@@ -14491,6 +14504,7 @@ static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cg
         compute_ctx = ggml_vk_get_compute_ctx(ctx);
         ctx->query_idx = 0;
         compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
+        ggml_vk_sync_buffers(ctx, compute_ctx);
     }
 
     ctx->prealloc_y_last_pipeline_used = nullptr;
@@ -14727,6 +14741,7 @@ static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cg
                 ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                 ctx->query_fusion_names[ctx->query_idx] = fusion_string;
                 compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
+                ggml_vk_sync_buffers(ctx, compute_ctx);
             } else {
                 // track a fusion string and number of fused ops for the current node_idx
                 ctx->query_fusion_names[i] = fusion_string;
@@ -15713,8 +15728,9 @@ static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggm
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
diff --git src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl
index e99108dc..bc580aee 100644
--- src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl
+++ src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl
@@ -296,13 +296,22 @@ vec2 get_dm_scale(uint ib, uint iqs) {
     const uint ib_k = ib / 8;
     const uint iqs_k = (ib % 8) * 8 + iqs;
     const uint is = iqs_k / 8;
-    u8vec2 scale_dm;
-    if (is < 4) {
-        scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
-    } else {
-        scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
-                          (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
-    }
+
+    const uvec3 scales = uvec3(data_a_packed32[ib_k].scales[0],
+                               data_a_packed32[ib_k].scales[1],
+                               data_a_packed32[ib_k].scales[2]);
+    const uint scalesoffs = (is & 3) * 8;
+
+    const uint scidx0 = (is < 4) ? 0 : 2;
+    const uint scidxshift0 = scalesoffs;
+    const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
+    const uint mbidx0 = (is < 4) ? 1 : 2;
+    const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
+    const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
+
+    const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
+    const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
+    u8vec2 scale_dm = u8vec2(sc, mbyte);
 
     return FLOAT_TYPEV2(data_a_packed32[ib_k].dm) * FLOAT_TYPEV2(scale_dm);
 }
diff --git src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl
index 6e4a29d2..73595168 100644
--- src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl
+++ src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl
@@ -201,19 +201,20 @@ void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uin
 
             const vec2 loadd = vec2(data_a[ib].dm);
 
-            const uint scidx0 = (is < 4) ? is : (is + 4);
-            const uint scidx1 = (is < 4) ? is : (is - 4);
-            const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
-            const uint scidxshift1 = (is < 4) ? 0 : 2;
-            const uint mbidx0 = is + 4;
-            const uint mbidx1 = (is < 4) ? is + 4 : is;
-            const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
-            const uint mbidxshift0 = (is < 4) ? 0 : 4;
-            const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
-            const uint mbidxshift1 = (is < 4) ? 0 : 2;
-
-            const uint8_t sc = uint8_t((data_a[ib].scales[scidx0] & 0xF) | ((data_a[ib].scales[scidx1] & scidxmask1) >> scidxshift1));
-            const uint8_t mbyte = uint8_t((data_a[ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));
+            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
+                                       data_a_packed32[ib].scales[1],
+                                       data_a_packed32[ib].scales[2]);
+            const uint scalesoffs = (is & 3) * 8;
+
+            const uint scidx0 = (is < 4) ? 0 : 2;
+            const uint scidxshift0 = scalesoffs;
+            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
+            const uint mbidx0 = (is < 4) ? 1 : 2;
+            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
+            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
+
+            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
+            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
 
             const float d = loadd.x * sc;
             const float m = -loadd.y * mbyte;
@@ -237,19 +238,20 @@ void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uin
 
             const vec2 loadd = vec2(data_a[ib].dm);
 
-            const uint scidx0 = (is < 4) ? is : (is + 4);
-            const uint scidx1 = (is < 4) ? is : (is - 4);
-            const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
-            const uint scidxshift1 = (is < 4) ? 0 : 2;
-            const uint mbidx0 = is + 4;
-            const uint mbidx1 = (is < 4) ? is + 4 : is;
-            const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
-            const uint mbidxshift0 = (is < 4) ? 0 : 4;
-            const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
-            const uint mbidxshift1 = (is < 4) ? 0 : 2;
-
-            const uint8_t sc    = uint8_t((data_a[ib].scales[scidx0] & 0xF)                         | ((data_a[ib].scales[scidx1] & scidxmask1) >> scidxshift1));
-            const uint8_t mbyte = uint8_t(((data_a[ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((data_a[ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));
+            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
+                                       data_a_packed32[ib].scales[1],
+                                       data_a_packed32[ib].scales[2]);
+            const uint scalesoffs = (is & 3) * 8;
+
+            const uint scidx0 = (is < 4) ? 0 : 2;
+            const uint scidxshift0 = scalesoffs;
+            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
+            const uint mbidx0 = (is < 4) ? 1 : 2;
+            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
+            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
+
+            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
+            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
 
             const float d = loadd.x * sc;
             const float m = -loadd.y * mbyte;
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
diff --git src/ggml-webgpu/ggml-webgpu-shader-lib.hpp src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
index 9d88f980..34cbf369 100644
--- src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
+++ src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
@@ -26,20 +26,23 @@
 // Matrix multiplication parameters
 
 // Register tiling parameters
-#define WEBGPU_MUL_MAT_TILE_M    8
-#define WEBGPU_MUL_MAT_TILE_N    8
-#define WEBGPU_MUL_MAT_WG_SIZE_M 8
-#define WEBGPU_MUL_MAT_WG_SIZE_N 8
-#define WEBGPU_MUL_MAT_TILE_K    32
+#define WEBGPU_MUL_MAT_TILE_M           4
+#define WEBGPU_MUL_MAT_TILE_N           4
+#define WEBGPU_MUL_MAT_WG_SIZE_M        8
+#define WEBGPU_MUL_MAT_WG_SIZE_N        8
+#define WEBGPU_MUL_MAT_REG_TILE_K_FLOAT 8
+#define WEBGPU_MUL_MAT_REG_TILE_K_QUANT 32
 
 // Subgroup matrix parameters
 // The number of subgroups in the M dimension
-#define WEBGPU_MUL_MAT_SUBGROUP_M        2
+#define WEBGPU_MUL_MAT_SUBGROUP_M            2
 // The number of subgroups in the N dimension
-#define WEBGPU_MUL_MAT_SUBGROUP_N        2
+#define WEBGPU_MUL_MAT_SUBGROUP_N            4
 // The number of subgroup matrices each subgroup accumulates over
-#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M 4
-#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N 2
+#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_M     4
+#define WEBGPU_MUL_MAT_SUBGROUP_MATRIX_N     2
+#define WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT 32
+#define WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT 32
 
 // Matrix-vector multiplication parameters
 #define WEBGPU_MUL_MAT_VEC_WG_SIZE 256
@@ -56,19 +59,32 @@ template <typename T> inline void ggml_webgpu_hash_combine(size_t & seed, const
     seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
 }
 
+// Calculates base address of a tensor ignoring the fake base pointer
+inline uintptr_t ggml_webgpu_tensor_addr(const ggml_tensor * tensor) {
+    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
+    return (uintptr_t) base_tensor->data + tensor->view_offs;
+}
+
+inline bool ggml_webgpu_tensor_equal(const ggml_tensor * a, const ggml_tensor * b) {
+    return a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) == ggml_webgpu_tensor_addr(b);
+}
+
+inline bool ggml_webgpu_tensor_overlap(const ggml_tensor * a, const ggml_tensor * b) {
+    return a->buffer == b->buffer && ggml_webgpu_tensor_addr(a) < ggml_webgpu_tensor_addr(b) + ggml_nbytes(b) &&
+           ggml_webgpu_tensor_addr(b) < ggml_webgpu_tensor_addr(a) + ggml_nbytes(a);
+}
+
 struct ggml_webgpu_shader_lib_context {
     ggml_tensor * src0;
     ggml_tensor * src1;
     ggml_tensor * src2;
     ggml_tensor * src3;
     ggml_tensor * src4;
+    ggml_tensor * src5;
     ggml_tensor * dst;
 
     uint32_t max_wg_size;
     size_t   wg_mem_limit_bytes       = 0;
-    bool     inplace                  = false;
-    bool     overlap                  = false;
-    bool     src_overlap              = false;
     bool     supports_subgroups       = false;
     bool     supports_subgroup_matrix = false;
     uint32_t sg_mat_m                 = 0;
@@ -85,6 +101,14 @@ struct webgpu_pipeline {
 
 struct ggml_webgpu_generic_shader_decisions {
     uint32_t wg_size = 0;
+    bool     inplace = false;
+};
+
+struct ggml_webgpu_binary_shader_decisions {
+    uint32_t wg_size     = 0;
+    bool     inplace     = false;
+    bool     overlap     = false;
+    bool     src_overlap = false;
 };
 
 struct ggml_webgpu_processed_shader {
@@ -98,6 +122,32 @@ struct ggml_webgpu_ssm_conv_shader_decisions {
     uint32_t tokens_per_wg;
 };
 
+struct ggml_webgpu_ssm_scan_pipeline_key {
+    int  type;
+    int  d_state;
+    bool xbc_overlap;
+
+    bool operator==(const ggml_webgpu_ssm_scan_pipeline_key & other) const {
+        return type == other.type && d_state == other.d_state && xbc_overlap == other.xbc_overlap;
+    }
+};
+
+struct ggml_webgpu_ssm_scan_pipeline_key_hash {
+    size_t operator()(const ggml_webgpu_ssm_scan_pipeline_key & key) const {
+        size_t seed = 0;
+        ggml_webgpu_hash_combine(seed, key.type);
+        ggml_webgpu_hash_combine(seed, key.d_state);
+        ggml_webgpu_hash_combine(seed, key.xbc_overlap);
+        return seed;
+    }
+};
+
+struct ggml_webgpu_ssm_scan_shader_decisions {
+    uint32_t wg_size;
+    uint32_t tokens_per_tile;
+    bool     xbc_overlap = false;
+};
+
 /** Argsort **/
 
 struct ggml_webgpu_argsort_shader_lib_context {
@@ -194,6 +244,35 @@ struct ggml_webgpu_row_norm_pipeline_key_hash {
     }
 };
 
+/** RMS_NORM + MUL **/
+
+struct ggml_webgpu_rms_norm_mul_pipeline_key {
+    bool inplace;      // rn_src == dst
+    bool overlap;      // mul_src == dst
+    bool src_overlap;  // rn_src == mul_src
+
+    bool operator==(const ggml_webgpu_rms_norm_mul_pipeline_key & other) const {
+        return inplace == other.inplace && overlap == other.overlap && src_overlap == other.src_overlap;
+    }
+};
+
+struct ggml_webgpu_rms_norm_mul_pipeline_key_hash {
+    size_t operator()(const ggml_webgpu_rms_norm_mul_pipeline_key & key) const {
+        size_t seed = 0;
+        ggml_webgpu_hash_combine(seed, key.inplace);
+        ggml_webgpu_hash_combine(seed, key.overlap);
+        ggml_webgpu_hash_combine(seed, key.src_overlap);
+        return seed;
+    }
+};
+
+struct ggml_webgpu_rms_norm_mul_shader_decisions {
+    uint32_t wg_size     = 0;
+    bool     inplace     = false;
+    bool     overlap     = false;
+    bool     src_overlap = false;
+};
+
 /** Pad **/
 struct ggml_webgpu_pad_pipeline_key {
     bool circular;
@@ -240,6 +319,46 @@ struct ggml_webgpu_ssm_conv_pipeline_key {
     }
 };
 
+/** CONV 2D */
+struct ggml_webgpu_conv2d_pipeline_key {
+    ggml_type weight_type;
+    ggml_type input_type;
+    ggml_type output_type;
+
+    bool operator==(const ggml_webgpu_conv2d_pipeline_key & other) const {
+        return weight_type == other.weight_type && input_type == other.input_type && output_type == other.output_type;
+    }
+};
+
+struct ggml_webgpu_conv2d_pipeline_key_hash {
+    size_t operator()(const ggml_webgpu_conv2d_pipeline_key & key) const {
+        size_t seed = 0;
+        ggml_webgpu_hash_combine(seed, key.weight_type);
+        ggml_webgpu_hash_combine(seed, key.input_type);
+        ggml_webgpu_hash_combine(seed, key.output_type);
+        return seed;
+    }
+};
+
+/** Im2Col **/
+struct ggml_webgpu_im2col_pipeline_key {
+    ggml_type input_type;
+    ggml_type output_type;
+
+    bool operator==(const ggml_webgpu_im2col_pipeline_key & other) const {
+        return input_type == other.input_type && output_type == other.output_type;
+    }
+};
+
+struct ggml_webgpu_im2col_pipeline_key_hash {
+    size_t operator()(const ggml_webgpu_im2col_pipeline_key & key) const {
+        size_t seed = 0;
+        ggml_webgpu_hash_combine(seed, key.input_type);
+        ggml_webgpu_hash_combine(seed, key.output_type);
+        return seed;
+    }
+};
+
 /** Gated Delta Net **/
 struct ggml_webgpu_gated_delta_net_pipeline_key {
     int type;
@@ -374,19 +493,27 @@ struct ggml_webgpu_unary_pipeline_key_hash {
 
 /** FlashAttention */
 
+enum ggml_webgpu_flash_attn_path : uint32_t {
+    GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX = 0u,
+    GGML_WEBGPU_FLASH_ATTN_PATH_TILE            = 1u,
+    GGML_WEBGPU_FLASH_ATTN_PATH_VEC             = 2u,
+};
+
 struct ggml_webgpu_flash_attn_pipeline_key {
     ggml_type kv_type;
     uint32_t  head_dim_qk;
     uint32_t  head_dim_v;
     bool      kv_direct;
+    bool      kv_overlap;
     bool      has_mask;
     bool      has_sinks;
     bool      uses_logit_softcap;
+    uint32_t  path;
 
     bool operator==(const ggml_webgpu_flash_attn_pipeline_key & other) const {
         return kv_type == other.kv_type && head_dim_qk == other.head_dim_qk && head_dim_v == other.head_dim_v &&
-               kv_direct == other.kv_direct && has_mask == other.has_mask && has_sinks == other.has_sinks &&
-               uses_logit_softcap == other.uses_logit_softcap;
+               kv_direct == other.kv_direct && kv_overlap == other.kv_overlap && has_mask == other.has_mask &&
+               has_sinks == other.has_sinks && uses_logit_softcap == other.uses_logit_softcap && path == other.path;
     }
 };
 
@@ -397,39 +524,71 @@ struct ggml_webgpu_flash_attn_pipeline_key_hash {
         ggml_webgpu_hash_combine(seed, key.head_dim_qk);
         ggml_webgpu_hash_combine(seed, key.head_dim_v);
         ggml_webgpu_hash_combine(seed, key.kv_direct);
+        ggml_webgpu_hash_combine(seed, key.kv_overlap);
         ggml_webgpu_hash_combine(seed, key.has_mask);
         ggml_webgpu_hash_combine(seed, key.has_sinks);
         ggml_webgpu_hash_combine(seed, key.uses_logit_softcap);
+        ggml_webgpu_hash_combine(seed, key.path);
         return seed;
     }
 };
 
 struct ggml_webgpu_flash_attn_decisions {
-    uint32_t q_tile  = 0;
-    uint32_t kv_tile = 0;
-    uint32_t wg_size = 0;
+    uint32_t path       = GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX;
+    uint32_t q_tile     = 0;
+    uint32_t kv_tile    = 0;
+    uint32_t wg_size    = 0;
+    bool     kv_direct  = false;
+    bool     kv_overlap = false;
 };
 
-struct ggml_webgpu_flash_attn_vec_decisions {
-    uint32_t kv_tile = 0;
-    uint32_t wg_size = 0;
-};
+inline constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH = 4u;
+inline constexpr uint32_t GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE       = 4u;
+
+inline uint32_t ggml_webgpu_flash_attn_pick_vec_ne(const ggml_webgpu_flash_attn_pipeline_key & key) {
+    if (key.path != GGML_WEBGPU_FLASH_ATTN_PATH_VEC || key.kv_type != GGML_TYPE_F16 ||
+        key.head_dim_qk != key.head_dim_v) {
+        return 1u;
+    }
+
+    switch (key.head_dim_qk) {
+        case 64:
+        case 192:
+        case 576:
+            return 2u;
+        case 96:
+            return 4u;
+        default:
+            return 1u;
+    }
+}
 
 inline ggml_webgpu_flash_attn_pipeline_key ggml_webgpu_flash_attn_make_pipeline_key(
-    const ggml_webgpu_shader_lib_context & context) {
+    const ggml_webgpu_shader_lib_context & context,
+    uint32_t                               path) {
     const bool has_mask  = context.src3 != nullptr;
     const bool has_sinks = context.src4 != nullptr;
-    const bool kv_direct = (context.src1->type == GGML_TYPE_F16) && (context.src0->ne[0] % context.sg_mat_k == 0) &&
-                           (context.src1->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
+    bool       kv_direct = false;
+    if (path != GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
+        uint32_t kv_direct_align = GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH;
+        if (path == GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX) {
+            kv_direct_align = context.sg_mat_k;
+        }
+        kv_direct = (context.src1->type == GGML_TYPE_F16) &&
+                    (context.src0->ne[0] % std::max(1u, kv_direct_align) == 0) &&
+                    (context.src1->ne[1] % GGML_WEBGPU_KV_SEQ_PAD == 0);
+    }
 
     ggml_webgpu_flash_attn_pipeline_key key = {};
     key.kv_type                             = context.src1->type;
     key.head_dim_qk                         = (uint32_t) context.src0->ne[0];
     key.head_dim_v                          = (uint32_t) context.src2->ne[0];
     key.kv_direct                           = kv_direct;
+    key.kv_overlap                          = ggml_webgpu_tensor_overlap(context.src1, context.src2);
     key.has_mask                            = has_mask;
     key.has_sinks                           = has_sinks;
     key.uses_logit_softcap                  = ggml_get_op_params_f32(context.dst, 2) != 0.0f;
+    key.path                                = path;
     return key;
 }
 
@@ -492,8 +651,16 @@ inline size_t ggml_webgpu_flash_attn_wg_mem_bytes(uint32_t q_tile,
 
 inline uint32_t ggml_webgpu_flash_attn_max_kv_tile(const ggml_webgpu_shader_lib_context &      context,
                                                    const ggml_webgpu_flash_attn_pipeline_key & key) {
-    const size_t limit_bytes  = context.wg_mem_limit_bytes;
-    const size_t q_tile       = context.sg_mat_m;
+    const size_t limit_bytes    = context.wg_mem_limit_bytes;
+    uint32_t     q_tile         = context.sg_mat_m;
+    uint32_t     kv_granularity = context.sg_mat_n;
+    if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
+        q_tile         = GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE;
+        kv_granularity = std::max(1u, context.max_subgroup_size);
+    } else if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
+        q_tile         = 1u;
+        kv_granularity = 8u;
+    }
     const size_t base_q_bytes = (key.head_dim_qk + key.head_dim_v) * q_tile * GGML_WEBGPU_F16_SIZE_BYTES +
                                 2 * q_tile * GGML_WEBGPU_F32_SIZE_BYTES;
     size_t bytes_per_kv = 0;
@@ -506,23 +673,90 @@ inline uint32_t ggml_webgpu_flash_attn_max_kv_tile(const ggml_webgpu_shader_lib_
     bytes_per_kv += q_tile;
     bytes_per_kv *= GGML_WEBGPU_F16_SIZE_BYTES;
     const uint32_t max_kv_tile = (limit_bytes - base_q_bytes) / bytes_per_kv;
-    return (max_kv_tile / context.sg_mat_n) * context.sg_mat_n;
+    return (max_kv_tile / kv_granularity) * kv_granularity;
 }
 
-inline uint32_t ggml_webgpu_flash_attn_vec_get_kv_tile(const ggml_webgpu_shader_lib_context & context) {
-    const ggml_webgpu_flash_attn_pipeline_key key         = ggml_webgpu_flash_attn_make_pipeline_key(context);
-    const uint32_t                            min_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(context, key);
-    uint32_t                                  kv_tile     = std::max(context.sg_mat_n, std::min(32u, min_kv_tile));
-    kv_tile                                               = (kv_tile / context.sg_mat_n) * context.sg_mat_n;
+inline ggml_webgpu_flash_attn_decisions ggml_webgpu_flash_attn_get_decisions(
+    const ggml_webgpu_shader_lib_context & context,
+    size_t                                 storage_offset_alignment) {
+    ggml_webgpu_flash_attn_decisions decisions = {};
+    const size_t                     alignment = std::max<size_t>(1u, storage_offset_alignment);
+    const auto *                     K         = context.src1;
+    const auto *                     V         = context.src2;
+    GGML_ASSERT(K != nullptr);
+    GGML_ASSERT(V != nullptr);
+
+    const auto flash_attn_tensor_offset = [](const ggml_tensor * tensor) -> size_t {
+        constexpr uintptr_t ptr_base_addr = 0x1000u;
+        const ggml_tensor * base          = tensor->view_src != nullptr ? tensor->view_src : tensor;
+        return reinterpret_cast<uintptr_t>(base->data) - ptr_base_addr + tensor->view_offs;
+    };
+
+    const uint32_t k_offset_elems =
+        (uint32_t) ((flash_attn_tensor_offset(K) & (alignment - 1)) / ggml_type_size(K->type));
+    const uint32_t v_offset_elems =
+        (uint32_t) ((flash_attn_tensor_offset(V) & (alignment - 1)) / ggml_type_size(V->type));
+    const bool f16_vec4_aligned = (k_offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u) &&
+                                  (v_offset_elems % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0u);
+    const bool kv_vec_type_supported =
+        K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0;
+    const bool use_vec = context.supports_subgroups && (context.src0->ne[1] < 20) && (context.src0->ne[0] % 32 == 0) &&
+                         (context.src2->ne[0] % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0) &&
+                         kv_vec_type_supported && (K->type != GGML_TYPE_F16 || f16_vec4_aligned) &&
+                         (context.src2->type == K->type);
+    const bool use_tile = context.supports_subgroups && !context.supports_subgroup_matrix && K->type == GGML_TYPE_F16 &&
+                          V->type == GGML_TYPE_F16 && f16_vec4_aligned &&
+                          (context.src0->ne[0] % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0) &&
+                          (context.src2->ne[0] % GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH == 0) && !use_vec;
+
+    decisions.path = use_vec  ? GGML_WEBGPU_FLASH_ATTN_PATH_VEC :
+                     use_tile ? GGML_WEBGPU_FLASH_ATTN_PATH_TILE :
+                                GGML_WEBGPU_FLASH_ATTN_PATH_SUBGROUP_MATRIX;
+
+    const ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context, decisions.path);
+    decisions.kv_direct                           = key.kv_direct;
+
+    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
+        const uint32_t min_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(context, key);
+        decisions.q_tile           = 1u;
+        decisions.kv_tile          = std::max(8u, std::min(32u, min_kv_tile));
+        decisions.kv_tile          = (decisions.kv_tile / 8u) * 8u;
+        decisions.wg_size          = std::max(1u, std::min<uint32_t>(32u, context.max_subgroup_size));
+        if (decisions.kv_direct) {
+            decisions.kv_tile = std::min(decisions.kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
+            while (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
+                decisions.kv_tile -= 8u;
+            }
+        }
+        return decisions;
+    }
+
+    decisions.q_tile =
+        decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ? GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE : context.sg_mat_m;
+    decisions.kv_tile = decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ?
+                            std::min(64u, ggml_webgpu_flash_attn_max_kv_tile(context, key)) :
+                            std::min(ggml_webgpu_flash_attn_max_kv_tile(context, key),
+                                     context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
+    decisions.wg_size = decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ?
+                            GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE :
+                            std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);
+
+    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
+        const uint32_t tile_kv_granularity = std::max(1u, context.max_subgroup_size);
+        decisions.kv_tile =
+            std::max(tile_kv_granularity, (decisions.kv_tile / tile_kv_granularity) * tile_kv_granularity);
+    }
 
-    if (key.kv_direct) {
-        kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
-        while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
-            kv_tile -= context.sg_mat_n;
+    if (decisions.kv_direct) {
+        GGML_ASSERT(decisions.kv_tile <= GGML_WEBGPU_KV_SEQ_PAD);
+        while (GGML_WEBGPU_KV_SEQ_PAD % decisions.kv_tile != 0) {
+            decisions.kv_tile -= decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ?
+                                     std::max(1u, context.max_subgroup_size) :
+                                     context.sg_mat_n;
         }
     }
 
-    return kv_tile;
+    return decisions;
 }
 
 /** Matrix Multiplication **/
@@ -734,16 +968,19 @@ class ggml_webgpu_shader_lib {
     std::unordered_map<int, webgpu_pipeline> cumsum_pipelines;         // key is fixed, no variants yet
     std::unordered_map<ggml_webgpu_row_norm_pipeline_key, webgpu_pipeline, ggml_webgpu_row_norm_pipeline_key_hash>
         row_norm_pipelines;                                            // op/inplace
+
     std::unordered_map<ggml_webgpu_get_rows_pipeline_key, webgpu_pipeline, ggml_webgpu_get_rows_pipeline_key_hash>
-        get_rows_pipelines;                                            // src_type, vectorized
+        get_rows_pipelines;   // src_type, vectorized
     std::unordered_map<ggml_webgpu_unary_pipeline_key, webgpu_pipeline, ggml_webgpu_unary_pipeline_key_hash>
-        unary_pipelines;                                               // type/op/inplace
+        unary_pipelines;      // type/op/inplace
     std::unordered_map<ggml_webgpu_scale_pipeline_key, webgpu_pipeline, ggml_webgpu_scale_pipeline_key_hash>
-        scale_pipelines;                                               // inplace
+        scale_pipelines;      // inplace
     std::unordered_map<ggml_webgpu_solve_tri_pipeline_key, webgpu_pipeline, ggml_webgpu_solve_tri_pipeline_key_hash>
-        solve_tri_pipelines;                                           // type
+        solve_tri_pipelines;  // type
     std::unordered_map<ggml_webgpu_ssm_conv_pipeline_key, webgpu_pipeline, ggml_webgpu_ssm_conv_pipeline_key_hash>
-        ssm_conv_pipelines;                                            // type/vectorized
+        ssm_conv_pipelines;   // type/vectorized
+    std::unordered_map<ggml_webgpu_ssm_scan_pipeline_key, webgpu_pipeline, ggml_webgpu_ssm_scan_pipeline_key_hash>
+        ssm_scan_pipelines;   // type/d_state
     std::unordered_map<ggml_webgpu_gated_delta_net_pipeline_key,
                        webgpu_pipeline,
                        ggml_webgpu_gated_delta_net_pipeline_key_hash>
@@ -758,8 +995,6 @@ class ggml_webgpu_shader_lib {
         repeat_pipelines;           // type
     std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
         flash_attn_pipelines;
-    std::unordered_map<ggml_webgpu_flash_attn_pipeline_key, webgpu_pipeline, ggml_webgpu_flash_attn_pipeline_key_hash>
-        flash_attn_vec_pipelines;
     std::unordered_map<ggml_webgpu_flash_attn_vec_reduce_pipeline_key,
                        webgpu_pipeline,
                        ggml_webgpu_flash_attn_vec_reduce_pipeline_key_hash>
@@ -789,6 +1024,15 @@ class ggml_webgpu_shader_lib {
         rope_pipelines;
     std::unordered_map<ggml_webgpu_soft_max_pipeline_key, webgpu_pipeline, ggml_webgpu_soft_max_pipeline_key_hash>
         soft_max_pipelines;
+    std::unordered_map<ggml_webgpu_conv2d_pipeline_key, webgpu_pipeline, ggml_webgpu_conv2d_pipeline_key_hash>
+        conv2d_pipelines;
+    std::unordered_map<ggml_webgpu_im2col_pipeline_key, webgpu_pipeline, ggml_webgpu_im2col_pipeline_key_hash>
+        im2col_pipelines;
+
+    std::unordered_map<ggml_webgpu_rms_norm_mul_pipeline_key,
+                       webgpu_pipeline,
+                       ggml_webgpu_rms_norm_mul_pipeline_key_hash>
+        rms_norm_mul_pipelines;
 
   public:
     ggml_webgpu_shader_lib(wgpu::Device device) { this->device = device; }
@@ -809,7 +1053,7 @@ class ggml_webgpu_shader_lib {
     webgpu_pipeline get_row_norm_pipeline(const ggml_webgpu_shader_lib_context & context) {
         ggml_webgpu_row_norm_pipeline_key key = {};
         key.op                                = context.dst->op;
-        key.inplace                           = context.inplace;
+        key.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);
 
         auto it = row_norm_pipelines.find(key);
         if (it != row_norm_pipelines.end()) {
@@ -839,8 +1083,12 @@ class ggml_webgpu_shader_lib {
         const uint32_t row_norm_wg_size = 128u;
         uint32_t       wg_size          = std::min(context.max_wg_size, row_norm_wg_size);
         defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
-        auto processed          = preprocessor.preprocess(wgsl_row_norm, defines);
-        row_norm_pipelines[key] = ggml_webgpu_create_pipeline(device, processed, variant);
+        auto processed                  = preprocessor.preprocess(wgsl_row_norm, defines);
+        auto decisions                  = std::make_shared<ggml_webgpu_generic_shader_decisions>();
+        decisions->wg_size              = wg_size;
+        decisions->inplace              = key.inplace;
+        row_norm_pipelines[key]         = ggml_webgpu_create_pipeline(device, processed, variant);
+        row_norm_pipelines[key].context = decisions;
         return row_norm_pipelines[key];
     }
 
@@ -915,7 +1163,7 @@ class ggml_webgpu_shader_lib {
     webgpu_pipeline get_set_pipeline(const ggml_webgpu_shader_lib_context & context) {
         ggml_webgpu_set_pipeline_key key = {};
         key.type                         = context.dst->type;
-        key.inplace                      = context.inplace;
+        key.inplace                      = ggml_webgpu_tensor_equal(context.src0, context.dst);
 
         auto it = set_pipelines.find(key);
         if (it != set_pipelines.end()) {
@@ -948,6 +1196,7 @@ class ggml_webgpu_shader_lib {
         auto processed           = preprocessor.preprocess(wgsl_set, defines);
         auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
         decisions->wg_size       = context.max_wg_size;
+        decisions->inplace       = key.inplace;
         webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
         pipeline.context         = decisions;
         set_pipelines[key]       = pipeline;
@@ -1075,6 +1324,7 @@ class ggml_webgpu_shader_lib {
                     std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);
 
                     switch (key.src_type) {
+                        case GGML_TYPE_Q1_0:
                         case GGML_TYPE_Q4_0:
                         case GGML_TYPE_Q5_0:
                         case GGML_TYPE_Q8_0:
@@ -1111,7 +1361,9 @@ class ggml_webgpu_shader_lib {
 
                     defines.push_back("DST_TYPE=f32");
 
-                    if ((key.src_type >= GGML_TYPE_Q4_0 && key.src_type <= GGML_TYPE_Q8_1) ||
+                    if (key.src_type == GGML_TYPE_Q1_0) {
+                        defines.push_back("BLOCK_SIZE=128u");
+                    } else if ((key.src_type >= GGML_TYPE_Q4_0 && key.src_type <= GGML_TYPE_Q8_1) ||
                         key.src_type == GGML_TYPE_IQ4_NL) {
                         defines.push_back("BLOCK_SIZE=32u");
                     } else if (key.src_type >= GGML_TYPE_Q2_K) {
@@ -1140,7 +1392,7 @@ class ggml_webgpu_shader_lib {
 
     webgpu_pipeline get_scale_pipeline(const ggml_webgpu_shader_lib_context & context) {
         ggml_webgpu_scale_pipeline_key key = {};
-        key.inplace                        = context.inplace;
+        key.inplace                        = ggml_webgpu_tensor_equal(context.src0, context.dst);
 
         auto it = scale_pipelines.find(key);
         if (it != scale_pipelines.end()) {
@@ -1160,6 +1412,7 @@ class ggml_webgpu_shader_lib {
         auto processed           = preprocessor.preprocess(wgsl_scale, defines);
         auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
         decisions->wg_size       = context.max_wg_size;
+        decisions->inplace       = key.inplace;
         webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
         pipeline.context         = decisions;
         scale_pipelines[key]     = pipeline;
@@ -1249,6 +1502,60 @@ class ggml_webgpu_shader_lib {
         return ssm_conv_pipelines[key];
     }
 
+    webgpu_pipeline get_ssm_scan_pipeline(const ggml_webgpu_shader_lib_context & context) {
+        ggml_webgpu_ssm_scan_pipeline_key key = {};
+        key.type                              = context.dst->type;
+        key.d_state                           = (int) context.src0->ne[0];
+        key.xbc_overlap                       = ggml_webgpu_tensor_overlap(context.src1, context.src4) &&
+                          ggml_webgpu_tensor_overlap(context.src1, context.src5);
+
+        auto it = ssm_scan_pipelines.find(key);
+        if (it != ssm_scan_pipelines.end()) {
+            return it->second;
+        }
+
+        std::vector<std::string> defines;
+        std::string              variant = "ssm_scan";
+
+        switch (key.type) {
+            case GGML_TYPE_F32:
+                variant += "_f32";
+                break;
+            default:
+                GGML_ABORT("Unsupported type for ssm_scan shader");
+        }
+
+        const uint32_t wg_size = (uint32_t) key.d_state;
+
+        constexpr uint32_t tokens_per_tile = 4u;
+
+        defines.push_back("WG_SIZE=" + std::to_string(wg_size) + "u");
+        defines.push_back("TOKENS_PER_TILE=" + std::to_string(tokens_per_tile) + "u");
+
+        if (context.supports_subgroups) {
+            defines.push_back("USE_SUBGROUP_REDUCTION");
+            variant += "_sg_reduce";
+        } else {
+            variant += "_wg_reduce";
+        }
+
+        if (key.xbc_overlap) {
+            defines.push_back("XBC_OVERLAP");
+        }
+
+        variant += "_d" + std::to_string(key.d_state);
+
+        auto processed             = preprocessor.preprocess(wgsl_ssm_scan, defines);
+        auto decisions             = std::make_shared<ggml_webgpu_ssm_scan_shader_decisions>();
+        decisions->wg_size         = wg_size;
+        decisions->tokens_per_tile = tokens_per_tile;
+        decisions->xbc_overlap     = key.xbc_overlap;
+        webgpu_pipeline pipeline   = ggml_webgpu_create_pipeline(device, processed, variant);
+        pipeline.context           = decisions;
+        ssm_scan_pipelines[key]    = pipeline;
+        return ssm_scan_pipelines[key];
+    }
+
     webgpu_pipeline get_gated_delta_net_pipeline(const ggml_webgpu_shader_lib_context & context) {
         ggml_webgpu_gated_delta_net_pipeline_key key = {};
         key.type                                     = context.dst->type;
@@ -1356,6 +1663,24 @@ class ggml_webgpu_shader_lib {
                     defines.push_back("MUL_ACC_" + type_upper);
                     defines.push_back("U32_DEQUANT_HELPERS");
                     defines.push_back("SRC0_INNER_TYPE=u32");
+                    switch (context.src0->type) {
+                        case GGML_TYPE_IQ1_S:
+                        case GGML_TYPE_IQ1_M:
+                        case GGML_TYPE_IQ2_S:
+                        case GGML_TYPE_IQ3_S:
+                        case GGML_TYPE_IQ4_NL:
+                        case GGML_TYPE_IQ4_XS:
+                            defines.push_back(type_upper + "_GRID");
+                            break;
+                        case GGML_TYPE_IQ2_XXS:
+                        case GGML_TYPE_IQ2_XS:
+                        case GGML_TYPE_IQ3_XXS:
+                            defines.push_back(type_upper + "_GRID");
+                            defines.push_back(type_upper + "_TABLES");
+                            break;
+                        default:
+                            break;
+                    }
                     break;
                 }
         }
@@ -1380,7 +1705,9 @@ class ggml_webgpu_shader_lib {
         uint32_t wg_size        = WEBGPU_MUL_MAT_VEC_WG_SIZE;
         uint32_t outputs_per_wg = WEBGPU_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG;
 
-        if (key.src0_type >= GGML_TYPE_Q2_K) {
+        if (key.src0_type == GGML_TYPE_Q1_0) {
+            outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
+        } else if (key.src0_type >= GGML_TYPE_Q2_K) {
             outputs_per_wg = WEBGPU_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG;
         } else if (key.src0_type >= GGML_TYPE_Q4_0) {
             outputs_per_wg = WEBGPU_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG;
@@ -1478,13 +1805,22 @@ class ggml_webgpu_shader_lib {
         // VEC/SCALAR controls
         defines.push_back(key.vectorized ? "VEC" : "SCALAR");
 
+        const bool is_quant = ggml_is_quantized(context.src0->type);
+
+        uint32_t tile_k;
+        if (key.use_subgroup_matrix) {
+            tile_k = is_quant ? WEBGPU_MUL_MAT_SUBGROUP_TILE_K_QUANT : WEBGPU_MUL_MAT_SUBGROUP_TILE_K_FLOAT;
+        } else {
+            tile_k = is_quant ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
+        }
+
         // Tiles
         defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
         defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
-        defines.push_back("TILE_K=" + std::to_string(WEBGPU_MUL_MAT_TILE_K) + "u");
 
         // Subgroup matrix specifics
         if (key.use_subgroup_matrix) {
+            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
             defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size) + "u");
             defines.push_back("SUBGROUP_M=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_M) + "u");
             defines.push_back("SUBGROUP_N=" + std::to_string(WEBGPU_MUL_MAT_SUBGROUP_N) + "u");
@@ -1504,12 +1840,13 @@ class ggml_webgpu_shader_lib {
         if (!key.use_subgroup_matrix) {
             defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
             defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
+            defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
         }
 
         auto processed = preprocessor.preprocess(shader_src, defines);
 
         auto decisions                 = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
-        decisions->tile_k              = WEBGPU_MUL_MAT_TILE_K;
+        decisions->tile_k              = tile_k;
         decisions->tile_m              = WEBGPU_MUL_MAT_TILE_M;
         decisions->tile_n              = WEBGPU_MUL_MAT_TILE_N;
         decisions->use_subgroup_matrix = key.use_subgroup_matrix;
@@ -1706,10 +2043,14 @@ class ggml_webgpu_shader_lib {
 
         defines.push_back("SCALAR");
 
+        // mul_mat_id is register-tile only.
+        const uint32_t tile_k =
+            ggml_is_quantized(context.src0->type) ? WEBGPU_MUL_MAT_REG_TILE_K_QUANT : WEBGPU_MUL_MAT_REG_TILE_K_FLOAT;
+
         // Tiles
         defines.push_back("TILE_M=" + std::to_string(WEBGPU_MUL_MAT_TILE_M) + "u");
         defines.push_back("TILE_N=" + std::to_string(WEBGPU_MUL_MAT_TILE_N) + "u");
-        defines.push_back("TILE_K=" + std::to_string(WEBGPU_MUL_MAT_TILE_K) + "u");
+        defines.push_back("TILE_K=" + std::to_string(tile_k) + "u");
 
         defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
         defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
@@ -1720,7 +2061,7 @@ class ggml_webgpu_shader_lib {
         auto processed = preprocessor.preprocess(wgsl_mul_mat_id, defines);
 
         auto decisions       = std::make_shared<ggml_webgpu_mul_mat_shader_decisions>();
-        decisions->tile_k    = WEBGPU_MUL_MAT_TILE_K;
+        decisions->tile_k    = tile_k;
         decisions->tile_m    = WEBGPU_MUL_MAT_TILE_M;
         decisions->tile_n    = WEBGPU_MUL_MAT_TILE_N;
         decisions->wg_size_m = WEBGPU_MUL_MAT_WG_SIZE_M;
@@ -1740,8 +2081,8 @@ class ggml_webgpu_shader_lib {
         key.type                                = context.dst->type;
         key.op                                  = op;
         key.is_unary                            = is_unary;
-        key.inplace                             = context.inplace;
-        key.ttype                               = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);
+        key.inplace = ggml_webgpu_tensor_equal(context.src0, context.dst) || context.dst->op == GGML_OP_FILL;
+        key.ttype   = (ggml_tri_type) ggml_get_op_params_i32(context.dst, 0);
 
         auto it = unary_pipelines.find(key);
         if (it != unary_pipelines.end()) {
@@ -1799,19 +2140,60 @@ class ggml_webgpu_shader_lib {
         auto processed           = preprocessor.preprocess(wgsl_unary, defines);
         auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
         decisions->wg_size       = context.max_wg_size;
+        decisions->inplace       = key.inplace;
         webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
         pipeline.context         = decisions;
         unary_pipelines[key]     = pipeline;
         return unary_pipelines[key];
     }
 
+    webgpu_pipeline get_rms_norm_mul_pipeline(const ggml_webgpu_shader_lib_context & context) {
+        ggml_webgpu_rms_norm_mul_pipeline_key key = {};
+        key.inplace                               = ggml_webgpu_tensor_equal(context.src0, context.dst);
+        key.overlap                               = ggml_webgpu_tensor_equal(context.src1, context.dst);
+        key.src_overlap                           = ggml_webgpu_tensor_overlap(context.src0, context.src1);
+
+        auto it = rms_norm_mul_pipelines.find(key);
+        if (it != rms_norm_mul_pipelines.end()) {
+            return it->second;
+        }
+
+        std::vector<std::string> defines;
+        std::string              op_name = "RMS_NORM_MUL";
+        std::string              variant = op_name;
+
+        if (key.inplace) {
+            defines.push_back("INPLACE");
+            variant += "_inplace";
+        } else if (key.overlap) {
+            defines.push_back("OVERLAP");
+            variant += "_overlap";
+        } else if (key.src_overlap) {
+            defines.push_back("SRC_OVERLAP");
+            variant += "_src_overlap";
+        }
+
+        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
+
+        auto processed                  = preprocessor.preprocess(wgsl_rms_norm_mul, defines);
+        auto pipeline_decisions         = std::make_shared<ggml_webgpu_rms_norm_mul_shader_decisions>();
+        pipeline_decisions->wg_size     = context.max_wg_size;
+        pipeline_decisions->inplace     = key.inplace;
+        pipeline_decisions->overlap     = key.overlap;
+        pipeline_decisions->src_overlap = key.src_overlap;
+        webgpu_pipeline pipeline        = ggml_webgpu_create_pipeline(device, processed, variant);
+        pipeline.context                = pipeline_decisions;
+        rms_norm_mul_pipelines[key]     = pipeline;
+        return rms_norm_mul_pipelines[key];
+    }
+
     webgpu_pipeline get_binary_pipeline(const ggml_webgpu_shader_lib_context & context) {
         ggml_webgpu_binary_pipeline_key key = {};
         key.type                            = context.dst->type;
         key.op                              = context.dst->op;
-        key.inplace                         = context.inplace;
-        key.overlap                         = context.overlap;
-        key.src_overlap                     = context.src_overlap;
+        key.inplace                         = ggml_webgpu_tensor_equal(context.src0, context.dst);
+        key.overlap                         = ggml_webgpu_tensor_equal(context.src1, context.dst);
+        key.src_overlap                     = ggml_webgpu_tensor_overlap(context.src0, context.src1);
 
         auto it = binary_pipelines.find(key);
         if (it != binary_pipelines.end()) {
@@ -1850,11 +2232,15 @@ class ggml_webgpu_shader_lib {
 
         defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
 
-        auto processed           = preprocessor.preprocess(wgsl_binary, defines);
-        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
-        decisions->wg_size       = context.max_wg_size;
+        auto processed                  = preprocessor.preprocess(wgsl_binary, defines);
+        auto pipeline_decisions         = std::make_shared<ggml_webgpu_binary_shader_decisions>();
+        pipeline_decisions->wg_size     = context.max_wg_size;
+        pipeline_decisions->inplace     = key.inplace;
+        pipeline_decisions->overlap     = key.overlap;
+        pipeline_decisions->src_overlap = key.src_overlap;
+
         webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
-        pipeline.context         = decisions;
+        pipeline.context         = pipeline_decisions;
         binary_pipelines[key]    = pipeline;
         return binary_pipelines[key];
     }
@@ -1935,14 +2321,19 @@ class ggml_webgpu_shader_lib {
         return repeat_pipelines[key];
     }
 
-    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_shader_lib_context & context) {
-        const ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context);
-        auto                                      it  = flash_attn_pipelines.find(key);
+    webgpu_pipeline get_flash_attn_pipeline(const ggml_webgpu_shader_lib_context & context,
+                                            size_t                                 storage_offset_alignment) {
+        const ggml_webgpu_flash_attn_decisions decisions =
+            ggml_webgpu_flash_attn_get_decisions(context, storage_offset_alignment);
+        ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context, decisions.path);
+        auto                                it  = flash_attn_pipelines.find(key);
         if (it != flash_attn_pipelines.end()) {
             return it->second;
         }
         std::vector<std::string> defines;
-        std::string              variant = "flash_attn";
+        std::string              variant = decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC  ? "flash_attn_vec" :
+                                           decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE ? "flash_attn_tile" :
+                                                                                                "flash_attn";
 
         switch (key.kv_type) {
             case GGML_TYPE_F32:
@@ -1964,7 +2355,12 @@ class ggml_webgpu_shader_lib {
 
         if (key.has_mask) {
             defines.push_back("MASK");
-            variant += "_mask";
+            if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
+                defines.push_back("BLK");
+                variant += "_mask_blk";
+            } else {
+                variant += "_mask";
+            }
         }
         if (key.has_sinks) {
             defines.push_back("SINKS");
@@ -1978,6 +2374,10 @@ class ggml_webgpu_shader_lib {
             defines.push_back("KV_DIRECT");
             variant += "_kvdirect";
         }
+        if (key.kv_overlap) {
+            defines.push_back("KV_OVERLAP");
+            variant += "_kv_overlap";
+        }
 
         defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(key.head_dim_qk));
         variant += std::string("_hsqk") + std::to_string(key.head_dim_qk);
@@ -1985,129 +2385,38 @@ class ggml_webgpu_shader_lib {
         defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
         variant += std::string("_hsv") + std::to_string(key.head_dim_v);
 
-        defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
-        defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
-        defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
-
-        auto decisions    = std::make_shared<ggml_webgpu_flash_attn_decisions>();
-        decisions->q_tile = context.sg_mat_m;
-
-        const uint32_t min_kv_tile = ggml_webgpu_flash_attn_max_kv_tile(context, key);
-        uint32_t       kv_tile = std::min(min_kv_tile, context.sg_mat_n * GGML_WEBGPU_FLASH_ATTN_PREFERRED_KV_SG_TILES);
-
-        if (key.kv_direct) {
-            kv_tile = std::min(kv_tile, GGML_WEBGPU_KV_SEQ_PAD);
-            while (GGML_WEBGPU_KV_SEQ_PAD % kv_tile != 0) {
-                kv_tile -= context.sg_mat_n;
-            }
+        const char * shader_src = wgsl_flash_attn;
+        if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
+            defines.push_back("KV_GRANULARITY=8");
+            defines.push_back(std::string("VEC_NE=") + std::to_string(ggml_webgpu_flash_attn_pick_vec_ne(key)) + "u");
+            shader_src = wgsl_flash_attn_vec_split;
+        } else if (key.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
+            shader_src = wgsl_flash_attn_tile;
+            defines.push_back("MAX_SUBGROUP_SIZE=" + std::to_string(context.max_subgroup_size));
+            defines.push_back("KV_STAGE_STRIDE=" + std::to_string(std::max(key.head_dim_qk, key.head_dim_v)));
+            variant += "_tile";
+        } else {
+            defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
+            defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
+            defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
         }
 
-        decisions->kv_tile = kv_tile;
-        decisions->wg_size = std::max(context.max_subgroup_size, GGML_WEBGPU_FLASH_ATTN_PREFERRED_WG_SIZE);
-
-        defines.push_back(std::string("Q_TILE=") + std::to_string(decisions->q_tile));
-        defines.push_back(std::string("KV_TILE=") + std::to_string(decisions->kv_tile));
-        defines.push_back(std::string("WG_SIZE=") + std::to_string(decisions->wg_size));
+        auto pipeline_decisions        = std::make_shared<ggml_webgpu_flash_attn_decisions>(decisions);
+        pipeline_decisions->kv_overlap = key.kv_overlap;
+        defines.push_back(std::string("Q_TILE=") + std::to_string(decisions.q_tile));
+        defines.push_back(std::string("KV_TILE=") + std::to_string(decisions.kv_tile));
+        defines.push_back(std::string("WG_SIZE=") + std::to_string(decisions.wg_size));
 
         webgpu_pipeline pipeline =
-            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn, defines), variant);
-        pipeline.context          = decisions;
+            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(shader_src, defines), variant);
+        pipeline.context          = pipeline_decisions;
         flash_attn_pipelines[key] = pipeline;
         return flash_attn_pipelines[key];
     }
 
-    webgpu_pipeline get_flash_attn_vec_pipeline(const ggml_webgpu_shader_lib_context & context) {
-        const ggml_webgpu_flash_attn_pipeline_key key = ggml_webgpu_flash_attn_make_pipeline_key(context);
-        auto                                      it  = flash_attn_vec_pipelines.find(key);
-        if (it != flash_attn_vec_pipelines.end()) {
-            return it->second;
-        }
-
-        std::vector<std::string> defines;
-        std::string              variant = "flash_attn_vec";
-
-        switch (key.kv_type) {
-            case GGML_TYPE_F32:
-                defines.push_back("KV_F32");
-                break;
-            case GGML_TYPE_F16:
-                defines.push_back("KV_F16");
-                break;
-            case GGML_TYPE_Q4_0:
-                defines.push_back("KV_Q4_0");
-                break;
-            case GGML_TYPE_Q8_0:
-                defines.push_back("KV_Q8_0");
-                break;
-            default:
-                GGML_ABORT("Unsupported KV type for flash attention shader");
-        }
-        variant += std::string("_") + ggml_type_name(key.kv_type);
-
-        if (key.has_mask) {
-            defines.push_back("MASK");
-            defines.push_back("BLK");
-            variant += "_mask_blk";
-        }
-        if (key.has_sinks) {
-            defines.push_back("SINKS");
-            variant += "_sinks";
-        }
-        if (key.uses_logit_softcap) {
-            defines.push_back("LOGIT_SOFTCAP");
-            variant += "_lgsc";
-        }
-        if (key.kv_direct) {
-            defines.push_back("KV_DIRECT");
-            variant += "_kvdirect";
-        }
-
-        defines.push_back(std::string("HEAD_DIM_QK=") + std::to_string(key.head_dim_qk));
-        variant += std::string("_hsqk") + std::to_string(key.head_dim_qk);
-
-        defines.push_back(std::string("HEAD_DIM_V=") + std::to_string(key.head_dim_v));
-        variant += std::string("_hsv") + std::to_string(key.head_dim_v);
-
-        defines.push_back(std::string("SG_MAT_M=") + std::to_string(context.sg_mat_m));
-        defines.push_back(std::string("SG_MAT_N=") + std::to_string(context.sg_mat_n));
-        defines.push_back(std::string("SG_MAT_K=") + std::to_string(context.sg_mat_k));
-        defines.push_back("Q_TILE=1");
-
-        auto decisions     = std::make_shared<ggml_webgpu_flash_attn_vec_decisions>();
-        decisions->kv_tile = ggml_webgpu_flash_attn_vec_get_kv_tile(context);
-        decisions->wg_size = std::max(1u, std::min<uint32_t>(32u, context.max_subgroup_size));
-        uint32_t vec_ne    = 1u;
-
-        // Keep conservative defaults unless this is the f16 vec-split shape family.
-        if (key.kv_type == GGML_TYPE_F16 && key.head_dim_qk == key.head_dim_v) {
-            switch (key.head_dim_qk) {
-                case 64:
-                case 192:
-                case 576:
-                    vec_ne = 2u;
-                    break;
-                case 96:
-                    vec_ne = 4u;
-                    break;
-                default:
-                    break;
-            }
-        }
-
-        defines.push_back(std::string("KV_TILE=") + std::to_string(decisions->kv_tile));
-        defines.push_back(std::string("WG_SIZE=") + std::to_string(decisions->wg_size));
-        defines.push_back(std::string("VEC_NE=") + std::to_string(vec_ne) + "u");
-
-        webgpu_pipeline pipeline =
-            ggml_webgpu_create_pipeline(device, preprocessor.preprocess(wgsl_flash_attn_vec_split, defines), variant);
-        pipeline.context              = decisions;
-        flash_attn_vec_pipelines[key] = pipeline;
-        return flash_attn_vec_pipelines[key];
-    }
-
-    webgpu_pipeline get_flash_attn_blk_pipeline(const ggml_webgpu_shader_lib_context & context) {
+    webgpu_pipeline get_flash_attn_blk_pipeline(const ggml_webgpu_shader_lib_context & context, uint32_t kv_tile) {
         ggml_webgpu_flash_attn_blk_pipeline_key key = {};
-        key.kv_tile                                 = ggml_webgpu_flash_attn_vec_get_kv_tile(context);
+        key.kv_tile                                 = kv_tile;
         auto it                                     = flash_attn_blk_pipelines.find(key);
         if (it != flash_attn_blk_pipelines.end()) {
             return it->second;
@@ -2285,7 +2594,7 @@ class ggml_webgpu_shader_lib {
     webgpu_pipeline get_rope_pipeline(const ggml_webgpu_shader_lib_context & context) {
         ggml_webgpu_rope_pipeline_key key = {};
         key.type                          = context.dst->type;
-        key.inplace                       = context.inplace;
+        key.inplace                       = ggml_webgpu_tensor_equal(context.src0, context.dst);
         key.has_ff                        = (context.src2 != nullptr);
 
         auto it = rope_pipelines.find(key);
@@ -2324,6 +2633,7 @@ class ggml_webgpu_shader_lib {
         auto processed           = preprocessor.preprocess(wgsl_rope, defines);
         auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
         decisions->wg_size       = context.max_wg_size;
+        decisions->inplace       = key.inplace;
         webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
         pipeline.context         = decisions;
         rope_pipelines[key]      = pipeline;
@@ -2335,7 +2645,7 @@ class ggml_webgpu_shader_lib {
         key.mask_type                         = context.src1 ? context.src1->type : GGML_TYPE_F32;
         key.has_mask                          = (context.src1 != nullptr);
         key.has_sink                          = (context.src2 != nullptr);
-        key.inplace                           = context.inplace;
+        key.inplace                           = ggml_webgpu_tensor_equal(context.src0, context.dst);
 
         auto it = soft_max_pipelines.find(key);
         if (it != soft_max_pipelines.end()) {
@@ -2376,12 +2686,91 @@ class ggml_webgpu_shader_lib {
         auto processed           = preprocessor.preprocess(wgsl_soft_max, defines);
         auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
         decisions->wg_size       = context.max_wg_size;
+        decisions->inplace       = key.inplace;
         webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
         pipeline.context         = decisions;
         soft_max_pipelines[key]  = pipeline;
         return soft_max_pipelines[key];
     }
 
+    webgpu_pipeline get_conv2d_pipeline(const ggml_webgpu_shader_lib_context & context) {
+        ggml_webgpu_conv2d_pipeline_key key = {};
+        key.weight_type                     = context.src0->type;
+        key.input_type                      = context.src1->type;
+        key.output_type                     = context.dst->type;
+
+        auto it = conv2d_pipelines.find(key);
+        if (it != conv2d_pipelines.end()) {
+            return it->second;
+        }
+
+        std::vector<std::string> defines;
+        std::string              variant = "conv_2d";
+
+        auto push_type_defines = [&](const char * prefix, ggml_type type) {
+            std::string s_prefix = prefix;
+            if (type == GGML_TYPE_F32) {
+                defines.push_back(s_prefix + "_F32");
+            } else if (type == GGML_TYPE_F16) {
+                defines.push_back(s_prefix + "_F16");
+            } else {
+                GGML_ABORT("Unsupported type for CONV_2D shader");
+            }
+        };
+
+        push_type_defines("WEIGHT", key.weight_type);
+        push_type_defines("INPUT", key.input_type);
+        push_type_defines("OUTPUT", key.output_type);
+
+        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
+
+        auto processed           = preprocessor.preprocess(wgsl_conv2d, defines);
+        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
+        decisions->wg_size       = context.max_wg_size;
+        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
+        pipeline.context         = decisions;
+        conv2d_pipelines[key]    = pipeline;
+        return conv2d_pipelines[key];
+    }
+
+    webgpu_pipeline get_im2col_pipeline(const ggml_webgpu_shader_lib_context & context) {
+        ggml_webgpu_im2col_pipeline_key key = {};
+        key.input_type                      = context.src1->type;
+        key.output_type                     = context.dst->type;
+
+        auto it = im2col_pipelines.find(key);
+        if (it != im2col_pipelines.end()) {
+            return it->second;
+        }
+
+        std::vector<std::string> defines;
+        std::string              variant = "im2col";
+
+        auto push_type_defines = [&](const char * prefix, ggml_type type) {
+            std::string s_prefix = prefix;
+            if (type == GGML_TYPE_F32) {
+                defines.push_back(s_prefix + "_F32");
+            } else if (type == GGML_TYPE_F16) {
+                defines.push_back(s_prefix + "_F16");
+            } else {
+                GGML_ABORT("Unsupported type for IM2COL shader");
+            }
+        };
+
+        push_type_defines("INPUT", key.input_type);
+        push_type_defines("OUTPUT", key.output_type);
+
+        defines.push_back(std::string("WG_SIZE=") + std::to_string(context.max_wg_size));
+
+        auto processed           = preprocessor.preprocess(wgsl_im2col, defines);
+        auto decisions           = std::make_shared<ggml_webgpu_generic_shader_decisions>();
+        decisions->wg_size       = context.max_wg_size;
+        webgpu_pipeline pipeline = ggml_webgpu_create_pipeline(device, processed, variant);
+        pipeline.context         = decisions;
+        im2col_pipelines[key]    = pipeline;
+        return im2col_pipelines[key];
+    }
+
   private:
     static webgpu_pipeline ggml_webgpu_create_pipeline(wgpu::Device & device,
                                                        std::string    shader_code,
diff --git src/ggml-webgpu/ggml-webgpu.cpp src/ggml-webgpu/ggml-webgpu.cpp
index aa20a745..762d9f8d 100644
--- src/ggml-webgpu/ggml-webgpu.cpp
+++ src/ggml-webgpu/ggml-webgpu.cpp
@@ -8,6 +8,7 @@
 #include "ggml-backend-impl.h"
 #include "ggml-impl.h"
 #include "ggml-webgpu-shader-lib.hpp"
+#include "ggml.h"
 
 #ifdef __EMSCRIPTEN__
 #    include <emscripten/emscripten.h>
@@ -107,12 +108,9 @@ static inline uint32_t ggml_webgpu_u32_from_f32(float value) {
 // their locations.
 static void * const webgpu_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT
 
-// Always returns the base offset of a tensor, regardless of views.
-static uint64_t webgpu_tensor_offset(const ggml_tensor * tensor) {
-    if (tensor->view_src) {
-        return (uint8_t *) tensor->view_src->data - (uint8_t *) webgpu_ptr_base;
-    }
-    return (uint8_t *) tensor->data - (uint8_t *) webgpu_ptr_base;
+static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
+    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
+    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) webgpu_ptr_base) + tensor->view_offs;
 }
 
 /* Struct definitions */
@@ -211,6 +209,7 @@ struct webgpu_global_context_struct {
     wgpu::Buffer    memset_params_buf;
     webgpu_pipeline memset_pipeline;
 
+    // TODO: We should rework the CPU profiling time handling to make it more useful. ref: https://github.com/ggml-org/llama.cpp/pull/22050
 #ifdef GGML_WEBGPU_CPU_PROFILE
     // Profiling: labeled CPU time in ms (total)
     std::unordered_map<std::string, double> cpu_time_ms;
@@ -218,11 +217,6 @@ struct webgpu_global_context_struct {
     std::unordered_map<std::string, double> cpu_detail_ms;
 #endif
 
-#ifdef GGML_WEBGPU_GPU_PROFILE
-    // Profiling: per-shader GPU time in ms
-    std::unordered_map<std::string, double> shader_gpu_time_ms;
-#endif
-
 #ifdef GGML_WEBGPU_DEBUG
     wgpu::Buffer debug_host_buf;
     wgpu::Buffer debug_dev_buf;
@@ -268,10 +262,12 @@ struct webgpu_context_struct {
     size_t memset_bytes_per_thread;
 
 #ifdef GGML_WEBGPU_GPU_PROFILE
-    wgpu::Buffer   profile_timestamp_dev_buf;
-    wgpu::Buffer   profile_timestamp_host_buf;
-    wgpu::QuerySet profile_timestamp_query_set;
-    uint32_t       profile_timestamp_query_count = 0;
+    // Profiling: per-shader GPU time in ms
+    std::unordered_map<std::string, double> shader_gpu_time_ms;
+    wgpu::Buffer                            profile_timestamp_dev_buf;
+    wgpu::Buffer                            profile_timestamp_host_buf;
+    wgpu::QuerySet                          profile_timestamp_query_set;
+    uint32_t                                profile_timestamp_query_count = 0;
 #endif
 
     ~webgpu_context_struct() {
@@ -376,10 +372,6 @@ static void ggml_webgpu_create_buffer(wgpu::Device &    device,
     buffer = device.CreateBuffer(&buffer_desc);
 }
 
-static size_t ggml_webgpu_tensor_offset(const ggml_tensor * tensor) {
-    return webgpu_tensor_offset(tensor) + tensor->view_offs;
-}
-
 static wgpu::Buffer ggml_webgpu_tensor_buf(const ggml_tensor * tensor) {
     ggml_backend_webgpu_buffer_context * ctx = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
     return ctx->buffer;
@@ -390,23 +382,6 @@ static size_t ggml_webgpu_tensor_misalignment(webgpu_context & ctx, const ggml_t
     return offset & (ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
 }
 
-static bool ggml_webgpu_flash_attn_use_vec(webgpu_global_context & global_ctx,
-                                           const ggml_tensor *     Q,
-                                           const ggml_tensor *     K,
-                                           const ggml_tensor *     V) {
-    const size_t   alignment = global_ctx->capabilities.limits.minStorageBufferOffsetAlignment;
-    const uint32_t k_offset_elems =
-        (uint32_t) ((ggml_webgpu_tensor_offset(K) & (alignment - 1)) / ggml_type_size(K->type));
-    const uint32_t v_offset_elems =
-        (uint32_t) ((ggml_webgpu_tensor_offset(V) & (alignment - 1)) / ggml_type_size(V->type));
-    const bool f16_vec4_aligned = (k_offset_elems % 4u == 0u) && (v_offset_elems % 4u == 0u);
-    const bool kv_vec_type_supported =
-        K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0;
-
-    return (Q->ne[1] < 20) && (Q->ne[0] % 32 == 0) && (V->ne[0] % 4 == 0) && kv_vec_type_supported &&
-           (K->type != GGML_TYPE_F16 || f16_vec4_aligned) && (V->type == K->type);
-}
-
 static size_t ggml_webgpu_tensor_align_offset(webgpu_context & ctx, const ggml_tensor * t) {
     size_t offset = ggml_webgpu_tensor_offset(t);
     return offset & ~(ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment - 1);
@@ -416,34 +391,31 @@ static size_t ggml_webgpu_tensor_binding_size(webgpu_context & ctx, ggml_tensor
     return ROUNDUP_POW2(ggml_nbytes(t) + ggml_webgpu_tensor_misalignment(ctx, t), WEBGPU_STORAGE_BUF_BINDING_MULT);
 }
 
-// Used to determine if two tensors are the same for in-place operations
-static bool ggml_webgpu_tensor_equal(ggml_tensor * a, ggml_tensor * b) {
-    return (ggml_webgpu_tensor_buf(a).Get() == ggml_webgpu_tensor_buf(b).Get()) &&
-           (ggml_webgpu_tensor_offset(a) == ggml_webgpu_tensor_offset(b));
-}
+struct ggml_webgpu_merged_binding_range {
+    size_t offset;
+    size_t size;
+};
 
-// Used to determine if two tensors share the same buffer and their byte ranges overlap,
-static bool ggml_webgpu_tensor_overlap(ggml_tensor * a, ggml_tensor * b) {
-    return (ggml_webgpu_tensor_buf(a).Get() == ggml_webgpu_tensor_buf(b).Get()) &&
-           ggml_webgpu_tensor_offset(a) < (ggml_webgpu_tensor_offset(b) + ggml_nbytes(b)) &&
-           ggml_webgpu_tensor_offset(b) < (ggml_webgpu_tensor_offset(a) + ggml_nbytes(a));
-}
+static ggml_webgpu_merged_binding_range ggml_webgpu_tensor_merged_binding_range(
+    webgpu_context &                     ctx,
+    std::initializer_list<ggml_tensor *> tensors) {
+    size_t merged_offset = SIZE_MAX;
+    size_t merged_end    = 0;
 
-struct binary_overlap_flags {
-    bool inplace;  // src0 == dst
-    bool overlap;  // src1 == dst
-    bool src_overlap;
-};
+    for (ggml_tensor * tensor : tensors) {
+        const size_t bind_offset = ggml_webgpu_tensor_align_offset(ctx, tensor);
+        const size_t bind_end    = bind_offset + ggml_webgpu_tensor_binding_size(ctx, tensor);
 
-static binary_overlap_flags ggml_webgpu_detect_binary_overlap(ggml_tensor * src0,
-                                                              ggml_tensor * src1,
-                                                              ggml_tensor * dst) {
-    binary_overlap_flags flags = {};
-    flags.inplace              = ggml_webgpu_tensor_equal(src0, dst);
-    flags.overlap              = ggml_webgpu_tensor_overlap(src1, dst);
-    flags.src_overlap          = ggml_webgpu_tensor_overlap(src0, src1);
+        merged_offset = std::min(merged_offset, bind_offset);
+        merged_end    = std::max(merged_end, bind_end);
+    }
+
+    return { merged_offset, merged_end - merged_offset };
+}
 
-    return flags;
+static uint32_t ggml_webgpu_tensor_merged_element_offset(const ggml_tensor *                      tensor,
+                                                         const ggml_webgpu_merged_binding_range & merged_range) {
+    return (uint32_t) ((ggml_webgpu_tensor_offset(tensor) - merged_range.offset) / ggml_type_size(tensor->type));
 }
 
 static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(uint32_t     binding,
@@ -713,12 +685,12 @@ static void ggml_backend_webgpu_free(ggml_backend_t backend) {
 #ifdef GGML_WEBGPU_GPU_PROFILE
     std::cout << "\n[ggml_webgpu gpu profiling summary]\n";
     double total_gpu = 0.0;
-    for (const auto & kv : ctx->webgpu_ctx->global_ctx->shader_gpu_time_ms) {
+    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
         total_gpu += kv.second;
     }
     std::cout << "ggml_webgpu: total gpu time (all shaders): " << total_gpu << " ms\n";
     std::cout << "\nggml_webgpu: gpu breakdown:\n";
-    for (const auto & kv : ctx->webgpu_ctx->global_ctx->shader_gpu_time_ms) {
+    for (const auto & kv : ctx->webgpu_ctx->shader_gpu_time_ms) {
         double pct = (total_gpu > 0.0) ? (kv.second / total_gpu * 100.0) : 0.0;
         std::cout << "ggml_webgpu:  " << kv.first << ": " << kv.second << " ms (" << std::fixed << std::setprecision(2)
                   << pct << "%)\n";
@@ -771,18 +743,16 @@ static webgpu_encoded_op ggml_webgpu_set(webgpu_context & ctx,
                                          ggml_tensor *    src0,
                                          ggml_tensor *    src1,
                                          ggml_tensor *    dst) {
-    const bool inplace = ggml_webgpu_tensor_equal(src0, dst);
-
     ggml_webgpu_shader_lib_context shader_lib_ctx = {};
     shader_lib_ctx.src0                           = src0;
     shader_lib_ctx.src1                           = src1;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = inplace;
 
     webgpu_pipeline pipeline = ctx->shader_lib->get_set_pipeline(shader_lib_ctx);
 
-    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+    const bool inplace   = decisions->inplace;
 
     const uint32_t ne            = inplace ? (uint32_t) ggml_nelements(src1) : (uint32_t) ggml_nelements(dst);
     const uint32_t dst_type_size = (uint32_t) ggml_type_size(dst->type);
@@ -923,6 +893,170 @@ static webgpu_encoded_op ggml_webgpu_solve_tri(webgpu_context & ctx,
     return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
 }
 
+static webgpu_encoded_op ggml_webgpu_conv_2d(webgpu_context & ctx,
+                                             ggml_tensor *    src0,
+                                             ggml_tensor *    src1,
+                                             ggml_tensor *    dst) {
+    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
+    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
+    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
+    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
+    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
+    const int32_t d1 = ggml_get_op_params_i32(dst, 5);
+
+    std::vector<uint32_t> params = {
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
+
+        (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
+        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
+        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
+        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
+
+        (uint32_t) (src1->nb[0] / ggml_type_size(src1->type)),
+        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
+        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
+        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
+
+        (uint32_t) (dst->nb[0] / ggml_type_size(dst->type)),
+        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
+        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
+        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
+
+        (uint32_t) src0->ne[0],
+        (uint32_t) src0->ne[1],
+        (uint32_t) src0->ne[2],
+
+        (uint32_t) src1->ne[0],
+        (uint32_t) src1->ne[1],
+
+        (uint32_t) dst->ne[0],
+        (uint32_t) dst->ne[1],
+        (uint32_t) dst->ne[2],
+        (uint32_t) dst->ne[3],
+
+        (uint32_t) s0,
+        (uint32_t) s1,
+        (uint32_t) p0,
+        (uint32_t) p1,
+        (uint32_t) d0,
+        (uint32_t) d1,
+    };
+
+    std::vector<wgpu::BindGroupEntry> entries = {
+        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
+        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1),
+        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst),
+    };
+
+    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
+    shader_lib_ctx.src0                           = src0;
+    shader_lib_ctx.src1                           = src1;
+    shader_lib_ctx.dst                            = dst;
+    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+
+    webgpu_pipeline pipeline = ctx->shader_lib->get_conv2d_pipeline(shader_lib_ctx);
+
+    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+
+    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
+    uint32_t wg_x     = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
+    uint32_t wg_y     = CEIL_DIV(total_wg, wg_x);
+
+    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
+}
+
+static webgpu_encoded_op ggml_webgpu_im2col(webgpu_context & ctx,
+                                            ggml_tensor *    src0,
+                                            ggml_tensor *    src1,
+                                            ggml_tensor *    dst) {
+    const int32_t s0    = ggml_get_op_params_i32(dst, 0);
+    const int32_t s1    = ggml_get_op_params_i32(dst, 1);
+    const int32_t p0    = ggml_get_op_params_i32(dst, 2);
+    const int32_t p1    = ggml_get_op_params_i32(dst, 3);
+    const int32_t d0    = ggml_get_op_params_i32(dst, 4);
+    const int32_t d1    = ggml_get_op_params_i32(dst, 5);
+    const bool    is_2D = ggml_get_op_params_i32(dst, 6) == 1;
+
+    const uint32_t KW = src0->ne[0];
+    const uint32_t KH = is_2D ? src0->ne[1] : 1;
+    const uint32_t IC = is_2D ? src0->ne[2] : src0->ne[1];
+
+    const uint32_t IW = src1->ne[0];
+    const uint32_t IH = is_2D ? src1->ne[1] : 1;
+    const uint32_t N  = is_2D ? src1->ne[3] : src1->ne[2];
+
+    const uint32_t OW = dst->ne[1];
+    const uint32_t OH = is_2D ? dst->ne[2] : 1;
+
+    const uint32_t si0 = (uint32_t) (src1->nb[0] / ggml_type_size(src1->type));
+    const uint32_t si1 = is_2D ? (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)) : 0;
+    const uint32_t si2 = is_2D ? (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)) :
+                                 (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
+    const uint32_t si3 = is_2D ? (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)) :
+                                 (uint32_t) (src1->nb[2] / ggml_type_size(src1->type));
+
+    const uint32_t so0 = (uint32_t) (dst->nb[0] / ggml_type_size(dst->type));
+    const uint32_t so1 = (uint32_t) (dst->nb[1] / ggml_type_size(dst->type));
+    const uint32_t so2 = is_2D ? (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
+    const uint32_t so3 = is_2D ? (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)) :
+                                 (uint32_t) (dst->nb[2] / ggml_type_size(dst->type));
+
+    std::vector<uint32_t> params = {
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
+
+        si0,
+        si1,
+        si2,
+        si3,
+        so0,
+        so1,
+        so2,
+        so3,
+
+        KW,
+        KH,
+        IC,
+
+        IW,
+        IH,
+        N,
+
+        OW,
+        OH,
+
+        (uint32_t) s0,
+        (uint32_t) s1,
+        (uint32_t) p0,
+        (uint32_t) p1,
+        (uint32_t) d0,
+        (uint32_t) d1,
+    };
+
+    std::vector<wgpu::BindGroupEntry> entries = {
+        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src1),
+        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst),
+    };
+
+    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
+    shader_lib_ctx.src0                           = src0;
+    shader_lib_ctx.src1                           = src1;
+    shader_lib_ctx.dst                            = dst;
+    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+
+    webgpu_pipeline pipeline = ctx->shader_lib->get_im2col_pipeline(shader_lib_ctx);
+
+    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+
+    uint32_t total_wg = CEIL_DIV((uint32_t) ggml_nelements(dst), decisions->wg_size);
+    uint32_t wg_x     = std::min(ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension, total_wg);
+    uint32_t wg_y     = CEIL_DIV(total_wg, wg_x);
+
+    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
+}
+
 static webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
                                               ggml_tensor *    src0,
                                               ggml_tensor *    src1,
@@ -969,6 +1103,113 @@ static webgpu_encoded_op ggml_webgpu_ssm_conv(webgpu_context & ctx,
     return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
 }
 
+static webgpu_encoded_op ggml_webgpu_ssm_scan(webgpu_context & ctx,
+                                              ggml_tensor *    src0,
+                                              ggml_tensor *    src1,
+                                              ggml_tensor *    src2,
+                                              ggml_tensor *    src3,
+                                              ggml_tensor *    src4,
+                                              ggml_tensor *    src5,
+                                              ggml_tensor *    src6,
+                                              ggml_tensor *    dst) {
+    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
+    shader_lib_ctx.src0                           = src0;
+    shader_lib_ctx.src1                           = src1;
+    shader_lib_ctx.src4                           = src4;
+    shader_lib_ctx.src5                           = src5;
+    shader_lib_ctx.dst                            = dst;
+    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+    shader_lib_ctx.supports_subgroups = ctx->global_ctx->capabilities.supports_subgroups;
+
+    webgpu_pipeline pipeline    = ctx->shader_lib->get_ssm_scan_pipeline(shader_lib_ctx);
+    auto *          decisions   = static_cast<ggml_webgpu_ssm_scan_shader_decisions *>(pipeline.context.get());
+    const bool      xbc_overlap = decisions->xbc_overlap;
+
+    uint32_t offset_x        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
+    uint32_t offset_B        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src4) / ggml_type_size(src4->type));
+    uint32_t offset_C        = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src5) / ggml_type_size(src5->type));
+    size_t   xbc_bind_offset = 0;
+    size_t   xbc_bind_size   = 0;
+    if (xbc_overlap) {
+        const ggml_webgpu_merged_binding_range merged_range =
+            ggml_webgpu_tensor_merged_binding_range(ctx, { src1, src4, src5 });
+        xbc_bind_offset = merged_range.offset;
+        xbc_bind_size   = merged_range.size;
+        offset_x        = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
+        offset_B        = ggml_webgpu_tensor_merged_element_offset(src4, merged_range);
+        offset_C        = ggml_webgpu_tensor_merged_element_offset(src5, merged_range);
+    }
+
+    std::vector<uint32_t> params = {
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
+        offset_x,
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src2) / ggml_type_size(src2->type)),
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src3) / ggml_type_size(src3->type)),
+        offset_B,
+        offset_C,
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src6) / ggml_type_size(src6->type)),
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
+
+        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
+        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
+        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
+
+        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
+        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
+        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
+
+        (uint32_t) (src2->nb[1] / ggml_type_size(src2->type)),
+        (uint32_t) (src2->nb[2] / ggml_type_size(src2->type)),
+
+        (uint32_t) src3->ne[0],
+        (uint32_t) (src3->nb[1] / ggml_type_size(src3->type)),
+
+        (uint32_t) (src4->nb[1] / ggml_type_size(src4->type)),
+        (uint32_t) (src4->nb[2] / ggml_type_size(src4->type)),
+        (uint32_t) (src4->nb[3] / ggml_type_size(src4->type)),
+
+        (uint32_t) (src5->nb[1] / ggml_type_size(src5->type)),
+        (uint32_t) (src5->nb[2] / ggml_type_size(src5->type)),
+        (uint32_t) (src5->nb[3] / ggml_type_size(src5->type)),
+
+        (uint32_t) src0->ne[0],
+        (uint32_t) src0->ne[1],
+        (uint32_t) src0->ne[2],
+        (uint32_t) src4->ne[1],
+        (uint32_t) src1->ne[2],
+        (uint32_t) src1->ne[3],
+        (uint32_t) ggml_nelements(src1),
+    };
+
+    std::vector<wgpu::BindGroupEntry> entries = {
+        ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0),
+    };
+    if (xbc_overlap) {
+        entries.push_back(
+            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1), xbc_bind_offset, xbc_bind_size));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src6));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, dst));
+    } else {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, src2));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 3, src3));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 4, src4));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 5, src5));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 6, src6));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 7, dst));
+    }
+
+    const uint32_t total_wg       = (uint32_t) (src0->ne[1] * src0->ne[2] * src1->ne[3]);
+    const uint32_t max_wg_per_dim = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupsPerDimension;
+    uint32_t       wg_x;
+    uint32_t       wg_y;
+    compute_2d_workgroups(total_wg, max_wg_per_dim, wg_x, wg_y);
+
+    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x, wg_y);
+}
+
 static webgpu_encoded_op ggml_webgpu_gated_delta_net(webgpu_context & ctx,
                                                      ggml_tensor *    src0,
                                                      ggml_tensor *    src1,
@@ -1169,8 +1410,20 @@ static webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
                 case GGML_TYPE_Q5_K:
                 case GGML_TYPE_Q3_K:
                 case GGML_TYPE_Q2_K:
+                case GGML_TYPE_Q1_0:
                     use_fast = true;
                     break;
+                case GGML_TYPE_IQ1_S:
+                case GGML_TYPE_IQ1_M:
+                case GGML_TYPE_IQ2_XXS:
+                case GGML_TYPE_IQ2_XS:
+                case GGML_TYPE_IQ2_S:
+                case GGML_TYPE_IQ3_XXS:
+                case GGML_TYPE_IQ3_S:
+                case GGML_TYPE_IQ4_NL:
+                case GGML_TYPE_IQ4_XS:
+                    use_fast = is_vec;
+                    break;
                 default:
                     break;
             }
@@ -1404,7 +1657,6 @@ static webgpu_encoded_op ggml_webgpu_mul_mat_id(webgpu_context & ctx,
     return ggml_backend_webgpu_build_multi(ctx, dispatches);
 }
 
-#ifndef __EMSCRIPTEN__
 static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
                                                 ggml_tensor *    Q,
                                                 ggml_tensor *    K,
@@ -1422,13 +1674,44 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
     float m0          = powf(2.0f, -(max_bias) / n_head_log2);
     float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
 
-    const int has_mask  = (mask != nullptr);
-    const int has_sinks = (sinks != nullptr);
+    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
+    shader_lib_ctx.src0                           = Q;
+    shader_lib_ctx.src1                           = K;
+    shader_lib_ctx.src2                           = V;
+    shader_lib_ctx.src3                           = mask;
+    shader_lib_ctx.src4                           = sinks;
+    shader_lib_ctx.dst                            = dst;
+    shader_lib_ctx.supports_subgroups             = ctx->global_ctx->capabilities.supports_subgroups;
+    shader_lib_ctx.supports_subgroup_matrix       = ctx->global_ctx->capabilities.supports_subgroup_matrix;
+    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
+    shader_lib_ctx.sg_mat_m           = ctx->global_ctx->capabilities.sg_mat_m;
+    shader_lib_ctx.sg_mat_n           = ctx->global_ctx->capabilities.sg_mat_n;
+    shader_lib_ctx.sg_mat_k           = ctx->global_ctx->capabilities.sg_mat_k;
+    shader_lib_ctx.max_subgroup_size  = ctx->global_ctx->capabilities.max_subgroup_size;
+    webgpu_pipeline pipeline          = ctx->shader_lib->get_flash_attn_pipeline(
+        shader_lib_ctx, ctx->global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
+    auto *     decisions  = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
+    const int  has_mask   = (mask != nullptr);
+    const int  has_sinks  = (sinks != nullptr);
+    const bool kv_overlap = decisions->kv_overlap;
+
+    uint32_t offset_k       = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type));
+    uint32_t offset_v       = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type));
+    size_t   kv_bind_offset = 0;
+    size_t   kv_bind_size   = 0;
+    if (kv_overlap) {
+        const ggml_webgpu_merged_binding_range merged_range = ggml_webgpu_tensor_merged_binding_range(ctx, { K, V });
+        kv_bind_offset                                      = merged_range.offset;
+        kv_bind_size                                        = merged_range.size;
+        offset_k                                            = ggml_webgpu_tensor_merged_element_offset(K, merged_range);
+        offset_v                                            = ggml_webgpu_tensor_merged_element_offset(V, merged_range);
+    }
 
     std::vector<uint32_t> params = {
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, Q) / ggml_type_size(Q->type)),
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, K) / ggml_type_size(K->type)),
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, V) / ggml_type_size(V->type)),
+        offset_k,
+        offset_v,
         has_mask ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)) : 0,
         has_sinks ? (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, sinks) / ggml_type_size(sinks->type)) : 0,
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
@@ -1456,10 +1739,15 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
     };
     std::vector<wgpu::BindGroupEntry> entries = {
         ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, Q),
-        ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K),
-        ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V),
     };
-    uint32_t binding_index = 3;
+    if (kv_overlap) {
+        entries.push_back(
+            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), kv_bind_offset, kv_bind_size));
+    } else {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, K));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, V));
+    }
+    uint32_t binding_index = kv_overlap ? 2u : 3u;
     if (has_mask) {
         entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, mask));
     }
@@ -1468,32 +1756,12 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
     }
     entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, binding_index++, dst));
 
-    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
-    shader_lib_ctx.src0                           = Q;
-    shader_lib_ctx.src1                           = K;
-    shader_lib_ctx.src2                           = V;
-    shader_lib_ctx.src3                           = mask;
-    shader_lib_ctx.src4                           = sinks;
-    shader_lib_ctx.dst                            = dst;
-    shader_lib_ctx.max_wg_size        = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.wg_mem_limit_bytes = ctx->global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
-    shader_lib_ctx.sg_mat_m           = ctx->global_ctx->capabilities.sg_mat_m;
-    shader_lib_ctx.sg_mat_n           = ctx->global_ctx->capabilities.sg_mat_n;
-    shader_lib_ctx.sg_mat_k           = ctx->global_ctx->capabilities.sg_mat_k;
-    shader_lib_ctx.max_subgroup_size  = ctx->global_ctx->capabilities.max_subgroup_size;
-    const bool      use_vec           = ggml_webgpu_flash_attn_use_vec(ctx->global_ctx, Q, K, V);
-    webgpu_pipeline pipeline          = use_vec ? ctx->shader_lib->get_flash_attn_vec_pipeline(shader_lib_ctx) :
-                                                  ctx->shader_lib->get_flash_attn_pipeline(shader_lib_ctx);
-
-    if (!use_vec) {
-        auto *   decisions   = static_cast<ggml_webgpu_flash_attn_decisions *>(pipeline.context.get());
+    if (decisions->path != GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
         uint32_t wg_per_head = CEIL_DIV(Q->ne[1], decisions->q_tile);
         uint32_t wg_x        = wg_per_head * Q->ne[2] * Q->ne[3];  // wg per head * number of heads * number of batches
         return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
     }
 
-    auto * decisions = static_cast<ggml_webgpu_flash_attn_vec_decisions *>(pipeline.context.get());
-
     wgpu::Buffer blk_buf         = {};
     uint64_t     blk_size_bytes  = 0;
     uint32_t     blk_nblk0       = 0;
@@ -1532,10 +1800,12 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
         tmp_bind_size   = tmp_size_bytes;
         scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
     } else {
-        // nwg==1 writes final dst directly in vec-split; keep tmp binding valid without extra allocation.
+        // nwg==1 writes final dst directly in vec-split; bind tmp to a tiny non-overlapping scratch region.
+        tmp_size_bytes  = WEBGPU_STORAGE_BUF_BINDING_MULT;
         tmp_buf         = ggml_webgpu_tensor_buf(dst);
-        tmp_bind_offset = ggml_webgpu_tensor_align_offset(ctx, dst);
-        tmp_bind_size   = ggml_webgpu_tensor_binding_size(ctx, dst);
+        tmp_bind_offset = scratch_offset;
+        tmp_bind_size   = tmp_size_bytes;
+        scratch_offset  = ROUNDUP_POW2(scratch_offset + tmp_size_bytes, align_bytes);
     }
 
     webgpu_pipeline                   blk_pipeline;
@@ -1550,7 +1820,7 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
         const uint64_t blk_elems    = (uint64_t) blk_nblk0 * blk_nblk1 * blk_batch_count;
         blk_size_bytes              = ROUNDUP_POW2(blk_elems * sizeof(uint32_t), WEBGPU_STORAGE_BUF_BINDING_MULT);
         const ggml_webgpu_shader_lib_context blk_shader_ctx = shader_lib_ctx;
-        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx);
+        blk_pipeline = ctx->shader_lib->get_flash_attn_blk_pipeline(blk_shader_ctx, decisions->kv_tile);
 
         blk_params = {
             (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mask) / ggml_type_size(mask->type)),  // offset_mask
@@ -1582,12 +1852,19 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
     std::vector<wgpu::BindGroupEntry> split_entries = {
         ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(Q), ggml_webgpu_tensor_align_offset(ctx, Q),
                                           ggml_webgpu_tensor_binding_size(ctx, Q)),
-        ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), ggml_webgpu_tensor_align_offset(ctx, K),
-                                          ggml_webgpu_tensor_binding_size(ctx, K)),
-        ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V), ggml_webgpu_tensor_align_offset(ctx, V),
-                                          ggml_webgpu_tensor_binding_size(ctx, V)),
     };
-    uint32_t split_binding_index = 3;
+    if (kv_overlap) {
+        split_entries.push_back(
+            ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K), kv_bind_offset, kv_bind_size));
+    } else {
+        split_entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(K),
+                                                                  ggml_webgpu_tensor_align_offset(ctx, K),
+                                                                  ggml_webgpu_tensor_binding_size(ctx, K)));
+        split_entries.push_back(ggml_webgpu_make_bind_group_entry(2, ggml_webgpu_tensor_buf(V),
+                                                                  ggml_webgpu_tensor_align_offset(ctx, V),
+                                                                  ggml_webgpu_tensor_binding_size(ctx, V)));
+    }
+    uint32_t split_binding_index = kv_overlap ? 2u : 3u;
     if (has_mask) {
         split_entries.push_back(ggml_webgpu_make_bind_group_entry(split_binding_index++, ggml_webgpu_tensor_buf(mask),
                                                                   ggml_webgpu_tensor_align_offset(ctx, mask),
@@ -1657,22 +1934,20 @@ static webgpu_encoded_op ggml_webgpu_flash_attn(webgpu_context & ctx,
 
     return ggml_backend_webgpu_build_multi(ctx, dispatches);
 }
-#endif  // __EMSCRIPTEN__
 
 static webgpu_encoded_op ggml_webgpu_unary_op(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
     bool is_unary = dst->op == GGML_OP_UNARY;
-    bool inplace  = ggml_webgpu_tensor_equal(src, dst) || (dst->op == GGML_OP_FILL);
 
     ggml_webgpu_shader_lib_context shader_lib_ctx = {};
     shader_lib_ctx.src0                           = src;
     shader_lib_ctx.src1                           = nullptr;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = inplace;
 
     webgpu_pipeline pipeline = ctx->shader_lib->get_unary_pipeline(shader_lib_ctx);
 
-    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+    auto *     decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+    const bool inplace   = decisions->inplace;
 
     uint32_t ne = (uint32_t) ggml_nelements(dst);
 
@@ -1734,41 +2009,38 @@ static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context & ctx,
                                                ggml_tensor *    src0,
                                                ggml_tensor *    src1,
                                                ggml_tensor *    dst) {
-    binary_overlap_flags flags = ggml_webgpu_detect_binary_overlap(src0, src1, dst);
-
     ggml_webgpu_shader_lib_context shader_lib_ctx = {};
     shader_lib_ctx.src0                           = src0;
     shader_lib_ctx.src1                           = src1;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = flags.inplace;
-    shader_lib_ctx.overlap     = flags.overlap;
-    shader_lib_ctx.src_overlap = flags.src_overlap;
 
-    webgpu_pipeline pipeline = ctx->shader_lib->get_binary_pipeline(shader_lib_ctx);
-
-    auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+    webgpu_pipeline pipeline  = ctx->shader_lib->get_binary_pipeline(shader_lib_ctx);
+    auto *          decisions = static_cast<ggml_webgpu_binary_shader_decisions *>(pipeline.context.get());
 
     uint32_t ne = (uint32_t) ggml_nelements(dst);
 
     size_t src0_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src0);
     size_t src1_webgpu_tensor_align_offset = ggml_webgpu_tensor_align_offset(ctx, src1);
 
-    uint32_t offset_merged_src0 = 0;
-    uint32_t offset_merged_src1 = 0;
-    if (flags.src_overlap) {
-        size_t min_off     = std::min(src0_webgpu_tensor_align_offset, src1_webgpu_tensor_align_offset);
-        offset_merged_src0 = (uint32_t) ((src0_webgpu_tensor_align_offset - min_off) / ggml_type_size(src0->type));
-        offset_merged_src1 = (uint32_t) ((src1_webgpu_tensor_align_offset - min_off) / ggml_type_size(src0->type));
+    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
+    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
+    size_t   merged_offset = 0;
+    size_t   merged_size   = 0;
+    if (decisions->src_overlap) {
+        const ggml_webgpu_merged_binding_range merged_range =
+            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
+        merged_offset = merged_range.offset;
+        merged_size   = merged_range.size;
+        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
+        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
     }
 
     std::vector<uint32_t> params = {
         ne,
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
+        offset_src0,
+        offset_src1,
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
-        offset_merged_src0,
-        offset_merged_src1,
         (uint32_t) (src0->nb[0] / ggml_type_size(src0->type)),
         (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
         (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
@@ -1788,12 +2060,9 @@ static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context & ctx,
 
     std::vector<wgpu::BindGroupEntry> entries;
 
-    if (flags.src_overlap) {
-        size_t merged_offset = std::min(src0_webgpu_tensor_align_offset, src1_webgpu_tensor_align_offset);
-        size_t merged_end    = std::max(src0_webgpu_tensor_align_offset + ggml_webgpu_tensor_binding_size(ctx, src0),
-                                        src1_webgpu_tensor_align_offset + ggml_webgpu_tensor_binding_size(ctx, src1));
-        entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset,
-                                                            merged_end - merged_offset));
+    if (decisions->src_overlap) {
+        entries.push_back(
+            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
         entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
     } else {
         entries.push_back(ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0),
@@ -1802,7 +2071,7 @@ static webgpu_encoded_op ggml_webgpu_binary_op(webgpu_context & ctx,
         entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(src1),
                                                             src1_webgpu_tensor_align_offset,
                                                             ggml_webgpu_tensor_binding_size(ctx, src1)));
-        if (!flags.inplace && !flags.overlap) {
+        if (!decisions->inplace && !decisions->overlap) {
             entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
         }
     }
@@ -1892,9 +2161,91 @@ static webgpu_encoded_op ggml_webgpu_repeat(webgpu_context & ctx, ggml_tensor *
     return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
 }
 
-static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
-    bool inplace = ggml_webgpu_tensor_equal(src, dst);
+static std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(webgpu_context & ctx,
+                                                                 ggml_tensor *    rn_src,
+                                                                 ggml_tensor *    rn_dst,
+                                                                 ggml_tensor *    mul_src0,
+                                                                 ggml_tensor *    mul_src1,
+                                                                 ggml_tensor *    dst) {
+    ggml_tensor * mul_src;
+
+    if (ggml_webgpu_tensor_equal(rn_dst, mul_src0)) {
+        mul_src = mul_src1;
+    } else if (ggml_webgpu_tensor_equal(rn_dst, mul_src1)) {
+        mul_src = mul_src0;
+    } else {
+        GGML_ABORT("rms_norm must be equal to the one of mul_src0 and mul_src1");
+    }
+
+    uint32_t offset_rn_src = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, rn_src) / ggml_type_size(rn_src->type));
+    uint32_t offset_mul_src =
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, mul_src) / ggml_type_size(mul_src->type));
+    size_t merged_offset = 0;
+    size_t merged_size   = 0;
 
+    std::vector<uint32_t> params = {
+        offset_rn_src,
+        offset_mul_src,
+        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
+        (uint32_t) (rn_src->nb[1] / ggml_type_size(rn_src->type)),
+        (uint32_t) (rn_src->nb[2] / ggml_type_size(rn_src->type)),
+        (uint32_t) (rn_src->nb[3] / ggml_type_size(rn_src->type)),
+        (uint32_t) (mul_src->nb[1] / ggml_type_size(mul_src->type)),
+        (uint32_t) (mul_src->nb[2] / ggml_type_size(mul_src->type)),
+        (uint32_t) (mul_src->nb[3] / ggml_type_size(mul_src->type)),
+        (uint32_t) (dst->nb[1] / ggml_type_size(dst->type)),
+        (uint32_t) (dst->nb[2] / ggml_type_size(dst->type)),
+        (uint32_t) (dst->nb[3] / ggml_type_size(dst->type)),
+        (uint32_t) mul_src->ne[0],
+        (uint32_t) mul_src->ne[1],
+        (uint32_t) mul_src->ne[2],
+        (uint32_t) mul_src->ne[3],
+        (uint32_t) dst->ne[0],
+        (uint32_t) dst->ne[1],
+        (uint32_t) dst->ne[2],
+        (uint32_t) dst->ne[3],
+        ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(rn_dst, 0))  // epsilon, treated as f32 in the shader
+    };
+
+    std::vector<wgpu::BindGroupEntry> entries;
+
+    ggml_webgpu_shader_lib_context shader_lib_ctx = {};
+    shader_lib_ctx.src0                           = rn_src;
+    shader_lib_ctx.src1                           = mul_src;
+    shader_lib_ctx.dst                            = dst;
+    shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+
+    webgpu_pipeline pipeline  = ctx->shader_lib->get_rms_norm_mul_pipeline(shader_lib_ctx);
+    auto *          decisions = static_cast<ggml_webgpu_rms_norm_mul_shader_decisions *>(pipeline.context.get());
+
+    if (decisions->src_overlap) {
+        const ggml_webgpu_merged_binding_range merged_range =
+            ggml_webgpu_tensor_merged_binding_range(ctx, { rn_src, mul_src });
+        merged_offset  = merged_range.offset;
+        merged_size    = merged_range.size;
+        offset_rn_src  = ggml_webgpu_tensor_merged_element_offset(rn_src, merged_range);
+        offset_mul_src = ggml_webgpu_tensor_merged_element_offset(mul_src, merged_range);
+        params[0]      = offset_rn_src;
+        params[1]      = offset_mul_src;
+    }
+
+    if (decisions->inplace || decisions->overlap) {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
+    } else if (decisions->src_overlap) {
+        entries.push_back(
+            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(rn_src), merged_offset, merged_size));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
+    } else {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, rn_src));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, mul_src));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
+    }
+
+    return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(dst));
+}
+
+static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
     std::vector<uint32_t> params = {
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src) / ggml_type_size(src->type)),
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
@@ -1911,18 +2262,18 @@ static webgpu_encoded_op ggml_webgpu_row_norm(webgpu_context & ctx, ggml_tensor
         ggml_webgpu_u32_from_f32(ggml_get_op_params_f32(dst, 0))  // epsilon, treated as f32 in the shader
     };
 
-    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
-    if (!inplace) {
-        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
-    }
-
     ggml_webgpu_shader_lib_context shader_lib_ctx = {};
     shader_lib_ctx.src0                           = src;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = inplace;
 
-    webgpu_pipeline pipeline = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
+    webgpu_pipeline pipeline  = ctx->shader_lib->get_row_norm_pipeline(shader_lib_ctx);
+    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
+
+    std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
+    if (!decisions->inplace) {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
+    }
     return ggml_backend_webgpu_build(ctx, pipeline, params, entries, ggml_nrows(src));
 }
 
@@ -1937,14 +2288,13 @@ static webgpu_encoded_op ggml_webgpu_rope(webgpu_context & ctx,
     shader_lib_ctx.src2                           = src2;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = ggml_webgpu_tensor_equal(src0, dst);
 
     webgpu_pipeline pipeline = ctx->shader_lib->get_rope_pipeline(shader_lib_ctx);
 
     auto * decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
 
-    const int inplace         = ggml_webgpu_tensor_equal(src0, dst);
-    const int has_freq_factor = (src2 != nullptr);
+    const bool inplace         = decisions->inplace;
+    const int  has_freq_factor = (src2 != nullptr);
 
     const int n_dims     = ((int32_t *) dst->op_params)[1];
     const int mode       = ((int32_t *) dst->op_params)[2];
@@ -2071,14 +2421,11 @@ static webgpu_encoded_op ggml_webgpu_glu(webgpu_context & ctx,
 }
 
 static webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * src, ggml_tensor * dst) {
-    bool inplace = ggml_webgpu_tensor_equal(src, dst);
-
     ggml_webgpu_shader_lib_context shader_lib_ctx = {};
     shader_lib_ctx.src0                           = src;
     shader_lib_ctx.src1                           = nullptr;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = inplace;
 
     webgpu_pipeline pipeline  = ctx->shader_lib->get_scale_pipeline(shader_lib_ctx);
     auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
@@ -2104,7 +2451,7 @@ static webgpu_encoded_op ggml_webgpu_scale(webgpu_context & ctx, ggml_tensor * s
     // bindgroups unchanged
     std::vector<wgpu::BindGroupEntry> entries = { ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src) };
 
-    if (!inplace) {
+    if (!decisions->inplace) {
         entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
     }
 
@@ -2123,17 +2470,17 @@ static webgpu_encoded_op ggml_webgpu_soft_max(webgpu_context & ctx,
     shader_lib_ctx.src2                           = src2;
     shader_lib_ctx.dst                            = dst;
     shader_lib_ctx.max_wg_size = ctx->global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
-    shader_lib_ctx.inplace     = ggml_webgpu_tensor_equal(src0, dst);
 
-    webgpu_pipeline pipeline = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
+    webgpu_pipeline pipeline  = ctx->shader_lib->get_soft_max_pipeline(shader_lib_ctx);
+    auto *          decisions = static_cast<ggml_webgpu_generic_shader_decisions *>(pipeline.context.get());
 
-    const int inplace     = ggml_webgpu_tensor_equal(src0, dst);
-    const int has_mask    = (src1 != nullptr);
-    const int has_sink    = (src2 != nullptr);
-    float     max_bias    = ggml_get_op_params_f32(dst, 1);
-    float     n_head_log2 = float(1u << (uint32_t) floor(log2(src0->ne[2])));
-    float     m0          = powf(2.0f, -(max_bias) / n_head_log2);
-    float     m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
+    const bool inplace     = decisions->inplace;
+    const int  has_mask    = (src1 != nullptr);
+    const int  has_sink    = (src2 != nullptr);
+    float      max_bias    = ggml_get_op_params_f32(dst, 1);
+    float      n_head_log2 = float(1u << (uint32_t) floor(log2(src0->ne[2])));
+    float      m0          = powf(2.0f, -(max_bias) / n_head_log2);
+    float      m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
 
     std::vector<uint32_t> params = {
         (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
@@ -2388,15 +2735,48 @@ static webgpu_encoded_op ggml_webgpu_sum_rows(webgpu_context & ctx, ggml_tensor
     return ggml_backend_webgpu_build(ctx, pipeline, params, entries, wg_x);
 }
 
+static bool ggml_webgpu_can_fuse_rms_norm_mul(const struct ggml_cgraph * cgraph, int node_idx) {
+    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
+        return false;
+    }
+
+    // additional constraints specific to this fusion
+    const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
+    const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
+
+    GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
+    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
+    // rms_norm only supports f32
+    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
+        return false;
+    }
+    // if rms_norm is the B operand, then we don't handle broadcast
+    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
+        return false;
+    }
+    // rms_norm shader assumes contiguous rows
+    if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
+        return false;
+    }
+
+    return true;
+}
+
 // Returns the encoded command, or std::nullopt if the operation is a no-op
-static std::optional<webgpu_encoded_op> ggml_webgpu_encode_node(webgpu_context ctx, ggml_tensor * node) {
+static std::optional<webgpu_encoded_op> ggml_webgpu_encode(webgpu_context ctx,
+                                                           ggml_cgraph *  cgraph,
+                                                           int            node_idx,
+                                                           int &          num_encoded_ops) {
+    ggml_tensor ** nodes = cgraph->nodes;
+    ggml_tensor *  node  = nodes[node_idx];
+
     if (ggml_is_empty(node)) {
         return std::nullopt;
     }
     if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
         return std::nullopt;
     }
-    WEBGPU_LOG_DEBUG("ggml_webgpu_encode_node(" << node << ", " << ggml_op_name(node->op) << ")");
+    WEBGPU_LOG_DEBUG("ggml_webgpu_encode(" << node << ", " << ggml_op_name(node->op) << ")");
 
     ggml_tensor * src0 = node->src[0];
     ggml_tensor * src1 = node->src[1];
@@ -2424,11 +2804,7 @@ static std::optional<webgpu_encoded_op> ggml_webgpu_encode_node(webgpu_context c
         case GGML_OP_MUL_MAT_ID:
             return ggml_webgpu_mul_mat_id(ctx, src0, src1, src2, node);
         case GGML_OP_FLASH_ATTN_EXT:
-#ifndef __EMSCRIPTEN__
             return ggml_webgpu_flash_attn(ctx, src0, src1, src2, node->src[3], node->src[4], node);
-#else
-            return std::nullopt;
-#endif
         case GGML_OP_ADD:
         case GGML_OP_SUB:
         case GGML_OP_MUL:
@@ -2439,6 +2815,13 @@ static std::optional<webgpu_encoded_op> ggml_webgpu_encode_node(webgpu_context c
         case GGML_OP_REPEAT:
             return ggml_webgpu_repeat(ctx, src0, node);
         case GGML_OP_RMS_NORM:
+            if (ggml_webgpu_can_fuse_rms_norm_mul(cgraph, node_idx)) {
+                num_encoded_ops        = 2;
+                ggml_tensor * mul_node = nodes[node_idx + 1];
+                return ggml_webgpu_rms_norm_mul(ctx, src0, node, mul_node->src[0], mul_node->src[1], mul_node);
+            } else {
+                return ggml_webgpu_row_norm(ctx, src0, node);
+            }
         case GGML_OP_L2_NORM:
             return ggml_webgpu_row_norm(ctx, src0, node);
         case GGML_OP_ROPE:
@@ -2464,6 +2847,9 @@ static std::optional<webgpu_encoded_op> ggml_webgpu_encode_node(webgpu_context c
             return ggml_webgpu_solve_tri(ctx, src0, src1, node);
         case GGML_OP_SSM_CONV:
             return ggml_webgpu_ssm_conv(ctx, src0, src1, node);
+        case GGML_OP_SSM_SCAN:
+            return ggml_webgpu_ssm_scan(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node->src[6],
+                                        node);
         case GGML_OP_GATED_DELTA_NET:
             return ggml_webgpu_gated_delta_net(ctx, src0, src1, src2, node->src[3], node->src[4], node->src[5], node);
         case GGML_OP_PAD:
@@ -2479,6 +2865,10 @@ static std::optional<webgpu_encoded_op> ggml_webgpu_encode_node(webgpu_context c
         case GGML_OP_SUM:
         case GGML_OP_SUM_ROWS:
             return ggml_webgpu_sum_rows(ctx, src0, node);
+        case GGML_OP_CONV_2D:
+            return ggml_webgpu_conv_2d(ctx, src0, src1, node);
+        case GGML_OP_IM2COL:
+            return ggml_webgpu_im2col(ctx, src0, src1, node);
         default:
             return std::nullopt;
     }
@@ -2511,14 +2901,17 @@ static void ggml_backend_webgpu_collect_profile_results(webgpu_context &
     for (size_t i = 0; i < pipeline_names.size(); ++i) {
         // WebGPU timestamps are in ns; convert to ms.
         const double elapsed_ms = double(ts_data[2 * i + 1] - ts_data[2 * i]) * 1e-6;
-        ctx->global_ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
+        ctx->shader_gpu_time_ms[pipeline_names[i]] += elapsed_ms;
     }
 
     ctx->profile_timestamp_host_buf.Unmap();
 }
 #endif
 
+// Don't bother checking set_rows index overflow for now, since practically the WebGPU doesn't need to support
+// models that would require it right now.
 static void ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t & num_inflight_batches) {
+#ifdef GGML_WEBGPU_CHECK_SET_ROWS
     wgpu::CommandEncoder encoder = ctx->global_ctx->device.CreateCommandEncoder();
     encoder.CopyBufferToBuffer(ctx->set_rows_dev_error_buf, 0, ctx->set_rows_host_error_buf, 0,
                                ctx->set_rows_host_error_buf.GetSize());
@@ -2531,6 +2924,10 @@ static void ggml_backend_webgpu_check_set_rows(webgpu_context & ctx, uint32_t &
         GGML_ABORT("ggml_webgpu: SET_ROWS index > 2^32, unsupported.");
     }
     ctx->set_rows_host_error_buf.Unmap();
+#else
+    GGML_UNUSED(ctx);
+    GGML_UNUSED(num_inflight_batches);
+#endif
 }
 
 static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
@@ -2547,6 +2944,8 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
     uint32_t num_inflight_batches = 0;
     bool     contains_set_rows    = false;
     bool     batch_compute_passes = true;
+    int      num_encoded_ops      = 1;
+    int      node_idx             = 0;
 
 #ifdef GGML_WEBGPU_GPU_PROFILE
     ctx->profile_timestamp_query_count = 0;
@@ -2559,11 +2958,11 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
         ctx->active_compute_pass = ctx->active_command_encoder.BeginComputePass();
     }
 
-    for (int i = 0; i < cgraph->n_nodes; i++) {
-        if (cgraph->nodes[i]->op == GGML_OP_SET_ROWS) {
+    while (node_idx < cgraph->n_nodes) {
+        if (cgraph->nodes[node_idx]->op == GGML_OP_SET_ROWS) {
             contains_set_rows = true;
         }
-        if (auto cmd = ggml_webgpu_encode_node(ctx, cgraph->nodes[i])) {
+        if (auto cmd = ggml_webgpu_encode(ctx, cgraph, node_idx, num_encoded_ops)) {
             commands.push_back(*cmd);
             num_batched_kernels += cmd.value().num_kernels;
 #ifdef GGML_WEBGPU_GPU_PROFILE
@@ -2588,6 +2987,9 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
             ctx->param_arena.reset();
             commands.clear();
         }
+
+        node_idx += num_encoded_ops;
+        num_encoded_ops = 1;
     }
 
     if (ctx->active_compute_pass) {
@@ -2611,28 +3013,111 @@ static ggml_status ggml_backend_webgpu_graph_compute(ggml_backend_t backend, str
         ggml_backend_webgpu_check_set_rows(ctx, num_inflight_batches);
     }
 
-    ggml_backend_webgpu_wait_queue(ctx->global_ctx);
-
     WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);
     return GGML_STATUS_SUCCESS;
 }
 
+struct ggml_backend_webgpu_event_context {
+    webgpu_global_context global_ctx;
+    wgpu::Future          future;
+    bool                  recorded = false;
+};
+
+static ggml_backend_event_t ggml_backend_webgpu_device_event_new(ggml_backend_dev_t device) {
+    ggml_backend_webgpu_device_context * dev_ctx = (ggml_backend_webgpu_device_context *) device->context;
+
+    auto * event_ctx      = new ggml_backend_webgpu_event_context();
+    event_ctx->global_ctx = dev_ctx->webgpu_global_ctx;
+
+    auto * event   = new ggml_backend_event;
+    event->device  = device;
+    event->context = event_ctx;
+    return event;
+}
+
+static void ggml_backend_webgpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
+    GGML_UNUSED(dev);
+    delete static_cast<ggml_backend_webgpu_event_context *>(event->context);
+    delete event;
+}
+
+static void ggml_backend_webgpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
+    GGML_UNUSED(dev);
+    ggml_backend_webgpu_event_context * event_ctx = (ggml_backend_webgpu_event_context *) event->context;
+    if (!event_ctx->recorded) {
+        return;
+    }
+    wgpu::WaitStatus status =
+        event_ctx->global_ctx->instance.WaitAny(event_ctx->future, WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
+    if (status == wgpu::WaitStatus::TimedOut) {
+        GGML_ABORT("ggml_webgpu: event_synchronize timed out after %u ms\n", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);
+    }
+    event_ctx->recorded = false;
+}
+
+static void ggml_backend_webgpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
+    ggml_backend_webgpu_context *       backend_ctx = (ggml_backend_webgpu_context *) backend->context;
+    ggml_backend_webgpu_event_context * event_ctx   = (ggml_backend_webgpu_event_context *) event->context;
+
+    event_ctx->future = backend_ctx->webgpu_ctx->global_ctx->queue.OnSubmittedWorkDone(
+        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {});
+    event_ctx->recorded = true;
+}
+
+static void ggml_backend_webgpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
+    GGML_UNUSED(backend);
+    ggml_backend_webgpu_device_event_synchronize(nullptr, event);
+}
+
+static void ggml_backend_webgpu_set_tensor_async(ggml_backend_t backend,
+                                                 ggml_tensor *  tensor,
+                                                 const void *   data,
+                                                 size_t         offset,
+                                                 size_t         size) {
+    GGML_UNUSED(backend);
+    auto * buf_ctx      = (ggml_backend_webgpu_buffer_context *) tensor->buffer->context;
+    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;
+
+    // Write aligned portion
+    buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);
+
+    if (size % 4 != 0) {
+        // If size is not a multiple of 4, we need to memset the remaining bytes
+        size_t remaining_size = size % 4;
+
+        // pack the remaining bytes into a uint32_t
+        uint32_t val32 = 0;
+
+        for (size_t i = 0; i < remaining_size; i++) {
+            ((uint8_t *) &val32)[i] = ((const uint8_t *) data)[size - remaining_size + i];
+        }
+        // memset the remaining bytes
+        ggml_backend_webgpu_buffer_memset(buf_ctx->global_ctx, buf_ctx->buffer, val32,
+                                          total_offset + (size - remaining_size), remaining_size);
+    }
+}
+
+static void ggml_backend_webgpu_synchronize(ggml_backend_t backend) {
+    ggml_backend_webgpu_context * backend_ctx = (ggml_backend_webgpu_context *) backend->context;
+    ggml_backend_webgpu_wait_queue(backend_ctx->webgpu_ctx->global_ctx);
+}
+
 static ggml_backend_i ggml_backend_webgpu_i = {
     /* .get_name                = */ ggml_backend_webgpu_name,
     /* .free                    = */ ggml_backend_webgpu_free,
-    /* .set_tensor_async        = */ NULL,
+    /* .set_tensor_async        = */ ggml_backend_webgpu_set_tensor_async,
     /* .get_tensor_async        = */ NULL,
     /* .get_tensor_2d_async     = */ NULL,
     /* .set_tensor_2d_async     = */ NULL,
     /* .cpy_tensor_async        = */ NULL,
-    /* .synchronize             = */ NULL,
+    /* .synchronize             = */ ggml_backend_webgpu_synchronize,
     /* .graph_plan_create       = */ NULL,
     /* .graph_plan_free         = */ NULL,
     /* .graph_plan_update       = */ NULL,
     /* .graph_plan_compute      = */ NULL,
     /* .graph_compute           = */ ggml_backend_webgpu_graph_compute,
-    /* .event_record            = */ NULL,
-    /* .event_wait              = */ NULL,
+    /* .event_record            = */ ggml_backend_webgpu_event_record,
+    /* .event_wait              = */ ggml_backend_webgpu_event_wait,
     /* .graph_optimize          = */ NULL,
 };
 
@@ -2673,7 +3158,7 @@ static void ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffe
     WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buf_ctx->label << ", " << tensor << ", " << value
                                                                  << ", " << offset << ", " << size << ")");
 
-    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;
+    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;
 
     // This is a trick to set all bytes of a u32 to the same 1 byte value.
     uint32_t val32 = (uint32_t) value * 0x01010101;
@@ -2692,7 +3177,7 @@ static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
     WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buf_ctx->label << ", " << tensor << ", " << data
                                                               << ", " << offset << ", " << size << ")");
 
-    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;
+    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;
 
     buf_ctx->global_ctx->queue.WriteBuffer(buf_ctx->buffer, total_offset, data, (size / 4) * 4);
 
@@ -2724,7 +3209,7 @@ static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                               << ", " << offset << ", " << size << ")");
     wgpu::Device device = buf_ctx->global_ctx->device;
 
-    size_t total_offset = webgpu_tensor_offset(tensor) + tensor->view_offs + offset;
+    size_t total_offset = ggml_webgpu_tensor_offset(tensor) + offset;
 
     size_t final_size = size;
     if (size % 4 != 0) {
@@ -2870,13 +3355,19 @@ static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer
                         ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
                     shader_lib_ctx.wg_mem_limit_bytes =
                         ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
+                    shader_lib_ctx.supports_subgroups = ctx->webgpu_global_ctx->capabilities.supports_subgroups;
+                    shader_lib_ctx.supports_subgroup_matrix =
+                        ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix;
                     shader_lib_ctx.sg_mat_m          = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
                     shader_lib_ctx.sg_mat_n          = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
                     shader_lib_ctx.sg_mat_k          = ctx->webgpu_global_ctx->capabilities.sg_mat_k;
                     shader_lib_ctx.max_subgroup_size = ctx->webgpu_global_ctx->capabilities.max_subgroup_size;
 
-                    if (ggml_webgpu_flash_attn_use_vec(ctx->webgpu_global_ctx, Q, K, V)) {
-                        const uint32_t kv_tile = ggml_webgpu_flash_attn_vec_get_kv_tile(shader_lib_ctx);
+                    const ggml_webgpu_flash_attn_decisions decisions = ggml_webgpu_flash_attn_get_decisions(
+                        shader_lib_ctx, ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
+
+                    if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
+                        const uint32_t kv_tile = decisions.kv_tile;
 
                         const uint32_t vec_nwg_cap = std::max(
                             1u, std::min<uint32_t>(32u, ctx->webgpu_global_ctx->capabilities.max_subgroup_size));
@@ -2896,6 +3387,8 @@ static size_t ggml_backend_webgpu_buffer_type_get_alloc_size(ggml_backend_buffer
                             const size_t   tmp_size_bytes  = ROUNDUP_POW2(
                                 (tmp_data_elems + tmp_stats_elems) * sizeof(float), WEBGPU_STORAGE_BUF_BINDING_MULT);
                             res += tmp_size_bytes + align;
+                        } else {
+                            res += WEBGPU_STORAGE_BUF_BINDING_MULT + align;
                         }
                         if (mask != nullptr) {
                             const uint32_t blk_nblk0       = CEIL_DIV((uint32_t) K->ne[1], kv_tile);
@@ -3044,12 +3537,12 @@ static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
     ctx->webgpu_global_ctx->capabilities.supports_subgroups =
         ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::Subgroups);
 
+    bool valid_subgroup_matrix_config = false;
 #ifndef __EMSCRIPTEN__
     // Accept f16 subgroup matrix configurations (square or non-square).
     // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
     // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
     // The shaders are already parameterized to handle any M/N/K dimensions.
-    bool valid_subgroup_matrix_config = false;
     if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
         for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
             const wgpu::SubgroupMatrixConfig config = subgroup_matrix_configs.configs[i];
@@ -3063,8 +3556,8 @@ static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
             }
         }
     }
-    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;
 #endif
+    ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix = valid_subgroup_matrix_config;
 
     // For subgroup matrix code to be the most efficient, we would like the subgroup size to be consistent and accurate.
     // Unfortunately, that is not possible, so we use the maximum subgroup size reported by the adapter.
@@ -3112,12 +3605,12 @@ static bool create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
     // Enable Dawn-specific toggles to increase native performance
     // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
     //       only for native performance?
-    const char * const deviceEnabledToggles[]  = { "skip_validation", "disable_robustness", "disable_workgroup_init",
-                                                   "disable_polyfills_on_integer_div_and_mod" };
-    const char * const deviceDisabledToggles[] = { "timestamp_quantization" };
+    const char * const          deviceEnabledToggles[]  = { "disable_robustness", "disable_workgroup_init",
+                                                            "disable_polyfills_on_integer_div_and_mod" };
+    const char * const          deviceDisabledToggles[] = { "timestamp_quantization" };
     wgpu::DawnTogglesDescriptor deviceTogglesDesc;
     deviceTogglesDesc.enabledToggles      = deviceEnabledToggles;
-    deviceTogglesDesc.enabledToggleCount  = 4;
+    deviceTogglesDesc.enabledToggleCount  = 3;
     deviceTogglesDesc.disabledToggles     = deviceDisabledToggles;
     deviceTogglesDesc.disabledToggleCount = 1;
 
@@ -3241,6 +3734,7 @@ static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggm
 
 static bool ggml_webgpu_supported_qtype(ggml_type type) {
     switch (type) {
+        case GGML_TYPE_Q1_0:
         case GGML_TYPE_Q4_0:
         case GGML_TYPE_Q4_1:
         case GGML_TYPE_Q5_0:
@@ -3335,6 +3829,7 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
                         switch (src0->type) {
                             case GGML_TYPE_F32:
                             case GGML_TYPE_F16:
+                            case GGML_TYPE_Q1_0:
                             case GGML_TYPE_Q4_0:
                             case GGML_TYPE_Q4_1:
                             case GGML_TYPE_Q5_0:
@@ -3373,6 +3868,7 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
                     switch (src0->type) {
                         case GGML_TYPE_F32:
                         case GGML_TYPE_F16:
+                        case GGML_TYPE_Q1_0:
                         case GGML_TYPE_Q4_0:
                         case GGML_TYPE_Q4_1:
                         case GGML_TYPE_Q5_0:
@@ -3395,33 +3891,63 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
             break;
         case GGML_OP_FLASH_ATTN_EXT:
             {
-#ifndef __EMSCRIPTEN__
-                if (!ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
+                supports_op = src0->type == GGML_TYPE_F32 &&
+                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
+                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
+                              src2->type == src1->type && op->type == GGML_TYPE_F32;
+                if (!supports_op) {
                     break;
                 }
-                // Head dimensions must be divisible by subgroup matrix dimensions
-                if (src0->ne[0] % ctx->webgpu_global_ctx->capabilities.sg_mat_k != 0 ||
-                    src2->ne[0] % ctx->webgpu_global_ctx->capabilities.sg_mat_n != 0) {
+                ggml_webgpu_shader_lib_context shader_lib_ctx = {};
+                shader_lib_ctx.src0                           = src0;
+                shader_lib_ctx.src1                           = src1;
+                shader_lib_ctx.src2                           = src2;
+                shader_lib_ctx.src3                           = op->src[3];
+                shader_lib_ctx.src4                           = op->src[4];
+                shader_lib_ctx.dst                            = const_cast<ggml_tensor *>(op);
+                shader_lib_ctx.supports_subgroups             = ctx->webgpu_global_ctx->capabilities.supports_subgroups;
+                shader_lib_ctx.supports_subgroup_matrix = ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix;
+                shader_lib_ctx.wg_mem_limit_bytes =
+                    ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
+                shader_lib_ctx.sg_mat_m          = ctx->webgpu_global_ctx->capabilities.sg_mat_m;
+                shader_lib_ctx.sg_mat_n          = ctx->webgpu_global_ctx->capabilities.sg_mat_n;
+                shader_lib_ctx.sg_mat_k          = ctx->webgpu_global_ctx->capabilities.sg_mat_k;
+                shader_lib_ctx.max_subgroup_size = ctx->webgpu_global_ctx->capabilities.max_subgroup_size;
+
+                const ggml_webgpu_flash_attn_decisions decisions = ggml_webgpu_flash_attn_get_decisions(
+                    shader_lib_ctx, ctx->webgpu_global_ctx->capabilities.limits.minStorageBufferOffsetAlignment);
+                const size_t limit_bytes = ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
+                const bool   has_mask    = op->src[3] != nullptr;
+                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_VEC) {
+                    const size_t min_bytes =
+                        ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
+                                                            (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
+                    if (min_bytes > limit_bytes) {
+                        supports_op = false;
+                    }
                     break;
                 }
-                // Head dimensions must fit in workgroup memory with minimum tile sizes
-                size_t     limit_bytes = ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize;
-                const bool has_mask    = op->src[3] != nullptr;
-                const bool kv_direct   = src1->type == GGML_TYPE_F16 &&
-                                       (src0->ne[0] % ctx->webgpu_global_ctx->capabilities.sg_mat_k) == 0 &&
-                                       (src1->ne[1] % GGML_WEBGPU_KV_SEQ_PAD) == 0;
-                const size_t min_bytes = ggml_webgpu_flash_attn_wg_mem_bytes(
-                    ctx->webgpu_global_ctx->capabilities.sg_mat_m, ctx->webgpu_global_ctx->capabilities.sg_mat_n,
-                    (uint32_t) src0->ne[0], (uint32_t) src2->ne[0], has_mask, kv_direct);
-                if (min_bytes > limit_bytes) {
+
+                if (decisions.path == GGML_WEBGPU_FLASH_ATTN_PATH_TILE) {
+                    const size_t min_bytes =
+                        ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
+                                                            (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
+                    if (min_bytes > limit_bytes) {
+                        supports_op = false;
+                    }
                     break;
                 }
 
-                supports_op = src0->type == GGML_TYPE_F32 &&
-                              (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 ||
-                               src1->type == GGML_TYPE_Q4_0 || src1->type == GGML_TYPE_Q8_0) &&
-                              src2->type == src1->type && op->type == GGML_TYPE_F32;
-#endif
+                if (!ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
+                    supports_op = false;
+                    break;
+                }
+                const size_t min_bytes =
+                    ggml_webgpu_flash_attn_wg_mem_bytes(decisions.q_tile, decisions.kv_tile, (uint32_t) src0->ne[0],
+                                                        (uint32_t) src2->ne[0], has_mask, decisions.kv_direct);
+                if (min_bytes > limit_bytes) {
+                    supports_op = false;
+                }
                 break;
             }
         case GGML_OP_RMS_NORM:
@@ -3497,9 +4023,22 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
         case GGML_OP_SOLVE_TRI:
             supports_op = op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
             break;
+        case GGML_OP_CONV_2D:
+            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
+                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
+                          (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
+            break;
+        case GGML_OP_IM2COL:
+            supports_op = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
+                          (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
+            break;
         case GGML_OP_SSM_CONV:
             supports_op = op->type == GGML_TYPE_F32;
             break;
+        case GGML_OP_SSM_SCAN:
+            supports_op = op->type == GGML_TYPE_F32 &&
+                          src0->ne[0] <= ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup;
+            break;
         case GGML_OP_GATED_DELTA_NET:
             {
                 const uint32_t s_v = (uint32_t) src2->ne[0];
@@ -3590,9 +4129,9 @@ static struct ggml_backend_device_i ggml_backend_webgpu_device_i = {
     /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
     /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
     /* .offload_op           = */ NULL,
-    /* .event_new            = */ NULL,
-    /* .event_free           = */ NULL,
-    /* .event_synchronize    = */ NULL,
+    /* .event_new            = */ ggml_backend_webgpu_device_event_new,
+    /* .event_free           = */ ggml_backend_webgpu_device_event_free,
+    /* .event_synchronize    = */ ggml_backend_webgpu_device_event_synchronize,
 };
 
 /* End GGML Backend Device Interface */
diff --git src/ggml-webgpu/wgsl-shaders/binary.wgsl src/ggml-webgpu/wgsl-shaders/binary.wgsl
index a748dc1b..605de7aa 100644
--- src/ggml-webgpu/wgsl-shaders/binary.wgsl
+++ src/ggml-webgpu/wgsl-shaders/binary.wgsl
@@ -7,8 +7,6 @@ struct Params {
     offset_src0: u32,
     offset_src1: u32,
     offset_dst: u32,
-    offset_merged_src0: u32,
-    offset_merged_src1: u32,
 
     stride_src0_0: u32,
     stride_src0_1: u32,
@@ -134,8 +132,8 @@ fn update(dst_i: u32, src0_i: u32, src1_i: u32) {
 @compute @workgroup_size(WG_SIZE)
 fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
     if (gid.x < params.ne) {
-        let src0_i = params.offset_src0 + params.offset_merged_src0 + src0_index(gid.x);
-        let src1_i = params.offset_src1 + params.offset_merged_src1 + src1_index(gid.x);
+        let src0_i = params.offset_src0 + src0_index(gid.x);
+        let src1_i = params.offset_src1 + src1_index(gid.x);
         update(params.offset_dst + gid.x, src0_i, src1_i);
     }
 }
diff --git src/ggml-webgpu/wgsl-shaders/conv2d.wgsl src/ggml-webgpu/wgsl-shaders/conv2d.wgsl
new file mode 100644
index 00000000..9eb131dc
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/conv2d.wgsl
@@ -0,0 +1,165 @@
+#include "common_decls.tmpl"
+enable f16;
+
+@group(0) @binding(0)
+#if defined(WEIGHT_F32)
+var<storage, read_write> weights: array<f32>;
+#elif defined(WEIGHT_F16)
+var<storage, read_write> weights: array<f16>;
+#endif
+
+@group(0) @binding(1)
+#if defined(INPUT_F32)
+var<storage, read_write> input: array<f32>;
+#elif defined(INPUT_F16)
+var<storage, read_write> input: array<f16>;
+#endif
+
+@group(0) @binding(2)
+#if defined(OUTPUT_F32)
+var<storage, read_write> output: array<f32>;
+#elif defined(OUTPUT_F16)
+var<storage, read_write> output: array<f16>;
+#endif
+
+struct Params {
+    offset_w: u32,
+    offset_i: u32,
+    offset_o: u32,
+
+    // element strides
+    sw0: u32, sw1: u32, sw2: u32, sw3: u32,
+    si0: u32, si1: u32, si2: u32, si3: u32,
+    so0: u32, so1: u32, so2: u32, so3: u32,
+
+    // kernel dimensions
+    KW: u32, KH: u32, IC: u32,
+    // input dimensions
+    IW: u32, IH: u32,
+    // output dimensions
+    OW: u32, OH: u32, OC_out: u32, N_out: u32,
+
+    // stride
+    s0: u32, s1: u32,
+    // padding
+    p0: u32, p1: u32,
+    // dilation
+    d0: u32, d1: u32,
+};
+
+@group(0) @binding(3)
+var<uniform> params: Params;
+
+fn load_weight(idx: u32) -> f32 {
+    #if defined(WEIGHT_F32)
+        return weights[idx];
+    #elif defined(WEIGHT_F16)
+        return f32(weights[idx]);
+    #endif
+}
+
+fn load_input(idx: u32) -> f32 {
+    #if defined(INPUT_F32)
+        return input[idx];
+    #elif defined(INPUT_F16)
+        return f32(input[idx]);
+    #endif
+}
+
+fn store_output(idx: u32, val: f32) {
+    #if defined(OUTPUT_F32)
+        output[idx] = val;
+    #elif defined(OUTPUT_F16)
+        output[idx] = f16(val);
+    #endif
+}
+
+fn ceil_div_u32(x: u32, y: u32) -> u32 {
+    return (x + y - 1) / y;
+}
+
+// returns the first valid kernel index k such that base + k * step >= 0
+fn first_valid_k(base: i32, step: u32) -> u32 {
+    if (base >= 0) {
+        return 0;
+    }
+
+    return ceil_div_u32(u32(-base), step);
+}
+
+// returns the first invalid kernel index k such that base + k * step >= limit so valid k are in [0, end_valid_k)
+fn end_valid_k(base: i32, step: u32, limit: u32, k_max: u32) -> u32 {
+    let remaining = i32(limit) - base;
+    if (remaining <= 0) {
+        return 0;
+    }
+
+    return min(k_max, ceil_div_u32(u32(remaining), step));
+}
+
+@compute @workgroup_size(WG_SIZE)
+fn main(
+    @builtin(global_invocation_id) gid: vec3<u32>,
+    @builtin(num_workgroups) num_wg: vec3<u32>
+) {
+
+    let threads_per_group = u32(WG_SIZE);
+    let i_out = gid.x + (num_wg.x * threads_per_group) * gid.y;
+    let n_out = params.OW * params.OH * params.OC_out * params.N_out;
+
+    var sum: f32 = 0.0;
+    if (i_out >= n_out) {
+        return;
+    }
+
+    // Kernel layout: [KW, KH, IC, ..]
+    // Input layout:  [IW, IH, .., ..]
+    // Output layout: [OW, OH, OC, N]
+
+    var i = i_out;
+    let n = i / (params.OC_out * params.OH * params.OW);
+    i = i % (params.OC_out * params.OH * params.OW);
+    let oc = i / (params.OH * params.OW);
+    i = i % (params.OH * params.OW);
+    let oh = i / params.OW;
+    let ow = i % params.OW;
+
+    let ow_base = i32(ow * params.s0) - i32(params.p0);
+    let oh_base = i32(oh * params.s1) - i32(params.p1);
+
+    // clip the valid kernel window once
+    let kw_begin = first_valid_k(ow_base, params.d0);
+    let kw_end = end_valid_k(ow_base, params.d0, params.IW, params.KW);
+    let kh_begin = first_valid_k(oh_base, params.d1);
+    let kh_end = end_valid_k(oh_base, params.d1, params.IH, params.KH);
+
+    // entire receptive field is out of bounds
+    if (kw_begin >= kw_end || kh_begin >= kh_end) {
+        let out_idx = params.offset_o + ow * params.so0 + oh * params.so1 + oc * params.so2 + n * params.so3;
+        store_output(out_idx, 0.0);
+        return;
+    }
+
+    let weight_oc_base = params.offset_w + oc * params.sw3;
+    let input_n_base = params.offset_i + n * params.si3;
+
+    for (var ic: u32 = 0; ic < params.IC; ic += 1) {
+        let w_base_ic = ic * params.sw2 + weight_oc_base;
+        let in_base = ic * params.si2 + input_n_base;
+
+        for (var kh: u32 = kh_begin; kh < kh_end; kh += 1) {
+            let ih = u32(oh_base + i32(kh * params.d1));
+            let w_row_base = w_base_ic + kh * params.sw1;
+            let in_row_base = in_base + ih * params.si1;
+            for (var kw: u32 = kw_begin; kw < kw_end; kw += 1) {
+                let iw = u32(ow_base + i32(kw * params.d0));
+                let w_idx = w_row_base + kw * params.sw0;
+                let in_idx = in_row_base + iw * params.si0;
+                sum += load_weight(w_idx) * load_input(in_idx);
+            }
+        }
+    }
+
+    let out_idx = params.offset_o + ow * params.so0 + oh * params.so1 + oc * params.so2 + n * params.so3;
+    store_output(out_idx, sum);
+}
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn.wgsl
index aa2d2e54..6d5d69fb 100644
--- src/ggml-webgpu/wgsl-shaders/flash_attn.wgsl
+++ src/ggml-webgpu/wgsl-shaders/flash_attn.wgsl
@@ -138,26 +138,55 @@ struct Params {
 };
 
 @group(0) @binding(0) var<storage, read_write> Q: array<f32>;
+#ifdef KV_OVERLAP
+@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
+#define V K
+#else
 @group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
 @group(0) @binding(2) var<storage, read_write> V: array<KV_TYPE>;
+#endif
 
 #if defined(MASK) && defined(SINKS)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
+@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#else
 @group(0) @binding(3) var<storage, read_write> mask: array<f16>;
 @group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
 #define DST_BINDING 5
 #define PARAMS_BINDING 6
+#endif
 #elif defined(MASK)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
+#define DST_BINDING 3
+#define PARAMS_BINDING 4
+#else
 @group(0) @binding(3) var<storage, read_write> mask: array<f16>;
 #define DST_BINDING 4
 #define PARAMS_BINDING 5
+#endif
 #elif defined(SINKS)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> sinks: array<f32>;
+#define DST_BINDING 3
+#define PARAMS_BINDING 4
+#else
 @group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
 #define DST_BINDING 4
 #define PARAMS_BINDING 5
+#endif
+#else
+#ifdef KV_OVERLAP
+#define DST_BINDING 2
+#define PARAMS_BINDING 3
 #else
 #define DST_BINDING 3
 #define PARAMS_BINDING 4
 #endif
+#endif
 
 @group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
 @group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl
new file mode 100644
index 00000000..37ea23b8
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl
@@ -0,0 +1,330 @@
+enable f16;
+enable subgroups;
+
+#define HEAD_DIM_QK 64
+#define HEAD_DIM_V 64
+#define KV_STAGE_STRIDE 64
+#define Q_TILE 4
+#define KV_TILE 64
+#define WG_SIZE 128
+
+struct Params {
+    offset_q: u32,
+    offset_k: u32,
+    offset_v: u32,
+    offset_mask: u32,
+    offset_sinks: u32,
+    offset_dst: u32,
+
+    n_heads: u32,
+    seq_len_q: u32,
+    seq_len_kv: u32,
+
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
+    q_per_kv: u32,
+
+    scale: f32,
+    max_bias: f32,
+    logit_softcap: f32,
+    n_head_log2: f32,
+    m0: f32,
+    m1: f32,
+};
+
+@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
+#ifdef KV_OVERLAP
+@group(0) @binding(1) var<storage, read_write> K: array<vec4<f16>>;
+#define V K
+#else
+@group(0) @binding(1) var<storage, read_write> K: array<vec4<f16>>;
+@group(0) @binding(2) var<storage, read_write> V: array<vec4<f16>>;
+#endif
+
+#if defined(MASK) && defined(SINKS)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
+@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#else
+@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
+@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
+#define DST_BINDING 5
+#define PARAMS_BINDING 6
+#endif
+#elif defined(MASK)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
+#define DST_BINDING 3
+#define PARAMS_BINDING 4
+#else
+@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#endif
+#elif defined(SINKS)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> sinks: array<f32>;
+#define DST_BINDING 3
+#define PARAMS_BINDING 4
+#else
+@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#endif
+#else
+#ifdef KV_OVERLAP
+#define DST_BINDING 2
+#define PARAMS_BINDING 3
+#else
+#define DST_BINDING 3
+#define PARAMS_BINDING 4
+#endif
+#endif
+
+@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
+@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;
+
+const FLOAT_MIN: f32 = -1.0e9;
+const Q_CHUNKS: u32 = HEAD_DIM_QK / 4u;
+const V_CHUNKS: u32 = HEAD_DIM_V / 4u;
+const SCORE_REGS_PER_LANE: u32 = (KV_TILE + MAX_SUBGROUP_SIZE - 1u) / MAX_SUBGROUP_SIZE;
+const OUT_REGS_PER_LANE: u32 = (V_CHUNKS + MAX_SUBGROUP_SIZE - 1u) / MAX_SUBGROUP_SIZE;
+
+var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;
+var<workgroup> kv_shmem: array<f16, KV_TILE * KV_STAGE_STRIDE>;
+var<workgroup> p_shmem: array<f32, Q_TILE * KV_TILE>;
+
+@compute @workgroup_size(WG_SIZE)
+fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
+        @builtin(local_invocation_id) local_id: vec3<u32>,
+        @builtin(subgroup_id) subgroup_id: u32,
+        @builtin(subgroup_size) subgroup_size: u32,
+        @builtin(num_subgroups) num_subgroups: u32,
+        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
+    if (subgroup_size == 0u || num_subgroups < Q_TILE) {
+        return;
+    }
+
+    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
+    let wg_per_batch = wg_per_head * params.n_heads;
+
+    let dst2_stride = HEAD_DIM_V * params.n_heads;
+    let dst3_stride = dst2_stride * params.seq_len_q;
+
+    let batch_idx = wg_id.x / wg_per_batch;
+    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
+    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
+    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
+    let dst_batch_offset = params.offset_dst + batch_idx * dst3_stride;
+    let wg_in_batch = wg_id.x % wg_per_batch;
+
+    let head_idx = wg_in_batch / wg_per_head;
+    let q_head_offset = q_batch_offset + head_idx * params.stride_q2;
+    let k_head_idx = head_idx / params.q_per_kv;
+    let v_head_offset = v_batch_offset + k_head_idx * params.stride_v2;
+    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
+
+    let wg_in_head = wg_in_batch % wg_per_head;
+    let q_row_start = wg_in_head * Q_TILE;
+    let global_q_row = q_row_start + subgroup_id;
+    let row_active = subgroup_id < Q_TILE && global_q_row < params.seq_len_q;
+
+#ifdef MASK
+    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row_start * params.seq_len_kv;
+#endif
+
+    let dst_global_offset = dst_batch_offset + q_row_start * dst2_stride + head_idx * HEAD_DIM_V;
+
+    let head = f32(head_idx);
+    let slope = select(1.0,
+                       select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0),
+                              pow(params.m0, head + 1.0),
+                              head < params.n_head_log2),
+                       params.max_bias > 0.0);
+
+    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
+        let q_tile_row = elem_idx / HEAD_DIM_QK;
+        let q_col = elem_idx % HEAD_DIM_QK;
+        let head_q_row = q_row_start + q_tile_row;
+        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
+        q_shmem[elem_idx] = f16(select(
+            0.0,
+            Q[global_q_row_offset + q_col] * params.scale,
+            head_q_row < params.seq_len_q));
+    }
+
+    workgroupBarrier();
+
+    var row_max = FLOAT_MIN;
+    var exp_sum = 0.0;
+    var out_regs: array<vec4<f32>, OUT_REGS_PER_LANE>;
+    for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
+        out_regs[reg_idx] = vec4<f32>(0.0);
+    }
+
+    let q_base = subgroup_id * HEAD_DIM_QK;
+    let subgroup_p_offset = subgroup_id * KV_TILE;
+
+    for (var kv_tile = 0u; kv_tile < params.seq_len_kv; kv_tile += KV_TILE) {
+        let kv_count = min(KV_TILE, params.seq_len_kv - kv_tile);
+        let score_slots = min(SCORE_REGS_PER_LANE, (kv_count + subgroup_size - 1u) / subgroup_size);
+        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
+        var local_scores: array<f32, SCORE_REGS_PER_LANE>;
+        for (var slot = 0u; slot < SCORE_REGS_PER_LANE; slot += 1u) {
+            local_scores[slot] = FLOAT_MIN;
+        }
+
+        for (var vec_idx_local = local_id.x; vec_idx_local < kv_count * Q_CHUNKS; vec_idx_local += WG_SIZE) {
+            let kv_local = vec_idx_local / Q_CHUNKS;
+            let chunk = vec_idx_local % Q_CHUNKS;
+            let global_k_row = kv_tile + kv_local;
+            let k_vec_index = (k_head_offset + global_k_row * params.stride_k1 + chunk * 4u) >> 2u;
+            let k4 = K[k_vec_index];
+            let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
+            kv_shmem[kv_off + 0u] = k4.x;
+            kv_shmem[kv_off + 1u] = k4.y;
+            kv_shmem[kv_off + 2u] = k4.z;
+            kv_shmem[kv_off + 3u] = k4.w;
+        }
+
+        workgroupBarrier();
+
+        var local_max = FLOAT_MIN;
+        if (row_active) {
+            for (var slot = 0u; slot < score_slots; slot += 1u) {
+                let kv_local = sg_inv_id + slot * subgroup_size;
+                if (kv_local >= kv_count) {
+                    continue;
+                }
+
+                let global_k_row = kv_tile + kv_local;
+                var dot_val = 0.0;
+                for (var chunk = 0u; chunk < Q_CHUNKS; chunk += 1u) {
+                    let q_off = q_base + chunk * 4u;
+                    let qv = vec4<f32>(
+                        f32(q_shmem[q_off + 0u]),
+                        f32(q_shmem[q_off + 1u]),
+                        f32(q_shmem[q_off + 2u]),
+                        f32(q_shmem[q_off + 3u]));
+                    let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
+                    let kv = vec4<f32>(
+                        f32(kv_shmem[kv_off + 0u]),
+                        f32(kv_shmem[kv_off + 1u]),
+                        f32(kv_shmem[kv_off + 2u]),
+                        f32(kv_shmem[kv_off + 3u]));
+                    dot_val += dot(qv, kv);
+                }
+#ifdef LOGIT_SOFTCAP
+                dot_val = params.logit_softcap * tanh(dot_val);
+#endif
+#ifdef MASK
+                let mask_idx = mask_global_offset + subgroup_id * params.seq_len_kv + global_k_row;
+                dot_val += slope * f32(mask[mask_idx]);
+#endif
+                local_scores[slot] = dot_val;
+                local_max = max(local_max, dot_val);
+            }
+        }
+
+        let tile_max = subgroupMax(local_max);
+        let new_max = max(row_max, tile_max);
+        let cur_exp = exp(row_max - new_max);
+        exp_sum *= cur_exp;
+        for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
+            out_regs[reg_idx] *= cur_exp;
+        }
+
+        var local_sum = 0.0;
+        for (var slot = 0u; slot < score_slots; slot += 1u) {
+            let kv_local = sg_inv_id + slot * subgroup_size;
+            if (row_active && kv_local < kv_count) {
+                let p = exp(local_scores[slot] - new_max);
+                p_shmem[subgroup_p_offset + kv_local] = p;
+                local_sum += p;
+            }
+        }
+
+        workgroupBarrier();
+
+        for (var vec_idx_local = local_id.x; vec_idx_local < kv_count * V_CHUNKS; vec_idx_local += WG_SIZE) {
+            let kv_local = vec_idx_local / V_CHUNKS;
+            let chunk = vec_idx_local % V_CHUNKS;
+            let global_v_row = kv_tile + kv_local;
+            let v_vec_index = (v_head_offset + global_v_row * params.stride_v1 + chunk * 4u) >> 2u;
+            let v4 = V[v_vec_index];
+            let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
+            kv_shmem[kv_off + 0u] = v4.x;
+            kv_shmem[kv_off + 1u] = v4.y;
+            kv_shmem[kv_off + 2u] = v4.z;
+            kv_shmem[kv_off + 3u] = v4.w;
+        }
+
+        workgroupBarrier();
+
+        let tile_sum = subgroupAdd(local_sum);
+        exp_sum += tile_sum;
+        row_max = new_max;
+
+        if (row_active) {
+            for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
+                let chunk = sg_inv_id + reg_idx * subgroup_size;
+                if (chunk >= V_CHUNKS) {
+                    continue;
+                }
+
+                var acc = out_regs[reg_idx];
+                for (var kv_local = 0u; kv_local < kv_count; kv_local += 1u) {
+                    let p = p_shmem[subgroup_p_offset + kv_local];
+                    let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
+                    let v4 = vec4<f32>(
+                        f32(kv_shmem[kv_off + 0u]),
+                        f32(kv_shmem[kv_off + 1u]),
+                        f32(kv_shmem[kv_off + 2u]),
+                        f32(kv_shmem[kv_off + 3u]));
+                    acc += p * v4;
+                }
+                out_regs[reg_idx] = acc;
+            }
+        }
+
+        workgroupBarrier();
+    }
+
+#ifdef SINKS
+    if (row_active) {
+        let sink_score = sinks[params.offset_sinks + head_idx];
+        let sink_max = max(row_max, sink_score);
+        let sink_scale = exp(row_max - sink_max);
+        for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
+            out_regs[reg_idx] *= sink_scale;
+        }
+        exp_sum = exp_sum * sink_scale + exp(sink_score - sink_max);
+        row_max = sink_max;
+    }
+#endif
+
+    if (row_active) {
+        let inv_exp_sum = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
+        let row_base = dst_global_offset + subgroup_id * dst2_stride;
+        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
+        for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
+            let chunk = sg_inv_id + reg_idx * subgroup_size;
+            if (chunk >= V_CHUNKS) {
+                continue;
+            }
+            let dst_vec_index = (row_base + chunk * 4u) >> 2u;
+            dst[dst_vec_index] = out_regs[reg_idx] * inv_exp_sum;
+        }
+    }
+}
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl
index 61107c6a..b4f7c16c 100644
--- src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl
+++ src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl
@@ -15,7 +15,7 @@ struct Params {
     nblk1: u32,
 };
 
-@group(0) @binding(0) var<storage, read> mask: array<f16>;
+@group(0) @binding(0) var<storage, read_write> mask: array<f16>;
 @group(0) @binding(1) var<storage, read_write> blk: array<u32>;
 @group(0) @binding(2) var<uniform> params: Params;
 
diff --git src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl
index a5257587..b1e23478 100644
--- src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl
+++ src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl
@@ -1,8 +1,6 @@
-diagnostic(off, chromium.subgroup_matrix_uniformity);
 diagnostic(off, subgroup_uniformity);
 enable f16;
 enable subgroups;
-enable chromium_experimental_subgroup_matrix;
 
 #ifdef KV_F32
 #define KV_TYPE f32
@@ -13,19 +11,14 @@ enable chromium_experimental_subgroup_matrix;
 #define HEAD_DIM_QK 64
 #define HEAD_DIM_V 64
 
-
-#define SG_MAT_M 8
-#define SG_MAT_N 8
-#define SG_MAT_K 8
-
-#define Q_TILE SG_MAT_M
+#define KV_GRANULARITY 8
 #define KV_TILE 16
 #define WG_SIZE 64
 #ifndef VEC_NE
 #define VEC_NE 4u
 #endif
 
-#define KV_BLOCKS (KV_TILE / SG_MAT_N)
+#define KV_BLOCKS (KV_TILE / KV_GRANULARITY)
 
 #define BLOCK_SIZE 32
 #define BLOCKS_K ((HEAD_DIM_QK + BLOCK_SIZE - 1) / BLOCK_SIZE)
@@ -97,6 +90,14 @@ struct Params {
 };
 
 @group(0) @binding(0) var<storage, read_write> Q: array<f32>;
+#ifdef KV_OVERLAP
+#if defined(KV_Q4_0) || defined(KV_Q8_0)
+@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
+#else
+@group(0) @binding(1) var<storage, read_write> K: array<vec4<KV_TYPE>>;
+#endif
+#define V K
+#else
 #if defined(KV_Q4_0) || defined(KV_Q8_0)
 @group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
 #else
@@ -107,7 +108,22 @@ struct Params {
 #else
 @group(0) @binding(2) var<storage, read_write> V: array<vec4<KV_TYPE>>;
 #endif
+#endif
 #if defined(MASK) && defined(SINKS)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
+@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
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
+#else
 @group(0) @binding(3) var<storage, read_write> mask: array<f16>;
 @group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
 #ifdef BLK
@@ -120,7 +136,21 @@ struct Params {
 #define DST_BINDING 6
 #define PARAMS_BINDING 7
 #endif
+#endif
 #elif defined(MASK)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
+#ifdef BLK
+#define BLK_BINDING 3
+#define TMP_BINDING 4
+#define DST_BINDING 5
+#define PARAMS_BINDING 6
+#else
+#define TMP_BINDING 3
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#endif
+#else
 @group(0) @binding(3) var<storage, read_write> mask: array<f16>;
 #ifdef BLK
 #define BLK_BINDING 4
@@ -132,16 +162,30 @@ struct Params {
 #define DST_BINDING 5
 #define PARAMS_BINDING 6
 #endif
+#endif
 #elif defined(SINKS)
+#ifdef KV_OVERLAP
+@group(0) @binding(2) var<storage, read_write> sinks: array<f32>;
+#define TMP_BINDING 3
+#define DST_BINDING 4
+#define PARAMS_BINDING 5
+#else
 @group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
 #define TMP_BINDING 4
 #define DST_BINDING 5
 #define PARAMS_BINDING 6
+#endif
+#else
+#ifdef KV_OVERLAP
+#define TMP_BINDING 2
+#define DST_BINDING 3
+#define PARAMS_BINDING 4
 #else
 #define TMP_BINDING 3
 #define DST_BINDING 4
 #define PARAMS_BINDING 5
 #endif
+#endif
 
 #ifdef BLK
 @group(0) @binding(BLK_BINDING) var<storage, read_write> blk: array<u32>;
@@ -153,7 +197,7 @@ struct Params {
 // Just a very small float value.
 const FLOAT_MIN: f32 = -1.0e9;
 
-var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;
+var<workgroup> q_shmem: array<f16, HEAD_DIM_QK>;
 
 #ifndef KV_DIRECT
 const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
@@ -161,31 +205,27 @@ const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
 var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
 #endif
 
-var<workgroup> o_shmem: array<f16, Q_TILE * HEAD_DIM_V>;
+var<workgroup> o_shmem: array<f16, HEAD_DIM_V>;
 
 #ifdef MASK
 // storage for mask values
-var<workgroup> mask_shmem: array<f16, Q_TILE * KV_TILE>;
+var<workgroup> mask_shmem: array<f16, KV_TILE>;
 #endif
 
 // note that we reuse the same storage for both since we only need one at a time
-var<workgroup> inter_shmem: array<f16, Q_TILE * KV_TILE>;
+var<workgroup> inter_shmem: array<f16, KV_TILE>;
 
 // Storage for row max and exp sum during online softmax
-var<workgroup> row_max_shmem: array<f32, Q_TILE>;
-var<workgroup> exp_sum_shmem: array<f32, Q_TILE>;
-var<workgroup> blk_state_wg: u32;
-
-fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32, has_bias: bool, apply_mask: bool) -> f32 {
+fn calc_softmax_term(kv_idx: u32, slope: f32, has_bias: bool, apply_mask: bool) -> f32 {
     var v = select(FLOAT_MIN,
-                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
+                   f32(inter_shmem[kv_idx]) * params.scale,
                    kv_idx < KV_TILE);
 #ifdef LOGIT_SOFTCAP
     v = params.logit_softcap * tanh(v);
 #endif
 #ifdef MASK
     if (apply_mask) {
-        var mask_val = select(0.0,f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
+        var mask_val = select(0.0, f32(mask_shmem[kv_idx]), kv_idx < KV_TILE);
         v += select(mask_val, slope * mask_val, has_bias);
     }
 #endif
@@ -199,19 +239,17 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
     @builtin(subgroup_size) subgroup_size: u32,
     @builtin(num_subgroups) num_subgroups: u32,
     @builtin(subgroup_invocation_id) sg_inv_id: u32) {
+    // Vec path processes exactly one query row per workgroup, so subgroup 0 can
+    // keep the running softmax state in private storage.
+    var row_max = FLOAT_MIN;
+    var exp_sum = 0.0;
 
-    // initialize row max for online softmax
-    for (var i = local_id.x; i < Q_TILE; i += WG_SIZE) {
-        row_max_shmem[i] = FLOAT_MIN;
-        exp_sum_shmem[i] = 0.0;
-    }
-
-    for (var i = local_id.x; i < Q_TILE * HEAD_DIM_V; i += WG_SIZE) {
+    for (var i = local_id.x; i < HEAD_DIM_V; i += WG_SIZE) {
         o_shmem[i] = 0.0;
     }
 
     // workgroups per head/batch
-    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
+    let wg_per_head = params.seq_len_q;
     let wg_per_batch = wg_per_head * params.n_heads;
 
     let dst2_stride = HEAD_DIM_V * params.n_heads;
@@ -235,9 +273,9 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
     let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
     let v_head_offset = v_batch_offset + v_head_idx * params.stride_v2;
 
-    // starting Q row for this workgroup
+    // Vec path handles one Q row per workgroup.
     let wg_in_head = wg_in_batch % wg_per_head;
-    let q_row_start = wg_in_head * Q_TILE;
+    let q_row_start = wg_in_head;
 
 #ifdef MASK
     // mask offset
@@ -248,21 +286,18 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
     let has_bias = params.max_bias > 0.0;
     let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), has_bias);
 
-    // load q tile into shared memory
-    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
-        let q_row = elem_idx / HEAD_DIM_QK;
-        let q_col = elem_idx % HEAD_DIM_QK;
-        let head_q_row = q_row_start + q_row;
-        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
+    // load the single Q row into shared memory
+    for (var elem_idx = local_id.x; elem_idx < HEAD_DIM_QK; elem_idx += WG_SIZE) {
+        let global_q_row_offset = q_head_offset + q_row_start * params.stride_q1;
         q_shmem[elem_idx] = f16(select(
             0.0,
-            Q[global_q_row_offset + q_col],
-            head_q_row < params.seq_len_q && q_col < HEAD_DIM_QK));
+            Q[global_q_row_offset + elem_idx],
+            q_row_start < params.seq_len_q));
     }
 
     for (var kv_tile = iwg * KV_TILE; kv_tile < params.seq_len_kv; kv_tile += KV_TILE * params.nwg) {
 #ifdef BLK
-        let q_blk = q_row_start / Q_TILE;
+        let q_blk = q_row_start;
         let kv_blk = kv_tile / KV_TILE;
         let blk_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
         let blk_idx = params.blk_base + (blk_batch * params.blk_nblk1 + q_blk) * params.blk_nblk0 + kv_blk;
@@ -270,13 +305,9 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
 #else
         let blk_state_local = 1u;
 #endif
-        if (local_id.x == 0u) {
-            blk_state_wg = blk_state_local;
-        }
-        workgroupBarrier();
-        let blk_state = blk_state_wg;
+        let blk_state = blk_state_local;
         let skip_tile = blk_state == 0u;
-        for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
+        for (var elem_idx = local_id.x; elem_idx < KV_TILE; elem_idx += WG_SIZE) {
             inter_shmem[elem_idx] = f16(0.0);
         }
 
@@ -360,20 +391,14 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
         let num_of_threads = subgroup_size / VEC_NE;
         let tx = sg_inv_id % num_of_threads;
         let ty = sg_inv_id / num_of_threads;
-          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
-              let global_q_row = q_row_start + q_tile_row;
-              if (global_q_row >= params.seq_len_q) {
-                  continue;
-              }
-              let local_q_row_offset = q_tile_row * HEAD_DIM_QK;
-
+          if (subgroup_id == 0u && q_row_start < params.seq_len_q) {
               for (var kv_base : u32 = 0u; kv_base < KV_TILE; kv_base += VEC_NE) {
                   let kv_idx = kv_base + ty;
                   var partial_sum: f32 = 0.0;
                   let kv_valid = kv_idx < KV_TILE && (kv_tile + kv_idx) < params.seq_len_kv;
                   if (kv_valid) {
                     for (var i = tx; i < (HEAD_DIM_QK / 4u); i += num_of_threads) {
-                        let q_off = local_q_row_offset + i * 4u;
+                        let q_off = i * 4u;
 
                         let qv = vec4<f32>(
                             f32(q_shmem[q_off + 0u]),
@@ -410,8 +435,7 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
 
                   let sum_bcast = subgroupShuffle(sum, num_of_threads * ty);
                   if (tx == 0u && kv_valid) {
-                      let dst_idx = q_tile_row * KV_TILE + kv_idx;
-                      inter_shmem[dst_idx] = f16(sum_bcast);
+                      inter_shmem[kv_idx] = f16(sum_bcast);
                   }
               }
           }
@@ -422,13 +446,10 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
       let apply_mask = !skip_tile && (blk_state != 2u);
       if (apply_mask) {
           // load mask tile into shared memory for this KV block
-          for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
-              let mask_row = elem_idx / KV_TILE;
-              let mask_col = elem_idx % KV_TILE;
-              let global_q_row = q_row_start + mask_row;
-              let global_k_col = kv_tile + mask_col;
-              let mask_in_bounds = global_q_row < params.seq_len_q && global_k_col < params.seq_len_kv;
-              let mask_idx = mask_global_offset + mask_row * params.seq_len_kv + global_k_col;
+          for (var elem_idx = local_id.x; elem_idx < KV_TILE; elem_idx += WG_SIZE) {
+              let global_k_col = kv_tile + elem_idx;
+              let mask_in_bounds = q_row_start < params.seq_len_q && global_k_col < params.seq_len_kv;
+              let mask_idx = mask_global_offset + global_k_col;
               mask_shmem[elem_idx] = select(0.0, mask[mask_idx], mask_in_bounds);
           }
       }
@@ -439,50 +460,40 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
       workgroupBarrier();
 
       // online softmax
-      if (!skip_tile) {
-          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
-              let global_q_row = q_row_start + q_tile_row;
-              if (global_q_row >= params.seq_len_q) {
-                  break;
-              }
-
-              var prev_max = row_max_shmem[q_tile_row];
-              var final_max = prev_max;
-              // pass 1: compute final max across the full KV tile in chunks
-              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
-                  let kv_idx = kv_offset + sg_inv_id;
-                  let kv_valid = kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE;
-                  let softmax_term = select(FLOAT_MIN,
-                                            calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask),
-                                            kv_valid);
-                  final_max = subgroupMax(max(final_max, softmax_term));
-              }
+      if (!skip_tile && subgroup_id == 0u && q_row_start < params.seq_len_q) {
+          var prev_max = row_max;
+          var final_max = prev_max;
+          // pass 1: compute final max across the full KV tile in chunks
+          for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
+              let kv_idx = kv_offset + sg_inv_id;
+              let kv_valid = kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE;
+              let softmax_term = select(FLOAT_MIN,
+                                        calc_softmax_term(kv_idx, slope, has_bias, apply_mask),
+                                        kv_valid);
+              final_max = subgroupMax(max(final_max, softmax_term));
+          }
 
-              var total_exp_term: f32 = 0.0;
-              // pass 2: compute exp sum and write P using final_max
-              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
-                  let kv_idx = kv_offset + sg_inv_id;
-                  let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask);
-                  let cur_p = select(0.0,
-                                     exp(softmax_term - final_max),
-                                     kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE);
-                  total_exp_term += subgroupAdd(cur_p);
-                  if (kv_idx < KV_TILE) {
-                      inter_shmem[kv_idx + q_tile_row * KV_TILE] = f16(cur_p);
-                  }
+          var total_exp_term: f32 = 0.0;
+          // pass 2: compute exp sum and write P using final_max
+          for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
+              let kv_idx = kv_offset + sg_inv_id;
+              let softmax_term = calc_softmax_term(kv_idx, slope, has_bias, apply_mask);
+              let cur_p = select(0.0,
+                                 exp(softmax_term - final_max),
+                                 kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE);
+              total_exp_term += subgroupAdd(cur_p);
+              if (kv_idx < KV_TILE) {
+                  inter_shmem[kv_idx] = f16(cur_p);
               }
+          }
 
-              let cur_exp = exp(prev_max - final_max);
+          let cur_exp = exp(prev_max - final_max);
 
-              if (sg_inv_id == 0) {
-                  row_max_shmem[q_tile_row] = final_max;
-                  exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * cur_exp + total_exp_term;
-              }
+          row_max = final_max;
+          exp_sum = exp_sum * cur_exp + total_exp_term;
 
-              for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
-                  let idx = q_tile_row * HEAD_DIM_V + elem_idx;
-                  o_shmem[idx] = f16(f32(o_shmem[idx]) * cur_exp);
-              }
+          for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
+              o_shmem[elem_idx] = f16(f32(o_shmem[elem_idx]) * cur_exp);
           }
       }
 
@@ -562,15 +573,13 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
       workgroupBarrier();
 
       if (!skip_tile) {
-          // we have P (Q_TILE x KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
+          // we have P (KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
           // we want to compute O += P * V across the full KV tile
           let ne_threads : u32 = VEC_NE;
           let nl_threads = max(1u, subgroup_size / ne_threads);
           let tx_pv = sg_inv_id % nl_threads;
           let ty_pv = sg_inv_id / nl_threads;
-          for (var q_tile_row = subgroup_id;
-               q_tile_row < Q_TILE;
-               q_tile_row += num_subgroups) {
+          if (subgroup_id == 0u && q_row_start < params.seq_len_q) {
               for (var vec_col = tx_pv; vec_col < (HEAD_DIM_V / 4u); vec_col += nl_threads) {
                   var lo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                   for (var cc = 0u; cc < KV_TILE / ne_threads; cc += 1u) {
@@ -580,7 +589,7 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
                           continue;
                       }
 
-                      let p = f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]);
+                      let p = f32(inter_shmem[kv_idx]);
 #ifdef KV_DIRECT
                       let v_idx = v_head_offset + v_row * params.stride_v1 + vec_col * 4u;
                       let v4 = vec4<f32>(V[v_idx >> 2u]);
@@ -621,11 +630,10 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
 
                   if (ty_pv == 0u) {
                       let elem_base = vec_col * 4u;
-                      let o_base_idx = q_tile_row * HEAD_DIM_V + elem_base;
-                      o_shmem[o_base_idx + 0u] = f16(f32(o_shmem[o_base_idx + 0u]) + lo_x);
-                      o_shmem[o_base_idx + 1u] = f16(f32(o_shmem[o_base_idx + 1u]) + lo_y);
-                      o_shmem[o_base_idx + 2u] = f16(f32(o_shmem[o_base_idx + 2u]) + lo_z);
-                      o_shmem[o_base_idx + 3u] = f16(f32(o_shmem[o_base_idx + 3u]) + lo_w);
+                      o_shmem[elem_base + 0u] = f16(f32(o_shmem[elem_base + 0u]) + lo_x);
+                      o_shmem[elem_base + 1u] = f16(f32(o_shmem[elem_base + 1u]) + lo_y);
+                      o_shmem[elem_base + 2u] = f16(f32(o_shmem[elem_base + 2u]) + lo_z);
+                      o_shmem[elem_base + 3u] = f16(f32(o_shmem[elem_base + 3u]) + lo_w);
                   }
               }
           }
@@ -637,70 +645,46 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
 
 #ifdef SINKS
     // Sinks are global terms and must be applied exactly once across split workgroups.
-    if (iwg == 0u) {
-        for (var q_tile_row = subgroup_id;
-             q_tile_row < Q_TILE;
-             q_tile_row += num_subgroups) {
-                let global_q_row = q_row_start + q_tile_row;
-                if (global_q_row >= params.seq_len_q) {
-                    break;
-                }
-
-                var prev_max = row_max_shmem[q_tile_row];
-
-                // for non-sink threads, exp(FLOAT_MIN) effectively zeroes out their contribution to the sum
-                let sink_val = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0);
-                let new_max = subgroupMax(max(prev_max, sink_val));
-                let max_exp = exp(prev_max - new_max);
-                let sink_exp = exp(sink_val - new_max);
-
-                let sink_exp_sum = subgroupAdd(sink_exp);
-
-                if (sg_inv_id == 0) {
-                    row_max_shmem[q_tile_row] = new_max;
-                    exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * max_exp + sink_exp_sum;
-                }
-
-            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
-                let idx = q_tile_row * HEAD_DIM_V + elem_idx;
-                o_shmem[idx] = f16(f32(o_shmem[idx]) * max_exp);
-            }
+    if (iwg == 0u && subgroup_id == 0u && q_row_start < params.seq_len_q) {
+        var prev_max = row_max;
+
+        // for non-sink threads, exp(FLOAT_MIN) effectively zeroes out their contribution to the sum
+        let sink_val = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0u);
+        let new_max = subgroupMax(max(prev_max, sink_val));
+        let max_exp = exp(prev_max - new_max);
+        let sink_exp = exp(sink_val - new_max);
+
+        let sink_exp_sum = subgroupAdd(sink_exp);
+
+        row_max = new_max;
+        exp_sum = exp_sum * max_exp + sink_exp_sum;
+
+        for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
+            o_shmem[elem_idx] = f16(f32(o_shmem[elem_idx]) * max_exp);
         }
-        workgroupBarrier();
     }
+    workgroupBarrier();
 #endif
     let rows_per_batch = params.n_heads * params.seq_len_q;
-    for (var q_tile_row = subgroup_id;
-         q_tile_row < Q_TILE;
-         q_tile_row += num_subgroups) {
-
-        let global_q_row = q_row_start + q_tile_row;
-        if (global_q_row >= params.seq_len_q) { break; }
-
+    if (subgroup_id == 0u && q_row_start < params.seq_len_q) {
         if (params.nwg == 1u) {
-            let exp_sum = exp_sum_shmem[q_tile_row];
             let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
-            let row_base: u32 =
-                params.offset_dst + batch_idx * dst3_stride + global_q_row * dst2_stride + head_idx * HEAD_DIM_V;
+            let row_base: u32 = params.offset_dst + batch_idx * dst3_stride + q_row_start * dst2_stride +
+                                head_idx * HEAD_DIM_V;
 
             for (var elem_base = sg_inv_id * 4u; elem_base < HEAD_DIM_V; elem_base += subgroup_size * 4u) {
-                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
-                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
-                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
-                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);
-
                 let v = vec4<f32>(
-                    f32(o_shmem[i0]) * scale,
-                    f32(o_shmem[i1]) * scale,
-                    f32(o_shmem[i2]) * scale,
-                    f32(o_shmem[i3]) * scale
+                    f32(o_shmem[elem_base + 0u]) * scale,
+                    f32(o_shmem[elem_base + 1u]) * scale,
+                    f32(o_shmem[elem_base + 2u]) * scale,
+                    f32(o_shmem[elem_base + 3u]) * scale
                 );
 
                 let dst_vec_index: u32 = (row_base + elem_base) >> 2u;
                 dst[dst_vec_index] = v;
             }
         } else {
-            let rid = batch_idx * rows_per_batch + head_idx * params.seq_len_q + global_q_row;
+            let rid = batch_idx * rows_per_batch + head_idx * params.seq_len_q + q_row_start;
             let tmp_row_data_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg) + iwg * HEAD_DIM_V;
             let tmp_row_stats_base = params.tmp_stats_base + rid * (2u * params.nwg) + 2u * iwg;
 
@@ -708,21 +692,16 @@ fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
                 elem_base < HEAD_DIM_V;
                 elem_base += subgroup_size * 4u) {
 
-                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
-                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
-                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
-                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);
-
                 let tbase = tmp_row_data_base + elem_base;
-                tmp[tbase + 0u] = f32(o_shmem[i0]);
-                tmp[tbase + 1u] = f32(o_shmem[i1]);
-                tmp[tbase + 2u] = f32(o_shmem[i2]);
-                tmp[tbase + 3u] = f32(o_shmem[i3]);
+                tmp[tbase + 0u] = f32(o_shmem[elem_base + 0u]);
+                tmp[tbase + 1u] = f32(o_shmem[elem_base + 1u]);
+                tmp[tbase + 2u] = f32(o_shmem[elem_base + 2u]);
+                tmp[tbase + 3u] = f32(o_shmem[elem_base + 3u]);
             }
 
             if (sg_inv_id == 0u) {
-                tmp[tmp_row_stats_base + 0u] = exp_sum_shmem[q_tile_row];
-                tmp[tmp_row_stats_base + 1u] = row_max_shmem[q_tile_row];
+                tmp[tmp_row_stats_base + 0u] = exp_sum;
+                tmp[tmp_row_stats_base + 1u] = row_max;
             }
         }
     }
diff --git src/ggml-webgpu/wgsl-shaders/get_rows.wgsl src/ggml-webgpu/wgsl-shaders/get_rows.wgsl
index 1415798f..5710cd35 100644
--- src/ggml-webgpu/wgsl-shaders/get_rows.wgsl
+++ src/ggml-webgpu/wgsl-shaders/get_rows.wgsl
@@ -27,6 +27,24 @@ fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
 }
 #endif
 
+#ifdef Q1_0
+fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
+    let block_byte_base = (src_base + offset) * 18;
+    let d = load_f16_as_f32_at_src(block_byte_base);
+    for (var j: u32 = 0u; j < 4u; j++) {
+        let q_packed = load_u32_at_src(block_byte_base + 2u + j * 4u);
+        let dst_base128 = dst_base + offset * 128u + j * 32u;
+        for (var k: u32 = 0; k < 4u; k++) {
+            let q_byte = get_byte(q_packed, k);
+            for (var bit: u32 = 0; bit < 8u; bit++) {
+                let w = select(-d, d, ((q_byte >> bit) & 1u) != 0u);
+                dst[dst_base128 + k * 8u + bit] = w;
+            }
+        }
+    }
+}
+#endif
+
 #ifdef Q4_0
 fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
     let block_byte_base = (src_base + offset) * 18; // Block stride: 18 bytes
diff --git src/ggml-webgpu/wgsl-shaders/im2col.wgsl src/ggml-webgpu/wgsl-shaders/im2col.wgsl
new file mode 100644
index 00000000..386ebab8
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/im2col.wgsl
@@ -0,0 +1,101 @@
+#include "common_decls.tmpl"
+enable f16;
+
+@group(0) @binding(0)
+#if defined(INPUT_F32)
+var<storage, read_write> input: array<f32>;
+#elif defined(INPUT_F16)
+var<storage, read_write> input: array<f16>;
+#endif
+
+@group(0) @binding(1)
+#if defined(OUTPUT_F32)
+var<storage, read_write> output: array<f32>;
+#elif defined(OUTPUT_F16)
+var<storage, read_write> output: array<f16>;
+#endif
+
+struct Params {
+    offset_i: u32,
+    offset_o: u32,
+
+    // element strides
+    si0: u32, si1: u32, si2: u32, si3: u32,
+    so0: u32, so1: u32, so2: u32, so3: u32,
+
+    KW: u32, KH: u32, IC: u32,
+    IW: u32, IH: u32, N: u32,
+    OW: u32, OH: u32,
+
+    // stride
+    s0: u32, s1: u32,
+    // padding
+    p0: u32, p1: u32,
+    // dilation
+    d0: u32, d1: u32,
+}
+
+@group(0) @binding(2)
+var<uniform> params: Params;
+
+fn load_input(idx: u32) -> f32 {
+    #if defined(INPUT_F32)
+        return input[idx];
+    #elif defined(INPUT_F16)
+        return f32(input[idx]);
+    #endif
+}
+
+fn store_output(idx: u32, val: f32) {
+    #if defined(OUTPUT_F32)
+        output[idx] = val;
+    #elif defined(OUTPUT_F16)
+        output[idx] = f16(val);
+    #endif
+}
+
+@compute @workgroup_size(WG_SIZE)
+fn main(
+    @builtin(global_invocation_id) gid: vec3<u32>,
+    @builtin(num_workgroups) num_wg: vec3<u32>
+) {
+
+    let threads_per_group = u32(WG_SIZE);
+    let i_out = gid.x + (num_wg.x * threads_per_group) * gid.y;
+    let K = params.KW * params.KH * params.IC;
+    let M = params.OW * params.OH;
+    let total = K * M * params.N;
+
+    if (i_out >= total) {
+        return;
+    }
+
+    // decode (k, m, n)
+    var i = i_out;
+    let n = i / (K * M);
+    i = i % (K * M);
+    let m = i / K;
+    let k = i % K;
+
+    // decode (oh, ow)
+    let oh = m / params.OW;
+    let ow = m % params.OW;
+
+    // decode (kw, kh, ic)
+    let kw = k % params.KW;
+    let tmp = k / params.KW;
+    let kh = tmp % params.KH;
+    let ic = tmp / params.KH;
+
+    let iw_i32 = i32(ow * params.s0 + kw * params.d0) - i32(params.p0);
+    let ih_i32 = i32(oh * params.s1 + kh * params.d1) - i32(params.p1);
+
+    if (iw_i32 >= 0 && iw_i32 < i32(params.IW) && ih_i32 >= 0 && ih_i32 < i32(params.IH)) {
+        let iw = u32(iw_i32);
+        let ih = u32(ih_i32);
+        let in_idx = params.offset_i + iw * params.si0 + ih * params.si1 + ic * params.si2 + n * params.si3;
+        store_output(params.offset_o + k * params.so0 + ow * params.so1 + oh * params.so2 + n * params.so3, load_input(in_idx));
+    } else {
+        store_output(params.offset_o + k * params.so0 + ow * params.so1 + oh * params.so2 + n * params.so3, 0.0);
+    }
+}
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl
index 5a323818..15b22c4f 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl
@@ -61,6 +61,39 @@ fn init_shmem_src1(thread_id: u32, batch_offset: u32, offset_n: u32, k_outer: u3
 #endif // INIT_SRC1_SHMEM_FLOAT
 #endif
 
+#ifdef INIT_SRC0_SHMEM_Q1_0
+const BLOCK_SIZE = 128u;
+const BLOCK_SIZE_BYTES = 18u;
+const NQ = 8u; // 8 weights (1 byte of qs) per thread per iteration
+
+fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u32) {
+    for (var i = thread_id * NQ; i < TILE_SRC0_SHMEM; i += TOTAL_WORKGROUP_SIZE * NQ) {
+        let tile_m = i / TILE_K;
+        let tile_k_start = i % TILE_K;
+        let global_m = offset_m + tile_m;
+        let global_k_start = k_outer + tile_k_start;
+
+        if (global_m >= params.m) {
+            break;
+        }
+
+        let block_k = global_k_start / BLOCK_SIZE;
+        let byte_in_block = (global_k_start % BLOCK_SIZE) / 8u;
+        let src0_idx = batch_offset + global_m * params.stride_01 + block_k;
+        let block_byte_base = src0_idx * BLOCK_SIZE_BYTES;
+        let d = load_f16_at_src0(block_byte_base);
+        let q_byte = load_u32_at_src0(block_byte_base + 2u + byte_in_block) & 0xFFu;
+
+        for (var bit = 0u; bit < NQ; bit++) {
+            let global_k = global_k_start + bit;
+            if (global_k < params.k) {
+                shmem[i + bit] = select(-d, d, ((q_byte >> bit) & 1u) != 0u);
+            }
+        }
+    }
+}
+#endif // INIT_SRC0_SHMEM_Q1_0
+
 #ifdef INIT_SRC0_SHMEM_Q4_0
 const BLOCK_SIZE = 32u;
 const BLOCK_SIZE_BYTES = 18u;
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl
index 97c9f6d7..a8000439 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl
@@ -128,6 +128,38 @@ fn main(
     }
 #endif
 
+#ifdef MUL_ACC_Q1_0
+#define BLOCK_SIZE 128
+#define BLOCK_SIZE_BYTES 18
+#define THREADS_PER_BLOCK 16
+#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)
+
+    let num_blocks = params.k / BLOCK_SIZE;
+    let thread_within_block = thread_id % THREADS_PER_BLOCK;
+    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
+        var x_block: array<f32, ELEMS_PER_THREAD>;
+        for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let q_byte = load_u32_at_src0(block_byte_base + 2u + thread_within_block) & 0xFFu;
+                var row_sum = 0.0;
+                for (var bit = 0u; bit < 8u; bit++) {
+                    let w = select(-d, d, ((q_byte >> bit) & 1u) != 0u);
+                    row_sum += w * x_block[bit];
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
 #ifdef MUL_ACC_Q4_0
 #define BLOCK_SIZE 32
 #define BLOCK_SIZE_BYTES 18
@@ -812,6 +844,520 @@ fn main(
     }
 #endif
 
+#ifdef MUL_ACC_IQ1_S
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 50
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+
+                let d     = f32(load_f16_at_src0(block_byte_base));
+                let qh    = load_u32_at_src0(block_byte_base + 34u + sub_blk * 2u) & 0xFFFFu;
+                let dl    = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
+                let delta = select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x8000u) != 0u);
+                let qs_w  = load_u32_at_src0(block_byte_base + 2u + sub_blk * 4u);
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l       = slot0 + ll;
+                    let qs_byte = get_byte(qs_w, l);
+                    let ig      = (qs_byte | (((qh >> (3u * l)) & 7u) << 8u)) * 8u;
+                    let gw      = iq1_grid[ig / 16u];
+                    let bit_base = (ig % 16u) * 2u;
+                    for (var j = 0u; j < 8u; j++) {
+                        let g  = (gw >> (bit_base + j * 2u)) & 3u;
+                        let gs = select(f32(g), f32(g) - 4.0, (g & 2u) != 0u);
+                        row_sum += dl * (gs + delta) * x_block[ll * 8u + j];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ1_M
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 56
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+
+                let sc_lo = load_u32_at_src0(block_byte_base + 48u);
+                let sc_hi = load_u32_at_src0(block_byte_base + 52u);
+                let sc0 = sc_lo & 0xFFFFu;
+                let sc1 = (sc_lo >> 16u) & 0xFFFFu;
+                let sc2 = sc_hi & 0xFFFFu;
+                let sc3 = (sc_hi >> 16u) & 0xFFFFu;
+                let d_bits = (sc0 >> 12u) | ((sc1 >> 8u) & 0xF0u) | ((sc2 >> 4u) & 0xF00u) | (sc3 & 0xF000u);
+                let d = f32(bitcast<vec2<f16>>(d_bits)[0]);
+
+                let sc_u16 = select(select(sc2, sc3, sub_blk >= 6u),
+                                    select(sc0, sc1, sub_blk >= 2u),
+                                    sub_blk < 4u);
+
+                let qs_w = load_u32_at_src0(block_byte_base + sub_blk * 4u);
+                let qh = load_u32_at_src0(block_byte_base + 32u + sub_blk * 2u) & 0xFFFFu;
+                let qh_lo = qh & 0xFFu;
+                let qh_hi = (qh >> 8u) & 0xFFu;
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l = slot0 + ll;
+                    let bit_off = 6u * (sub_blk % 2u) + 3u * (l / 2u);
+                    let sub_scale = (sc_u16 >> bit_off) & 0x7u;
+                    let dl = d * f32(2u * sub_scale + 1u);
+                    let qh_byte = select(qh_lo, qh_hi, l >= 2u);
+                    let ll2 = l % 2u;
+                    let grid_idx = get_byte(qs_w, l) | (((qh_byte >> (4u * ll2)) & 7u) << 8u);
+                    let delta = select(IQ1_DELTA, -IQ1_DELTA, ((qh_byte >> (3u + 4u * ll2)) & 1u) != 0u);
+                    let ig = grid_idx * 8u;
+                    let gw = iq1_grid[ig / 16u];
+                    let bit_base = (ig % 16u) * 2u;
+                    for (var j = 0u; j < 8u; j++) {
+                        let g  = (gw >> (bit_base + j * 2u)) & 3u;
+                        let gs = select(f32(g), f32(g) - 4.0, (g & 2u) != 0u);
+                        row_sum += dl * (gs + delta) * x_block[ll * 8u + j];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ2_XXS
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 66
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let aux_lo = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u);
+                let aux_hi = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u + 4u);
+                let ls = aux_hi >> 28u;
+                let db = d * (0.5 + f32(ls)) * 0.25;
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l = slot0 + ll;
+                    let grid_idx = (aux_lo >> (8u * l)) & 0xFFu;
+                    let signs_idx = (aux_hi >> (7u * l)) & 0x7Fu;
+                    let signs = (ksigns_iq2xs[signs_idx / 4u] >> ((signs_idx % 4u) * 8u)) & 0xFFu;
+                    let gw_lo = iq2xxs_grid[grid_idx * 2u];
+                    let gw_hi = iq2xxs_grid[grid_idx * 2u + 1u];
+                    for (var j = 0u; j < 8u; j++) {
+                        let gw = select(gw_hi, gw_lo, j < 4u);
+                        let b = f32((gw >> ((j & 3u) * 8u)) & 0xFFu);
+                        let s = select(1.0, -1.0, ((signs >> j) & 1u) != 0u);
+                        row_sum += db * b * s * x_block[ll * 8u + j];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ2_XS
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 74
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let qs_lo = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u);
+                let qs_hi = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u + 4u);
+                let scales_word = load_u32_at_src0(block_byte_base + 66u + (sub_blk / 4u) * 4u);
+                let scales_byte = get_byte(scales_word, sub_blk % 4u);
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l = slot0 + ll;
+                    let qs_word = select(qs_hi, qs_lo, l < 2u);
+                    let half2 = (l % 2u) * 16u;
+                    let qs_val = (qs_word >> half2) & 0xFFFFu;
+                    let grid_idx = qs_val & 0x1FFu;
+                    let signs_idx = (qs_val >> 9u) & 0x7Fu;
+                    let sub_scale = (scales_byte >> (4u * (l / 2u))) & 0xFu;
+                    let db = d * (0.5 + f32(sub_scale)) * 0.25;
+                    let signs = (ksigns_iq2xs[signs_idx / 4u] >> ((signs_idx % 4u) * 8u)) & 0xFFu;
+                    let gw_lo = iq2xs_grid[grid_idx * 2u];
+                    let gw_hi = iq2xs_grid[grid_idx * 2u + 1u];
+                    for (var j = 0u; j < 8u; j++) {
+                        let gw = select(gw_hi, gw_lo, j < 4u);
+                        let b = f32((gw >> ((j & 3u) * 8u)) & 0xFFu);
+                        let s = select(1.0, -1.0, ((signs >> j) & 1u) != 0u);
+                        row_sum += db * b * s * x_block[ll * 8u + j];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ2_S
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 82
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let qs_w = load_u32_at_src0(block_byte_base + 2u + sub_blk * 4u);
+                let sg_w = load_u32_at_src0(block_byte_base + 34u + sub_blk * 4u);
+                let qh_word = load_u32_at_src0(block_byte_base + 66u + (sub_blk / 4u) * 4u);
+                let qh_byte = get_byte(qh_word, sub_blk % 4u);
+                let sc_word = load_u32_at_src0(block_byte_base + 74u + (sub_blk / 4u) * 4u);
+                let scales_byte = get_byte(sc_word, sub_blk % 4u);
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l = slot0 + ll;
+                    let qs_byte = get_byte(qs_w, l);
+                    let sign_byte = get_byte(sg_w, l);
+                    let grid_idx = qs_byte | (((qh_byte >> (2u * l)) & 3u) << 8u);
+                    let sub_scale = (scales_byte >> (4u * (l / 2u))) & 0xFu;
+                    let db = d * (0.5 + f32(sub_scale)) * 0.25;
+                    let gw_lo = iq2s_grid[grid_idx * 2u];
+                    let gw_hi = iq2s_grid[grid_idx * 2u + 1u];
+                    for (var j = 0u; j < 8u; j++) {
+                        let gw = select(gw_hi, gw_lo, j < 4u);
+                        let b = f32((gw >> ((j & 3u) * 8u)) & 0xFFu);
+                        let s = select(1.0, -1.0, ((sign_byte >> j) & 1u) != 0u);
+                        row_sum += db * b * s * x_block[ll * 8u + j];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ3_XXS
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 98
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let qs_lo = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u);
+                let qs_hi = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u + 4u);
+                let aux = load_u32_at_src0(block_byte_base + 66u + sub_blk * 4u);
+                let ls = aux >> 28u;
+                let db = d * (0.5 + f32(ls)) * 0.5;
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l = slot0 + ll;
+                    let qs_word = select(qs_hi, qs_lo, l < 2u);
+                    let byte_pos = (l % 2u) * 2u;
+                    let grid_idx_0 = (qs_word >> (byte_pos * 8u)) & 0xFFu;
+                    let grid_idx_1 = (qs_word >> ((byte_pos + 1u) * 8u)) & 0xFFu;
+                    let signs_idx = (aux >> (7u * l)) & 0x7Fu;
+                    let signs = (ksigns_iq2xs[signs_idx / 4u] >> ((signs_idx % 4u) * 8u)) & 0xFFu;
+                    let grid1 = iq3xxs_grid[grid_idx_0];
+                    let grid2 = iq3xxs_grid[grid_idx_1];
+                    for (var j = 0u; j < 4u; j++) {
+                        let b1 = f32((grid1 >> (j * 8u)) & 0xFFu);
+                        let b2 = f32((grid2 >> (j * 8u)) & 0xFFu);
+                        let s1 = select(1.0, -1.0, ((signs >> j) & 1u) != 0u);
+                        let s2 = select(1.0, -1.0, ((signs >> (j + 4u)) & 1u) != 0u);
+                        row_sum += db * b1 * s1 * x_block[ll * 8u + j];
+                        row_sum += db * b2 * s2 * x_block[ll * 8u + j + 4u];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ3_S
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 110
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let slot0   = half * 2u;
+    let y_offset = sub_blk * 32u + slot0 * 8u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let qs_lo = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u);
+                let qs_hi = load_u32_at_src0(block_byte_base + 2u + sub_blk * 8u + 4u);
+                let qh_word = load_u32_at_src0(block_byte_base + 66u + (sub_blk / 4u) * 4u);
+                let qh_byte = get_byte(qh_word, sub_blk % 4u);
+                let sg_w = load_u32_at_src0(block_byte_base + 74u + sub_blk * 4u);
+                let sc_word = load_u32_at_src0(block_byte_base + 106u);
+                let scales_byte = get_byte(sc_word, sub_blk / 2u);
+                let sub_scale = (scales_byte >> (4u * (sub_blk % 2u))) & 0xFu;
+                let db = d * (1.0 + 2.0 * f32(sub_scale));
+
+                var row_sum = 0.0;
+                for (var ll = 0u; ll < 2u; ll++) {
+                    let l = slot0 + ll;
+                    let qs_word = select(qs_hi, qs_lo, l < 2u);
+                    let byte_pos = (l % 2u) * 2u;
+                    let qs0 = (qs_word >> (byte_pos * 8u)) & 0xFFu;
+                    let qs1 = (qs_word >> ((byte_pos + 1u) * 8u)) & 0xFFu;
+                    let grid_idx_1 = qs0 | (((qh_byte >> (2u * l)) & 1u) << 8u);
+                    let grid_idx_2 = qs1 | (((qh_byte >> (2u * l + 1u)) & 1u) << 8u);
+                    let sign_byte = get_byte(sg_w, l);
+                    let grid1 = iq3s_grid[grid_idx_1];
+                    let grid2 = iq3s_grid[grid_idx_2];
+                    for (var j = 0u; j < 4u; j++) {
+                        let b1 = f32((grid1 >> (j * 8u)) & 0xFFu);
+                        let b2 = f32((grid2 >> (j * 8u)) & 0xFFu);
+                        let s1 = select(1.0, -1.0, ((sign_byte >> j) & 1u) != 0u);
+                        let s2 = select(1.0, -1.0, ((sign_byte >> (j + 4u)) & 1u) != 0u);
+                        row_sum += db * b1 * s1 * x_block[ll * 8u + j];
+                        row_sum += db * b2 * s2 * x_block[ll * 8u + j + 4u];
+                    }
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ4_NL
+#define BLOCK_SIZE 32
+#define BLOCK_SIZE_BYTES 18
+#define THREADS_PER_BLOCK 4
+#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)
+
+    let num_blocks = params.k / BLOCK_SIZE;
+    let thread_within_block = thread_id % THREADS_PER_BLOCK;
+    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4u;
+        var x_block: array<f32, ELEMS_PER_THREAD>;
+        for (var i = 0u; i < ELEMS_PER_THREAD / 2u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+            x_block[i + 4u] = f32(src1[x_base + i + 16u]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                var row_sum = 0.0;
+
+                let q_packed = load_u32_at_src0(block_byte_base + 2u + 4u * thread_within_block);
+                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
+                    let q_byte = get_byte(q_packed, byte_idx);
+                    let q_lo = f32(kvalues_iq4nl[q_byte & 0xFu]) * d;
+                    let q_hi = f32(kvalues_iq4nl[(q_byte >> 4u) & 0xFu]) * d;
+                    row_sum += q_lo * x_block[byte_idx];
+                    row_sum += q_hi * x_block[byte_idx + 4u];
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
+#ifdef MUL_ACC_IQ4_XS
+#define BLOCK_SIZE 256
+#define BLOCK_SIZE_BYTES 136
+#define THREADS_PER_BLOCK 16
+
+    let tid = thread_id % THREADS_PER_BLOCK;
+    let block_group = thread_id / THREADS_PER_BLOCK;
+    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;
+
+    let sub_blk = tid / 2u;
+    let half    = tid % 2u;
+    let y_offset = sub_blk * 32u + half * 16u;
+
+    let num_blocks = params.k / BLOCK_SIZE;
+
+    for (var block = block_group; block < num_blocks; block += num_block_groups) {
+        let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
+        var x_block: array<f32, 16>;
+        for (var i = 0u; i < 16u; i++) {
+            x_block[i] = f32(src1[x_base + i]);
+        }
+
+        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
+            let output_row = row_base + row;
+            if (output_row < params.m) {
+                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
+                let d = f32(load_f16_at_src0(block_byte_base));
+                let scales_h = load_u16_at_src0(block_byte_base + 2u);
+                let scales_l_word = load_u32_at_src0(block_byte_base + 4u);
+                let sl_byte = get_byte(scales_l_word, sub_blk / 2u);
+                let sl = (sl_byte >> (4u * (sub_blk % 2u))) & 0xFu;
+                let sh_bits = (scales_h >> (2u * sub_blk)) & 3u;
+                let ls = i32(sl | (sh_bits << 4u));
+                let dl = d * f32(ls - 32);
+
+                let qs_byte_off = 8u + sub_blk * 16u;
+                let q_w0 = load_u32_at_src0(block_byte_base + qs_byte_off);
+                let q_w1 = load_u32_at_src0(block_byte_base + qs_byte_off + 4u);
+                let q_w2 = load_u32_at_src0(block_byte_base + qs_byte_off + 8u);
+                let q_w3 = load_u32_at_src0(block_byte_base + qs_byte_off + 12u);
+
+                var row_sum = 0.0;
+                for (var i = 0u; i < 16u; i++) {
+                    let q_word = select(
+                        select(q_w0, q_w1, i >= 4u),
+                        select(q_w2, q_w3, i >= 12u),
+                        i >= 8u);
+                    let q_byte = get_byte(q_word, i % 4u);
+                    let nib = select(q_byte & 0xFu, (q_byte >> 4u) & 0xFu, half == 1u);
+                    row_sum += f32(kvalues_iq4nl[nib]) * dl * x_block[i];
+                }
+                acc[row] += row_sum;
+            }
+        }
+    }
+#endif
+
 #ifdef USE_SUBGROUP_REDUCTION
     for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
         let subgroup_total = subgroupAdd(acc[row]);
diff --git src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl
new file mode 100644
index 00000000..fd20a4e5
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl
@@ -0,0 +1,152 @@
+#ifdef OVERLAP
+
+@group(0) @binding(0)
+var<storage, read_write> rn_src: array<f32>;
+
+@group(0) @binding(1)
+var<storage, read_write> mul_src: array<f32>;
+
+@group(0) @binding(2)
+var<uniform> params: Params;
+
+fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
+    mul_src[dst_offset] = scale * rn_src[rn_src_offset] * mul_src[mul_src_offset];
+}
+
+#elif INPLACE
+
+@group(0) @binding(0)
+var<storage, read_write> rn_src: array<f32>;
+
+@group(0) @binding(1)
+var<storage, read_write> mul_src: array<f32>;
+
+@group(0) @binding(2)
+var<uniform> params: Params;
+
+fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
+    rn_src[dst_offset] = scale * rn_src[rn_src_offset] * mul_src[mul_src_offset];
+}
+
+#elif SRC_OVERLAP
+
+@group(0) @binding(0)
+var<storage, read_write> merged_src: array<f32>;
+
+@group(0) @binding(1)
+var<storage, read_write> dst: array<f32>;
+
+@group(0) @binding(2)
+var<uniform> params: Params;
+
+fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
+    dst[dst_offset] = scale * merged_src[rn_src_offset] * merged_src[mul_src_offset];
+}
+
+#else
+
+@group(0) @binding(0)
+var<storage, read_write> rn_src: array<f32>;
+
+@group(0) @binding(1)
+var<storage, read_write> mul_src: array<f32>;
+
+@group(0) @binding(2)
+var<storage, read_write> dst: array<f32>;
+
+@group(0) @binding(3)
+var<uniform> params: Params;
+
+fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
+    dst[dst_offset] = scale * rn_src[rn_src_offset] * mul_src[mul_src_offset];
+}
+
+#endif
+
+struct Params {
+    offset_rn_src: u32,
+    offset_mul_src: u32,
+    offset_dst: u32,
+
+    stride_rn_src1: u32,
+    stride_rn_src2: u32,
+    stride_rn_src3: u32,
+
+    stride_mul_src1: u32,
+    stride_mul_src2: u32,
+    stride_mul_src3: u32,
+
+    stride_dst1: u32,
+    stride_dst2: u32,
+    stride_dst3: u32,
+
+    mul_src_ne0: u32,
+    mul_src_ne1: u32,
+    mul_src_ne2: u32,
+    mul_src_ne3: u32,
+
+    ne0: u32,
+    ne1: u32,
+    ne2: u32,
+    ne3: u32,
+
+    eps: f32
+};
+
+var<workgroup> scratch: array<f32, WG_SIZE>;
+
+@compute @workgroup_size(WG_SIZE)
+fn main(@builtin(workgroup_id) wid: vec3<u32>,
+        @builtin(local_invocation_id) lid: vec3<u32>) {
+
+    // one thread per row
+    var i = wid.x;
+    let i3 = i / (params.ne2 * params.ne1);
+    i = i % (params.ne2 * params.ne1);
+    let i2 = i / params.ne1;
+    let i1 = i % params.ne1;
+    let i_rn_src_row = params.offset_rn_src + i3 * params.stride_rn_src3 + i2 * params.stride_rn_src2 + i1 * params.stride_rn_src1;
+    let i_mul_src_row = params.offset_mul_src + (i3 % params.mul_src_ne3) * params.stride_mul_src3 + (i2 % params.mul_src_ne2) * params.stride_mul_src2 + (i1 % params.mul_src_ne1) * params.stride_mul_src1;
+    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
+
+    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;
+
+    var sum = 0.0f;
+    var col = lid.x;
+    for (var j: u32 = 0; j < elems; j++) {
+        if (col >= params.ne0) {
+            break;
+        }
+#ifdef SRC_OVERLAP
+        sum += pow(merged_src[i_rn_src_row + col], 2.0);
+#else
+        sum += pow(rn_src[i_rn_src_row + col], 2.0);
+#endif
+        col += WG_SIZE;
+    }
+
+    scratch[lid.x] = sum;
+
+    workgroupBarrier();
+
+    var offset: u32 = WG_SIZE / 2;
+    while (offset > 0) {
+        if (lid.x < offset) {
+            scratch[lid.x] += scratch[lid.x + offset];
+        }
+        offset = offset / 2;
+        workgroupBarrier();
+    }
+    sum = scratch[0];
+
+    let scale = 1.0/sqrt(sum/f32(params.ne0) + params.eps);
+
+    col = lid.x;
+    for (var j: u32 = 0; j < elems; j++) {
+        if (col >= params.ne0) {
+            break;
+        }
+        update(i_rn_src_row + col, i_dst_row + col, scale, i_mul_src_row + col % params.mul_src_ne0);
+        col += WG_SIZE;
+    }
+}
diff --git src/ggml-webgpu/wgsl-shaders/ssm_scan.wgsl src/ggml-webgpu/wgsl-shaders/ssm_scan.wgsl
new file mode 100644
index 00000000..05761dec
--- /dev/null
+++ src/ggml-webgpu/wgsl-shaders/ssm_scan.wgsl
@@ -0,0 +1,193 @@
+#ifdef USE_SUBGROUP_REDUCTION
+enable subgroups;
+#endif
+
+struct Params {
+    offset_s: u32,
+    offset_x: u32,
+    offset_dt: u32,
+    offset_A: u32,
+    offset_B: u32,
+    offset_C: u32,
+    offset_ids: u32,
+    offset_dst: u32,
+
+    stride_s1: u32,
+    stride_s2: u32,
+    stride_s3: u32,
+
+    stride_x1: u32,
+    stride_x2: u32,
+    stride_x3: u32,
+
+    stride_dt1: u32,
+    stride_dt2: u32,
+
+    a_ne0: u32,
+    stride_A1: u32,
+
+    stride_B1: u32,
+    stride_B2: u32,
+    stride_B3: u32,
+
+    stride_C1: u32,
+    stride_C2: u32,
+    stride_C3: u32,
+
+    d_state: u32,
+    d_inner: u32,
+    n_head: u32,
+    n_group: u32,
+    n_seq_tokens: u32,
+    n_seqs: u32,
+
+    y_elems: u32,
+};
+
+@group(0) @binding(0) var<storage, read_write> s_in: array<f32>;
+#ifdef XBC_OVERLAP
+@group(0) @binding(1) var<storage, read_write> x_B_C_merged: array<f32>;
+@group(0) @binding(2) var<storage, read_write> dt: array<f32>;
+@group(0) @binding(3) var<storage, read_write> A: array<f32>;
+@group(0) @binding(4) var<storage, read_write> ids: array<i32>;
+@group(0) @binding(5) var<storage, read_write> dst: array<f32>;
+@group(0) @binding(6) var<uniform> params: Params;
+#else
+@group(0) @binding(1) var<storage, read_write> x: array<f32>;
+@group(0) @binding(2) var<storage, read_write> dt: array<f32>;
+@group(0) @binding(3) var<storage, read_write> A: array<f32>;
+@group(0) @binding(4) var<storage, read_write> B: array<f32>;
+@group(0) @binding(5) var<storage, read_write> C: array<f32>;
+@group(0) @binding(6) var<storage, read_write> ids: array<i32>;
+@group(0) @binding(7) var<storage, read_write> dst: array<f32>;
+@group(0) @binding(8) var<uniform> params: Params;
+#endif
+
+var<workgroup> shared_x_dt: array<f32, TOKENS_PER_TILE>;
+var<workgroup> shared_dtsp: array<f32, TOKENS_PER_TILE>;
+var<workgroup> shared_reduce: array<f32, TOKENS_PER_TILE * WG_SIZE>;
+
+fn reduce_base(token_in_tile: u32) -> u32 {
+    return token_in_tile * WG_SIZE;
+}
+
+@compute @workgroup_size(WG_SIZE)
+fn main(
+    @builtin(local_invocation_id) local_id: vec3<u32>,
+    @builtin(workgroup_id) wg_id: vec3<u32>,
+    @builtin(num_workgroups) num_wg: vec3<u32>
+#ifdef USE_SUBGROUP_REDUCTION
+  , @builtin(subgroup_id) subgroup_id: u32,
+    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
+    @builtin(num_subgroups) num_subgroups: u32
+#endif
+) {
+    let tid = local_id.x;
+    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
+
+    let i1 = wg_linear % params.d_inner;
+    let head_seq = wg_linear / params.d_inner;
+    let ir = head_seq % params.n_head;
+    let i3 = head_seq / params.n_head;
+
+    let state_slot = u32(ids[params.offset_ids + i3]);
+    let g = ir / (params.n_head / params.n_group);
+
+    let s_idx = params.offset_s + tid + i1 * params.stride_s1 + ir * params.stride_s2 + state_slot * params.stride_s3;
+    var s_prev = s_in[s_idx];
+
+    let A0 = A[params.offset_A + (tid % params.a_ne0) + ir * params.stride_A1];
+
+    for (var token_base = 0u; token_base < params.n_seq_tokens; token_base += TOKENS_PER_TILE) {
+        if (tid < TOKENS_PER_TILE) {
+            let token = token_base + tid;
+            if (token < params.n_seq_tokens) {
+                let x_idx = params.offset_x + i1 + ir * params.stride_x1 + token * params.stride_x2 + i3 * params.stride_x3;
+                let dt_idx = params.offset_dt + ir + token * params.stride_dt1 + i3 * params.stride_dt2;
+                let dt0 = dt[dt_idx];
+                let dtsp = select(log(1.0 + exp(dt0)), dt0, dt0 > 20.0);
+                shared_dtsp[tid] = dtsp;
+#ifdef XBC_OVERLAP
+                shared_x_dt[tid] = x_B_C_merged[x_idx] * dtsp;
+#else
+                shared_x_dt[tid] = x[x_idx] * dtsp;
+#endif
+            }
+        }
+
+        workgroupBarrier();
+
+        for (var token_in_tile = 0u; token_in_tile < TOKENS_PER_TILE; token_in_tile++) {
+            let token = token_base + token_in_tile;
+            if (token >= params.n_seq_tokens) {
+                break;
+            }
+
+            let x_dt = shared_x_dt[token_in_tile];
+            let dA = exp(shared_dtsp[token_in_tile] * A0);
+            let reduce_idx = reduce_base(token_in_tile) + tid;
+
+            let b_idx = params.offset_B + tid + g * params.stride_B1 + token * params.stride_B2 + i3 * params.stride_B3;
+            let c_idx = params.offset_C + tid + g * params.stride_C1 + token * params.stride_C2 + i3 * params.stride_C3;
+#ifdef XBC_OVERLAP
+            let s = s_prev * dA + x_B_C_merged[b_idx] * x_dt;
+#else
+            let s = s_prev * dA + B[b_idx] * x_dt;
+#endif
+            s_prev = s;
+
+#ifdef USE_SUBGROUP_REDUCTION
+#ifdef XBC_OVERLAP
+            let subgroup_partial = subgroupAdd(s * x_B_C_merged[c_idx]);
+#else
+            let subgroup_partial = subgroupAdd(s * C[c_idx]);
+#endif
+            if (subgroup_invocation_id == 0u) {
+                shared_reduce[reduce_idx - tid + subgroup_id] = subgroup_partial;
+            }
+#else
+#ifdef XBC_OVERLAP
+            shared_reduce[reduce_idx] = s * x_B_C_merged[c_idx];
+#else
+            shared_reduce[reduce_idx] = s * C[c_idx];
+#endif
+#endif
+
+            workgroupBarrier();
+
+#ifdef USE_SUBGROUP_REDUCTION
+            if (tid == 0u) {
+                var sum = 0.0;
+                for (var sg = 0u; sg < num_subgroups; sg++) {
+                    sum += shared_reduce[reduce_base(token_in_tile) + sg];
+                }
+                let y_idx =
+                    params.offset_dst + i1 + ir * params.d_inner + token * (params.n_head * params.d_inner) +
+                    i3 * (params.n_seq_tokens * params.n_head * params.d_inner);
+                dst[y_idx] = sum;
+            }
+#else
+            for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
+                if (tid < stride) {
+                    shared_reduce[reduce_idx] += shared_reduce[reduce_idx + stride];
+                }
+                workgroupBarrier();
+            }
+
+            if (tid == 0u) {
+                let y_idx =
+                    params.offset_dst + i1 + ir * params.d_inner + token * (params.n_head * params.d_inner) +
+                    i3 * (params.n_seq_tokens * params.n_head * params.d_inner);
+                dst[y_idx] = shared_reduce[reduce_base(token_in_tile)];
+            }
+#endif
+
+            workgroupBarrier();
+        }
+    }
+
+    let state_idx =
+        params.offset_dst + params.y_elems + tid + i1 * params.d_state + ir * (params.d_state * params.d_inner) +
+        i3 * (params.d_state * params.d_inner * params.n_head);
+    dst[state_idx] = s_prev;
+}
diff --git src/ggml.c src/ggml.c
index eda041f4..54d3eae3 100644
--- src/ggml.c
+++ src/ggml.c
@@ -7656,7 +7656,7 @@ size_t ggml_quantize_chunk(
                int64_t   nrows,
                int64_t   n_per_row,
            const float * imatrix) {
-    const int64_t n = (int64_t) nrows * n_per_row;
+    const int64_t n = nrows * n_per_row;
 
     if (ggml_quantize_requires_imatrix(type)) {
         GGML_ASSERT(imatrix != NULL);
@@ -7673,21 +7673,21 @@ size_t ggml_quantize_chunk(
     size_t result = 0;
 
     switch (type) {
-        case GGML_TYPE_Q1_0:    result = quantize_q1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q4_0:    result = quantize_q4_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q4_1:    result = quantize_q4_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q5_0:    result = quantize_q5_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q5_1:    result = quantize_q5_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q8_0:    result = quantize_q8_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_MXFP4:   result = quantize_mxfp4(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_NVFP4:   result = quantize_nvfp4(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q2_K:    result = quantize_q2_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q3_K:    result = quantize_q3_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q4_K:    result = quantize_q4_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q5_K:    result = quantize_q5_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_Q6_K:    result = quantize_q6_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_TQ1_0:   result = quantize_tq1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
-        case GGML_TYPE_TQ2_0:   result = quantize_tq2_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
+        case GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
         case GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
         case GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
         case GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
@@ -7752,9 +7752,9 @@ struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads) {
 }
 
 bool ggml_threadpool_params_match(const struct ggml_threadpool_params * p0, const struct ggml_threadpool_params * p1) {
-    if (p0->n_threads      != p1->n_threads  )    return false;
-    if (p0->prio           != p1->prio       )    return false;
-    if (p0->poll           != p1->poll       )    return false;
-    if (p0->strict_cpu     != p1->strict_cpu )    return false;
+    if (p0->n_threads  != p1->n_threads  ) return false;
+    if (p0->prio       != p1->prio       ) return false;
+    if (p0->poll       != p1->poll       ) return false;
+    if (p0->strict_cpu != p1->strict_cpu ) return false;
     return memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
 }
