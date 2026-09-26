diff --git src/ggml-cuda/common.cuh src/ggml-cuda/common.cuh
index 2e78ae4f..5363dd00 100644
--- src/ggml-cuda/common.cuh
+++ src/ggml-cuda/common.cuh
@@ -111,9 +111,9 @@
 #define GGML_CUDA_CC_IS_QY2(cc)      (cc >= GGML_CUDA_CC_QY2 && cc < GGML_CUDA_CC_PH1)
 #define GGML_CUDA_CC_IS_PH1(cc)      (cc >= GGML_CUDA_CC_PH1)
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
+#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
 #    define GGML_CUDA_USE_CUB
-#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
+#endif  // !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
 
 // PDL host-side support (cudaLaunchKernelEx) requires CUDART >= 11.8.
 // However, this has been bugged in CTK < 12.3 for MSVC builds, see
@@ -237,7 +237,7 @@ static const char * cu_get_error_str(CUresult err) {
 #define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
 #endif
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#if !defined(GGML_USE_HIP)
 #    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
         do {                                                                                                   \
             static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
@@ -252,7 +252,7 @@ static const char * cu_get_error_str(CUresult err) {
         do {                                             \
             GGML_UNUSED(nbytes);                         \
         } while (0)
-#endif // !(defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#endif // !defined(GGML_USE_HIP)
 
 #if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
 #define GGML_CUDA_ASSUME(x) __builtin_assume(x)
@@ -397,7 +397,7 @@ static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
 
 // Maximum number of bytes that can be copied in a single instruction.
 static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
-#ifdef GGML_USE_HIP
+#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
     return 16;
 #else
 #if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
@@ -405,7 +405,7 @@ static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
 #else
     return 8;
 #endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
-#endif // GGML_USE_HIP
+#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
 }
 
 
@@ -424,10 +424,6 @@ static __device__ void no_device_code(
     __trap();
 
     GGML_UNUSED(no_device_code); // suppress unused function warning
-
-#if defined(GGML_USE_MUSA)
-    __builtin_unreachable();
-#endif // defined(GGML_USE_MUSA)
 }
 
 #ifdef __CUDA_ARCH__
@@ -696,16 +692,11 @@ static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const hal
 
 template<int width = WARP_SIZE>
 static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
-#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
 #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
-#else
-   GGML_UNUSED(x);
-   NO_DEVICE_CODE;
-#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL || defined(GGML_USE_HIP)
 }
 
 #if (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || \
diff --git src/ggml-cuda/conv2d-dw.cu src/ggml-cuda/conv2d-dw.cu
index 7583233b..6371d0a8 100644
--- src/ggml-cuda/conv2d-dw.cu
+++ src/ggml-cuda/conv2d-dw.cu
@@ -1,4 +1,5 @@
 #include "conv2d-dw.cuh"
+#include "convert.cuh"
 
 struct conv_params {
     int in_w, in_h;
@@ -79,7 +80,7 @@ struct cwhn_layout {
 };
 
 template <typename T, typename Layout>
-__global__ void conv2d_dw_kernel(const T * __restrict__ input, const T * __restrict__ kernel, T * __restrict__ output,
+__global__ void conv2d_dw_kernel(const float * __restrict__ input, const T * __restrict__ kernel, float * __restrict__ output,
                                  const int in_w, const int in_h, const int out_w, const int out_h,
                                  const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
                                  const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
@@ -97,7 +98,7 @@ __global__ void conv2d_dw_kernel(const T * __restrict__ input, const T * __restr
     int batch_idx, channel_idx, out_y_idx, out_x_idx;
     Layout::unpack_indices(global_idx, params, batch_idx, channel_idx, out_y_idx, out_x_idx);
 
-    T accumulator = 0;
+    float accumulator = 0.0f;
     kernel_bounds bounds = calculate_kernel_bounds(out_x_idx, out_y_idx, params);
 
     for (int kern_y = bounds.y_min; kern_y < bounds.y_max; ++kern_y) {
@@ -106,10 +107,10 @@ __global__ void conv2d_dw_kernel(const T * __restrict__ input, const T * __restr
         for (int kern_x = bounds.x_min; kern_x < bounds.x_max; ++kern_x) {
             int in_x_idx = calculate_input_coord(out_x_idx, kern_x, params.stride_x, params.dilation_x, params.padding_x);
 
-            const T input_val  = input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)];
-            const T kernel_val = kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];
+            const float input_val  = input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)];
+            const T     kernel_val = kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];
 
-            accumulator += input_val * kernel_val;
+            accumulator += input_val * ggml_cuda_cast<float>(kernel_val);
         }
     }
 
@@ -120,8 +121,9 @@ void ggml_cuda_op_conv2d_dw(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
     const ggml_tensor * kernel = dst->src[0];
     const ggml_tensor * input  = dst->src[1];
 
-    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
-    const float * w_d = (const float *) kernel->data;
+    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
+    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
+    const void *  w_d = kernel->data;
     const float * x_d = (const float *) input->data;
     float *       y_d = (float *) dst->data;
 
@@ -148,13 +150,25 @@ void ggml_cuda_op_conv2d_dw(ggml_backend_cuda_context & ctx, ggml_tensor * dst)
     const int blocks = (total + CUDA_CONV2D_DW_BLOCK_SIZE - 1) / CUDA_CONV2D_DW_BLOCK_SIZE;
 
     if (ggml_is_contiguous(input)) {
-        conv2d_dw_kernel<float, whcn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
-            x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
-            dilation_x, dilation_y, channels, batches);
+        if (kernel->type == GGML_TYPE_F16) {
+            conv2d_dw_kernel<half, whcn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
+                x_d, (const half *) w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y,
+                padding_x, padding_y, dilation_x, dilation_y, channels, batches);
+        } else {
+            conv2d_dw_kernel<float, whcn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
+                x_d, (const float *) w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y,
+                padding_x, padding_y, dilation_x, dilation_y, channels, batches);
+        }
     } else if (ggml_is_contiguous_channels(input)) {
-        conv2d_dw_kernel<float, cwhn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
-            x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
-            dilation_x, dilation_y, channels, batches);
+        if (kernel->type == GGML_TYPE_F16) {
+            conv2d_dw_kernel<half, cwhn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
+                x_d, (const half *) w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y,
+                padding_x, padding_y, dilation_x, dilation_y, channels, batches);
+        } else {
+            conv2d_dw_kernel<float, cwhn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
+                x_d, (const float *) w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y,
+                padding_x, padding_y, dilation_x, dilation_y, channels, batches);
+        }
     } else {
         GGML_ABORT("Unsupported memory layout for conv_2d_dw");
     }
diff --git src/ggml-cuda/fattn-mma-f16.cuh src/ggml-cuda/fattn-mma-f16.cuh
index 449a77c5..abb99a35 100644
--- src/ggml-cuda/fattn-mma-f16.cuh
+++ src/ggml-cuda/fattn-mma-f16.cuh
@@ -2091,26 +2091,22 @@ void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml
             constexpr bool use_sparse_kernel = false;
             fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>;
 
-#if !defined(GGML_USE_MUSA)
             static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
             if (!shared_memory_limit_raised[id]) {
                 CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                 shared_memory_limit_raised[id] = true;
             }
-#endif // !defined(GGML_USE_MUSA)
         }
     } else {
         constexpr bool use_logit_softcap = true;
         constexpr bool use_sparse_kernel = false;
         fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel>;
 
-#if !defined(GGML_USE_MUSA)
         static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
         if (!shared_memory_limit_raised[id]) {
             CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
             shared_memory_limit_raised[id] = true;
         }
-#endif // !defined(GGML_USE_MUSA)
     }
 
     launch_fattn<DV, ncols1, ncols2>
diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index 9dd34c86..e76ff312 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -310,13 +310,9 @@ static ggml_cuda_device_info ggml_cuda_init() {
         info.devices[id].smpb       = prop.sharedMemPerBlock;
         info.devices[id].warp_size  = prop.warpSize;
 
-#ifndef GGML_USE_MUSA
         int supports_coop_launch = 0;
         CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
         info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
-#else
-        info.devices[id].supports_cooperative_launch = false;
-#endif // !(GGML_USE_MUSA)
 
 #if defined(GGML_USE_HIP)
         info.devices[id].smpbo = prop.sharedMemPerBlock;
@@ -337,8 +333,6 @@ static ggml_cuda_device_info ggml_cuda_init() {
                       device_vmm ? "yes" : "no", prop.warpSize,
                       device_vram_mib);
 #elif defined(GGML_USE_MUSA)
-        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
-        info.devices[id].warp_size = 32;
         info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
         info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
         info.devices[id].cc += prop.minor * 0x10;
@@ -3316,6 +3310,19 @@ static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
         return true;
     }
 
+    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
+        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
+        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];
+
+        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
+        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
+
+        float bias;
+        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));
+
+        return bias == 0.0f && scale->type == GGML_TYPE_F32;
+    }
+
     if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
      && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
         const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
@@ -4157,6 +4164,11 @@ static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph
         return 1;
     }
 
+    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
+        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
+        return 1;
+    }
+
     if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
         ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
         return 2;
@@ -5521,7 +5533,8 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
                    op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
         case GGML_OP_CONV_2D_DW:
-            return op->src[0]->type == GGML_TYPE_F32;
+            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
+                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
         case GGML_OP_CONV_TRANSPOSE_2D:
         case GGML_OP_POOL_1D:
         case GGML_OP_POOL_2D:
@@ -5562,12 +5575,7 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
         case GGML_OP_RWKV_WKV7:
             return true;
         case GGML_OP_GATED_DELTA_NET:
-            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
-#ifdef GGML_USE_MUSA
-            return false;
-#else
             return true;
-#endif // GGML_USE_MUSA
         case GGML_OP_DSV4_HC_COMB:
             return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                 op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
diff --git src/ggml-cuda/mmq-load-tiles.cuh src/ggml-cuda/mmq-load-tiles.cuh
index 7f00bad9..e19f4f24 100644
--- src/ggml-cuda/mmq-load-tiles.cuh
+++ src/ggml-cuda/mmq-load-tiles.cuh
@@ -1631,16 +1631,16 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
 template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4_fp4(
         const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
     constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
-    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
-    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
-    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
+    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q4) / warp_size;
+    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q4);
+    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q4);
 
     int *      x_qs = (int *) x_tile;
     uint32_t * x_sc = (uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
 
     const int txi = threadIdx.x;
 
-    constexpr int iter_k = ggml_cuda_mmq_get_K_vram(type, J, fallback);
+    constexpr int iter_k = ggml_cuda_mmq_get_K_vram(type, J, fallback, GGML_PREC_Q4);
 
     constexpr int threads_per_row = iter_k / QK_MXFP4;  // each thread processes 1 block
     constexpr int rows_per_warp   = warp_size / threads_per_row;
@@ -1670,12 +1670,12 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
     }
 }
 
-template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
         const char * __restrict__ x, int * __restrict__ x_tile, const int kb0, const int i_max, const int stride) {
     constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
-    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
-    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
-    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
+    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
+    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
+    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
 
 #if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
     int   * x_qs = (int   *) x_tile;
@@ -1729,12 +1729,12 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
 template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4_nvfp4(
         const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
     constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
-    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
-    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
-    constexpr int iter_k          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
+    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q4) / warp_size;
+    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q4);
+    constexpr int iter_k          = ggml_cuda_mmq_get_K_vram(type, J, fallback, GGML_PREC_Q4);
     constexpr int threads_per_row = iter_k / QK_NVFP4; // each thread processes 1 block
     constexpr int rows_per_warp   = warp_size / threads_per_row;
-    constexpr int sram_stride     = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
+    constexpr int sram_stride     = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q4);
 
     uint32_t * x_u32 = (uint32_t *) x_tile;
 
diff --git src/ggml-cuda/mmq-vec-dot.cuh src/ggml-cuda/mmq-vec-dot.cuh
index 4d1c398f..4ca6542d 100644
--- src/ggml-cuda/mmq-vec-dot.cuh
+++ src/ggml-cuda/mmq-vec-dot.cuh
@@ -474,7 +474,7 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
 }
 
 // Used for Q3_K, IQ2_S, and IQ2_XS:
-template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(
         const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
 #if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
     constexpr data_layout input_layout = get_input_data_layout();
@@ -482,7 +482,7 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
     typedef tile<16,  4, int, input_layout>        tile_B;
     typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
 
-    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
+    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
     constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
     constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
@@ -532,7 +532,7 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
     typedef tile< 8, 4, int> tile_B;
     typedef tile<16, 8, int> tile_C;
 
-    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
+    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
     constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
     constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.
 
@@ -1180,7 +1180,7 @@ template <ggml_type type, int J, bool fallback> static __device__ __forceinline_
     typedef tile<8,  8, int>   tile_B;
     typedef tile<16, 8, float> tile_C;
 
-    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
+    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q4);
     constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
     constexpr int ntx           = rows_per_warp / tile_C::I;
     constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;
diff --git src/ggml-cuda/mmq.cu src/ggml-cuda/mmq.cu
index b13b34ee..3e1721bb 100644
--- src/ggml-cuda/mmq.cu
+++ src/ggml-cuda/mmq.cu
@@ -5,7 +5,7 @@
 
 #include <cstdint>
 
-static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
+static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
     switch (args.type_x) {
         case GGML_TYPE_Q1_0:
             mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
@@ -71,9 +71,18 @@ static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, con
             break;
 // -----------------------------------------------------------------------
         case GGML_TYPE_MXFP4:
+            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
+            if (prec_src1 == GGML_PREC_Q4) {
+                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
+                break;
+            }
             mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
             break;
         case GGML_TYPE_NVFP4:
+            if (prec_src1 == GGML_PREC_Q4) {
+                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
+                break;
+            }
             mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
             break;
         default:
@@ -82,6 +91,47 @@ static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, con
     }
 }
 
+// overrides the src1 precision requested by the graph, "auto" keeps the requested one
+static ggml_prec ggml_cuda_mmq_get_prec_env() {
+    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
+    if (env_c == nullptr) {
+        return GGML_PREC_UNDEFINED;
+    }
+    std::string env_cpp = env_c;
+    for (char & c : env_cpp) {
+        c = std::tolower(c);
+    }
+    if (env_cpp == "q4") {
+        return GGML_PREC_Q4;
+    }
+    if (env_cpp == "q8") {
+        return GGML_PREC_Q8;
+    }
+    if (env_cpp != "auto") {
+        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
+    }
+    return GGML_PREC_UNDEFINED;
+}
+
+// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
+// default to the native W4A4 instructions on Blackwell.
+static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
+    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();
+
+    ggml_prec prec = prec_env;
+    if (prec == GGML_PREC_UNDEFINED) {
+        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
+    }
+
+    // Q4 only for the FP4 types on Blackwell
+    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
+    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
+    if (prec == GGML_PREC_Q8 || !can_use_q4) {
+        return GGML_PREC_Q8;
+    }
+    return GGML_PREC_Q4;
+}
+
 void ggml_cuda_mul_mat_q(
         ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
     GGML_ASSERT(        src1->type == GGML_TYPE_F32);
@@ -128,7 +178,9 @@ void ggml_cuda_mul_mat_q(
 
     const bool fallback = ne01 % 128 != 0;
 
-    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
+    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);
+
+    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
     const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
     const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;
 
@@ -172,7 +224,7 @@ void ggml_cuda_mul_mat_q(
             ne02, ne12, s02, s12, s2,
             ne03, ne13, s03, s13, s3,
             ne1, ne1};
-        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
+        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
         return;
     }
 
@@ -260,7 +312,7 @@ void ggml_cuda_mul_mat_q(
         ne03, ne13, s03, s13, s3,
         ne12, ncols_opt};
 
-    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
+    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
 }
 
 bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
@@ -389,5 +441,10 @@ bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t
         return n_experts > 0;
     }
 
+    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
+    if (cc == GGML_CUDA_CC_PH1) {
+        return false;
+    }
+
     return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
 }
diff --git src/ggml-cuda/mmq.cuh src/ggml-cuda/mmq.cuh
index 6923f351..4b50d1dc 100644
--- src/ggml-cuda/mmq.cuh
+++ src/ggml-cuda/mmq.cuh
@@ -227,7 +227,7 @@ struct ggml_cuda_mmq_config {
 
 #undef CASE
 
-static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc) {
+static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1 = GGML_PREC_Q8) {
     if (GGML_CUDA_CC_IS_AMD(cc)) {
         if (GGML_CUDA_CC_IS_GCN(cc)) {
             return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
@@ -247,6 +247,10 @@ static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type ty
         return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
     }
     if (blackwell_mma_available(cc)) {
+        // only src1 at Q4 uses the native FP4 config, higher precisions keep src1 at Q8_1
+        if (prec_src1 != GGML_PREC_Q4 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4)) {
+            return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
+        }
         return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
     }
     if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {
@@ -258,7 +262,7 @@ static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type ty
     return ggml_cuda_mmq_get_config_pascal_older(type, J, fallback);
 }
 
-static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback) {
+static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
 #ifdef GGML_USE_HIP
 #ifdef GCN
     return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
@@ -275,6 +279,10 @@ static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_t
 #endif // CDNA
 #else
 #ifdef BLACKWELL_MMA_AVAILABLE
+    // only src1 at Q4 uses the native FP4 config, higher precisions keep src1 at Q8_1
+    if (prec_src1 != GGML_PREC_Q4 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4)) {
+        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
+    }
     return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
 #elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
     return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
@@ -284,79 +292,71 @@ static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_t
     return ggml_cuda_mmq_get_config_pascal_older(type, J, fallback);
 #endif // BLACKWELL_MMA_AVAILABLE
 #endif // GGML_USE_HIP
-    GGML_UNUSED_VARS(type, J, fallback);
+    GGML_UNUSED_VARS(type, J, fallback, prec_src1);
 }
 
 static __host__ int ggml_cuda_mmq_get_type(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).type;
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_type(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).type;
-}
-
-static __host__ int ggml_cuda_mmq_get_nthreads(const ggml_type type, const int J, const bool fallback, const int cc) {
-    return ggml_cuda_mmq_get_config(type, J, fallback, cc).nthreads;
-}
-
-static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).nthreads;
+static constexpr __device__ int ggml_cuda_mmq_get_type(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type;
 }
 
-static __host__ int ggml_cuda_mmq_get_occupancy(const ggml_type type, const int J, const bool fallback, const int cc) {
-    return ggml_cuda_mmq_get_config(type, J, fallback, cc).occupancy;
+static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).nthreads;
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_occupancy(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).occupancy;
+static constexpr __device__ int ggml_cuda_mmq_get_occupancy(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).occupancy;
 }
 
 static __host__ int ggml_cuda_mmq_get_I(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).I;
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).I;
+static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).I;
 }
 
 static __host__ int ggml_cuda_mmq_get_J(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).J;
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_J(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).J;
+static constexpr __device__ int ggml_cuda_mmq_get_J(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).J;
 }
 
 static __host__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).sram_layout;
 }
 
-static constexpr __device__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).sram_layout;
+static constexpr __device__ ggml_cuda_mmq_sram_layout ggml_cuda_mmq_get_sram_layout(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).sram_layout;
 }
 
 static __host__ int ggml_cuda_mmq_get_K_vram(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).K_vram;
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_K_vram(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).K_vram;
+static constexpr __device__ int ggml_cuda_mmq_get_K_vram(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).K_vram;
 }
 
 static __host__ bool ggml_cuda_mmq_get_stream_k(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).stream_k;
 }
 
-static constexpr __device__ bool ggml_cuda_mmq_get_stream_k(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).stream_k;
+static constexpr __device__ bool ggml_cuda_mmq_get_stream_k(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).stream_k;
 }
 
 static __host__ int ggml_cuda_mmq_get_fallback(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).fallback;
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_fallback(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_config(type, J, fallback).fallback;
+static constexpr __device__ int ggml_cuda_mmq_get_fallback(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).fallback;
 }
 
 // ---------------------------------------------------------------------------------------------
@@ -365,8 +365,8 @@ static __host__ int ggml_cuda_mmq_get_sram_stride(const ggml_type type, const in
     return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, cc));
 }
 
-static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int J, bool fallback) {
-    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback));
+static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8) {
+    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, prec_src1));
 }
 
 static __host__ int ggml_cuda_mmq_get_J_max(const ggml_type type, const bool fallback, const int cc, const int64_t ne11) {
@@ -541,9 +541,9 @@ struct ggml_cuda_mmq_util_funcs {
         vdr(vdr), load_tiles(load_tiles), vec_dot(vec_dot), write_back(write_back) {}
 };
 
-template <ggml_type type, int J, bool fallback>
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_funcs() {
-    if (!ggml_cuda_mmq_get_config(type, J, fallback).use_mma_data_layout()) {
+    if (!ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).use_mma_data_layout()) {
         switch (type) {
             case GGML_TYPE_Q1_0:
                 return ggml_cuda_mmq_util_funcs(
@@ -690,17 +690,23 @@ static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_func
 #ifdef BLACKWELL_MMA_AVAILABLE
     switch (type) {
         case GGML_TYPE_MXFP4:
-            return ggml_cuda_mmq_util_funcs(
-                -1,
-                ggml_cuda_mmq_load_tiles_mxfp4_fp4<type, J, fallback>,
-                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
-                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
+            if (prec_src1 == GGML_PREC_Q4) {
+                return ggml_cuda_mmq_util_funcs(
+                    -1,
+                    ggml_cuda_mmq_load_tiles_mxfp4_fp4<type, J, fallback>,
+                    ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
+                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
+            }
+            break;
         case GGML_TYPE_NVFP4:
-            return ggml_cuda_mmq_util_funcs(
-                -1,
-                ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
-                ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
-                ggml_cuda_mmq_write_back_mma<type, J, fallback>);
+            if (prec_src1 == GGML_PREC_Q4) {
+                return ggml_cuda_mmq_util_funcs(
+                    -1,
+                    ggml_cuda_mmq_load_tiles_nvfp4_nvfp4<type, J, fallback>,
+                    ggml_cuda_mmq_vec_dot_fp4_fp4_mma<type, J, fallback>,
+                    ggml_cuda_mmq_write_back_mma<type, J, fallback>);
+            }
+            break;
         default:
             break;
     }
@@ -841,37 +847,37 @@ static constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_func
         case GGML_TYPE_NVFP4:
             return ggml_cuda_mmq_util_funcs(
                 -1,
-                ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback>,
-                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,
+                ggml_cuda_mmq_load_tiles_nvfp4<type, J, fallback, prec_src1>,
+                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,
                 ggml_cuda_mmq_write_back_mma<type, J, fallback>);
         default:
             return ggml_cuda_mmq_util_funcs(1, nullptr, nullptr, nullptr);
     }
 }
 
-template <ggml_type type, int J, bool fallback>
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 static constexpr __device__ int ggml_cuda_mmq_get_vdr() {
-    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().vdr;
+    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().vdr;
 }
 
-template <ggml_type type, int J, bool fallback>
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 static constexpr __device__ ggml_cuda_mmq_load_tiles_t ggml_cuda_mmq_get_load_tiles() {
-    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().load_tiles;
+    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().load_tiles;
 }
 
-template <ggml_type type, int J, bool fallback>
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 static constexpr __device__ ggml_cuda_mmq_vec_dot_t ggml_cuda_mmq_get_vec_dot() {
-    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().vec_dot;
+    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().vec_dot;
 }
 
-template <ggml_type type, int J, bool fallback>
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 static constexpr __device__ ggml_cuda_mmq_write_back_t ggml_cuda_mmq_get_write_back() {
-    return ggml_cuda_mmq_get_util_funcs<type, J, fallback>().write_back;
+    return ggml_cuda_mmq_get_util_funcs<type, J, fallback, prec_src1>().write_back;
 }
 
 // ---------------------------------------------------------------------------------------------
 
-template <ggml_type type, int J, bool fallback, bool fixup>
+template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1 = GGML_PREC_Q8>
 static __device__ __forceinline__ void mul_mat_q_process_tile(
         const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
         const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
@@ -880,25 +886,27 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
         const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {
 
     constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
-    constexpr int              nwarps     = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
+    constexpr int              nwarps     = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
     constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
-    constexpr int              I          = ggml_cuda_mmq_get_I(type, J, fallback);
-    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback>();
-    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback>();
-    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback>();
+    constexpr int              I          = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
+    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback, prec_src1>();
+    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback, prec_src1>();
+    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback, prec_src1>();
 
     extern __shared__ int data_mul_mat_q[];
     int * tile_y = data_mul_mat_q + J;
     int * tile_x = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);
 
 #if defined(BLACKWELL_MMA_AVAILABLE)
-    // FP4 tile stores 8 blocks
-    constexpr int ne_block = (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) ? QK_FP4_MMQ : QK8_1_MMQ;
+    // FP4 tile stores 8 blocks. src1 above Q4 uses the generic
+    // Q8_1 tile layout instead of the packed FP4 tile.
+    constexpr int ne_block = ((type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) && prec_src1 == GGML_PREC_Q4) ?
+        QK_FP4_MMQ : QK8_1_MMQ;
 #else
     constexpr int ne_block = QK8_1_MMQ;
 #endif  // defined(BLACKWELL_MMA_AVAILABLE)
 
-    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
+    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
     constexpr int blocks_per_iter = ITER_K / qk;
 
     float sum[J*I / (nwarps*warp_size)] = {0.0f};
@@ -950,8 +958,8 @@ static __device__ __forceinline__ void mul_mat_q_process_tile(
 
 // The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598
 
-template <ggml_type type, int J, bool fallback>
-__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback), ggml_cuda_mmq_get_occupancy(type, J, fallback))
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
+__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
 static __global__ void mul_mat_q(
         const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
         const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
@@ -962,15 +970,15 @@ static __global__ void mul_mat_q(
         const uint3 ntx) {
 
     // Skip unused template specializations for faster compilation:
-    if (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT) {
+    if (ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type == GGML_TYPE_COUNT) {
         NO_DEVICE_CODE;
         return;
     }
 
     constexpr int warp_size = ggml_cuda_get_physical_warp_size();
-    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
+    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
     constexpr int qk        = ggml_cuda_type_traits<type>::qk;
-    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);
+    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
 
     const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y
 
@@ -990,7 +998,7 @@ static __global__ void mul_mat_q(
     }
     __syncthreads();
 
-    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
+    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1)) {
         const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
         const int wt = tmp2.x;
         const int zt = tmp2.y;
@@ -1053,14 +1061,14 @@ static __global__ void mul_mat_q(
         const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;
 
         constexpr bool fixup = false;
-        mul_mat_q_process_tile<type, J, fallback, fixup>
+        mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
             (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
              stride_row_x, ncols_y, stride_col_dst,
              tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
         return;
     }
 
-    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
+    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
     constexpr int blocks_per_iter = ITER_K / qk;
 
     // kbc == k block continuous, current index in continuous ijk space.
@@ -1147,7 +1155,7 @@ static __global__ void mul_mat_q(
         const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;
 
         constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
-        mul_mat_q_process_tile<type, J, fallback, fixup>
+        mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
             (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
              stride_row_x, ncols_y, stride_col_dst,
              tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
@@ -1231,24 +1239,24 @@ static __global__ void mul_mat_q(
     const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;
 
     constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
-    mul_mat_q_process_tile<type, J, fallback, fixup>
+    mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
         (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
          stride_row_x, ncols_y, stride_col_dst,
          tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
 }
 
-template <ggml_type type, int J, bool fallback>
-__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback)/2, 1)
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
+__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1)/2, 1)
 static __global__ void mul_mat_q_stream_k_fixup(
         const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
         float * __restrict__ tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
         const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
         const int stride_sample_dst, const uint3 ntx) {
     constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
-    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2) / warp_size;
-    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
+    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / 2) / warp_size;
+    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
     constexpr int qk              = ggml_cuda_type_traits<type>::qk;
-    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
+    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
     constexpr int blocks_per_iter = ITER_K / qk;
 
     float sum[J / nwarps] = {0.0f};
@@ -1392,22 +1400,22 @@ static size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, const i
     return nbs_ids + nbs_x + GGML_PAD(nbs_y, config.nthreads*sizeof(int));
 }
 
-template <ggml_type type, int J, bool fallback>
+template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
     const int id = ggml_cuda_get_device();
     const int cc = ggml_cuda_info().devices[id].cc;
     const int nsm = ggml_cuda_info().devices[id].nsm;
     const int warp_size = ggml_cuda_info().devices[id].warp_size;
 
-    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
+    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
     GGML_ASSERT(config.nthreads % warp_size == 0);
     const int nwarps = config.nthreads / warp_size;
     const int nbytes_shared = mmq_get_nbytes_shared(config, cc);
 
     const dim3 block_dims(warp_size, nwarps, 1);
 
-    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false>), nbytes_shared);
-    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J,  true>), nbytes_shared);
+    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false, prec_src1>), nbytes_shared);
+    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J,  true, prec_src1>), nbytes_shared);
 
     const int nty  = (args.nrows_x   + config.I - 1) / config.I;
     const int ntx  = (args.ncols_max + config.J - 1) / config.J;
@@ -1426,8 +1434,8 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
     const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
     const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);
 
-    if (!ggml_cuda_mmq_get_stream_k(type, J, fallback, cc)) {
-        mul_mat_q<type, J, fallback><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
+    if (!config.stream_k) {
+        mul_mat_q<type, J, fallback, prec_src1><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
             (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale,
              blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
              channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
@@ -1456,7 +1464,7 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
     const dim3 block_nums_fixup(block_nums_stream_k.x, config.I/warp_size, 1);
     const dim3 block_dims_fixup(block_dims.x, block_dims.y/2, block_dims.z);
 
-    mul_mat_q<type, J, fallback><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
+    mul_mat_q<type, J, fallback, prec_src1><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
         (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, args.y_scale,
          blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
          channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
@@ -1468,13 +1476,13 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
     }
 
     CUDA_CHECK(cudaGetLastError());
-    mul_mat_q_stream_k_fixup<type, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
+    mul_mat_q_stream_k_fixup<type, J, fallback, prec_src1><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
         (args.ids_dst, args.expert_bounds, args.dst, tmp_fixup.ptr, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
          args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
          ntx_fd);
 }
 
-template <ggml_type type, bool fallback>
+template <ggml_type type, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
     const int    id    = ggml_cuda_get_device();
     const int    cc    = ggml_cuda_info().devices[id].cc;
@@ -1484,7 +1492,7 @@ void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
     int ntiles_J_best = INT_MAX;
 
     for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
-        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
+        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
         if (config.type == GGML_TYPE_COUNT) {
             continue;
         }
@@ -1503,52 +1511,52 @@ void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
 
     switch (J_best) {
         case   8:
-            launch_mul_mat_q<type,   8, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,   8, fallback, prec_src1>(ctx, args, stream);
             break;
         case  16:
-            launch_mul_mat_q<type,  16, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  16, fallback, prec_src1>(ctx, args, stream);
             break;
         case  24:
-            launch_mul_mat_q<type,  24, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  24, fallback, prec_src1>(ctx, args, stream);
             break;
         case  32:
-            launch_mul_mat_q<type,  32, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  32, fallback, prec_src1>(ctx, args, stream);
             break;
         case  40:
-            launch_mul_mat_q<type,  40, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  40, fallback, prec_src1>(ctx, args, stream);
             break;
         case  48:
-            launch_mul_mat_q<type,  48, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  48, fallback, prec_src1>(ctx, args, stream);
             break;
         case  56:
-            launch_mul_mat_q<type,  56, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  56, fallback, prec_src1>(ctx, args, stream);
             break;
         case  64:
-            launch_mul_mat_q<type,  64, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  64, fallback, prec_src1>(ctx, args, stream);
             break;
         case  72:
-            launch_mul_mat_q<type,  72, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  72, fallback, prec_src1>(ctx, args, stream);
             break;
         case  80:
-            launch_mul_mat_q<type,  80, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  80, fallback, prec_src1>(ctx, args, stream);
             break;
         case  88:
-            launch_mul_mat_q<type,  88, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  88, fallback, prec_src1>(ctx, args, stream);
             break;
         case  96:
-            launch_mul_mat_q<type,  96, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type,  96, fallback, prec_src1>(ctx, args, stream);
             break;
         case 104:
-            launch_mul_mat_q<type, 104, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type, 104, fallback, prec_src1>(ctx, args, stream);
             break;
         case 112:
-            launch_mul_mat_q<type, 112, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type, 112, fallback, prec_src1>(ctx, args, stream);
             break;
         case 120:
-            launch_mul_mat_q<type, 120, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type, 120, fallback, prec_src1>(ctx, args, stream);
             break;
         case 128:
-            launch_mul_mat_q<type, 128, fallback>(ctx, args, stream);
+            launch_mul_mat_q<type, 128, fallback, prec_src1>(ctx, args, stream);
             break;
         default:
             fprintf(stderr, "J_best=%d\n", J_best);
@@ -1557,20 +1565,24 @@ void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
     }
 }
 
-template <ggml_type type>
+template <ggml_type type, ggml_prec prec_src1 = GGML_PREC_Q8>
 void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
     if (args.nrows_x % 128 == 0) {
         constexpr bool fallback = false;
-        mul_mat_q_switch_J<type, fallback>(ctx, args, stream);
+        mul_mat_q_switch_J<type, fallback, prec_src1>(ctx, args, stream);
     } else {
         constexpr bool fallback = true;
-        mul_mat_q_switch_J<type, fallback>(ctx, args, stream);
+        mul_mat_q_switch_J<type, fallback, prec_src1>(ctx, args, stream);
     }
 }
 
 #define DECL_MMQ_CASE(type)                                                        \
     template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \
 
+// FP4 variant: uses native FP4 MMA instead of keeping src1 at Q8_1.
+#define DECL_MMQ_CASE_W4A4(type)                                                   \
+    template void mul_mat_q_case<type, GGML_PREC_Q4>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \
+
 extern DECL_MMQ_CASE(GGML_TYPE_Q1_0);
 extern DECL_MMQ_CASE(GGML_TYPE_Q2_0);
 extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
@@ -1596,6 +1608,8 @@ extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
 // -----------------------------------------
 extern DECL_MMQ_CASE(GGML_TYPE_MXFP4);
 extern DECL_MMQ_CASE(GGML_TYPE_NVFP4);
+extern DECL_MMQ_CASE_W4A4(GGML_TYPE_MXFP4);
+extern DECL_MMQ_CASE_W4A4(GGML_TYPE_NVFP4);
 
 // -------------------------------------------------------------------------------------------------------------------------
 
diff --git src/ggml-cuda/norm.cu src/ggml-cuda/norm.cu
index c3758cd5..5543307b 100644
--- src/ggml-cuda/norm.cu
+++ src/ggml-cuda/norm.cu
@@ -73,7 +73,7 @@ static __global__ void group_norm_f32(const float * x, float * dst, const int gr
     }
 }
 
-template <int block_size, bool do_multiply = false, bool do_add = false>
+template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false>
 static __global__ void rms_norm_f32(const float * x,
                                     float *       dst,
                                     const int     ncols,
@@ -96,7 +96,8 @@ static __global__ void rms_norm_f32(const float * x,
                                     const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                     const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                     const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
-                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
+                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
+                                    const float   scale_out            = 1.0f) {
     ggml_cuda_pdl_lc();
     const int nrows     = gridDim.x;
     const int nchannels = gridDim.y;
@@ -107,6 +108,7 @@ static __global__ void rms_norm_f32(const float * x,
     const int tid       = threadIdx.x;
 
     static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
+    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");
 
     x   += sample*stride_sample + channel*stride_channel + row*stride_row;
     dst += ((sample*nchannels + channel)*nrows + row)*ncols;
@@ -148,6 +150,8 @@ static __global__ void rms_norm_f32(const float * x,
         } else if constexpr (do_multiply) {
             const int mul_col = fastmodulo(col, mul_ncols_packed);
             dst[col]          = scale * x[col] * mul[mul_col];
+        } else if constexpr (do_scale) {
+            dst[col] = scale_out * (scale * x[col]);
         } else {
             dst[col] = scale * x[col];
         }
@@ -301,25 +305,27 @@ static void group_norm_f32_cuda(
     }
 }
 
+template <bool do_scale = false>
 static void rms_norm_f32_cuda(
         const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
-        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
+        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
+        const float scale_out = 1.0f) {
     const dim3 blocks_num(nrows, nchannels, nsamples);
     if (ncols < 1024) {
         const dim3 block_dims(256, 1, 1);
         const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
-        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
+        ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, do_scale>, launch_params,
             x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
         // underlying cudaLaunchKernelEx does not support default params
         nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
-        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
+        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
     } else {
         const dim3 block_dims(1024, 1, 1);
         const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
-        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
+        ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, do_scale>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
         // underlying cudaLaunchKernelEx does not support default params
         nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
-        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
+        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
     }
 }
 
@@ -367,7 +373,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
                 x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 // underlying cudaLaunchKernelEx does not support default params
-            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
+            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);
         } else {
             const dim3 block_dims(1024, 1, 1);
             const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
@@ -375,7 +381,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
                 x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 // underlying cudaLaunchKernelEx does not support default params
-            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
+            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);
         }
     } else {
         const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
@@ -394,7 +400,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
                 x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                 add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
-                add_nchannels_packed, add_nsamples_packed);
+                add_nchannels_packed, add_nsamples_packed, 1.0f);
         } else {
             const dim3 block_dims(1024, 1, 1);
             const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
@@ -402,7 +408,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
                 x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                 add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
-                add_nchannels_packed, add_nsamples_packed);
+                add_nchannels_packed, add_nsamples_packed, 1.0f);
         }
     }
 }
@@ -499,6 +505,33 @@ void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
 }
 
+void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
+    const ggml_tensor * src0   = dst->src[0];
+    const float *       src0_d = (const float *) src0->data;
+    float *             dst_d  = (float *) scale_tensor->data;
+    cudaStream_t        stream = ctx.stream();
+
+    GGML_ASSERT(src0->type == GGML_TYPE_F32);
+    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);
+
+    GGML_TENSOR_UNARY_OP_LOCALS;
+
+    float eps;
+    memcpy(&eps, dst->op_params, sizeof(float));
+    GGML_ASSERT(eps >= 0.0f);
+
+    float scale;
+    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));
+
+    const size_t ts0 = ggml_type_size(src0->type);
+    GGML_ASSERT(nb00 == ts0);
+    const int64_t s01 = nb01 / ts0;
+    const int64_t s02 = nb02 / ts0;
+    const int64_t s03 = nb03 / ts0;
+
+    rms_norm_f32_cuda<true>(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
+}
+
 void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
     const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
     float eps = 0.0f;
diff --git src/ggml-cuda/norm.cuh src/ggml-cuda/norm.cuh
index a74f6376..95618df5 100644
--- src/ggml-cuda/norm.cuh
+++ src/ggml-cuda/norm.cuh
@@ -8,6 +8,8 @@ void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
 
 void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);
 
+void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);
+
 void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                      ggml_tensor *               dst,
                                      ggml_tensor *               mul_tensor,
diff --git src/ggml-cuda/ssm-scan.cu src/ggml-cuda/ssm-scan.cu
index 40cb38de..e9a1043f 100644
--- src/ggml-cuda/ssm-scan.cu
+++ src/ggml-cuda/ssm-scan.cu
@@ -1,6 +1,6 @@
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
+#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
 #define USE_CUB
-#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
+#endif // !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
 
 #ifdef USE_CUB
 #include <cub/cub.cuh>
@@ -342,7 +342,7 @@ static void ssm_scan_f32_cuda(const float * src0, const float * src1, const floa
     }
 }
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#if !defined(GGML_USE_HIP)
 // ============================================================================
 // SSD (State Space Duality) kernels for Mamba-2 prefill (n_tok > SSM_SSD_MIN_TOKENS)
 //
@@ -821,7 +821,7 @@ void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     GGML_ASSERT(src5->nb[2] <= (size_t)INT_MAX);
     GGML_ASSERT(src5->nb[3] <= (size_t)INT_MAX);
 
-#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
+#if !defined(GGML_USE_HIP)
     // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
     // Requires NVIDIA Turing+ otherwise fallback to scan.
     const bool is_mamba2 = (src3->nb[1] == sizeof(float));
diff --git src/ggml-cuda/template-instances/generate_cu_files.py src/ggml-cuda/template-instances/generate_cu_files.py
index d7cd2716..7be409bd 100755
--- src/ggml-cuda/template-instances/generate_cu_files.py
+++ src/ggml-cuda/template-instances/generate_cu_files.py
@@ -50,6 +50,12 @@ SOURCE_MMQ = """// This file has been autogenerated by generate_cu_files.py, do
 DECL_MMQ_CASE({type});
 """
 
+TYPES_MMQ_W4A4 = ["GGML_TYPE_MXFP4", "GGML_TYPE_NVFP4"]
+
+SOURCE_MMQ_W4A4 = """
+DECL_MMQ_CASE_W4A4({type});
+"""
+
 SOURCE_MMF = """// This file has been autogenerated by generate_cu_files.py, do not edit manually.
 
 #include "../mmf.cuh"
@@ -105,6 +111,8 @@ for ncols in [8, 16, 32, 64]:
 for type in TYPES_MMQ:
     with open(f"mmq-instance-{get_short_name(type)}.cu", "w") as f:
         f.write(SOURCE_MMQ.format(type=type))
+        if type in TYPES_MMQ_W4A4:
+            f.write(SOURCE_MMQ_W4A4.format(type=type))
 
 for type in range(1, 17):
     with open(f"mmf-instance-ncols_{type}.cu", "w") as f:
diff --git src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu
index c14624c5..0c1c7f29 100644
--- src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu
+++ src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu
@@ -3,3 +3,5 @@
 #include "../mmq.cuh"
 
 DECL_MMQ_CASE(GGML_TYPE_MXFP4);
+
+DECL_MMQ_CASE_W4A4(GGML_TYPE_MXFP4);
diff --git src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu
index 2cb140d3..bed8c8e2 100644
--- src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu
+++ src/ggml-cuda/template-instances/mmq-instance-nvfp4.cu
@@ -3,3 +3,5 @@
 #include "../mmq.cuh"
 
 DECL_MMQ_CASE(GGML_TYPE_NVFP4);
+
+DECL_MMQ_CASE_W4A4(GGML_TYPE_NVFP4);
diff --git src/ggml-cuda/topk-moe.cu src/ggml-cuda/topk-moe.cu
index dadcd601..ee903bf2 100644
--- src/ggml-cuda/topk-moe.cu
+++ src/ggml-cuda/topk-moe.cu
@@ -98,7 +98,12 @@ __global__ void topk_moe_cuda(const float *         logits,
                               const float           clamp_val,
                               const float           scale_val,
                               const topk_moe_config config) {
+#if defined(GGML_USE_MUSA)
+    // MUSA: every warp of a partially filled block must reach the barrier below.
+    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
+#else
     const int row = blockIdx.x * blockDim.y + threadIdx.y;
+#endif // defined(GGML_USE_MUSA)
     if (row >= n_rows) {
         return;
     }
diff --git src/ggml-cuda/vendors/hip.h src/ggml-cuda/vendors/hip.h
index 0a2f2829..da7e3da0 100644
--- src/ggml-cuda/vendors/hip.h
+++ src/ggml-cuda/vendors/hip.h
@@ -248,11 +248,11 @@
 typedef __hip_bfloat16 nv_bfloat16;
 typedef __hip_bfloat162 nv_bfloat162;
 
-#if HIP_VERSION >= 60200000
+#if HIP_VERSION >= 60300000
 #include <hip/hip_fp8.h>
 typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
 #define FP8_AVAILABLE
-#endif // HIP_VERSION >= 60200000
+#endif // HIP_VERSION >= 60300000
 
 typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
 typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
diff --git src/ggml-cuda/vendors/musa.h src/ggml-cuda/vendors/musa.h
index 6d725c7e..4243caab 100644
--- src/ggml-cuda/vendors/musa.h
+++ src/ggml-cuda/vendors/musa.h
@@ -44,6 +44,7 @@
 #define cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
 #define cudaDeviceProp musaDeviceProp
 #define cudaDeviceSynchronize musaDeviceSynchronize
+#define cudaDeviceGetAttribute musaDeviceGetAttribute
 #define cudaError_t musaError_t
 #define cudaErrorMemoryAllocation musaErrorMemoryAllocation
 #define cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
@@ -114,6 +115,7 @@
 #define cuMemRelease muMemRelease
 #define cuMemSetAccess muMemSetAccess
 #define cuMemUnmap muMemUnmap
+#define cudaDevAttrCooperativeLaunch musaDevAttrCooperativeLaunch
 #define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
 #define cudaFuncSetAttribute musaFuncSetAttribute
 #define cudaMemcpy3DPeerParms musaMemcpy3DPeerParms
@@ -145,6 +147,9 @@
 #define cudaStreamCaptureModeRelaxed musaStreamCaptureModeRelaxed
 #define cudaStreamBeginCapture musaStreamBeginCapture
 #define cudaStreamEndCapture musaStreamEndCapture
+#define cudaStreamCaptureStatus musaStreamCaptureStatus
+#define cudaStreamCaptureStatusNone musaStreamCaptureStatusNone
+#define cudaStreamIsCapturing musaStreamIsCapturing
 #define cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor
 
 typedef __mt_bfloat16 nv_bfloat16;
diff --git src/ggml-cuda/wkv.cu src/ggml-cuda/wkv.cu
index 23611121..0bf99776 100644
--- src/ggml-cuda/wkv.cu
+++ src/ggml-cuda/wkv.cu
@@ -79,9 +79,7 @@ static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, cons
     float state[head_size];
     __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];
 
-#ifndef GGML_USE_MUSA
     #pragma unroll
-#endif
     for (int i = 0; i < head_size; i++) {
         state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
     }
diff --git src/ggml-hexagon/ggml-hexagon.cpp src/ggml-hexagon/ggml-hexagon.cpp
index 9a350025..581f07aa 100644
--- src/ggml-hexagon/ggml-hexagon.cpp
+++ src/ggml-hexagon/ggml-hexagon.cpp
@@ -267,10 +267,10 @@ static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
     return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
            type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q6_K ||
-           type == GGML_TYPE_Q4_K;
+           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
 }
 
-// Size of one repacked row in the DSP tiled layout. The Q6_K and Q4_K tiles store uncompressed scales/mins,
+// Size of one repacked row in the DSP tiled layout. The Q6_K, Q5_K and Q4_K tiles store uncompressed scales/mins,
 // so they are larger than the ggml blocks. For the other repack types the tile has the same size as the ggml blocks.
 static inline size_t ggml_hexagon_tiled_row_size(enum ggml_type type, int64_t ne0) {
     if (type == GGML_TYPE_Q6_K) {
@@ -279,6 +279,9 @@ static inline size_t ggml_hexagon_tiled_row_size(enum ggml_type type, int64_t ne
     if (type == GGML_TYPE_Q4_K) {
         return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q4_1 / 32);
     }
+    if (type == GGML_TYPE_Q5_K) {
+        return (size_t) (ne0 / 32) * (HTP_MM_WEIGHT_TILE_SIZE_Q5_K / 32);
+    }
     return ggml_row_size(type, ne0);
 }
 
@@ -1742,6 +1745,202 @@ static void repack_tiled_q4_K(void * data, const ggml_tensor * t, size_t offset,
     GGML_UNUSED(size);
 }
 
+// tile layout: see HTP_MM_WEIGHT_TILE_SIZE_Q5_K in htp/matmul-ops.h
+static void repack_q5_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
+    GGML_ASSERT(offset == 0);
+
+    const block_q5_K * src_matrix = (const block_q5_K *) data;
+    int64_t ne0 = t->ne[0];
+    int64_t ne1 = t->ne[1];
+    int64_t ne2 = t->ne[2];
+    int64_t ne3 = t->ne[3];
+    int64_t ne0_padded = hex_round_up(ne0, 32);
+    int64_t ne1_padded = hex_round_up(ne1, 32);
+
+    GGML_ASSERT(ne0 % QK_K == 0);
+
+    const int n_col_tiles = ne1_padded / 32;
+    const int n_k_tiles   = ne0_padded / 32;
+    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q5_K;
+    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
+
+    const int64_t sb_per_row = ne0 / QK_K;
+
+    for (int i3 = 0; i3 < ne3; i3++) {
+        for (int i2 = 0; i2 < ne2; i2++) {
+            const block_q5_K * src_slice = src_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
+            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;
+
+            memset(matrix_dst, 0, matrix_size);
+
+            for (int64_t r = 0; r < ne1; r++) {
+                const int ct  = (int) (r / 32);
+                const int row = (int) (r % 32);
+                const block_q5_K * src_row = src_slice + r * sb_per_row;
+
+                for (int kt = 0; kt < n_k_tiles; kt++) {
+                    const int kt_local = kt % 8;
+                    const block_q5_K * b = &src_row[kt / 8];
+                    const float d = GGML_FP16_TO_FP32(b->d);
+                    const float dmin = GGML_FP16_TO_FP32(b->dmin);
+
+                    uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
+                    uint8_t * plane    = tile_dst + 640;
+
+                    uint8_t sc, m;
+                    get_scale_min_k4(kt_local, b->scales, &sc, &m);
+
+                    const float D = d * (float) sc;
+                    const float M = -dmin * (float) m;
+
+                    const uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
+                    const int shift = (kt_local & 1) ? 4 : 0;
+                    const uint8_t hbit = (uint8_t) (1 << kt_local);
+
+                    for (int cp = 0; cp < 16; cp++) {
+                        const uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
+                        const uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
+                        tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);
+
+                        const int i    = cp / 4;
+                        const int lane = (cp % 4) * 32 + row;
+                        if (b->qh[2 * cp + 0] & hbit) {
+                            plane[lane] |= (uint8_t) (1 << (2 * i));
+                        }
+                        if (b->qh[2 * cp + 1] & hbit) {
+                            plane[lane] |= (uint8_t) (1 << (2 * i + 1));
+                        }
+                    }
+
+                    ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
+                    scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(D);
+                    scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(M);
+                }
+            }
+        }
+    }
+
+    GGML_UNUSED(size);
+}
+
+// Reverse of repack_q5_K_tiled. Unpacks quants losslessly and normalizes scales/mins. Read-back only.
+static void repack_tiled_q5_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
+    GGML_ASSERT(offset == 0);
+
+    block_q5_K * dst_matrix = (block_q5_K *) data;
+    int64_t ne0 = t->ne[0];
+    int64_t ne1 = t->ne[1];
+    int64_t ne2 = t->ne[2];
+    int64_t ne3 = t->ne[3];
+    int64_t ne0_padded = hex_round_up(ne0, 32);
+    int64_t ne1_padded = hex_round_up(ne1, 32);
+
+    GGML_ASSERT(ne0 % QK_K == 0);
+
+    const int n_col_tiles = ne1_padded / 32;
+    const int n_k_tiles   = ne0_padded / 32;
+    const size_t tile_size   = HTP_MM_WEIGHT_TILE_SIZE_Q5_K;
+    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
+
+    const int64_t sb_per_row = ne0 / QK_K;
+
+    for (int i3 = 0; i3 < ne3; i3++) {
+        for (int i2 = 0; i2 < ne2; i2++) {
+            block_q5_K * dst_slice = dst_matrix + (i3 * ne2 + i2) * (ne1 * sb_per_row);
+            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;
+
+            for (int64_t r = 0; r < ne1; r++) {
+                const int ct  = (int) (r / 32);
+                const int row = (int) (r % 32);
+                block_q5_K * dst_row = dst_slice + r * sb_per_row;
+
+                for (int64_t sb = 0; sb < sb_per_row; sb++) {
+                    block_q5_K * b = &dst_row[sb];
+                    memset(b, 0, sizeof(block_q5_K));
+
+                    float sub_scales[8];
+                    float sub_mins[8];
+
+                    for (int kt_local = 0; kt_local < 8; kt_local++) {
+                        const int kt = sb * 8 + kt_local;
+                        const uint8_t * tile_src  = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
+                        const uint8_t * plane     = tile_src + 640;
+                        const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);
+
+                        uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
+                        const int shift = (kt_local & 1) ? 4 : 0;
+                        const uint8_t hbit = (uint8_t) (1 << kt_local);
+
+                        for (int cp = 0; cp < 16; cp++) {
+                            const uint8_t val = tile_src[cp * 32 + row];
+                            const uint8_t q0 = val & 0x0F;
+                            const uint8_t q1 = val >> 4;
+                            qs_sub[2 * cp + 0] |= (uint8_t) (q0 << shift);
+                            qs_sub[2 * cp + 1] |= (uint8_t) (q1 << shift);
+
+                            const int i    = cp / 4;
+                            const int lane = (cp % 4) * 32 + row;
+                            if (plane[lane] & (1 << (2 * i))) {
+                                b->qh[2 * cp + 0] |= hbit;
+                            }
+                            if (plane[lane] & (1 << (2 * i + 1))) {
+                                b->qh[2 * cp + 1] |= hbit;
+                            }
+                        }
+
+                        const float D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
+                        const float M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
+                        sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
+                        sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
+                    }
+
+                    float max_scale = 0.0f;
+                    float max_min   = 0.0f;
+                    for (int j = 0; j < 8; j++) {
+                        if (sub_scales[j] > max_scale) max_scale = sub_scales[j];
+                        if (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
+                    }
+
+                    float inv_scale = 0.0f;
+                    if (max_scale > 0.0f) {
+                        b->d = GGML_FP32_TO_FP16(max_scale / 63.0f);
+                        const float d_actual = GGML_FP16_TO_FP32(b->d);
+                        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
+                    } else {
+                        b->d = GGML_FP32_TO_FP16(0.0f);
+                    }
+
+                    float inv_min = 0.0f;
+                    if (max_min > 0.0f) {
+                        b->dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
+                        const float dmin_actual = GGML_FP16_TO_FP32(b->dmin);
+                        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
+                    } else {
+                        b->dmin = GGML_FP32_TO_FP16(0.0f);
+                    }
+
+                    for (int j = 0; j < 8; j++) {
+                        uint8_t ls = (uint8_t) roundf(inv_scale * sub_scales[j]);
+                        uint8_t lm = (uint8_t) roundf(inv_min * sub_mins[j]);
+                        ls = (std::min)((uint8_t) 63, ls);
+                        lm = (std::min)((uint8_t) 63, lm);
+                        if (j < 4) {
+                            b->scales[j]     = ls;
+                            b->scales[j + 4] = lm;
+                        } else {
+                            b->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
+                            b->scales[j - 4] |= ((ls >> 4) << 6);
+                            b->scales[j - 0] |= ((lm >> 4) << 6);
+                        }
+                    }
+                }
+            }
+        }
+    }
+
+    GGML_UNUSED(size);
+}
+
 static void repack_tensor_tiled(ggml_tensor * tensor, const void * data, size_t size) {
     switch (tensor->type) {
         case GGML_TYPE_Q4_0:
@@ -1768,6 +1967,10 @@ static void repack_tensor_tiled(ggml_tensor * tensor, const void * data, size_t
             repack_mxfp4_tiled(tensor, data, 0, size);
             break;
 
+        case GGML_TYPE_Q5_K:
+            repack_q5_K_tiled(tensor, data, 0, size);
+            break;
+
         case GGML_TYPE_Q6_K:
             repack_q6_K_tiled(tensor, data, 0, size);
             break;
@@ -1856,6 +2059,12 @@ static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
             repack_tiled_q4_K(data, tensor, offset, size);
             break;
 
+        case GGML_TYPE_Q5_K:
+            GGML_ASSERT(offset == 0);
+            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
+            repack_tiled_q5_K(data, tensor, offset, size);
+            break;
+
         case GGML_TYPE_Q8_0:
             GGML_ASSERT(offset == 0);
             GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
@@ -1989,6 +2198,10 @@ static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buff
             repack_tiled_q4_K(temp_buf.data(), tensor, offset, temp_size);
             break;
 
+        case GGML_TYPE_Q5_K:
+            repack_tiled_q5_K(temp_buf.data(), tensor, offset, temp_size);
+            break;
+
         case GGML_TYPE_Q8_0:
             repack_tiled_q8_0(temp_buf.data(), tensor, offset, temp_size);
             break;
@@ -4494,8 +4707,7 @@ static bool ggml_hexagon_precompute_hmx_mm_params(
 
     if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
         // Try grouped path first
-        const bool use_dma_activation = (src1->nb[1]/sizeof(float) > (size_t)ne00_padded);
-        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, ne11, group_size, use_dma_activation, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
+        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, ne11, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
             use_grouped = true;
         }
     }
@@ -5404,12 +5616,13 @@ static bool ggml_hexagon_supported_mul_mat(const struct ggml_hexagon_session * s
         case GGML_TYPE_IQ4_NL:
         case GGML_TYPE_MXFP4:
         case GGML_TYPE_Q4_K:
+        case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
             if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                 return false;
             }
 
-            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
+            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
                 return false;
             }
 
@@ -5487,12 +5700,13 @@ static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session
         case GGML_TYPE_IQ4_NL:
         case GGML_TYPE_MXFP4:
         case GGML_TYPE_Q4_K:
+        case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
             if (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                 return false;
             }
 
-            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
+            if (src0->ne[0] % ((src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q4_K) ? QK_K : 32)) {
                 return false;
             }
 
@@ -7089,18 +7303,21 @@ static bool ggml_hexagon_supported_cpy(const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0 = op->src[0];
     const struct ggml_tensor * dst  = op;
 
-    // for now we can do f32 -> f16 and f16 -> f32 (without reshaping)
-    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
-    if ( dst->type != GGML_TYPE_F32 &&  dst->type != GGML_TYPE_F16) return false;
+    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
+        src0->type != GGML_TYPE_I32) return false;
+    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 &&
+        dst->type != GGML_TYPE_I32) return false;
 
     const bool sametype   = (src0->type == dst->type);
     const bool transposed = ggml_is_transposed(src0) || ggml_is_transposed(dst);
     const bool sameshape  = !transposed && ggml_are_same_shape(src0, dst);
 
-    // can handle any shape and any same-type (pretty slow if reshaping is required)
+    // Same-type copies also support I32.
     if (sametype) return true;
 
-    // cannot handle re-shaping and type conversion at the same time
+    // Type conversion is only supported between F32 and F16.
+    if (src0->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_I32) return false;
+
     if (!sameshape) return false;
 
     return true;
@@ -7110,8 +7327,9 @@ static bool ggml_hexagon_supported_cont(const struct ggml_hexagon_session * sess
     GGML_UNUSED(sess);
     const struct ggml_tensor * src0 = op->src[0];
 
-    // CONT is same-type only, supports f32 and f16
-    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
+    // CONT is same-type only and supports F32, F16, and I32.
+    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
+        src0->type != GGML_TYPE_I32) return false;
 
     return true;
 }
@@ -7754,6 +7972,8 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
                   "please update hexagon_type to match ggml_type");
     static_assert((unsigned int) HTP_TYPE_Q4_K == (unsigned int) GGML_TYPE_Q4_K,
                   "please update hexagon_type to match ggml_type");
+    static_assert((unsigned int) HTP_TYPE_Q5_K == (unsigned int) GGML_TYPE_Q5_K,
+                  "please update hexagon_type to match ggml_type");
     static_assert((unsigned int) HTP_TYPE_Q6_K == (unsigned int) GGML_TYPE_Q6_K,
                   "please update hexagon_type to match ggml_type");
 
diff --git src/ggml-hexagon/htp/concat-ops.c src/ggml-hexagon/htp/concat-ops.c
index 1fa6ec1b..382a9466 100644
--- src/ggml-hexagon/htp/concat-ops.c
+++ src/ggml-hexagon/htp/concat-ops.c
@@ -20,6 +20,7 @@ struct htp_concat_context {
     uint32_t nrows;
     uint32_t elem_start;
     uint32_t nelems;
+    uint32_t nplanes;
     struct fastdiv_values div_ne0;
     struct fastdiv_values div_ne1;
     struct fastdiv_values div_ne2;
@@ -60,39 +61,47 @@ static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void *
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
-    for (uint32_t i = start_i; i < end_i; i += block_i) {
-        uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;
-
-        uint32_t src1_width_bytes = current_block_i * sizeof(float);
-        const dma_addr_t src1_addr = src1->data + i * src1->nb[1];
-        dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);
-
-        uint32_t src0_row_bytes = src0_ne0 * sizeof(float);
-        const dma_addr_t src0_addr = src0->data + i * src0->nb[1];
-        dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);
-
-        dma_queue_pop(dma_q); // src1
-
-        HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);
-
-        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
-        for (uint32_t j = 0; j < src1_ne0_padded; j += 32) {
-            #pragma unroll(4)
-            for (uint32_t ii = 0; ii < current_block_i; ii++) {
-                size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(float));
-                Q6_vgather_ARMVw(&vtcm_tmp[ii], rt, mu, vv);
-                uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(float);
-                hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
+    for (uint32_t p = 0; p < cctx->nplanes; p++) {
+        const uint32_t i3 = p / dst->ne[2];
+        const uint32_t i2 = p - i3 * dst->ne[2];
+        const dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
+        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
+        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];
+
+        for (uint32_t i = start_i; i < end_i; i += block_i) {
+            uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;
+
+            uint32_t src1_width_bytes = current_block_i * sizeof(float);
+            const dma_addr_t src1_addr = src1_plane + i * src1->nb[1];
+            dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);
+
+            uint32_t src0_row_bytes = src0_ne0 * sizeof(float);
+            const dma_addr_t src0_addr = src0_plane + i * src0->nb[1];
+            dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);
+
+            dma_queue_pop(dma_q); // src1
+
+            HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);
+
+            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
+            for (uint32_t j = 0; j < src1_ne0_padded; j += 32) {
+                #pragma unroll(4)
+                for (uint32_t ii = 0; ii < current_block_i; ii++) {
+                    size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(float));
+                    Q6_vgather_ARMVw(&vtcm_tmp[ii], rt, mu, vv);
+                    uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(float);
+                    hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
+                }
             }
-        }
-        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
+            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
 
-        dma_queue_pop(dma_q); // src0
+            dma_queue_pop(dma_q); // src0
 
-        const dma_addr_t dst_addr = dst->data + i * dst->nb[1];
-        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);
+            const dma_addr_t dst_addr = dst_plane + i * dst->nb[1];
+            dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);
 
-        dma_queue_pop(dma_q);
+            dma_queue_pop(dma_q);
+        }
     }
 }
 
@@ -131,39 +140,47 @@ static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void *
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
-    for (uint32_t i = start_i; i < end_i; i += block_i) {
-        uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;
-
-        uint32_t src1_width_bytes = current_block_i * sizeof(__fp16);
-        const dma_addr_t src1_addr = src1->data + i * src1->nb[1];
-        dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);
-
-        uint32_t src0_row_bytes = src0_ne0 * sizeof(__fp16);
-        const dma_addr_t src0_addr = src0->data + i * src0->nb[1];
-        dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);
-
-        dma_queue_pop(dma_q); // src1
-
-        HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);
-
-        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
-        for (uint32_t j = 0; j < src1_ne0_padded; j += 64) {
-            #pragma unroll(4)
-            for (uint32_t ii = 0; ii < current_block_i; ii++) {
-                size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(__fp16));
-                Q6_vgather_ARMVh(&vtcm_tmp[ii], rt, mu, vv);
-                uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(__fp16);
-                hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
+    for (uint32_t p = 0; p < cctx->nplanes; p++) {
+        const uint32_t i3 = p / dst->ne[2];
+        const uint32_t i2 = p - i3 * dst->ne[2];
+        const dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
+        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
+        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];
+
+        for (uint32_t i = start_i; i < end_i; i += block_i) {
+            uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;
+
+            uint32_t src1_width_bytes = current_block_i * sizeof(__fp16);
+            const dma_addr_t src1_addr = src1_plane + i * src1->nb[1];
+            dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);
+
+            uint32_t src0_row_bytes = src0_ne0 * sizeof(__fp16);
+            const dma_addr_t src0_addr = src0_plane + i * src0->nb[1];
+            dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);
+
+            dma_queue_pop(dma_q); // src1
+
+            HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);
+
+            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
+            for (uint32_t j = 0; j < src1_ne0_padded; j += 64) {
+                #pragma unroll(4)
+                for (uint32_t ii = 0; ii < current_block_i; ii++) {
+                    size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(__fp16));
+                    Q6_vgather_ARMVh(&vtcm_tmp[ii], rt, mu, vv);
+                    uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(__fp16);
+                    hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
+                }
             }
-        }
-        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
+            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
 
-        dma_queue_pop(dma_q); // src0
+            dma_queue_pop(dma_q); // src0
 
-        const dma_addr_t dst_addr = dst->data + i * dst->nb[1];
-        dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);
+            const dma_addr_t dst_addr = dst_plane + i * dst->nb[1];
+            dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);
 
-        dma_queue_pop(dma_q);
+            dma_queue_pop(dma_q);
+        }
     }
 }
 
@@ -230,6 +247,61 @@ static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
     }
 }
 
+static bool concat_dim1_contiguous_dma(struct htp_ops_context * octx, int dim, uint32_t type_size) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * src1 = octx->src[1];
+    const struct htp_tensor * dst  = octx->dst;
+
+    if (dim != 1 || octx->ctx->mdev.count > 1 ||
+        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_I32) ||
+        src0->type != dst->type || src1->type != dst->type ||
+        src0->ne[0] != dst->ne[0] || src1->ne[0] != dst->ne[0] ||
+        src0->ne[2] != dst->ne[2] || src1->ne[2] != dst->ne[2] ||
+        src0->ne[3] != dst->ne[3] || src1->ne[3] != dst->ne[3] ||
+        dst->ne[1] != src0->ne[1] + src1->ne[1] ||
+        !htp_tensor_is_contiguous(src0, type_size) ||
+        !htp_tensor_is_contiguous(src1, type_size) ||
+        !htp_tensor_is_contiguous(dst, type_size)) {
+        return false;
+    }
+
+    const uint32_t src0_row_size = src0->ne[0] * type_size;
+    const uint32_t src1_row_size = src1->ne[0] * type_size;
+
+    // v75+ dma_queue_push() writes a 2D descriptor directly and does not split overflow.
+#if __HVX_ARCH__ >= 75
+    if (src0_row_size > 0xffffffu || src1_row_size > 0xffffffu ||
+        src0->nb[1] > 0xffffffu || src1->nb[1] > 0xffffffu || dst->nb[1] > 0xffffffu ||
+        src0->ne[1] > UINT16_MAX || src1->ne[1] > UINT16_MAX) {
+        return false;
+    }
+#endif
+
+    dma_queue * q = octx->ctx->dma[0];
+
+    for (uint32_t i3 = 0; i3 < dst->ne[3]; ++i3) {
+        for (uint32_t i2 = 0; i2 < dst->ne[2]; ++i2) {
+            dma_addr_t dst_addr  = dst->data  + i3 * dst->nb[3]  + i2 * dst->nb[2];
+            dma_addr_t src0_addr = src0->data + i3 * src0->nb[3] + i2 * src0->nb[2];
+            dma_addr_t src1_addr = src1->data + i3 * src1->nb[3] + i2 * src1->nb[2];
+
+            if (!dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1])) {
+                dma_queue_flush(q);
+                dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1]);
+            }
+
+            dst_addr += src0->ne[1] * dst->nb[1];
+            if (!dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1])) {
+                dma_queue_flush(q);
+                dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1]);
+            }
+        }
+    }
+
+    dma_queue_flush(q);
+    return true;
+}
+
 int op_concat(struct htp_ops_context * octx) {
     const struct htp_tensor * src0 = octx->src[0];
     const struct htp_tensor * src1 = octx->src[1];
@@ -237,12 +309,14 @@ int op_concat(struct htp_ops_context * octx) {
 
     int dim = octx->op_params[0];
 
-    bool is_2d = dst->ne[2] == 1 && dst->ne[3] == 1;
-
     const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
     bool is_src1_transposed  = (src1->nb[0] > src1->nb[1]);
     bool is_src0_transposed  = (src0->nb[0] > src0->nb[1]);
 
+    if (concat_dim1_contiguous_dma(octx, dim, type_size)) {
+        return HTP_STATUS_OK;
+    }
+
     uint32_t n_threads = octx->n_threads;
     struct htp_concat_context cctx;
     cctx.octx = octx;
@@ -253,7 +327,9 @@ int op_concat(struct htp_ops_context * octx) {
 
     void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;
 
-    if (dim == 0 && is_2d && is_src1_transposed && !is_src0_transposed) {
+    const bool rows_ok = src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size;
+
+    if (dim == 0 && is_src1_transposed && !is_src0_transposed && rows_ok) {
         const uint32_t total_rows = dst->ne[1];
         const size_t dst_data_row_size = dst->ne[0] * type_size;
         uint32_t row_start = 0;
@@ -272,6 +348,7 @@ int op_concat(struct htp_ops_context * octx) {
 
         cctx.row_start = row_start;
         cctx.nrows     = nrows;
+        cctx.nplanes   = dst->ne[2] * dst->ne[3];
 
         uint32_t block_i = (type_size == 4) ? 32 : 64;
 
diff --git src/ggml-hexagon/htp/cpy-ops.c src/ggml-hexagon/htp/cpy-ops.c
index 4453dda3..785e3e49 100644
--- src/ggml-hexagon/htp/cpy-ops.c
+++ src/ggml-hexagon/htp/cpy-ops.c
@@ -140,6 +140,7 @@ static void cpy_thread_##NAME##_sameshape(unsigned int nth, unsigned int ith, vo
 
 DEFINE_CPY_SAMESHAPE(f32,  float, 4)
 DEFINE_CPY_SAMESHAPE(f16, __fp16, 2)
+DEFINE_CPY_SAMESHAPE(i32, int32_t, 4)
 
 #define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                                \
 static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void * data) {            \
@@ -226,6 +227,7 @@ static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void
 
 DEFINE_CPY_RESHAPE(f32,  float, 4)
 DEFINE_CPY_RESHAPE(f16, __fp16, 2)
+DEFINE_CPY_RESHAPE(i32, int32_t, 4)
 
 static void cpy_thread_f16_f32_sameshape(unsigned int nth, unsigned int ith, void * data) {
     struct htp_copy_context * ct = (struct htp_copy_context *) data;
@@ -370,6 +372,7 @@ static int exec_cpy(struct htp_ops_context * octx, bool * use_dma) {
     switch (src0->type) {
     case HTP_TYPE_F32: ct.src0_type_size = 4; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
     case HTP_TYPE_F16: ct.src0_type_size = 2; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
+    case HTP_TYPE_I32: ct.src0_type_size = 4; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
     default:
         return HTP_STATUS_NO_SUPPORT;
     }
@@ -377,6 +380,7 @@ static int exec_cpy(struct htp_ops_context * octx, bool * use_dma) {
     switch (dst->type) {
     case HTP_TYPE_F32: ct.dst_type_size = 4; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
     case HTP_TYPE_F16: ct.dst_type_size = 2; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
+    case HTP_TYPE_I32: ct.dst_type_size = 4; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
     default:
         return HTP_STATUS_NO_SUPPORT;
     }
@@ -436,7 +440,12 @@ static int exec_cpy(struct htp_ops_context * octx, bool * use_dma) {
         } else {
             work_queue_func_t copy_fun = NULL;
             if (sametype) {
-                copy_fun = (src0->type == HTP_TYPE_F32) ? cpy_thread_f32_sameshape : cpy_thread_f16_sameshape;
+                switch (src0->type) {
+                    case HTP_TYPE_F32: copy_fun = cpy_thread_f32_sameshape; break;
+                    case HTP_TYPE_F16: copy_fun = cpy_thread_f16_sameshape; break;
+                    case HTP_TYPE_I32: copy_fun = cpy_thread_i32_sameshape; break;
+                    default: return HTP_STATUS_NO_SUPPORT;
+                }
             } else if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
                 copy_fun = cpy_thread_f16_f32_sameshape;
             } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
@@ -482,7 +491,13 @@ static int exec_cpy(struct htp_ops_context * octx, bool * use_dma) {
         ct.nelem           = nelem;
         ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);
 
-        work_queue_func_t copy_fun = (src0->type == HTP_TYPE_F32) ? cpy_thread_f32_reshape : cpy_thread_f16_reshape;
+        work_queue_func_t copy_fun = NULL;
+        switch (src0->type) {
+            case HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; break;
+            case HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; break;
+            case HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; break;
+            default: return HTP_STATUS_NO_SUPPORT;
+        }
         work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
     } else {
         return HTP_STATUS_NO_SUPPORT;
diff --git src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
index d6d40586..5b7f3402 100644
--- src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
+++ src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
@@ -506,6 +506,111 @@ static void dequantize_tiled_weight_to_fp16_task_q8_0(
     }
 }
 
+static void dequantize_tiled_weight_to_fp16_task_q5_k(
+        const tiled_dequantize_state_t *state,
+        uint32_t start_tile, uint32_t end_tile) {
+
+    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
+
+    for (uint32_t t = start_tile; t < end_tile; t++) {
+        const uint8_t * tile_src = state->src + t * state->aligned_tile_size;
+        __fp16 * dst_ptr = state->dst + t * HTP_MM_HMX_TILE_N_ELMS;
+
+        HVX_Vector vscale_offset = hvx_vmem(tile_src + 512);
+        HVX_VectorPair dm_deal = Q6_W_vdeal_VVR(vscale_offset, vscale_offset, -2);
+        HVX_Vector vd = Q6_V_lo_W(dm_deal);
+        HVX_Vector vm = Q6_V_hi_W(dm_deal);
+
+        HVX_Vector v_scale_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vd, vd, -2));
+        HVX_Vector v_offset_duplicated = Q6_V_lo_W(Q6_W_vshuff_VVR(vm, vm, -2));
+
+        // Load all 4 groups in parallel
+        HVX_Vector vq0 = hvx_vmem(tile_src + 0 * 128);
+        HVX_Vector vq1 = hvx_vmem(tile_src + 1 * 128);
+        HVX_Vector vq2 = hvx_vmem(tile_src + 2 * 128);
+        HVX_Vector vq3 = hvx_vmem(tile_src + 3 * 128);
+
+        // Nibble extraction
+        HVX_Vector v_lo0 = Q6_V_vand_VV(vq0, mask_h4);
+        HVX_Vector v_hi0 = Q6_Vub_vlsr_VubR(vq0, 4);
+        HVX_Vector v_lo1 = Q6_V_vand_VV(vq1, mask_h4);
+        HVX_Vector v_hi1 = Q6_Vub_vlsr_VubR(vq1, 4);
+        HVX_Vector v_lo2 = Q6_V_vand_VV(vq2, mask_h4);
+        HVX_Vector v_hi2 = Q6_Vub_vlsr_VubR(vq2, 4);
+        HVX_Vector v_lo3 = Q6_V_vand_VV(vq3, mask_h4);
+        HVX_Vector v_hi3 = Q6_Vub_vlsr_VubR(vq3, 4);
+
+        // Q5_K: OR in the 5th bit from the plane
+        HVX_Vector v_plane = hvx_vmem(tile_src + 640);
+        v_lo0 = hvx_q5k_or_hibit(v_lo0, v_plane, 0);
+        v_hi0 = hvx_q5k_or_hibit(v_hi0, v_plane, 1);
+        v_lo1 = hvx_q5k_or_hibit(v_lo1, v_plane, 2);
+        v_hi1 = hvx_q5k_or_hibit(v_hi1, v_plane, 3);
+        v_lo2 = hvx_q5k_or_hibit(v_lo2, v_plane, 4);
+        v_hi2 = hvx_q5k_or_hibit(v_hi2, v_plane, 5);
+        v_lo3 = hvx_q5k_or_hibit(v_lo3, v_plane, 6);
+        v_hi3 = hvx_q5k_or_hibit(v_hi3, v_plane, 7);
+
+        // Shuffling
+        HVX_VectorPair vp_shuf0 = Q6_W_vshuff_VVR(v_hi0, v_lo0, -1);
+        HVX_VectorPair vp_shuf1 = Q6_W_vshuff_VVR(v_hi1, v_lo1, -1);
+        HVX_VectorPair vp_shuf2 = Q6_W_vshuff_VVR(v_hi2, v_lo2, -1);
+        HVX_VectorPair vp_shuf3 = Q6_W_vshuff_VVR(v_hi3, v_lo3, -1);
+
+        // Unpack to 16-bit
+        HVX_VectorPair vp_int16_lo0 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf0));
+        HVX_VectorPair vp_int16_hi0 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf0));
+        HVX_VectorPair vp_int16_lo1 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf1));
+        HVX_VectorPair vp_int16_hi1 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf1));
+        HVX_VectorPair vp_int16_lo2 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf2));
+        HVX_VectorPair vp_int16_hi2 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf2));
+        HVX_VectorPair vp_int16_lo3 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(vp_shuf3));
+        HVX_VectorPair vp_int16_hi3 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(vp_shuf3));
+
+        // Convert, multiply, add offset
+        HVX_Vector v_grp0_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp0_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo0)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp0_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp0_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi0)), v_scale_duplicated), v_offset_duplicated));
+
+        HVX_Vector v_grp1_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp1_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo1)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp1_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp1_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi1)), v_scale_duplicated), v_offset_duplicated));
+
+        HVX_Vector v_grp2_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp2_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo2)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp2_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp2_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi2)), v_scale_duplicated), v_offset_duplicated));
+
+        HVX_Vector v_grp3_0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp3_1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_lo3)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp3_2 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));
+        HVX_Vector v_grp3_3 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_int16_hi3)), v_scale_duplicated), v_offset_duplicated));
+
+        // Parallel Stores
+        hvx_vmem(dst_ptr +  0 * 64) = v_grp0_0;
+        hvx_vmem(dst_ptr +  1 * 64) = v_grp0_1;
+        hvx_vmem(dst_ptr +  2 * 64) = v_grp0_2;
+        hvx_vmem(dst_ptr +  3 * 64) = v_grp0_3;
+
+        hvx_vmem(dst_ptr +  4 * 64) = v_grp1_0;
+        hvx_vmem(dst_ptr +  5 * 64) = v_grp1_1;
+        hvx_vmem(dst_ptr +  6 * 64) = v_grp1_2;
+        hvx_vmem(dst_ptr +  7 * 64) = v_grp1_3;
+
+        hvx_vmem(dst_ptr +  8 * 64) = v_grp2_0;
+        hvx_vmem(dst_ptr +  9 * 64) = v_grp2_1;
+        hvx_vmem(dst_ptr + 10 * 64) = v_grp2_2;
+        hvx_vmem(dst_ptr + 11 * 64) = v_grp2_3;
+
+        hvx_vmem(dst_ptr + 12 * 64) = v_grp3_0;
+        hvx_vmem(dst_ptr + 13 * 64) = v_grp3_1;
+        hvx_vmem(dst_ptr + 14 * 64) = v_grp3_2;
+        hvx_vmem(dst_ptr + 15 * 64) = v_grp3_3;
+    }
+}
+
 // Q6_K stores 6-bit weights and one fp16 scale per 16 k, see HTP_MM_WEIGHT_TILE_SIZE_Q6_K.
 // A k-group holds 4 k per row, the HMX tile holds 2, so each group is dealt into two tiles.
 static void dequantize_tiled_weight_to_fp16_task_q6_k(
@@ -914,98 +1019,6 @@ typedef struct {
 
 // activations : fp32 -> fp16
 
-static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, uint32_t n_rows, uint32_t k_block, uint32_t k_stride, uint32_t k_valid) {
-    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
-    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;
-
-    uint32_t r = 0;
-
-    #pragma unroll(2)
-    for (r = 0; r < n_rows_tiled; r += 2) {
-        uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
-        uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx
-
-        const float *ptr_in0 = src + (r + 0) * k_stride;
-        const float *ptr_in1 = src + (r + 1) * k_stride;
-
-        uint32_t c = 0;
-        for (; c + 32 <= k_valid; c += 32) {
-            HVX_Vector v0 = *(const HVX_Vector *)(ptr_in0 + c);
-            HVX_Vector v1 = *(const HVX_Vector *)(ptr_in1 + c);
-            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);
-
-            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
-            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;
-
-            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
-            tile[r1 / 2]     = v_out;
-        }
-        if (c < k_block) {
-            HVX_Vector v0 = *(const HVX_Vector *)(ptr_in0 + c);
-            HVX_Vector v1 = *(const HVX_Vector *)(ptr_in1 + c);
-
-            uint32_t rem = k_valid - c;
-            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
-            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
-            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
-
-            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);
-
-            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
-            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;
-
-            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
-            tile[r1 / 2]     = v_out;
-        }
-    }
-
-    for (; r < n_rows_padded; r += 2) {
-        uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
-        uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx
-
-        const bool row0_valid = r       < n_rows;
-        const bool row1_valid = (r + 1) < n_rows;
-
-        const float *ptr_in0 = row0_valid ? (src + (r + 0) * k_stride) : NULL;
-        const float *ptr_in1 = row1_valid ? (src + (r + 1) * k_stride) : NULL;
-
-        uint32_t c = 0;
-        for (; c + 32 <= k_valid; c += 32) {
-            HVX_Vector v0 = Q6_V_vzero();
-            HVX_Vector v1 = Q6_V_vzero();
-            if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
-            if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);
-
-            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);
-
-            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
-            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;
-
-            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
-            tile[r1 / 2]     = v_out;
-        }
-        if (c < k_block) {
-            HVX_Vector v0 = Q6_V_vzero();
-            HVX_Vector v1 = Q6_V_vzero();
-            if (row0_valid) v0 = *(const HVX_Vector *)(ptr_in0 + c);
-            if (row1_valid) v1 = *(const HVX_Vector *)(ptr_in1 + c);
-
-            uint32_t rem = k_valid - c;
-            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
-            v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
-            v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());
-
-            HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);
-
-            uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
-            uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;
-
-            HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
-            tile[r1 / 2]     = v_out;
-        }
-    }
-}
-
 static void transfer_activation_row_pair_fp32_to_fp16(
         __fp16 *restrict vtcm_dst,
         const float *restrict row0,
diff --git src/ggml-hexagon/htp/htp-ops.h src/ggml-hexagon/htp/htp-ops.h
index ee5b9244..a03c8550 100644
--- src/ggml-hexagon/htp/htp-ops.h
+++ src/ggml-hexagon/htp/htp-ops.h
@@ -23,6 +23,7 @@ enum htp_data_type {
     HTP_TYPE_Q4_1   = 3,
     HTP_TYPE_Q8_0   = 8,
     HTP_TYPE_Q4_K   = 12,
+    HTP_TYPE_Q5_K   = 13,
     HTP_TYPE_Q6_K   = 14,
     HTP_TYPE_IQ4_NL = 20,
     HTP_TYPE_I32    = 26,
diff --git src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h
index 4d6110ff..4564b152 100644
--- src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h
+++ src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h
@@ -121,29 +121,43 @@ static inline void quantize_block_f32_q8_0_tiled(float * restrict x, uint8_t * r
     HVX_Vector * vx = (HVX_Vector *) x;
     HVX_Vector zero   = Q6_V_vzero();
 
+    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
+    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
+    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
+    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));
+
     HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
     HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
     HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
     HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);
 
+    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
+    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
+    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
+    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);
+
+    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
+    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));
+
     HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
     HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));
 
-    HVX_Vector vmax_hf = hvx_vec_reduce_max_f16(hvx_vec_abs_f16(vx01_hf));
-    vmax_hf            = hvx_vec_reduce_max2_f16(hvx_vec_abs_f16(vx23_hf), vmax_hf);
-
-    HVX_Vector vd_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax_hf, Q6_Vh_vsplat_R(0x2008));
-    HVX_Vector vd_hf   = Q6_Vhf_equals_Vqf16(vd_qf16);
+    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
+    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
+    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
+    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);
 
-    HVX_Vector vd_inv_hf = hvx_vec_inverse_f16(vd_hf);
-    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd_inv_hf));
-    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd_inv_hf));
+    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
+    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
+    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
+    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));
 
     HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
     HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
     HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);
 
-    HVX_Vector r_scale = hvx_vec_repl_f16(vd_hf);
+    HVX_VectorPair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
+    HVX_VectorPair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);
 
     static const uint8_t __attribute__((aligned(128))) repl[128] = {
         0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
@@ -157,8 +171,19 @@ static inline void quantize_block_f32_q8_0_tiled(float * restrict x, uint8_t * r
     };
     HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;
 
+    #pragma unroll
     for (int b = 0; b < 4; b++) {
         HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
+        HVX_Vector r_scale;
+        if (b == 0) {
+            r_scale = Q6_V_lo_W(vp01);
+        } else if (b == 1) {
+            r_scale = Q6_V_hi_W(vp01);
+        } else if (b == 2) {
+            r_scale = Q6_V_lo_W(vp23);
+        } else {
+            r_scale = Q6_V_hi_W(vp23);
+        }
 
         HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
         HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
@@ -371,6 +396,78 @@ static inline HVX_VectorPair accum_q8_0_32x2(
     return Q6_W_vcombine_VV(v_sum1, v_sum0);
 }
 
+// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_MM_WEIGHT_TILE_SIZE_Q5_K)
+static inline HVX_Vector hvx_q5k_or_hibit(HVX_Vector v, HVX_Vector v_plane, int j) {
+    HVX_VectorPred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
+    return Q6_V_vandor_VQR(v, q, 0x10101010);
+}
+
+// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
+static inline HVX_VectorPair unpack_and_interleave_5bit_x2(HVX_Vector v_src, HVX_Vector v_plane, int i, HVX_Vector mask_h4) {
+    HVX_Vector v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, mask_h4), v_plane, 2 * i);
+    HVX_Vector v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
+    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
+    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
+    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);
+
+    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
+    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));
+
+    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
+    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));
+
+    return Q6_W_vcombine_VV(v_W1, v_W0);
+}
+
+static inline HVX_Vector accum_5bit_32x1(
+    const HVX_Vector * restrict vptr,
+    const HVX_Vector * restrict v_act,
+    HVX_Vector i8
+) {
+    HVX_Vector v_sum0 = Q6_V_vzero();
+    HVX_Vector v_sum1 = Q6_V_vzero();
+    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
+    HVX_Vector v_plane = vptr[5];
+
+    #pragma unroll
+    for (int i = 0; i < 4; i++) {
+        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
+        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
+        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
+        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
+        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
+    }
+
+    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
+}
+
+static inline HVX_VectorPair accum_5bit_32x2(
+    const HVX_Vector * restrict vptr,
+    const HVX_Vector * restrict v_act0,
+    const HVX_Vector * restrict v_act1,
+    HVX_Vector i8
+) {
+    HVX_Vector v_sum0 = Q6_V_vzero();
+    HVX_Vector v_sum1 = Q6_V_vzero();
+    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
+    HVX_Vector v_plane = vptr[5];
+
+    #pragma unroll
+    for (int i = 0; i < 4; i++) {
+        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
+        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
+        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
+
+        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
+        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);
+
+        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
+        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
+    }
+
+    return Q6_W_vcombine_VV(v_sum1, v_sum0);
+}
+
 // Q6_K weights are stored unsigned (0..63), see HTP_MM_WEIGHT_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
 static inline HVX_Vector unpack_q6_k_group(const HVX_Vector * restrict vptr, int g, HVX_Vector mask_0f, HVX_Vector mask_03, HVX_Vector i32) {
     HVX_Vector v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], mask_0f);
@@ -689,6 +786,102 @@ static void tiled_vec_dot_q8_0_32x2(const uint32_t n, float * restrict s0, float
     }
 }
 
+static void tiled_vec_dot_q5_k_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
+    const uint8_t * restrict tile_ptr = vx;
+    const uint8_t * restrict y_q = vy;
+
+    HVX_Vector v_sum_float = Q6_V_vzero();
+
+    uint32_t n_k_tiles = n / 32;
+    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
+        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 768);
+        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);
+
+        HVX_Vector v_sum = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
+        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);
+
+        HVX_Vector v_scale_offset = vptr[4];
+        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
+        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
+        HVX_Vector v_offset = Q6_V_hi_W(p_deal);
+
+        HVX_Vector v_scale_a = v_act[8];
+        HVX_Vector v_sum_a   = v_act[9];
+
+        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
+        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);
+
+        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
+        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);
+
+        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
+    }
+
+    if (sz) {
+        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
+    } else {
+        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
+    }
+}
+
+static void tiled_vec_dot_q5_k_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
+    const uint8_t * restrict tile_ptr = vx;
+    const uint8_t * restrict y0_q = vy0;
+    const uint8_t * restrict y1_q = vy1;
+
+    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
+    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
+
+    uint32_t n_k_tiles = n / 32;
+    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
+        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 768);
+        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
+        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);
+
+        HVX_VectorPair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
+        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
+        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);
+
+        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
+        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);
+
+        HVX_Vector v_scale_offset = vptr[4];
+        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
+        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
+        HVX_Vector v_offset = Q6_V_hi_W(p_deal);
+
+        HVX_Vector v_scale_a_c0 = v_act0[8];
+        HVX_Vector v_sum_a_c0   = v_act0[9];
+        HVX_Vector v_scale_a_c1 = v_act1[8];
+        HVX_Vector v_sum_a_c1   = v_act1[9];
+
+        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
+        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
+        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
+        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);
+
+        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
+        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);
+
+        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
+        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);
+
+        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
+        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
+    }
+
+    if (sz0) {
+        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
+    } else {
+        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
+    }
+    if (sz1) {
+        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
+    } else {
+        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
+    }
+}
+
 static void tiled_vec_dot_q6_k_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
     const uint8_t * restrict tile_ptr = vx;
     const uint8_t * restrict y_q = vy;
@@ -941,14 +1134,9 @@ static inline void quantize_f32_q8_0_tiled_kernel(
     size_t src_row_size,
     size_t dst_row_size
 ) {
-    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0_TILED * sizeof(float));
-    hvx_splat_f32_a(tmp_data, 0.0f, src_row_size_padded / sizeof(float));
-
+    (void) tmp_data;
     for (uint32_t i = 0; i < nrows; ++i) {
-        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
-        hvx_copy_f32_aa(tmp_data, src_data, ne0);
-
-        quantize_row_f32_q8_0_tiled((float *) tmp_data, dst_data, ne0);
+        quantize_row_f32_q8_0_tiled((float *) src_data, dst_data, ne0);
         dst_data += dst_row_size;
         src_data += src_row_size;
     }
@@ -963,14 +1151,9 @@ static inline void quantize_f32_q8_1_tiled_kernel(
     size_t src_row_size,
     size_t dst_row_size
 ) {
-    const size_t src_row_size_padded = hex_round_up(src_row_size, QK_Q8_0_TILED * sizeof(float));
-    hvx_splat_f32_a(tmp_data, 0.0f, src_row_size_padded / sizeof(float));
-
+    (void) tmp_data;
     for (uint32_t i = 0; i < nrows; ++i) {
-        hex_l2fetch(src_data, src_row_size, src_row_size, 2);
-        hvx_copy_f32_aa(tmp_data, src_data, ne0);
-
-        quantize_row_f32_q8_1_tiled((float *) tmp_data, dst_data, ne0);
+        quantize_row_f32_q8_1_tiled((float *) src_data, dst_data, ne0);
         dst_data += dst_row_size;
         src_data += src_row_size;
     }
@@ -988,24 +1171,15 @@ static inline void quantize_f32_q8_0_tiled_block_kernel(
     uint32_t r,
     uint32_t c
 ) {
+    (void) tmp_data;
     const uint32_t qk = QK_Q8_0_TILED;
     const uint32_t nb = (ne0 + qk - 1) / qk;
 
     for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
-        const uint8_t * restrict src_ptr = (const uint8_t *) src + r * src_row_size + c * qk * sizeof(float);
+        const float * restrict src_ptr = (const float *) ((const uint8_t *) src + r * src_row_size + c * qk * sizeof(float));
         uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1152;
 
-        hex_l2fetch(src_ptr, qk * sizeof(float), qk * sizeof(float), 1);
-
-        if (c == nb - 1) {
-            uint32_t active_elements = ne0 - c * qk;
-            hvx_splat_f32_a(tmp_data, 0.0f, qk);
-            hvx_copy_f32_aa(tmp_data, src_ptr, active_elements);
-        } else {
-            hvx_copy_f32_aa(tmp_data, src_ptr, qk);
-        }
-
-        quantize_block_f32_q8_0_tiled((float *) tmp_data, dst_ptr);
+        quantize_block_f32_q8_0_tiled((float *) src_ptr, dst_ptr);
 
         c++;
         if (c == nb) {
@@ -1027,24 +1201,15 @@ static inline void quantize_f32_q8_1_tiled_block_kernel(
     uint32_t r,
     uint32_t c
 ) {
+    (void) tmp_data;
     const uint32_t qk = QK_Q8_0_TILED;
     const uint32_t nb = (ne0 + qk - 1) / qk;
 
     for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
-        const uint8_t * restrict src_ptr = (const uint8_t *) src + r * src_row_size + c * qk * sizeof(float);
+        const float * restrict src_ptr = (const float *) ((const uint8_t *) src + r * src_row_size + c * qk * sizeof(float));
         uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1280;
 
-        hex_l2fetch(src_ptr, qk * sizeof(float), qk * sizeof(float), 1);
-
-        if (c == nb - 1) {
-            uint32_t active_elements = ne0 - c * qk;
-            hvx_splat_f32_a(tmp_data, 0.0f, qk);
-            hvx_copy_f32_aa(tmp_data, src_ptr, active_elements);
-        } else {
-            hvx_copy_f32_aa(tmp_data, src_ptr, qk);
-        }
-
-        quantize_block_f32_q8_1_tiled((float *) tmp_data, dst_ptr);
+        quantize_block_f32_q8_1_tiled((float *) src_ptr, dst_ptr);
 
         c++;
         if (c == nb) {
diff --git src/ggml-hexagon/htp/matmul-ops.c src/ggml-hexagon/htp/matmul-ops.c
index a09bc7a2..727dc281 100644
--- src/ggml-hexagon/htp/matmul-ops.c
+++ src/ggml-hexagon/htp/matmul-ops.c
@@ -29,7 +29,7 @@ typedef struct {
     float        *dst;
     dma_addr_t    src2_addr;
     size_t        src2_bytes;
-    const float  *activation;
+    dma_addr_t    act_dma_addr;
     dma_addr_t    weight;
     dma_queue *   weight_dma;
     int           m;
@@ -45,8 +45,8 @@ typedef struct {
     int           ne13;
     size_t        src0_nb2;
     size_t        src0_nb3;
-    size_t        src1_nb2;
-    size_t        src1_nb3;
+    size_t        act_nb2;
+    size_t        act_nb3;
     size_t        src2_nb2;
     size_t        src2_nb3;
     size_t        dst_nb2;
@@ -93,7 +93,7 @@ struct htp_mm_context {
     uint32_t src0_row_start;
     uint32_t src0_row_end;
     uint32_t src0_row_size_padded;
-    uint32_t src1_nrows;
+    uint32_t act_nrows;
     uint32_t cur_m_start;
     uint32_t cur_m_rows;
 
@@ -103,16 +103,13 @@ struct htp_mm_context {
     struct fastdiv_values mm_div_r3;
     struct fastdiv_values mm_div_ne11;
 
-    // Per thread quant tasks
     // Precomputed block-parallel quantization values
-    worker_callback_t quant_task_func;
     uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
     uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
     uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
     uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
     uint32_t          n_quant_tasks;
     uint32_t          n_quant_rows_per_thread;
-    atomic_uint       quant_barrier;
 
     // Fields for scattered mapping & HMX support in MUL_MAT_ID
     const uint32_t * matrix_row_counts;
@@ -125,12 +122,14 @@ struct htp_mm_context {
     uint8_t * vtcm_src2;
     uint8_t * vtcm_src3;
     uint8_t * vtcm_dst;
+    uint8_t * vtcm_act_raw;
 
     // Cached strides
     uint32_t vtcm_src0_stride;
     uint32_t vtcm_src1_stride;
     uint32_t vtcm_src2_stride;
     uint32_t vtcm_src3_stride;
+    uint32_t vtcm_act_raw_stride;
 
     // Cached thread offsets/sizes
     uint32_t vtcm_src0_size_per_thread;
@@ -234,17 +233,6 @@ static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
     uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
     htp_matmul_tensors_preamble;
 
-static inline void hvx_mm_run_quant_task(struct htp_mm_context * mmctx, unsigned int ith) {
-    if (mmctx->quant_task_func) {
-        if (ith < mmctx->n_quant_tasks) {
-            mmctx->quant_task_func(mmctx->n_quant_tasks, ith, mmctx);
-            atomic_fetch_sub(&mmctx->quant_barrier, 1);
-        }
-        while (atomic_load(&mmctx->quant_barrier) > 0) {
-            // spin
-        }
-    }
-}
 
 
 
@@ -301,8 +289,6 @@ static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
         }                                                                                                                                  \
     }                                                                                                                                      \
                                                                                                                                            \
-    hvx_mm_run_quant_task(mmctx, ith);                                                                                                     \
-                                                                                                                                           \
     if (src0_start_row >= src0_end_row) {                                                                                                  \
         return;                                                                                                                            \
     }                                                                                                                                      \
@@ -416,8 +402,6 @@ static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
         }                                                                                                                \
     }                                                                                                                    \
                                                                                                                          \
-    hvx_mm_run_quant_task(mmctx, ith);                                                                                   \
-                                                                                                                         \
     if (src0_start_row >= src0_end_row) {                                                                                \
         return;                                                                                                          \
     }                                                                                                                    \
@@ -479,8 +463,6 @@ static void hvx_mm_nx_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, v
     uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
     uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                   \
-    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
-                                                                                                                                  \
     for (uint32_t widx = 0; widx < n_weights; widx++) {                                                                           \
         const struct htp_tensor * restrict src_w = octx->src[widx];                                                               \
         const struct htp_tensor * restrict dst   = octx->dsts[widx];                                                              \
@@ -562,63 +544,124 @@ MATMUL_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_do
 MATMUL_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
 MATMUL_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
 MATMUL_2D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x2,  tiled_vec_dot_q6_k_32x1)
+MATMUL_2D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)
 MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
 MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
 
-#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                                                                               \
-static void name(unsigned int nth, unsigned int ith, void * data) {                                                                                               \
-    struct htp_mm_context * mmctx = data;                                                                                                                         \
-    struct htp_ops_context * octx = mmctx->octx;                                                                                                                  \
-    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                                                      \
-    const struct htp_tensor * src = mmctx->act;                                                                                                                   \
-    const uint32_t ne0 = src->ne[0];                                                                                                                              \
-    const uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->src1_nrows;                                                                             \
-    const uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                                                                             \
-                                                                                                                                                                  \
-    const uint32_t ir_first = nrows_per_thread * ith;                                                                                                             \
-    if (ir_first >= nrows) {                                                                                                                                      \
-        return;                                                                                                                                                   \
-    }                                                                                                                                                             \
-                                                                                                                                                                  \
-    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                                        \
-    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                                                                               \
-                                                                                                                                                                  \
-    uint8_t * restrict dst = mmctx->vtcm_src1;                                                                                                                    \
-    const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                                                                             \
-    const size_t src_row_size = src->nb[1];                                                                                                                       \
-    const size_t dst_row_size = (dst_row_size_expr);                                                                                                              \
-    uint8_t * restrict tmp_data = (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith);                                                          \
-                                                                                                                                                                  \
-    const bool is_contiguous = (src->nb[2] == src->ne[1] * src->nb[1]) && (src->nb[3] == src->ne[2] * src->nb[2]);                                                \
-    if (is_contiguous) {                                                                                                                                          \
-        const uint8_t * restrict src_data = (const uint8_t *) src->data + (src_row_size * (mmctx->cur_m_start + ir_first));                                       \
-        uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                                                                                \
-        kernel_fn(src_data, dst_data, tmp_data, ne0, ir_last - ir_first, src_row_size, dst_row_size);                                                             \
-    } else {                                                                                                                                                      \
-        const uint32_t ne12_ne1 = src->ne[2] * src->ne[1];                                                                                                        \
-        for (uint32_t ir = ir_first; ir < ir_last; ++ir) {                                                                                                        \
-            const uint32_t ir1 = mmctx->cur_m_start + ir;                                                                                                         \
-            const uint32_t i13 = fastdiv(ir1, &kparams->div_ne12_ne1);                                                                                            \
-            const uint32_t rem = ir1 - i13 * ne12_ne1;                                                                                                            \
-            const uint32_t i12 = fastdiv(rem, &kparams->div_ne1);                                                                                                 \
-            const uint32_t i11 = rem - i12 * src->ne[1];                                                                                                          \
-            const uint8_t * restrict row_src = (const uint8_t *) src->data + ((size_t) i11 * src->nb[1] + (size_t) i12 * src->nb[2] + (size_t) i13 * src->nb[3]); \
-            uint8_t * restrict row_dst = dst + (dst_row_size * ir);                                                                                               \
-            kernel_fn(row_src, row_dst, tmp_data, ne0, 1, src_row_size, dst_row_size);                                                                            \
-        }                                                                                                                                                         \
-    }                                                                                                                                                             \
-                                                                                                                                                                  \
-    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                                                                                \
+static void hvx_mm_transfer_src1_dma(
+    struct htp_ops_context * octx,
+    const struct htp_mm_kernel_params * kparams,
+    const struct htp_tensor * src1,
+    uint8_t * dst_base,
+    size_t dst_row_size,
+    uint32_t m_start,
+    uint32_t m_rows
+) {
+    if (m_rows == 0) {
+        return;
+    }
+
+    dma_queue * dma_q = octx->ctx->dma[0];
+    const uint32_t ne0 = src1->ne[0];
+    const size_t elem_size = (src1->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
+    const size_t row_bytes = ne0 * elem_size;
+    const size_t src1_nb1 = src1->nb[1];
+    const dma_addr_t src_base = src1->data;
+
+    const bool is_contiguous = (src1->nb[2] == src1->ne[1] * src1_nb1) &&
+                               (src1->nb[3] == src1->ne[2] * src1->nb[2]);
+
+    if (is_contiguous) {
+        const dma_addr_t src_addr = src_base + m_start * src1_nb1;
+        dma_queue_push(dma_q, dma_make_data(dst_base, src_addr),
+                       dst_row_size, src1_nb1, row_bytes, m_rows);
+        dma_queue_pop(dma_q);
+    } else {
+        const uint32_t ne12_ne1 = src1->ne[2] * src1->ne[1];
+        const bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
+        for (uint32_t ir = 0; ir < m_rows; ++ir) {
+            const uint32_t ir1 = m_start + ir;
+            uint32_t i11, i12, i13;
+            if (use_fastdiv) {
+                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
+                const uint32_t rem = ir1 - i13 * ne12_ne1;
+                i12 = fastdiv(rem, &kparams->div_ne1);
+                i11 = rem - i12 * src1->ne[1];
+            } else {
+                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
+                const uint32_t rem = ir1 - i13 * ne12_ne1;
+                i12 = src1->ne[1] ? rem / src1->ne[1] : 0;
+                i11 = rem - i12 * src1->ne[1];
+            }
+            const dma_addr_t row_src = src_base + (i11 * src1->nb[1] +
+                                                   i12 * src1->nb[2] +
+                                                   i13 * src1->nb[3]);
+            uint8_t * row_dst = dst_base + ir * dst_row_size;
+            dma_queue_push(dma_q, dma_make_data(row_dst, row_src),
+                           dst_row_size, src1_nb1, row_bytes, 1);
+            dma_queue_pop(dma_q);
+        }
+    }
+
+    if (dst_row_size > row_bytes) {
+        struct htp_thread_trace * tr = &octx->ctx->trace[0];
+        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
+        const uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
+        if (elem_size == sizeof(float)) {
+            for (uint32_t ir = 0; ir < m_rows; ++ir) {
+                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
+            }
+        } else {
+            for (uint32_t ir = 0; ir < m_rows; ++ir) {
+                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
+            }
+        }
+        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
+    }
+}
+
+#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
+static void name(unsigned int nth, unsigned int ith, void * data) {                                        \
+    (void) nth;                                                                                            \
+    struct htp_mm_context * mmctx = data;                                                                  \
+    struct htp_ops_context * octx = mmctx->octx;                                                           \
+    const struct htp_tensor * src = mmctx->act;                                                            \
+    const uint32_t ne0 = src->ne[0];                                                                       \
+    const uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
+    const uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
+                                                                                                           \
+    const uint32_t ir_first = nrows_per_thread * ith;                                                      \
+    if (ir_first >= nrows) {                                                                               \
+        return;                                                                                            \
+    }                                                                                                      \
+                                                                                                           \
+    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
+    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
+                                                                                                           \
+    uint8_t * restrict dst = mmctx->vtcm_src1;                                                             \
+    const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
+    const size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
+    const size_t dst_row_size = (dst_row_size_expr);                                                       \
+                                                                                                           \
+    const uint8_t * restrict src_data = (const uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
+    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
+    kernel_fn(src_data, dst_data, NULL, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
+                                                                                                           \
+    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
 }
 
 QUANTIZE_IMPL(quantize_f32_q8_0_tiled, "quantize-f32-q8_0_tiled", quantize_f32_q8_0_tiled_kernel, htp_mm_q8_0_tiled_row_size(ne0))
 QUANTIZE_IMPL(quantize_f32_q8_1_tiled, "quantize-f32-q8_1_tiled", quantize_f32_q8_1_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
-QUANTIZE_IMPL(quantize_f32_f32,       "quantize-f32-f32",       quantize_f32_f32_kernel,       mmctx->vtcm_src1_stride)
-QUANTIZE_IMPL(quantize_f32_f16,       "quantize-f32-f16",       quantize_f32_f16_kernel,       mmctx->vtcm_src1_stride)
-QUANTIZE_IMPL(quantize_f16_f16,       "quantize-f16-f16",       quantize_f16_f16_kernel,       mmctx->vtcm_src1_stride)
+QUANTIZE_IMPL(quantize_f32_f32,        "quantize-f32-f32",        quantize_f32_f32_kernel,        mmctx->vtcm_src1_stride)
+QUANTIZE_IMPL(quantize_f32_f16,        "quantize-f32-f16",        quantize_f32_f16_kernel,        mmctx->vtcm_src1_stride)
+QUANTIZE_IMPL(quantize_f16_f16,        "quantize-f16-f16",        quantize_f16_f16_kernel,        mmctx->vtcm_src1_stride)
 
 static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
+    (void) nth;
     struct htp_mm_context * mmctx = data;
+    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
+        return;
+    }
     struct htp_ops_context * octx = mmctx->octx;
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
     htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
@@ -626,13 +669,13 @@ static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, vo
     const struct htp_tensor * src = mmctx->act;
 
     quantize_f32_q8_0_tiled_block_kernel(
-        (const float *) src->data,
+        (const float *) mmctx->vtcm_act_raw,
         mmctx->vtcm_src1,
-        (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith),
+        NULL,
         src->ne[0],
         mmctx->quant_ib_first[ith],
         mmctx->quant_ib_last[ith],
-        src->nb[1],
+        mmctx->vtcm_act_raw_stride,
         htp_mm_q8_0_tiled_row_size(src->ne[0]),
         mmctx->quant_r[ith],
         mmctx->quant_c[ith]
@@ -642,7 +685,11 @@ static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, vo
 }
 
 static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, void * data) {
+    (void) nth;
     struct htp_mm_context * mmctx = data;
+    if (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
+        return;
+    }
     struct htp_ops_context * octx = mmctx->octx;
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
     htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
@@ -650,13 +697,13 @@ static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, vo
     const struct htp_tensor * src = mmctx->act;
 
     quantize_f32_q8_1_tiled_block_kernel(
-        (const float *) src->data,
+        (const float *) mmctx->vtcm_act_raw,
         mmctx->vtcm_src1,
-        (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith),
+        NULL,
         src->ne[0],
         mmctx->quant_ib_first[ith],
         mmctx->quant_ib_last[ith],
-        src->nb[1],
+        mmctx->vtcm_act_raw_stride,
         htp_mm_q8_1_tiled_row_size(src->ne[0]),
         mmctx->quant_r[ith],
         mmctx->quant_c[ith]
@@ -668,15 +715,17 @@ static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, vo
 MATVEC_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x1)
 MATVEC_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x1)
 MATVEC_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x1)
+MATVEC_2D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x1)
 MATVEC_2D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x1)
 MATVEC_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x1)
 MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)
 
-MATMUL_NX_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
-MATMUL_NX_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
-MATMUL_NX_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
-MATMUL_NX_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
-MATMUL_NX_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
+MATMUL_NX_2D_REPACKED_IMPL(q4_0,    576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
+MATMUL_NX_2D_REPACKED_IMPL(q4_1,    640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
+MATMUL_NX_2D_REPACKED_IMPL(q8_0,    1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
+MATMUL_NX_2D_REPACKED_IMPL(iq4nl,   576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
+MATMUL_NX_2D_REPACKED_IMPL(mxfp4,   544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
+MATMUL_NX_2D_REPACKED_IMPL(q5_k,    768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)
 
 #define MATMUL_4D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                                                        \
 static void hvx_mm_4d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                                                  \
@@ -713,7 +762,6 @@ static void hvx_mm_4d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     const uint32_t ct_start = src0_start_row / 32;                                                                                                          \
     const uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                                     \
                                                                                                                                                             \
-    hvx_mm_run_quant_task(mmctx, ith);                                                                                                                      \
                                                                                                                                                             \
     if (src0_start_row >= src0_end_row || cur_m_rows == 0) {                                                                                                \
         return;                                                                                                                                             \
@@ -810,6 +858,7 @@ MATMUL_4D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_do
 MATMUL_4D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
 MATMUL_4D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
 MATMUL_4D_REPACKED_IMPL(q6_k,       896,  tiled_vec_dot_q6_k_32x2,  tiled_vec_dot_q6_k_32x1)
+MATMUL_4D_REPACKED_IMPL(q5_k,       768,  tiled_vec_dot_q5_k_32x2,  tiled_vec_dot_q5_k_32x1)
 MATMUL_4D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
 MATMUL_4D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
 
@@ -822,7 +871,7 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t prefetch_mask = n_prefetch - 1;
 
     const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
-    const uint32_t src1_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->src1_nrows;                          // src1 rows
+    const uint32_t src1_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                          // src1 rows
     const uint32_t cur_m_start = mmctx->cur_m_start;
 
     const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
@@ -845,6 +894,7 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
 
     const dma_addr_t src0_row = src0->data;
 
+
     // Prefill vtcm with src0 rows
     if (src0_start_row < src0_end_row) {
         for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
@@ -857,8 +907,6 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
         }
     }
 
-    hvx_mm_run_quant_task(mmctx, ith);
-
     if (src0_start_row >= src0_end_row) {
         return;
     }
@@ -974,7 +1022,6 @@ static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
         }
     }
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     if (src0_start_row >= src0_end_row) {
         return;
@@ -1049,7 +1096,6 @@ static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
     uint8_t * restrict src1_data     = mmctx->vtcm_src1;
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     if (src0_start_row >= src0_end_row || cur_m_rows == 0) {
         return;
@@ -1183,7 +1229,6 @@ static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
     const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     if (src0_start_row >= src0_end_row) {
         return;
@@ -1272,7 +1317,6 @@ static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
     const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     if (src0_start_row >= src0_end_row) {
         return;
@@ -1351,7 +1395,6 @@ static void hvx_mv_id_nx(unsigned int nth, unsigned int ith, void * data) {
     const struct htp_tensor * restrict act  = octx->src[n_weights];
     const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
@@ -1442,7 +1485,6 @@ static void hvx_mm_id_nx(unsigned int nth, unsigned int ith, void * data) {
     const struct htp_tensor * restrict act  = octx->src[n_weights];
     const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
@@ -1550,6 +1592,10 @@ static int hvx_mm_init_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type
             mmctx->type         = "q8_0_tiled-f32";
             mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
             return 0;
+        case HTP_TYPE_Q5_K:
+            mmctx->type         = "q5_k_tiled-f32";
+            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
+            return 0;
         case HTP_TYPE_Q6_K:
             mmctx->type         = "q6_k_tiled-f32";
             mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
@@ -1582,7 +1628,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
     const uint32_t src0_nrows = ne01;
     const uint32_t src1_nrows = ne11 * ne12 * ne13;
-    mmctx->src1_nrows = src1_nrows;
+    mmctx->act_nrows = src1_nrows;
 
     uint32_t src0_row_start = 0;
     uint32_t src0_row_end   = src0_nrows;
@@ -1605,7 +1651,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
     bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                         src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                         src0->type == HTP_TYPE_MXFP4 || src0->type == HTP_TYPE_Q6_K ||
-                        src0->type == HTP_TYPE_Q4_K);
+                        src0->type == HTP_TYPE_Q4_K || src0->type == HTP_TYPE_Q5_K);
 
     // Compute src0_nrows_per_thread
     mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
@@ -1634,6 +1680,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
                 case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_4d_repacked_q4_1;   break;
                 case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_4d_repacked_q8_0;   break;
                 case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mm_4d_repacked_q6_k;   break;
+                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_4d_repacked_q5_k;   break;
                 case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_4d_repacked_iq4nl;  break;
                 case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_4d_repacked_mxfp4;  break;
                 default:              return HTP_STATUS_NO_SUPPORT;
@@ -1649,6 +1696,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
                 case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_2d_repacked_q4_1;   break;
                 case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;
                 case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mm_2d_repacked_q6_k;   break;
+                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_2d_repacked_q5_k;   break;
                 case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl;  break;
                 case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4;  break;
                 default:              return HTP_STATUS_NO_SUPPORT;
@@ -1663,6 +1711,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
                 case HTP_TYPE_Q4_1:
                 case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mv_2d_repacked_q4_1;   break;
                 case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;
+                case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mv_2d_repacked_q5_k;   break;
                 case HTP_TYPE_Q6_K:   matmul_job_func = hvx_mv_2d_repacked_q6_k;   break;
                 case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl;  break;
                 case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4;  break;
@@ -1678,7 +1727,8 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
     switch (kparams->kernel_type) {
         case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
             quant_task_func        = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
-            mmctx->type            = "f16-f16";
+            need_quant             = (src1->type == HTP_TYPE_F32);
+            mmctx->type            = (src1->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
             mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
             mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
             mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
@@ -1686,7 +1736,8 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
             break;
 
         case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
-            quant_task_func        = quantize_f32_f32;
+            quant_task_func        = NULL;
+            need_quant             = false;
             mmctx->type            = "f32-f32";
             mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
             mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
@@ -1707,7 +1758,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
             if (src1_nrows < octx->n_threads && !is_batched) {
                 n_quant_tasks = MIN(total_nb, octx->n_threads);
-                quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
+                quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
                 for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                     uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                     uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
@@ -1718,9 +1769,9 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
                 }
             } else {
                 n_quant_tasks = MIN(src1_nrows, octx->n_threads);
-                quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
+                quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
             }
-            src1_row_size = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+            src1_row_size = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
             break;
     }
 
@@ -1760,10 +1811,11 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
     }
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
-    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
-    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
-    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
+    mmctx->vtcm_src2     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
+    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
     octx->src1_spad.src  = NULL;
     octx->src0_spad.src  = NULL;
@@ -1771,46 +1823,68 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
     mmctx->vtcm_src0_stride = src0_row_size_padded;
     mmctx->vtcm_src1_stride = src1_row_size;
+    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
+        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
+    } else if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
+        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), 128);
+    } else {
+        mmctx->vtcm_act_raw_stride = 0;
+    }
 
-    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < src1_nrows) {
-        atomic_init(&mmctx->quant_barrier, 0);
-        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
+    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
+    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < src1_nrows) {
         for (uint32_t m_start = 0; m_start < src1_nrows; m_start += m_chunk) {
             const uint32_t cur_m_rows = MIN(src1_nrows - m_start, m_chunk);
             mmctx->cur_m_start = m_start;
             mmctx->cur_m_rows  = cur_m_rows;
 
             if (need_quant) {
-                const uint32_t quant_tasks = MIN(cur_m_rows, octx->n_threads);
-                mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
+                hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);
+
+                const uint32_t qk = QK_Q8_0_TILED;
+                const uint32_t nb = (ne10 + qk - 1) / qk;
+                const uint32_t total_nb = cur_m_rows * nb;
+                uint32_t quant_tasks;
+                work_queue_func_t q_func;
+                if (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
+                    quant_tasks = MIN(total_nb, octx->n_threads);
+                    q_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
+                    for (uint32_t ith = 0; ith < quant_tasks; ++ith) {
+                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
+                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
+                        mmctx->quant_ib_first[ith] = ib_first;
+                        mmctx->quant_ib_last[ith]  = ib_last;
+                        mmctx->quant_r[ith]        = ib_first / nb;
+                        mmctx->quant_c[ith]        = ib_first % nb;
+                    }
+                } else {
+                    quant_tasks = MIN(cur_m_rows, octx->n_threads);
+                    q_func = quant_task_func;
+                    mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
+                }
                 mmctx->n_quant_tasks = quant_tasks;
-                atomic_store(&mmctx->quant_barrier, quant_tasks);
-                mmctx->quant_task_func = quant_task_func;
+                work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
             } else {
-                mmctx->quant_task_func = NULL;
-                mmctx->n_quant_tasks = 0;
+                hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_src1, mmctx->vtcm_src1_stride, m_start, cur_m_rows);
             }
 
-            worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, octx->n_threads);
+            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
         }
     } else {
         mmctx->cur_m_start = 0;
         mmctx->cur_m_rows  = src1_nrows;
 
         if (need_quant) {
+            hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, src1_nrows);
             mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
-            mmctx->quant_task_func = quant_task_func;
             mmctx->n_quant_tasks = n_quant_tasks;
-            atomic_init(&mmctx->quant_barrier, n_quant_tasks);
+            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
         } else {
-            mmctx->quant_task_func = NULL;
-            mmctx->n_quant_tasks = 0;
+            hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_src1, mmctx->vtcm_src1_stride, 0, src1_nrows);
         }
 
-        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
-
-        worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, octx->n_threads);
+        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
     }
 
     return HTP_STATUS_OK;
@@ -1835,7 +1909,6 @@ static void hvx_mm_nx_2d(unsigned int nth, unsigned int ith, void * data) {
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
-    hvx_mm_run_quant_task(mmctx, ith);
 
     for (uint32_t widx = 0; widx < n_weights; widx++) {
         const struct htp_tensor * restrict src_w = octx->src[widx];
@@ -1938,6 +2011,7 @@ DEQUANTIZE_WORKER_LOOP_IMPL(iq4_nl)
 DEQUANTIZE_WORKER_LOOP_IMPL(mxfp4)
 DEQUANTIZE_WORKER_LOOP_IMPL(q8_0)
 DEQUANTIZE_WORKER_LOOP_IMPL(q6_k)
+DEQUANTIZE_WORKER_LOOP_IMPL(q5_k)
 
 static void convert_f16_worker_loop(unsigned int n, unsigned int i, void *data) {
     tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;
@@ -1990,8 +2064,7 @@ typedef struct {
     struct htp_context            * ctx;
     struct htp_thread_trace       * traces;
     __fp16                        * dst;
-    const float                   * src;
-    const struct mmid_row_mapping * matrix_rows;
+    dma_addr_t                      act_dma_addr;
     float                         * vtcm_f32_act;
     uint32_t                        n_tasks;
     uint32_t                        n_tot_chunks;
@@ -2008,7 +2081,7 @@ typedef struct {
     struct htp_context            * ctx;
     struct htp_thread_trace       * traces;
     __fp16                        * dst;
-    const float                   * src;
+    dma_addr_t                      act_dma_addr;
     float                         * vtcm_f32_act;
     uint32_t                        n_rows;
     uint32_t                        k_block;
@@ -2024,7 +2097,7 @@ typedef struct {
 static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
         dma_queue *dma_q,
         __fp16 *restrict vtcm_dst,
-        const float *restrict src,
+        dma_addr_t act_dma_addr,
         uint32_t n_rows,
         uint32_t k_block,
         uint32_t k_stride,
@@ -2044,7 +2117,7 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
     // Push step 0
     if (n_steps > 0 && n_rows > 0) {
         uint32_t nrows_to_fetch = hex_smin(n_rows, R);
-        dma_queue_push(dma_q, dma_make_data(thread_f32_act, src + c_first),
+        dma_queue_push(dma_q, dma_make_data(thread_f32_act, act_dma_addr + (size_t) c_first * sizeof(float)),
                        c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
     }
     // Push step 1
@@ -2052,9 +2125,8 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
         uint32_t next_r = R * 1;
         if (next_r < n_rows) {
             uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
-            const float *next_src = src + next_r * k_stride + c_first;
             float *next_buf = thread_f32_act + 1 * R * c_len;
-            dma_queue_push(dma_q, dma_make_data(next_buf, next_src),
+            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * sizeof(float)),
                            c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
         }
     }
@@ -2084,50 +2156,12 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
         uint32_t next_r = next_s << dma_step_rows_shift;
         if (next_r < n_rows) {
             uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
-            const float *next_src = src + next_r * k_stride + c_first;
-            dma_queue_push(dma_q, dma_make_data(curr_buf, next_src),
+            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * sizeof(float)),
                            c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
         }
     }
 }
 
-static void transfer_activation_chunk_fp32_to_fp16_col_chunk(
-        __fp16 *restrict vtcm_dst,
-        const float *restrict src,
-        uint32_t n_rows,
-        uint32_t k_block,
-        uint32_t k_stride,
-        uint32_t c_first,
-        uint32_t c_len,
-        uint32_t k_chunk_valid) {
-    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);
-    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_HMX_TILE_N_ROWS) * HTP_MM_HMX_TILE_N_ROWS;
-
-    uint32_t r = 0;
-
-    #pragma unroll(2)
-    for (r = 0; r < n_rows_tiled; r += 2) {
-        const float *ptr_in0 = src + (r + 0) * k_stride + c_first;
-        const float *ptr_in1 = src + (r + 1) * k_stride + c_first;
-
-        transfer_activation_row_pair_fp32_to_fp16_col_chunk(
-            vtcm_dst, ptr_in0, ptr_in1, r, k_block, c_first, c_len, k_chunk_valid, true, true
-        );
-    }
-
-    for (; r < n_rows_padded; r += 2) {
-        const bool row0_valid = r       < n_rows;
-        const bool row1_valid = (r + 1) < n_rows;
-
-        const float *ptr_in0 = row0_valid ? (src + (r + 0) * k_stride + c_first) : NULL;
-        const float *ptr_in1 = row1_valid ? (src + (r + 1) * k_stride + c_first) : NULL;
-
-        transfer_activation_row_pair_fp32_to_fp16_col_chunk(
-            vtcm_dst, ptr_in0, ptr_in1, r, k_block, c_first, c_len, k_chunk_valid, row0_valid, row1_valid
-        );
-    }
-}
-
 static void transfer_activation_chunk_col_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
     activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
     struct htp_thread_trace * tr = &st->traces[i];
@@ -2149,29 +2183,20 @@ static void transfer_activation_chunk_col_chunk_worker_fn(unsigned int n, unsign
     }
 
     __fp16 *dst = st->dst;
-    const float *src = st->src;
 
-    if (st->vtcm_f32_act) {
-        size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
-        float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * thread_scratch_bytes);
+    size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
+    float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * thread_scratch_bytes);
 
-        transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
-            st->ctx->dma[i], dst, src, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
-            c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
-        );
-    } else {
-        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, c_first);
-        transfer_activation_chunk_fp32_to_fp16_col_chunk(
-            dst, src, st->n_rows, st->k_block, st->k_stride, c_first, c_len, k_chunk_valid
-        );
-        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, c_first);
-    }
+    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
+        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
+        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
+    );
 }
 
 static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
         dma_queue *dma_q,
         __fp16 *restrict vtcm_dst,
-        const float *restrict src,
+        dma_addr_t act_dma_addr,
         uint32_t n_rows,
         uint32_t k_block,
         uint32_t k_stride,
@@ -2189,7 +2214,7 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
     // Push step 0
     if (n_steps > 0 && n_rows > 0) {
         uint32_t nrows_to_fetch = hex_smin(n_rows, R);
-        dma_queue_push(dma_q, dma_make_data(thread_f32_act, src),
+        dma_queue_push(dma_q, dma_make_data(thread_f32_act, act_dma_addr),
                        k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
     }
     // Push step 1 (if valid)
@@ -2197,9 +2222,8 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
         uint32_t next_r = R * 1;
         if (next_r < n_rows) {
             uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
-            const float *next_src = src + next_r * k_stride;
             float *next_buf = thread_f32_act + 1 * R * k_block;
-            dma_queue_push(dma_q, dma_make_data(next_buf, next_src),
+            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * k_stride * sizeof(float)),
                            k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
         }
     }
@@ -2227,8 +2251,7 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
         uint32_t next_r = next_s << dma_step_rows_shift;
         if (next_r < n_rows) {
             uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
-            const float *next_src = src + next_r * k_stride;
-            dma_queue_push(dma_q, dma_make_data(curr_buf, next_src),
+            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * k_stride * sizeof(float)),
                            k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
         }
     }
@@ -2244,18 +2267,12 @@ static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i,
         size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);
 
         __fp16      *dst = st->dst + chunk_idx * st->k_block;
-        const float *src = st->src + chunk_idx * st->k_stride;
+        const dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * sizeof(float);
 
-        if (st->vtcm_f32_act) {
-            float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread);
-            transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
-                st->ctx->dma[i], dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
-            );
-        } else {
-            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
-            transfer_activation_chunk_fp32_to_fp16(dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid);
-            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
-        }
+        float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread);
+        transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
+            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
+        );
     }
 }
 
@@ -2494,7 +2511,7 @@ static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst,
 struct activation_transfer_params {
     struct htp_context *          ctx;
     __fp16 *                      dst;
-    const float *                 src;
+    dma_addr_t                    act_dma_addr;
     int                           n_rows;
     int                           k_block;
     int                           k_stride;
@@ -2509,7 +2526,7 @@ struct activation_transfer_params {
 static void transfer_activation_chunk_threaded(const struct activation_transfer_params * params) {
     struct htp_context *          ctx                = params->ctx;
     __fp16 *                      dst                = params->dst;
-    const float *                 src                = params->src;
+    const dma_addr_t              act_dma_addr       = params->act_dma_addr;
     int                           n_rows             = params->n_rows;
     int                           k_block            = params->k_block;
     int                           k_stride           = params->k_stride;
@@ -2529,7 +2546,7 @@ static void transfer_activation_chunk_threaded(const struct activation_transfer_
         // Calculate step rows parameters for column-chunked dma pipelining
         uint32_t dma_step_rows = 2;
         uint32_t dma_step_rows_shift = 1;
-        if (vtcm_f32_act && vtcm_f32_act_bytes > 0 && k_block > 0) {
+        if (vtcm_f32_act_bytes > 0) {
             size_t thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
             size_t thread_scratch_elements = thread_scratch_bytes / sizeof(float);
             size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
@@ -2541,7 +2558,7 @@ static void transfer_activation_chunk_threaded(const struct activation_transfer_
 
         activation_transfer_col_chunk_state_t col_state;
         col_state.dst = dst;
-        col_state.src = src;
+        col_state.act_dma_addr = act_dma_addr;
         col_state.n_rows = n_rows;
         col_state.k_block = k_block;
         col_state.k_stride = k_stride;
@@ -2569,7 +2586,7 @@ static void transfer_activation_chunk_threaded(const struct activation_transfer_
     state.n_tot_chunks       = n_tot_chunks;
     state.n_chunks_per_task  = n_chunks_per_task;
     state.dst                = dst;
-    state.src                = src;
+    state.act_dma_addr       = act_dma_addr;
     state.k_block            = k_block;
     state.k_stride           = k_stride;
     state.k_valid            = k_valid;
@@ -2581,7 +2598,7 @@ static void transfer_activation_chunk_threaded(const struct activation_transfer_
 
     uint32_t dma_step_rows = 2;
     uint32_t dma_step_rows_shift = 1;
-    if (vtcm_f32_act && state.vtcm_f32_act_bytes_per_thread > 0 && k_block > 0) {
+    if (state.vtcm_f32_act_bytes_per_thread > 0) {
         size_t thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / sizeof(float);
         size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
         if (dma_step_rows_max >= 4) {
@@ -2639,7 +2656,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
                                   float *restrict dst,
                                   dma_addr_t src2_addr,
                                   size_t src2_bytes,
-                                  const float *activation,
+                                  dma_addr_t act_dma_addr,
                                   dma_addr_t weight,
                                   int m, int k, int n,
                                   int act_stride,
@@ -2663,7 +2680,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
     htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);
 
     if (k % 32 != 0 || n % 32 != 0) { return -1; }
-    if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }
+    if (!hex_is_aligned(dst, VLEN) || (act_dma_addr & (VLEN - 1)) != 0) { return -1; }
 
     size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
     if (row_stride == 0) {
@@ -2678,6 +2695,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
         case HTP_TYPE_Q4_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
         case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
         case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
+        case HTP_TYPE_Q5_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q5_k; break;
         case HTP_TYPE_Q6_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q6_k; break;
         case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
         case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
@@ -2703,7 +2721,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
     const size_t qweight_row_stride = is_quant ? (size_t)(n_k_tiles * aligned_tile_size) / 32 : 0;
 
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, false, pipeline, act_threads, aligned_tile_size, src2_bytes);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, src2_bytes);
 
     vtcm_used = L.total_bytes;
     if (vtcm_used > vtcm_budget) {
@@ -2755,7 +2773,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .src = activation + mr * act_stride,
+                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -2846,7 +2864,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .src = activation + mr * act_stride,
+                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -2925,10 +2943,10 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
     const int k_valid     = (int) act->ne[0];
     const int m           = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
     const int act_stride  = (int) (act->nb[1] / sizeof(float));
-    const float * activation = (const float *) act->data;
+    const dma_addr_t act_dma_addr = act->data;
 
     if (k % 32 != 0) { return HTP_STATUS_NO_SUPPORT; }
-    if (!hex_is_aligned(activation, VLEN)) { return HTP_STATUS_NO_SUPPORT; }
+    if ((act_dma_addr & (VLEN - 1)) != 0) { return HTP_STATUS_NO_SUPPORT; }
 
     size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
     if (row_stride == 0) {
@@ -2943,6 +2961,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
         case HTP_TYPE_Q4_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
         case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
         case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
+        case HTP_TYPE_Q5_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q5_k; break;
         case HTP_TYPE_Q6_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q6_k; break;
         case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
         case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
@@ -2970,7 +2989,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
     const uint32_t dma_width_bytes = is_quant ? tile_size : row_stride;
 
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, false, pipeline, act_threads, aligned_tile_size, 0);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, 0);
 
     if (L.total_bytes > vtcm_budget) {
         FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
@@ -3026,7 +3045,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .src = activation + mr * act_stride,
+                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -3124,7 +3143,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .src = activation + mr * act_stride,
+                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -3202,11 +3221,10 @@ static inline dma_addr_t hmx_mm_weight_batch_data(const hmx_mm_f16_f32_batched_p
     return params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
 }
 
-static inline const float *hmx_mm_activation_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
-                                                           int dst_b2, int dst_b3) {
-    return (const float *) ((const uint8_t *) params->activation +
-                            (size_t) dst_b2 * params->src1_nb2 +
-                            (size_t) dst_b3 * params->src1_nb3);
+static inline dma_addr_t hmx_mm_act_batch_addr(const hmx_mm_f16_f32_batched_params_t *params,
+                                                int dst_b2, int dst_b3) {
+    return params->act_dma_addr + dst_b2 * params->act_nb2 +
+           dst_b3 * params->act_nb3;
 }
 
 static inline float *hmx_mm_dst_batch_ptr(const hmx_mm_f16_f32_batched_params_t *params,
@@ -3224,11 +3242,11 @@ static int hmx_mm_f16_f32_batched_simple(struct htp_context *ctx,
     for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
         for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
             dma_addr_t cur_src2_addr = params->src2_addr ? (params->src2_addr +
-                                       (dma_addr_t) b2 * params->src2_nb2 +
-                                       (dma_addr_t) b3 * params->src2_nb3) : 0;
+                                       b2 * params->src2_nb2 +
+                                       b3 * params->src2_nb3) : 0;
             ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                 cur_src2_addr, params->src2_bytes,
-                                hmx_mm_activation_batch_ptr(params, b2, b3),
+                                hmx_mm_act_batch_addr(params, b2, b3),
                                 hmx_mm_weight_batch_data(params, b2, b3),
                                 params->m, params->k, params->n,
                                 params->act_stride, params->weight_stride * (int)sizeof(__fp16),
@@ -3249,7 +3267,7 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
     if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
     if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
     if (params->k % 32 != 0 || params->n % 32 != 0) { return -1; }
-    if (!hex_is_aligned(params->dst, VLEN) || !hex_is_aligned(params->activation, VLEN)) { return -1; }
+    if (!hex_is_aligned(params->dst, VLEN) || (params->act_dma_addr & (VLEN - 1)) != 0) { return -1; }
 
     const int group_size = params->r2;
     const size_t vtcm_budget  = ctx->vtcm_size;
@@ -3267,16 +3285,12 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
 
     const size_t vec_dot_size = params->k * sizeof(__fp16);
 
-    const bool use_dma_activation = (params->act_stride > params->k);
-    const size_t f32_scratch_size = use_dma_activation
-        ? hex_align_up((size_t)act_threads * HTP_MM_DMA_ACT_MULTIPLIER * (size_t) params->k * sizeof(float), HTP_MM_HMX_TILE_SIZE) : 0;
-
     size_t m_chunk_n_rows = m_chunk;
     size_t n_chunk_n_cols = n_chunk;
     size_t vtcm_used = vtcm_size;
 
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, use_dma_activation, false, act_threads, 0, params->src2_bytes);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->src2_bytes);
 
     if (L.total_bytes > vtcm_budget) {
         FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
@@ -3291,7 +3305,7 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
     void    *vtcm_scratch0   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[0]);
     void    *vtcm_scratch1   = VTCM_LAYOUT_PTR(void, base, L.off_scratch[1]);
     __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
-    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_act_f32, use_dma_activation);
+    float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
 
     const bool has_src2      = (params->src2_bytes > 0 && params->src2_addr != 0);
     float   *vtcm_src2       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_src2, has_src2);
@@ -3329,12 +3343,13 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
                 // converts from the contiguous VTCM buffer.  This avoids L2 cache
                 // thrashing from HVX loads at large strides.
                 for (int g = 0; g < group_size; ++g) {
-                    const float *activation_chunk = hmx_mm_activation_batch_ptr(params, b2_base + g, b3) + mr * params->act_stride;
+                    const dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
+                                                      mr * params->act_stride * sizeof(float);
                     __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                     struct activation_transfer_params act_params = {
                         .ctx = ctx,
                         .dst = vtcm_act_g,
-                        .src = activation_chunk,
+                        .act_dma_addr = act_dma_addr,
                         .n_rows = (int) n_rows,
                         .k_block = params->k,
                         .k_stride = params->act_stride,
@@ -3542,6 +3557,7 @@ static int hmx_mm_id_2d_f32(struct htp_context *ctx,
         case HTP_TYPE_Q4_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q4_1; break;
         case HTP_TYPE_MXFP4:  dequant_worker_fn = dequantize_tiled_worker_loop_mxfp4; break;
         case HTP_TYPE_Q8_0:   dequant_worker_fn = dequantize_tiled_worker_loop_q8_0; break;
+        case HTP_TYPE_Q5_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q5_k; break;
         case HTP_TYPE_Q6_K:   dequant_worker_fn = dequantize_tiled_worker_loop_q6_k; break;
         case HTP_TYPE_F16:    dequant_worker_fn = convert_f16_worker_loop; break;
         case HTP_TYPE_F32:    dequant_worker_fn = quantize_f32_worker_loop; break;
@@ -3692,15 +3708,15 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
     size_t src2_nb3 = 0;
     if (src2) {
         src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / sizeof(float));
-        src2_addr = src2->data + (dma_addr_t) m_start * src2_stride * sizeof(float);
+        src2_addr = src2->data + m_start * src2_stride * sizeof(float);
         src2_bytes = (size_t) kparams->vtcm_src2_size;
         src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
         src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
     }
 
     const int dst_stride = (int)(dst->nb[1] / sizeof(float));
-    float       * dst_ptr = (float *)       dst->data  + m_start * dst_stride;
-    const float * act_ptr = (const float *) src1->data + m_start * act_stride;
+    float       * dst_ptr = (float *) dst->data + m_start * dst_stride;
+    const dma_addr_t act_addr = src1->data + m_start * act_stride * sizeof(float);
 
     int ret = -1;
     const int n_threads = kparams->n_threads;
@@ -3709,7 +3725,7 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
             .dst             = dst_ptr,
             .src2_addr       = src2_addr,
             .src2_bytes      = src2_bytes,
-            .activation      = act_ptr,
+            .act_dma_addr    = act_addr,
             .weight          = src0->data,
             .weight_dma      = octx->ctx->dma[0],
             .m               = m_rows,
@@ -3725,8 +3741,8 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
             .ne13            = ne13,
             .src0_nb2        = src0->nb[2],
             .src0_nb3        = src0->nb[3],
-            .src1_nb2        = src1->nb[2],
-            .src1_nb3        = src1->nb[3],
+            .act_nb2         = src1->nb[2],
+            .act_nb3         = src1->nb[3],
             .dst_nb2         = dst->nb[2],
             .dst_nb3         = dst->nb[3],
             .src2_nb2        = src2_nb2,
@@ -3745,7 +3761,8 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
                                      kparams->vtcm_size);
     } else {
         ret = hmx_mm_2d_f32(
-            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes, act_ptr, src0->data,
+            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
+            act_addr, src0->data,
             m_rows, k, n, act_stride, (int) src0->nb[1], (int) src0->type, (int) src1->ne[0],
             dst_stride, src2_stride, (int)dst->ne[0],
             kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
@@ -3829,7 +3846,7 @@ static int hvx_mm_matmul_id(
 ) {
     htp_matmul_tensors_preamble;
     const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
-    const uint32_t src1_nrows           = mmctx->src1_nrows;
+    const uint32_t act_nrows            = mmctx->act_nrows;
 
     struct htp_thread_trace * tr = &octx->ctx->trace[0];
     htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);
@@ -3840,13 +3857,13 @@ static int hvx_mm_matmul_id(
 
     const uint32_t qk = QK_Q8_0_TILED;
     const uint32_t nb = (ne10 + qk - 1) / qk;
-    const uint32_t total_nb = src1_nrows * nb;
+    const uint32_t total_nb = act_nrows * nb;
 
     work_queue_func_t quant_task_func;
     uint32_t n_quant_tasks = 1;
-    if (src1_nrows < octx->n_threads) {
+    if (act_nrows < octx->n_threads) {
         n_quant_tasks = MIN(total_nb, octx->n_threads);
-        quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
+        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
         for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
             uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
             uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
@@ -3856,13 +3873,13 @@ static int hvx_mm_matmul_id(
             mmctx->quant_c[ith]        = ib_first % nb;
         }
     } else {
-        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
-        quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
+        n_quant_tasks = MIN(act_nrows, octx->n_threads);
+        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
     }
-    size_t src1_row_size  = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+    size_t src1_row_size  = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
 
     struct htp_mm_hvx_vtcm_layout L;
-    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
+    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                  0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);
 
     const size_t vtcm_size = L.total_bytes;
@@ -3882,18 +3899,20 @@ static int hvx_mm_matmul_id(
     }
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
-    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
-    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src2 = NULL;
-    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
+    mmctx->vtcm_src2     = NULL;
+    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
     octx->src1_spad.src  = NULL;
     octx->src0_spad.src  = NULL;
     octx->src2_spad.src  = NULL;
     octx->dst_spad.src   = NULL;
 
-    mmctx->vtcm_src0_stride = src0_row_size_padded;
-    mmctx->vtcm_src1_stride = src1_row_size;
+    mmctx->vtcm_src0_stride    = src0_row_size_padded;
+    mmctx->vtcm_src1_stride    = src1_row_size;
+    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
     mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
@@ -3901,16 +3920,17 @@ static int hvx_mm_matmul_id(
     mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);
 
     mmctx->cur_m_start = 0;
-    mmctx->cur_m_rows  = src1_nrows;
-
-    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
-    mmctx->quant_task_func = quant_task_func;
-    mmctx->n_quant_tasks = n_quant_tasks;
-    atomic_init(&mmctx->quant_barrier, n_quant_tasks);
+    mmctx->cur_m_rows  = act_nrows;
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    worker_pool_run_func(octx->ctx->worker_pool, hvx_mmid_task_func, mmctx, octx->n_threads);
+    hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+
+    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
+    mmctx->n_quant_tasks = n_quant_tasks;
+    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
+
+    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);
 
     return HTP_STATUS_OK;
 }
@@ -3976,7 +3996,7 @@ static int hvx_mm_matmul_id_nx(
     work_queue_func_t hvx_mmid_task_func
 ) {
     const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
-    const uint32_t src1_nrows           = mmctx->src1_nrows;
+    const uint32_t act_nrows            = mmctx->act_nrows;
 
     struct htp_thread_trace * tr = &octx->ctx->trace[0];
     htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);
@@ -3990,13 +4010,13 @@ static int hvx_mm_matmul_id_nx(
 
     const uint32_t qk = QK_Q8_0_TILED;
     const uint32_t nb = (act->ne[0] + qk - 1) / qk;
-    const uint32_t total_nb = src1_nrows * nb;
+    const uint32_t total_nb = act_nrows * nb;
 
     work_queue_func_t quant_task_func;
     uint32_t n_quant_tasks = 1;
-    if (src1_nrows < octx->n_threads) {
+    if (act_nrows < octx->n_threads) {
         n_quant_tasks = MIN(total_nb, octx->n_threads);
-        quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
+        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
         for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
             uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
             uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
@@ -4006,13 +4026,13 @@ static int hvx_mm_matmul_id_nx(
             mmctx->quant_c[ith]        = ib_first % nb;
         }
     } else {
-        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
-        quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
+        n_quant_tasks = MIN(act_nrows, octx->n_threads);
+        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
     }
-    size_t src1_row_size = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);
+    size_t src1_row_size = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);
 
     struct htp_mm_hvx_vtcm_layout L;
-    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], src1_nrows, octx->n_threads,
+    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                  0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);
 
     const size_t vtcm_size = L.total_bytes;
@@ -4024,9 +4044,10 @@ static int hvx_mm_matmul_id_nx(
     }
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
-    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
-    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
+    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
     octx->src0_spad.src = NULL;
     octx->src1_spad.src = NULL;
@@ -4034,29 +4055,31 @@ static int hvx_mm_matmul_id_nx(
     octx->src3_spad.src = NULL;
     octx->dst_spad.src  = NULL;
 
-    mmctx->vtcm_src0_stride = 0;
-    mmctx->vtcm_src1_stride = src1_row_size;
+    mmctx->vtcm_src0_stride    = 0;
+    mmctx->vtcm_src1_stride    = src1_row_size;
+    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
     mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
     mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);
 
     mmctx->cur_m_start = 0;
-    mmctx->cur_m_rows  = src1_nrows;
-
-    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
-    mmctx->quant_task_func         = quant_task_func;
-    mmctx->n_quant_tasks           = n_quant_tasks;
-    atomic_init(&mmctx->quant_barrier, n_quant_tasks);
+    mmctx->cur_m_rows  = act_nrows;
 
     FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, src1 %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
          src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
-         act->ne[0], act->ne[1], act->ne[2], src1_nrows,
+         act->ne[0], act->ne[1], act->ne[2], act_nrows,
          L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    worker_pool_run_func(octx->ctx->worker_pool, hvx_mmid_task_func, mmctx, octx->n_threads);
+    hvx_mm_transfer_src1_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+
+    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
+    mmctx->n_quant_tasks           = n_quant_tasks;
+    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
+
+    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);
 
     return HTP_STATUS_OK;
 }
@@ -4202,7 +4225,7 @@ int op_matmul_id(struct htp_ops_context * octx) {
     mmctx->mapping_stride       = mapping_stride;
     mmctx->mm_div_ne11          = kparams->div_ne1;
     mmctx->src0_row_size_padded = src0_row_size_padded;
-    mmctx->src1_nrows           = src1_nrows;
+    mmctx->act_nrows            = src1_nrows;
     mmctx->cur_m_start          = 0;
     mmctx->cur_m_rows           = src1_nrows;
 
@@ -4281,7 +4304,7 @@ int op_matmul_id_nx(struct htp_ops_context * octx) {
     const size_t src0_row_size = src0->nb[1];
     const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
 
-    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];
+    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
 
     const int n_ids = ids->ne[0];
     const int n_as  = src0->ne[2];
@@ -4291,7 +4314,7 @@ int op_matmul_id_nx(struct htp_ops_context * octx) {
     uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
     struct mmid_row_mapping * matrix_rows = NULL;
 
-    if (src1_nrows > 1) {
+    if (act_nrows > 1) {
         const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
         assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);
 
@@ -4325,9 +4348,9 @@ int op_matmul_id_nx(struct htp_ops_context * octx) {
     mmctx->mapping_stride       = mapping_stride;
     mmctx->mm_div_ne11          = kparams->div_ne1;
     mmctx->src0_row_size_padded = src0_row_size_padded;
-    mmctx->src1_nrows           = src1_nrows;
+    mmctx->act_nrows            = act_nrows;
     mmctx->cur_m_start          = 0;
-    mmctx->cur_m_rows           = src1_nrows;
+    mmctx->cur_m_rows           = act_nrows;
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
@@ -4336,7 +4359,7 @@ int op_matmul_id_nx(struct htp_ops_context * octx) {
         s = hmx_mm_op_matmul_id_nx(octx, mmctx);
     } else {
         if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
-            s = hvx_mm_matmul_id_nx(octx, mmctx, src1_nrows > 1 ? hvx_mm_id_nx : hvx_mv_id_nx);
+            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx : hvx_mv_id_nx);
         } else {
             s = HTP_STATUS_NO_SUPPORT;
         }
@@ -4370,17 +4393,18 @@ int op_matmul_nx(struct htp_ops_context * octx) {
 
     bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                         src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
-                        src0->type == HTP_TYPE_MXFP4 || src0->type == HTP_TYPE_Q4_K);
+                        src0->type == HTP_TYPE_MXFP4 || src0->type == HTP_TYPE_Q4_K ||
+                        src0->type == HTP_TYPE_Q5_K);
 
     struct htp_mm_context mmctx_struct = {0};
     struct htp_mm_context * mmctx = &mmctx_struct;
     mmctx->octx = octx;
     mmctx->act  = act;
 
-    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];
-    mmctx->src1_nrows  = src1_nrows;
+    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
+    mmctx->act_nrows   = act_nrows;
     mmctx->cur_m_start = 0;
-    mmctx->cur_m_rows  = src1_nrows;
+    mmctx->cur_m_rows  = act_nrows;
 
     const size_t src0_row_size = src0->nb[1];
     const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
@@ -4391,13 +4415,13 @@ int op_matmul_nx(struct htp_ops_context * octx) {
 
     const uint32_t qk = QK_Q8_0_TILED;
     const uint32_t nb = (act->ne[0] + qk - 1) / qk;
-    const uint32_t total_nb = src1_nrows * nb;
+    const uint32_t total_nb = act_nrows * nb;
 
     worker_callback_t quant_task_func;
     uint32_t n_quant_tasks = 1;
-    if (src1_nrows < octx->n_threads) {
+    if (act_nrows < octx->n_threads) {
         n_quant_tasks = MIN(total_nb, octx->n_threads);
-        quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
+        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
         for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
             uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
             uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
@@ -4407,16 +4431,16 @@ int op_matmul_nx(struct htp_ops_context * octx) {
             mmctx->quant_c[ith]        = ib_first % nb;
         }
     } else {
-        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
-        quant_task_func = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
+        n_quant_tasks = MIN(act_nrows, octx->n_threads);
+        quant_task_func = htp_mm_weight_has_offset(src0->type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
     }
 
-    const size_t src1_row_size = (src0->type == HTP_TYPE_Q4_1 || src0->type == HTP_TYPE_Q4_K)
+    const size_t src1_row_size = htp_mm_weight_has_offset(src0->type)
                                ? htp_mm_q8_1_tiled_row_size(act->ne[0])
                                : htp_mm_q8_0_tiled_row_size(act->ne[0]);
 
     struct htp_mm_hvx_vtcm_layout L;
-    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], src1_nrows, octx->n_threads,
+    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                  0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, false, true);
 
     const size_t vtcm_size = L.total_bytes;
@@ -4428,9 +4452,10 @@ int op_matmul_nx(struct htp_ops_context * octx) {
     }
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
-    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
-    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
+    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
+    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
     octx->src0_spad.src  = NULL;
     octx->src1_spad.src  = NULL;
@@ -4438,18 +4463,14 @@ int op_matmul_nx(struct htp_ops_context * octx) {
     octx->src3_spad.src  = NULL;
     octx->dst_spad.src   = NULL;
 
-    mmctx->vtcm_src0_stride = is_repacked ? 0 : src0_row_size_padded;
-    mmctx->vtcm_src1_stride = src1_row_size;
+    mmctx->vtcm_src0_stride    = is_repacked ? 0 : src0_row_size_padded;
+    mmctx->vtcm_src1_stride    = src1_row_size;
+    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
     mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
     mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);
 
-    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
-    mmctx->quant_task_func = quant_task_func;
-    mmctx->n_quant_tasks = n_quant_tasks;
-    atomic_init(&mmctx->quant_barrier, n_quant_tasks);
-
     // Run fused matmul
     const uint32_t n_matmul_jobs = octx->n_threads;
     worker_callback_t matmul_job_func;
@@ -4458,10 +4479,13 @@ int op_matmul_nx(struct htp_ops_context * octx) {
             case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_0;   break;
             case HTP_TYPE_Q4_1:
             case HTP_TYPE_Q4_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_1;   break;
+            case HTP_TYPE_Q5_K:   matmul_job_func = hvx_mm_nx_2d_repacked_q5_k;   break;
             case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q8_0;   break;
             case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_nx_2d_repacked_iq4nl;  break;
             case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_nx_2d_repacked_mxfp4;  break;
-            default:              return HTP_STATUS_NO_SUPPORT;
+            default:
+                htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
+                return HTP_STATUS_NO_SUPPORT;
         }
     } else {
         matmul_job_func = hvx_mm_nx_2d;
@@ -4469,7 +4493,13 @@ int op_matmul_nx(struct htp_ops_context * octx) {
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);
+    hvx_mm_transfer_src1_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+
+    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
+    mmctx->n_quant_tasks = n_quant_tasks;
+    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
+
+    work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, n_matmul_jobs);
 
     return HTP_STATUS_OK;
 }
diff --git src/ggml-hexagon/htp/matmul-ops.h src/ggml-hexagon/htp/matmul-ops.h
index fe9dbb61..cfb3bfbb 100644
--- src/ggml-hexagon/htp/matmul-ops.h
+++ src/ggml-hexagon/htp/matmul-ops.h
@@ -25,6 +25,9 @@ extern "C" {
 #define HTP_MM_WEIGHT_TILE_SIZE_Q8_0   1088
 #define HTP_MM_WEIGHT_TILE_SIZE_IQ4_NL 576
 #define HTP_MM_WEIGHT_TILE_SIZE_MXFP4  544
+// Q5_K: the Q4_1 tile (640) followed by a 128-byte plane with the 5th bit of every quant, transposed so that
+//   plane byte l holds the eight flags of lane l: bit 2i = low nibble of nibble vector i, bit 2i+1 = high nibble
+#define HTP_MM_WEIGHT_TILE_SIZE_Q5_K   768
 // Q6_K native 6-bit tile (32 rows x 32 k), vrmpy-ready: byte 4*row+b of a vector holds k = 4*group+b
 //   vectors 0..3: low nibbles, vector i holds group 2i (low nibble) and group 2i+1 (high nibble)
 //   vectors 4..5: high 2 bits, vector m holds groups 4m..4m+3 at bit offsets 0,2,4,6
@@ -37,6 +40,7 @@ extern "C" {
 #define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0   1152
 #define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_IQ4_NL 640
 #define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_MXFP4  640
+#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q5_K   768
 #define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q6_K   896
 
 // --- Activation Tiled Block Sizes (including padding) ---
@@ -199,6 +203,8 @@ static inline uint32_t htp_mm_get_weight_tile_size(int weight_type) {
             return HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
         case HTP_TYPE_Q8_0:
             return HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
+        case HTP_TYPE_Q5_K:
+            return HTP_MM_WEIGHT_TILE_SIZE_Q5_K;
         case HTP_TYPE_Q6_K:
             return HTP_MM_WEIGHT_TILE_SIZE_Q6_K;
         case HTP_TYPE_MXFP4:
@@ -218,6 +224,8 @@ static inline uint32_t htp_mm_get_weight_aligned_tile_size(int weight_type) {
             return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_1;
         case HTP_TYPE_Q8_0:
             return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
+        case HTP_TYPE_Q5_K:
+            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q5_K;
         case HTP_TYPE_Q6_K:
             return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q6_K;
         case HTP_TYPE_MXFP4:
@@ -227,6 +235,11 @@ static inline uint32_t htp_mm_get_weight_aligned_tile_size(int weight_type) {
     }
 }
 
+// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
+static inline bool htp_mm_weight_has_offset(int weight_type) {
+    return weight_type == HTP_TYPE_Q4_1 || weight_type == HTP_TYPE_Q4_K || weight_type == HTP_TYPE_Q5_K;
+}
+
 // --- Activation/Row Size Helpers ---
 static inline size_t htp_mm_q8_0_tiled_row_size(uint32_t ne) {
     const uint32_t ne_padded = ((ne + 127) / 128) * 128;
@@ -248,6 +261,7 @@ static inline size_t htp_mm_get_tiled_row_stride(int weight_type, uint32_t k) {
         case HTP_TYPE_Q4_1:
         case HTP_TYPE_Q4_K:
         case HTP_TYPE_Q8_0:
+        case HTP_TYPE_Q5_K:
         case HTP_TYPE_Q6_K:
         case HTP_TYPE_MXFP4:
             return (size_t) nb * htp_mm_get_weight_tile_size(weight_type);
@@ -332,6 +346,7 @@ struct htp_mm_hvx_vtcm_layout {
     size_t off_src2;          // vtcm_src2 (Wq / fused only)
     size_t off_src3;          // vtcm_src3 (Wv / fused only)
     size_t off_dst;           // vtcm_dst (output scratch)
+    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)
 
     // Cached sizes
     size_t src0_bytes;
@@ -339,6 +354,7 @@ struct htp_mm_hvx_vtcm_layout {
     size_t src2_bytes;
     size_t src3_bytes;
     size_t dst_bytes;
+    size_t act_raw_bytes;
 
     size_t total_bytes;
 };
@@ -351,7 +367,6 @@ static inline void htp_mm_hmx_vtcm_layout_build(
     size_t mc,
     size_t nc,
     uint32_t group_size,
-    bool use_dma_activation,
     bool pipeline,
     uint32_t act_threads,
     uint32_t aligned_tile_size,
@@ -366,8 +381,7 @@ static inline void htp_mm_hmx_vtcm_layout_build(
         const size_t activation_area_size = hex_align_up(group_size * act_head_stride * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
         const size_t output_area_size  = hex_align_up(group_size * mc * nc * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
         const size_t scratch_area_size = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
-        const size_t min_f32_size = use_dma_activation
-            ? hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), 128) : 0;
+        const size_t min_f32_size = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), 128);
 
         // Group A: Permanent activation tiles and scales
         size_t off_group_a = 0;
@@ -388,10 +402,9 @@ static inline void htp_mm_hmx_vtcm_layout_build(
 
         // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
         const size_t max_f32_size = act_threads * 64 * k * sizeof(float);
-        const size_t act_f32_size = use_dma_activation
-            ? hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128) : 0;
+        const size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
         size_t off_group_c = off_group_a;
-        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_f32, act_f32_size, use_dma_activation);
+        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);
 
         const size_t group_c_size = off_group_c - off_group_a;
 
@@ -478,20 +491,20 @@ static inline void htp_mm_hvx_vtcm_layout_build(
     bool is_fused_nx
 ) {
     (void)src1_row_size;
-    size_t src0_sz = 0;
-    size_t src1_sz = 0;
-    size_t src2_sz = src2_row_size > 0 ? htp_mm_round_up(src2_row_size, 128) : 0;
-    size_t src3_sz = 0;
-    size_t dst_sz  = 0;
+    size_t src0_sz    = 0;
+    size_t src1_sz    = 0;
+    size_t src2_sz    = src2_row_size > 0 ? htp_mm_round_up(src2_row_size, 128) : 0;
+    size_t src3_sz    = 0;
+    size_t dst_sz     = 0;
+    size_t act_raw_sz = 0;
 
     const bool is_repack = (wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 ||
                             wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL ||
                             wtype == HTP_TYPE_MXFP4 || wtype == HTP_TYPE_Q6_K ||
-                            wtype == HTP_TYPE_Q4_K);
+                            wtype == HTP_TYPE_Q4_K || wtype == HTP_TYPE_Q5_K);
 
     if (is_fused_nx) {
         const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
-        const size_t quant_scratch_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float)) * n_threads;
 
         size_t weight_sz_per_thread = 0;
 
@@ -505,17 +518,19 @@ static inline void htp_mm_hvx_vtcm_layout_build(
             weight_sz_per_thread = hex_round_up(n_prefetch * src0_row_size_padded, 128);
         }
 
-        size_t tiled_act_row_size = (wtype == HTP_TYPE_Q4_1 || wtype == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+        size_t tiled_act_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
         size_t act_sz = hex_round_up(tiled_act_row_size * src1_nrows, 128);
-
-        src0_sz = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
-        src1_sz = act_sz;                           // quantized activation buffer
-        src2_sz = 0;
-        src3_sz = 0;
-        dst_sz  = quant_scratch_size;
+        size_t raw_row_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
+
+        src0_sz    = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
+        src1_sz    = act_sz;                           // quantized activation buffer
+        src2_sz    = 0;
+        src3_sz    = 0;
+        dst_sz     = 0;
+        act_raw_sz = hex_round_up(raw_row_size * src1_nrows, 128);
     } else if (is_matmul_id) {
         const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
-        const size_t src1_row_size_tiled = (wtype == HTP_TYPE_Q4_1 || wtype == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10)
+        const size_t src1_row_size_tiled = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10)
                                                                                                : htp_mm_q8_0_tiled_row_size(ne10);
 
         size_t src0_sz_per_thread = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
@@ -529,10 +544,13 @@ static inline void htp_mm_hvx_vtcm_layout_build(
             src0_sz_per_thread               = repacked_vtcm_size;
         }
 
-        src0_sz = src0_sz_per_thread * n_threads;
-        dst_sz  = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float)) * n_threads;
-        src2_sz = 0;
-        src3_sz = 0;
+        size_t raw_row_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
+
+        src0_sz    = src0_sz_per_thread * n_threads;
+        dst_sz     = 0;
+        src2_sz    = 0;
+        src3_sz    = 0;
+        act_raw_sz = hex_round_up(raw_row_size * src1_nrows, 128);
     } else {
         const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
         const size_t dst_nrows = (src1_nrows > 1) ? 0 : 1;
@@ -540,21 +558,23 @@ static inline void htp_mm_hvx_vtcm_layout_build(
         switch (kernel_type) {
             case HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                 size_t f16_src1_row_size = htp_mm_round_up(ne10 * 2, 128);
-                src1_sz = htp_mm_round_up(f16_src1_row_size * src1_nrows, 256);
-                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
-                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
+                src1_sz    = htp_mm_round_up(f16_src1_row_size * src1_nrows, 256);
+                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
+                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
+                act_raw_sz = hex_round_up(hex_round_up(ne10 * sizeof(float), 128) * src1_nrows, 128);
                 break;
             }
             case HTP_MM_KERNEL_HVX_F32_F32_VTCM: {
                 size_t f32_src1_row_size = htp_mm_round_up(ne10 * 4, 128);
-                src1_sz = htp_mm_round_up(f32_src1_row_size * src1_nrows, 256);
-                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
-                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
+                src1_sz    = htp_mm_round_up(f32_src1_row_size * src1_nrows, 256);
+                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
+                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
+                act_raw_sz = 0;
                 break;
             }
             case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
             case HTP_MM_KERNEL_HVX_QUANT_ROW: {
-                size_t q_src1_row_size = (wtype == HTP_TYPE_Q4_1 || wtype == HTP_TYPE_Q4_K) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+                size_t q_src1_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
 
                 src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
                 src1_sz = htp_mm_round_up(q_src1_row_size * src1_nrows, 256);
@@ -569,10 +589,10 @@ static inline void htp_mm_hvx_vtcm_layout_build(
                     src0_sz = repacked_vtcm_size * n_threads;
                 }
 
-                size_t quant_scratch_size_per_thread = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
                 size_t dst_slice_per_thread = (dst_nrows > 0 && src1_nrows == 1) ? htp_mm_round_up((dst_row_size + n_threads - 1) / n_threads, 128) : 0;
-                size_t dst_size_per_thread = (dst_slice_per_thread > quant_scratch_size_per_thread) ? dst_slice_per_thread : quant_scratch_size_per_thread;
-                dst_sz = dst_size_per_thread * n_threads;
+                dst_sz = dst_slice_per_thread * n_threads;
+                size_t raw_row_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
+                act_raw_sz = hex_round_up(raw_row_size * src1_nrows, 128);
                 break;
             }
             default:
@@ -580,19 +600,30 @@ static inline void htp_mm_hvx_vtcm_layout_build(
         }
     }
 
-    size_t off = 0;
-    VTCM_LAYOUT_ALLOC(off, off_src0, src0_sz);
-    VTCM_LAYOUT_ALLOC(off, off_src1, src1_sz);
-    VTCM_LAYOUT_ALLOC(off, off_src2, src2_sz);
-    VTCM_LAYOUT_ALLOC(off, off_src3, src3_sz);
-    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_sz);
-
-    L->src0_bytes = src0_sz;
-    L->src1_bytes = src1_sz;
-    L->src2_bytes = src2_sz;
-    L->src3_bytes = src3_sz;
-    L->dst_bytes  = dst_sz;
-    L->total_bytes = off;
+    // Group A: Persistent buffers across chunk compute
+    size_t off_group_a = 0;
+    VTCM_LAYOUT_ALLOC(off_group_a, off_src1, src1_sz);
+    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_src2, src2_sz, src2_sz > 0);
+    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_src3, src3_sz, src3_sz > 0);
+
+    // Group B: Compute-only buffers (starts at off_group_a)
+    size_t off_group_b = off_group_a;
+    VTCM_LAYOUT_ALLOC(off_group_b, off_src0, src0_sz);
+    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst, dst_sz, dst_sz > 0);
+    const size_t group_b_size = off_group_b - off_group_a;
+
+    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
+    size_t off_group_c = off_group_a;
+    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
+    const size_t group_c_size = off_group_c - off_group_a;
+
+    L->src0_bytes    = src0_sz;
+    L->src1_bytes    = src1_sz;
+    L->src2_bytes    = src2_sz;
+    L->src3_bytes    = src3_sz;
+    L->dst_bytes     = dst_sz;
+    L->act_raw_bytes = act_raw_sz;
+    L->total_bytes   = off_group_a + hex_smax(group_b_size, group_c_size);
 }
 
 static inline bool htp_mm_hvx_solve_vtcm_params(
@@ -630,7 +661,7 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
     const size_t avail_act = vtcm_budget - fixed_bytes;
     size_t row_size = 0;
     if (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
-        row_size = (wtype == HTP_TYPE_Q4_1 || wtype == HTP_TYPE_Q4_K)
+        row_size = htp_mm_weight_has_offset(wtype)
                  ? htp_mm_q8_1_tiled_row_size(ne10)
                  : htp_mm_q8_0_tiled_row_size(ne10);
     } else if (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
@@ -642,7 +673,12 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
         return false;
     }
 
-    uint32_t m_chunk = (uint32_t) (avail_act / row_size);
+    size_t eff_row_size = row_size;
+    if (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
+        eff_row_size += hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
+    }
+
+    uint32_t m_chunk = (uint32_t) (avail_act / eff_row_size);
     if (m_chunk > 1) {
         m_chunk &= ~1U;
     }
@@ -679,15 +715,15 @@ static inline size_t htp_mm_hmx_get_2d_vtcm_size(
     int wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_size, size_t src2_size
 ) {
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, false, pipeline, act_threads, aligned_tile_size, src2_size);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_threads, aligned_tile_size, src2_size);
     return L.total_bytes;
 }
 
 static inline size_t htp_mm_hmx_get_batched_vtcm_size(
-    int wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool use_dma_activation, bool pipeline, uint32_t act_threads, size_t src2_size) {
+    int wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool pipeline, uint32_t act_threads, size_t src2_size) {
     (void)pipeline;
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, wtype, k, mc, nc, group_size, use_dma_activation, false, act_threads, 0, src2_size);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, wtype, k, mc, nc, group_size, false, act_threads, 0, src2_size);
     return L.total_bytes;
 }
 
@@ -697,7 +733,6 @@ static inline bool htp_mm_hmx_solve_batched_params(
     uint32_t ne01_padded,
     uint32_t ne11,
     uint32_t group_size,
-    bool use_dma_activation,
     int n_threads,
     bool pipeline,
     size_t src2_size,
@@ -726,7 +761,7 @@ static inline bool htp_mm_hmx_solve_batched_params(
         if (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_size_per_n, group_size_per_m, group_size_per_mn, hex_align_up(ne11, 32), ne01_padded,
                                (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
-            size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, group_size, use_dma_activation, pipeline, act_threads, src2_size);
+            size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, group_size, pipeline, act_threads, src2_size);
             if (exact_size <= vtcm_budget) {
                 size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                 if (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
diff --git src/ggml-metal/CMakeLists.txt src/ggml-metal/CMakeLists.txt
index e7afdb69..68532a98 100644
--- src/ggml-metal/CMakeLists.txt
+++ src/ggml-metal/CMakeLists.txt
@@ -29,9 +29,29 @@ set(METALLIB_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-common.h")
 set(METALLIB_KERNELS_COMMON     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.h")
 set(METALLIB_KERNELS_DEQUANTIZE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/dequantize.h")
 set(METALLIB_KERNELS_QUANTIZE   "${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantize.h")
+set(METALLIB_KERNELS_FA_COMMON     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fa_common.metal")
+set(METALLIB_KERNELS_FA_VEC_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fa_vec_common.metal")
+set(METALLIB_KERNELS_FA_SHARED
+    ${METALLIB_KERNELS_FA_COMMON}
+    ${METALLIB_KERNELS_FA_VEC_COMMON}
+)
 
 set(METALLIB_KERNEL_SOURCES
-    kernels/fa.metal
+    kernels/fa_aux.metal
+    kernels/fa_f16.metal
+    kernels/fa_f32.metal
+    kernels/fa_q4_0.metal
+    kernels/fa_q4_1.metal
+    kernels/fa_q5_0.metal
+    kernels/fa_q5_1.metal
+    kernels/fa_q8_0.metal
+    kernels/fa_vec_f16.metal
+    kernels/fa_vec_f32.metal
+    kernels/fa_vec_q4_0.metal
+    kernels/fa_vec_q4_1.metal
+    kernels/fa_vec_q5_0.metal
+    kernels/fa_vec_q5_1.metal
+    kernels/fa_vec_q8_0.metal
     kernels/mul_mv.metal
     kernels/mul_mm.metal
     kernels/quantize.metal
@@ -82,13 +102,21 @@ if (GGML_METAL_EMBED_LIBRARY)
         if(_has_quantize)
             list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_QUANTIZE})
         endif()
+        file(STRINGS ${SRC} _has_fa_common REGEX "#include \"fa_common\\.metal\"")
+        file(STRINGS ${SRC} _has_fa_vec_common REGEX "#include \"fa_vec_common\\.metal\"")
+        if(_has_fa_common)
+            list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_FA_COMMON})
+        endif()
+        if(_has_fa_vec_common)
+            list(APPEND HEADERS_FOR_SRC ${METALLIB_KERNELS_FA_VEC_COMMON})
+        endif()
 
         add_custom_command(
             OUTPUT "${ASM}"
             # Step 1: concatenate shared headers + this kernel source
             COMMAND cat ${HEADERS_FOR_SRC} ${SRC} > "${EMBED}.tmp1"
             # Step 2: remove internal #include and #pragma once
-            COMMAND sed -e "/\#include \"common.h\"/d" -e "/\#include \"dequantize.h\"/d" -e "/\#include \"quantize.h\"/d" -e "/\#pragma once/d" < "${EMBED}.tmp1" > "${EMBED}.tmp2"
+            COMMAND sed -e "/\#include \"common.h\"/d" -e "/\#include \"dequantize.h\"/d" -e "/\#include \"quantize.h\"/d" -e "/\#include \"fa_common.metal\"/d" -e "/\#include \"fa_vec_common.metal\"/d" -e "/\#pragma once/d" < "${EMBED}.tmp1" > "${EMBED}.tmp2"
             # Step 3: inline ggml-common.h (replacing __embed_ggml-common.h__ sentinel)
             COMMAND sed -e "/__embed_ggml-common.h__/r ${METALLIB_COMMON}" -e "/__embed_ggml-common.h__/d" < "${EMBED}.tmp2" > "${EMBED}.tmp3"
             # Step 4: inline ggml-metal-impl.h
@@ -103,8 +131,7 @@ if (GGML_METAL_EMBED_LIBRARY)
             COMMAND echo .incbin "\"${EMBED}\""                                  >> "${ASM}"
             COMMAND echo ".globl _ggml_metallib_${kind_sym}_end"                 >> "${ASM}"
             COMMAND echo "_ggml_metallib_${kind_sym}_end:"                       >> "${ASM}"
-            DEPENDS ../ggml-common.h ggml-metal-impl.h
-                    kernels/common.h kernels/dequantize.h kernels/quantize.h
+            DEPENDS ${HEADERS_FOR_SRC} ../ggml-common.h ggml-metal-impl.h
                     kernels/${kind}.metal
             COMMENT "Generate embedded Metal library for ${kind}"
             VERBATIM
@@ -123,6 +150,10 @@ else()
     configure_file(kernels/common.h     ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/common.h     COPYONLY)
     configure_file(kernels/dequantize.h ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/dequantize.h COPYONLY)
     configure_file(kernels/quantize.h   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/quantize.h   COPYONLY)
+    foreach(hdr ${METALLIB_KERNELS_FA_SHARED})
+        get_filename_component(hdr_name ${hdr} NAME)
+        configure_file(${hdr} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/${hdr_name} COPYONLY)
+    endforeach()
 
     foreach(src ${METALLIB_KERNEL_SOURCES})
         configure_file(${src} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${src} COPYONLY)
@@ -180,7 +211,7 @@ else()
         add_custom_command(
             OUTPUT ${AIR}
             COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS} -I ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${src} -o ${AIR}
-            DEPENDS ${src} kernels/common.h kernels/dequantize.h kernels/quantize.h ${METALLIB_COMMON} ggml-metal-impl.h
+            DEPENDS ${src} ${METALLIB_KERNELS_FA_SHARED} kernels/common.h kernels/dequantize.h kernels/quantize.h ${METALLIB_COMMON} ggml-metal-impl.h
             COMMENT "Compiling ${src}"
             VERBATIM
         )
diff --git src/ggml-metal/ggml-metal-common.cpp src/ggml-metal/ggml-metal-common.cpp
index 9c0b9474..e43023cc 100644
--- src/ggml-metal/ggml-metal-common.cpp
+++ src/ggml-metal/ggml-metal-common.cpp
@@ -7,22 +7,29 @@
 
 #include <vector>
 
-// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal
-static bool ggml_metal_fwht_supported_size(int64_t n) {
-    return n == 64 || n == 128 || n == 256 || n == 512;
+// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
+// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
+// float[N] per threadgroup, so they are only available where that fits.
+static bool ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
+    if (n == 64 || n == 128 || n == 256 || n == 512) {
+        return true;
+    }
+
+    if (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
+        return (size_t) n * sizeof(float) <= max_tg_mem;
+    }
+
+    return false;
 }
 
 // the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
 // and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
 // through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
-bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
-    return ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
-           op->type == GGML_TYPE_F32 &&
-           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
-           ggml_is_contiguous(op->src[1]) &&
-           ggml_is_contiguous(op) &&
-           ggml_are_same_shape(op->src[1], op) &&
-           ggml_metal_fwht_supported_size(op->src[1]->ne[0]);
+bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op, size_t max_tg_mem) {
+    return ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD && op->type == GGML_TYPE_F32 &&
+           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && ggml_is_contiguous(op->src[1]) &&
+           ggml_is_contiguous(op) && ggml_are_same_shape(op->src[1], op) &&
+           ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
 }
 
 bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
diff --git src/ggml-metal/ggml-metal-common.h src/ggml-metal/ggml-metal-common.h
index e6a28d03..6b5a1883 100644
--- src/ggml-metal/ggml-metal-common.h
+++ src/ggml-metal/ggml-metal-common.h
@@ -3,6 +3,7 @@
 #pragma once
 
 #include <stdbool.h>
+#include <stddef.h>
 
 #ifdef __cplusplus
 extern "C" {
@@ -48,7 +49,7 @@ bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * ten
 void ggml_graph_optimize(struct ggml_cgraph * gf);
 
 // mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
-bool ggml_metal_op_mul_mat_use_fwht (const struct ggml_tensor * op);
+bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op, size_t max_tg_mem);
 bool ggml_metal_op_mul_mat_use_mm   (const struct ggml_tensor * op, bool has_simdgroup_mm);
 bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm);
 
diff --git src/ggml-metal/ggml-metal-context.m src/ggml-metal/ggml-metal-context.m
index 442ed2a0..e9064666 100644
--- src/ggml-metal/ggml-metal-context.m
+++ src/ggml-metal/ggml-metal-context.m
@@ -477,6 +477,10 @@ enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph *
         return GGML_STATUS_FAILED;
     }
 
+    if (gf->n_nodes == 0) {
+        return GGML_STATUS_SUCCESS;
+    }
+
     // number of nodes encoded by the main thread (empirically determined)
     const int n_main = MAX(64, 0.1*gf->n_nodes);
 
@@ -514,8 +518,6 @@ enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph *
 
         const bool use_capture = ctx->capture_compute == 0;
         if (use_capture) {
-            ctx->capture_compute = -1;
-
             // make sure all previous computations have finished before starting the capture
             if (ctx->cmd_buf_last) {
                 [ctx->cmd_buf_last waitUntilCompleted];
@@ -538,7 +540,7 @@ enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph *
 
                 NSError * error = nil;
                 if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
-                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
+                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                 } else {
                     [ctx->capture_scope beginScope];
                     ctx->capture_started = true;
@@ -749,7 +751,7 @@ void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
             idx_start,
             idx_end,
             ctx->use_concurrency,
-            ctx->capture_compute,
+            ctx->capture_compute == 0,
             ctx->debug_graph);
 
         for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
diff --git src/ggml-metal/ggml-metal-device.m src/ggml-metal/ggml-metal-device.m
index 81ea7f9d..fa58b896 100644
--- src/ggml-metal/ggml-metal-device.m
+++ src/ggml-metal/ggml-metal-device.m
@@ -110,7 +110,21 @@ int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_wi
 //   X(suffix, name): name is both the kernels/<name>.metal basename and the
 //   ggml_metallib_<name>_{start,end} embed-symbol stem.
 #define GGML_METAL_LIBS \
-    X(FA,              fa)             \
+    X(FA_AUX,          fa_aux)         \
+    X(FA_F16,          fa_f16)         \
+    X(FA_F32,          fa_f32)         \
+    X(FA_Q4_0,         fa_q4_0)        \
+    X(FA_Q4_1,         fa_q4_1)        \
+    X(FA_Q5_0,         fa_q5_0)        \
+    X(FA_Q5_1,         fa_q5_1)        \
+    X(FA_Q8_0,         fa_q8_0)        \
+    X(FA_VEC_F16,      fa_vec_f16)     \
+    X(FA_VEC_F32,      fa_vec_f32)     \
+    X(FA_VEC_Q4_0,     fa_vec_q4_0)    \
+    X(FA_VEC_Q4_1,     fa_vec_q4_1)    \
+    X(FA_VEC_Q5_0,     fa_vec_q5_0)    \
+    X(FA_VEC_Q5_1,     fa_vec_q5_1)    \
+    X(FA_VEC_Q8_0,     fa_vec_q8_0)    \
     X(MUL_MV,          mul_mv)         \
     X(MUL_MM,          mul_mm)         \
     X(QUANTIZE,        quantize)       \
@@ -1840,7 +1854,7 @@ bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_te
             // the FWHT kernels read an F16 source directly; every other F16 src1 path
             // still goes through ggml_metal_supports_mul_mat_op
             if (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F16 &&
-                ggml_metal_op_mul_mat_use_fwht(op)) {
+                ggml_metal_op_mul_mat_use_fwht(op, dev->props.max_theadgroup_memory_size)) {
                 return has_simdgroup_reduction;
             }
             return ggml_metal_supports_mul_mat_op(
diff --git src/ggml-metal/ggml-metal-impl.h src/ggml-metal/ggml-metal-impl.h
index 490dd83a..eed85f28 100644
--- src/ggml-metal/ggml-metal-impl.h
+++ src/ggml-metal/ggml-metal-impl.h
@@ -1235,6 +1235,11 @@ typedef struct {
     int32_t  top_k;  // k
 } ggml_metal_kargs_top_k;
 
+// widths at or above this use the threadgroup FWHT kernel, one row per threadgroup
+// with GGML_METAL_FWHT_TG_NT threads, instead of one row per simdgroup
+#define GGML_METAL_FWHT_TG_MIN_N 1024
+#define GGML_METAL_FWHT_TG_NT    256
+
 typedef struct {
     int32_t  ne01;      // n_tokens
     uint64_t nb01;      // logits row stride
diff --git src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal-ops.cpp
index 708703e5..8a46ec66 100644
--- src/ggml-metal/ggml-metal-ops.cpp
+++ src/ggml-metal/ggml-metal-ops.cpp
@@ -2343,6 +2343,13 @@ int ggml_metal_op_fwht(ggml_metal_op_t ctx, int idx) {
     const int th_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
     const int simd_size = 32;
 
+    if (n >= GGML_METAL_FWHT_TG_MIN_N) {
+        GGML_ASSERT(th_max >= GGML_METAL_FWHT_TG_NT);
+        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, GGML_METAL_FWHT_TG_NT, 1, 1);
+
+        return 1;
+    }
+
     int sg_per_tg = 2;
     sg_per_tg = std::min(sg_per_tg, th_max/simd_size);
     sg_per_tg = std::max(sg_per_tg, 1);
@@ -2419,10 +2426,11 @@ int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
     ggml_metal_library_t lib = ctx->lib;
     ggml_metal_encoder_t enc = ctx->enc;
 
-    if (ggml_metal_op_mul_mat_use_fwht(op)) {
+    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
+
+    if (ggml_metal_op_mul_mat_use_fwht(op, props_dev->max_theadgroup_memory_size)) {
         return ggml_metal_op_fwht(ctx, idx);
     }
-    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
 
     GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
     GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
@@ -3494,34 +3502,20 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
 
         const int is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;
 
-        // 2*(2*ncpsg)
-        // ncpsg soft_max values + ncpsg mask values
-        //
-        // 16*32*(nsg)
-        // the shared memory needed for the simdgroups to load the KV cache
-        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
-        //
-#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + is_q*(16*32*(nsg)))*(sizeof(float)/2), 16))
+        // shared memory layout (halfs unless noted):
+        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
+        //   quantized KV scratch: 16*32*NSG (only when is_q)
+        const int64_t dv_pad = GGML_PAD(ne20, 64);
 
-        //int64_t nsgmax = 4;
-        //
-        //if (is_q) {
-        //    nsgmax = 2;
-        //    while (true) {
-        //        const size_t smem = FATTN_SMEM(nsgmax);
-        //        if (smem > props_dev->max_theadgroup_memory_size) {
-        //            break;
-        //        }
-        //        nsgmax *= 2;
-        //    }
-        //    nsgmax /= 2;
-        //}
+        auto fa_smem = [&](int32_t nsg) -> size_t {
+            const size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
+            return GGML_PAD(smem_half*sizeof(ggml_fp16_t), 16);
+        };
 
         // simdgroups per threadgroup (a.k.a. warps)
-        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
         int32_t nsg = ne00 >= 512 ? 8 : 4;
 
-        const size_t smem = FATTN_SMEM(nsg);
+        const size_t smem = fa_smem(nsg);
 
         const int32_t ns10 = nb11_attn/nb10_attn;
         const int32_t ns20 = nb21_attn/nb20_attn;
@@ -3577,7 +3571,6 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
         ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
 
         ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
-#undef FATTN_SMEM
     } else {
         // half4x4 kernel
         // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
@@ -3679,14 +3672,18 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
         // note: for simplicity assume the K is larger or equal than V
         GGML_ASSERT(ne10 >= ne20);
 
-        // ne00 + 2*ncpsg*(nsg)
-        // for each query, we load it as f16 in shared memory (ne00)
-        // and store the soft_max values and the mask
-        //
-        // ne20*(nsg)
-        // each simdgroup has a full f32 head vector in shared mem to accumulate results
-        //
-#define FATTN_SMEM(nsg) (GGML_PAD(((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128))*(nsg)*nqptg)*(sizeof(float)/2), 16))
+        // shared memory layout (halfs unless noted):
+        //   queries:      Q*NSG*PAD2(ne00, 128)
+        //   attn + mask:  NSG*4*Q*C
+        //   results:      2*NSG*Q*PAD2(ne20, 128)
+        //   sparse idx:   NSG*C ints (only when use_sparse)
+        const int64_t dk_pad = GGML_PAD(ne00, 128);
+        const int64_t dv_pad = GGML_PAD(ne20, 128);
+
+        auto fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
+            const size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
+            return GGML_PAD(smem_half*sizeof(ggml_fp16_t) + (use_sparse ? (size_t) nsg*ncpsg*sizeof(int) : 0), 16);
+        };
 
         int64_t nsg = 1;
 
@@ -3722,7 +3719,7 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
         }
 
         // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
-        if ((size_t) FATTN_SMEM(nsg) > props_dev->max_theadgroup_memory_size) {
+        if (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_memory_size) {
             cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20);
             nqptg = cfg.Q;  // = 1
         }
@@ -3779,9 +3776,8 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
         ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
         ggml_metal_encoder_set_buffer  (enc, use_sparse ? bid_idx : bid_src0, 8);
 
-        const size_t smem = FATTN_SMEM(nsg);
+        const size_t smem = fa_vec_smem(nsg, nqptg);
 
-        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
         GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);
 
         if (nwg == 1) {
@@ -3827,7 +3823,6 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
                 ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
             }
         }
-#undef FATTN_SMEM
     }
 
     return 1;
diff --git src/ggml-metal/ggml-metal-tuning.h src/ggml-metal/ggml-metal-tuning.h
index 003b4d6b..d36a0b63 100644
--- src/ggml-metal/ggml-metal-tuning.h
+++ src/ggml-metal/ggml-metal-tuning.h
@@ -17,7 +17,7 @@ constexpr int FA_VEC_NE01_BUCKETS[] = { 2, 3, 4, 5 };
 int fa_vec_ne11_bucket(int64_t ne11);
 int fa_vec_ne01_bucket(int64_t ne01);
 
-// NE baked into each (dk,dv) baseline instantiation in kernels/fa.metal.
+// NE baked into each (dk,dv) baseline instantiation in kernels/fa_vec_*.metal.
 // Hand-maintained mirror; keep in sync with those instantiations.
 // The Metal test slice covers every legal config for dk=128 and dk=576.
 int fa_vec_baseline_ne(int dk, int dv);
diff --git src/ggml-metal/kernels/fa_aux.metal src/ggml-metal/kernels/fa_aux.metal
new file mode 100644
index 00000000..89cf0bcd
--- /dev/null
+++ src/ggml-metal/kernels/fa_aux.metal
@@ -0,0 +1,479 @@
+#include "common.h"
+#include "dequantize.h"
+
+// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
+// - one thread per block; dispatched separately for K and V
+// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
+template <
+    typename block_t,
+    short QK,
+    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
+kernel void kernel_flash_attn_ext_kv_f16(
+        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
+        device const char * x,
+        device       half * x_dst,
+        uint gid [[thread_position_in_grid]]) {
+    if (gid >= (uint) args.nblocks) {
+        return;
+    }
+
+    const uint nb = args.ne0/QK;
+    const uint i0 = gid%nb;
+    uint ib       = gid/nb;
+    const uint i1 = ib%args.ne1;
+    ib /= args.ne1;
+    const uint i2 = ib%args.ne2;
+    const uint i3 = ib/args.ne2;
+
+    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;
+
+    device const block_t * src = (device const block_t *) (x + offs);
+    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;
+
+    for (short i = 0; i < QK/16; ++i) {
+        float4x4 reg;
+        deq_t4x4(src, i, reg);
+        dst[4*i + 0] = (half4) reg[0];
+        dst[4*i + 1] = (half4) reg[1];
+        dst[4*i + 2] = (half4) reg[2];
+        dst[4*i + 3] = (half4) reg[3];
+    }
+}
+
+typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;
+
+template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
+template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
+template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
+template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
+template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;
+
+constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
+
+constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];
+
+// pad the last chunk of C elements of k and v into a an extra pad buffer
+kernel void kernel_flash_attn_ext_pad(
+        constant ggml_metal_kargs_flash_attn_ext_pad & args,
+        device const char * k,
+        device const char * v,
+        device const char * mask,
+        device       char * dst,
+        uint3   tgpig[[threadgroup_position_in_grid]],
+        ushort  tiitg[[thread_index_in_threadgroup]],
+        ushort3   ntg[[threads_per_threadgroup]]) {
+    const int32_t C = FC_flash_attn_ext_pad_ncpsg;
+
+    device char * k_pad    = dst;
+    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
+    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;
+
+    const int32_t icp = args.ne11 % C;
+    const int32_t ic0 = args.ne11 - icp;
+
+    const int32_t i1 = tgpig[0];
+    const int32_t i2 = tgpig[1];
+    const int32_t i3 = tgpig[2];
+
+    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
+        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
+        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;
+
+        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
+        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;
+
+        if (i1 >= icp) {
+            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
+            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
+                k_dst[i] = 0;
+            }
+            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
+                v_dst[i] = 0;
+            }
+        } else {
+            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
+                k_dst[i] = k_src[i];
+            }
+            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
+                v_dst[i] = v_src[i];
+            }
+        }
+    }
+
+    if (FC_flash_attn_ext_pad_has_mask) {
+        if (i2 < args.ne32 && i3 < args.ne33) {
+            for (int ib = i1; ib < args.ne31; ib += C) {
+                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
+                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;
+
+                for (int i = tiitg; i < C; i += ntg.x) {
+                    if (i >= icp) {
+                        mask_dst[i] = -MAXHALF;
+                    } else {
+                        mask_dst[i] = mask_src[i];
+                    }
+                }
+            }
+        }
+    }
+}
+
+constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
+constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];
+
+// scan the blocks of the mask that are not masked
+// 0 -     masked (i.e. full of -INF, skip)
+// 1 - not masked (i.e. at least one element of the mask is not -INF)
+// 2 - all zero
+kernel void kernel_flash_attn_ext_blk(
+        constant ggml_metal_kargs_flash_attn_ext_blk & args,
+        device const char * mask,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]]) {
+    // block size C x Q
+    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
+    const int32_t C = FC_flash_attn_ext_blk_ncpsg;
+
+    constexpr short NW  = N_SIMDWIDTH;
+
+    const int32_t i3 = tgpig[2]/args.ne32;
+    const int32_t i2 = tgpig[2]%args.ne32;
+    const int32_t i1 = tgpig[1];
+    const int32_t i0 = tgpig[0];
+
+    char res = i0*C + C > args.ne30 || i1*Q + Q > args.ne31 ? 1 : 0;
+
+    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;
+
+    // detailed check of the elements of the block
+    if ((C > NW || Q > 1) && res == 0) {
+        half mmin =  MAXHALF;
+        half mmax = -MAXHALF;
+
+        FOR_UNROLL (short j = 0; j < Q; ++j) {
+            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
+                mmin = min(mmin, mask_src[ii*NW]);
+                mmax = max(mmax, mask_src[ii*NW]);
+            }
+
+            mask_src += args.nb31/2;
+        }
+
+        mmin = simd_min(mmin);
+        mmax = simd_max(mmax);
+
+        if (mmax > -MAXHALF) {
+            if (mmin == 0.0 && mmax == 0.0) {
+                res = 2;
+            } else {
+                res = 1;
+            }
+        }
+    }
+
+    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
+    const int32_t nblk0 = ((args.ne30 + C - 1)/C);
+
+    if (tiisg == 0) {
+        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
+    }
+}
+// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
+// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
+// one threadgroup per mask row; the mask remains the single source of truth for the values
+kernel void kernel_flash_attn_ext_vec_idx(
+        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
+        device const half * mask,
+        device       int  * idx,
+        uint3   tgpig[[threadgroup_position_in_grid]],
+        ushort  tiitg[[thread_index_in_threadgroup]],
+        ushort3 ntg[[threads_per_threadgroup]]) {
+    constexpr short NW = N_SIMDWIDTH;
+    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread
+
+    const int i1 = tgpig[0];
+    const int i2 = tgpig[1];
+    const int i3 = tgpig[2];
+
+    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
+    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;
+
+    const int n  = args.ne30;
+    const int q  = n/ntg.x;
+    const int r  = n%ntg.x;
+
+    // each thread handles a contiguous slice of the mask row
+    const int r0 = q*tiitg + min((int) tiitg, r);
+    const int r1 = r0 + q + (tiitg < r ? 1 : 0);
+
+    // count the finite entries in the slice and keep their positions in registers (single mask read)
+    int cnt = 0;  // total finite entries in the slice
+    int nloc = 0; // finite entries kept in registers
+    int local[NLOCAL];
+    for (int i = r0; i < r1; ++i) {
+        if (isfinite((float) pm[i])) {
+            if (nloc < NLOCAL) {
+                local[nloc] = i;
+                nloc++;
+            }
+            cnt++;
+        }
+    }
+
+    const short sgitg = tiitg/NW;
+    const short tiisg = tiitg%NW;
+
+    threadgroup int tcount[8];
+
+    // simd_sum is a collective: all lanes must evaluate it
+    const int sg_sum = simd_sum(cnt);
+    if (tiisg == 0) {
+        tcount[sgitg] = sg_sum;
+    }
+
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+
+    int total = 0;
+    for (short s = 0; s < ntg.x/NW; ++s) {
+        total += tcount[s];
+    }
+
+    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
+    int sg_base = 0;
+    for (short s = 0; s < sgitg; ++s) {
+        sg_base += tcount[s];
+    }
+
+    // exclusive prefix scan of the per-thread counts within the simdgroup
+    int incl = cnt;
+    for (int d = 1; d < NW; d <<= 1) {
+        const int v = simd_shuffle_up(incl, d);
+        if (tiisg >= d) {
+            incl += v;
+        }
+    }
+    const int base = sg_base + (incl - cnt);
+
+    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
+    int j = 0;
+    for (; j < nloc && base + j < args.n_kv_max; ++j) {
+        pidx[base + j] = local[j];
+    }
+
+    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
+    if (cnt > nloc && base + nloc < args.n_kv_max) {
+        int j2 = 0;
+        for (int i = r0; i < r1; ++i) {
+            if (isfinite((float) pm[i])) {
+                if (j2 >= nloc) {
+                    pidx[base + j2] = i;
+                }
+                j2++;
+                if (base + j2 >= args.n_kv_max) {
+                    break;
+                }
+            }
+        }
+    }
+
+    // pad the tail of the list with -1
+    const int count = min(total, args.n_kv_max);
+    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
+        pidx[i] = -1;
+    }
+}
+
+constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
+constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];
+
+kernel void kernel_flash_attn_ext_vec_reduce(
+        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
+        device  const char * htmp,
+        device        char * dst,
+        uint   tgpig[[threadgroup_position_in_grid]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
+#define DV  (FC_flash_attn_ext_vec_reduce_DV)
+
+    const uint64_t rid = tgpig;
+
+    const short iwg = tiisg;
+
+    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;
+
+    float S = ss[rid*(2*NWG) + 2*iwg + 0];
+    float M = ss[rid*(2*NWG) + 2*iwg + 1];
+
+    const float m  = simd_max(M);
+    const float ms = exp(M - m);
+
+    S = simd_sum(S*ms);
+    S = S == 0.0f ? 0.0f : 1.0f/S;
+
+    const short DV4 = DV/4;
+
+    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
+    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;
+
+    for (short i = sgitg; i < DV4; i += NWG) {
+        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);
+
+        if (iwg == 0) {
+            dst4[i] = v*S;
+        }
+    }
+
+#undef NWG
+#undef DV
+}
+
+template<
+    typename kd4x4_t,
+    short nl_k,
+    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
+kernel void kernel_lightning_indexer(
+        constant ggml_metal_kargs_lightning_indexer & args,
+        device const char * q,
+        device const char * k,
+        device const char * w,
+        device const char * m,
+        device       char * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort tiitg[[thread_index_in_threadgroup]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
+    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
+    constexpr short NH    = OP_LIGHTNING_INDEXER_NH;
+    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
+    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
+    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
+    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;
+
+    constexpr short DK4  = DK/4;
+    constexpr short DK8  = DK/8;
+    constexpr short DK16 = DK/16;
+
+    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
+    constexpr short NTG = 32*NSG;    // threads per threadgroup
+
+    const int i_stream = tgpig.z;
+    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
+    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup
+
+    threadgroup half sk[NK * DK16 * 16];
+    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;
+
+    for (short i = tiitg; i < NK*DK16; i += NTG) {
+        const short ik  = i/DK16;
+        const short i16 = i%DK16;
+
+        half4x4 tmp;
+
+        if (i_kv_0 + ik < args.n_kv) {
+            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);
+
+            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
+        } else {
+            FOR_UNROLL (short j = 0; j < 4; ++j) {
+                tmp[j] = half4(0.0h);
+            }
+        }
+
+        sk4x4[i] = tmp;
+    }
+
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+
+    // K tile of this simdgroup, transposed to [DK, NKPSG]
+    simdgroup_half8x8 mk[DK8];
+
+    FOR_UNROLL (short i = 0; i < DK8; ++i) {
+        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
+    }
+
+    threadgroup half4   sq4[NHPTG*DK4];
+    threadgroup half  * sq = (threadgroup half *) sq4;
+
+    threadgroup float sw [NHPTG];
+    threadgroup float sqk[NSG*NHPTG*NKPSG];
+
+    const int i_batch_0 = tgpig.y*NBPTG;
+    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);
+
+    for (short ib = 0; ib < n_batch; ++ib) {
+        const int i_batch = i_batch_0 + ib;
+
+        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
+        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;
+
+        float score = 0.0f;
+
+        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
+            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights
+            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
+                const short ih = i/DK4;
+                const short i4 = i%DK4;
+
+                device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);
+
+                sq4[ih*DK4 + i4] = half4(q4[i4]);
+            }
+
+            if (tiitg < NHPTG) {
+                sw[tiitg] = ((device const float *) pw)[i_head + tiitg];
+            }
+
+            threadgroup_barrier(mem_flags::mem_threadgroup);
+
+            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
+
+            FOR_UNROLL (short i = 0; i < DK8; ++i) {
+                simdgroup_half8x8 mq;
+
+                simdgroup_load(mq, sq + 8*i, DK, 0, false);
+                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
+            }
+
+            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;
+
+            simdgroup_store(mqk, pqk, NKPSG, 0, false);
+            simdgroup_barrier(mem_flags::mem_threadgroup);
+
+            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
+            if (tiisg < NKPSG) {
+                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
+                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
+                }
+            }
+
+            threadgroup_barrier(mem_flags::mem_threadgroup);
+        }
+
+        if (tiisg < NKPSG) {
+            const int ik = i_kv + tiisg;
+            if (ik < args.n_kv) {
+                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
+                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);
+
+                pd[ik] = score + (float) pm[ik];
+            }
+        }
+    }
+}
+
+typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;
+
+template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
+template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;
+
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
+#endif
+
+template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
+template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
+template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
+template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
+template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;
diff --git src/ggml-metal/kernels/fa_common.metal src/ggml-metal/kernels/fa_common.metal
new file mode 100644
index 00000000..5e08bf85
--- /dev/null
+++ src/ggml-metal/kernels/fa_common.metal
@@ -0,0 +1,710 @@
+constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
+constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
+constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
+constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
+constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];
+
+constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];
+
+//constant float FC_flash_attn_ext_scale         [[function_constant(FC_FLASH_ATTN_EXT + 10)]];
+//constant float FC_flash_attn_ext_max_bias      [[function_constant(FC_FLASH_ATTN_EXT + 11)]];
+//constant float FC_flash_attn_ext_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT + 12)]];
+
+constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
+constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
+constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];
+
+// ref: https://arxiv.org/pdf/2307.08691.pdf
+template<
+    typename q_t,     // query types in shared memory
+    typename q4_t,
+    typename q8x8_t,
+    typename k_t,     // key types in shared memory
+    typename k4x4_t,
+    typename k8x8_t,
+    typename v_t,     // value types in shared memory
+    typename v4x4_t,
+    typename v8x8_t,
+    typename qk_t,    // Q*K types
+    typename qk8x8_t,
+    typename s_t,     // soft-max types
+    typename s2_t,
+    typename s8x8_t,
+    typename o_t,     // attention accumulation types
+    typename o4_t,
+    typename o8x8_t,
+    typename kd4x4_t, // key type in device memory
+    short nl_k,
+    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
+    typename vd4x4_t, // value type in device memory
+    short nl_v,
+    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
+    short DK,         // K head size
+    short DV,         // V head size
+    short Q,          // queries per threadgroup
+    short C,          // cache items per threadgroup
+    short NSG>        // number of simd groups
+void kernel_flash_attn_ext_impl(
+        constant ggml_metal_kargs_flash_attn_ext & args,
+        device const char * q,
+        device const char * k,
+        device const char * v,
+        device const char * mask,
+        device const char * sinks,
+        device const char * pad,
+        device const char * blk,
+        device       char * dst,
+        threadgroup  half * shmem_f16,
+        uint3   tgpig,
+        ushort  tiisg,
+        ushort  sgitg) {
+    const ushort iq3 = tgpig[2];
+    const ushort iq2 = tgpig[1];
+    const ushort iq1 = tgpig[0]*Q;
+
+#define NS10 (FC_flash_attn_ext_ns10)
+#define NS20 (FC_flash_attn_ext_ns20)
+
+    // note: I had some concerns that using this instead of the ugly macros above was affecting performance
+    //       need to re-check carefully and if no regressions are observerd - remove the macros
+    //       the concerns is that maybe using const variables requires extra registers? but not sure if the compiler
+    //         is clever enough to avoid this. unfortunately, using constexpr is not possible with FC
+    //const short NS10 = FC_flash_attn_ext_ns10;
+    //const short NS20 = FC_flash_attn_ext_ns20;
+
+    constexpr short KV   = 8;
+
+    constexpr short DK4  = DK/4;
+    constexpr short DK8  = DK/8;
+    constexpr short DK16 = DK/16;
+    constexpr short DV4  = DV/4;
+  //constexpr short DV8  = DV/8;
+    constexpr short DV16 = DV/16;
+
+    constexpr short PV   = PAD2(DV, 64);
+    constexpr short PV4  = PV/4;
+    constexpr short PV8  = PV/8;
+  //constexpr short PV16 = PV/16;
+
+    constexpr short NW  = N_SIMDWIDTH;
+    constexpr short NQ  = Q/NSG;
+    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)
+
+    constexpr short TS = 2*SH;
+    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)
+
+    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data
+    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t
+    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)
+    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
+    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix
+    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t
+
+    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load K in shared memory
+    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in k4x4_t
+
+    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load V in shared memory
+    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in v4x4_t
+
+    // mask storage in shared mem
+    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);
+
+    // per-query mask pointers
+    device const half2 * pm2[NQ];
+
+    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+        const short j = jj*NSG + sgitg;
+
+        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
+    }
+
+    {
+        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
+        const int32_t nblk0 = ((args.ne11 + C - 1)/C);
+
+        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
+    }
+
+    {
+        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;
+
+        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
+        const short ikv3 = iq3/(args.ne03/args.ne_12_3);
+
+        k += ikv2*args.nb12 + ikv3*args.nb13;
+        v += ikv2*args.nb22 + ikv3*args.nb23;
+    }
+
+    // load heads from Q to shared memory
+    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+        const short j = jj*NSG + sgitg;
+
+        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);
+
+        for (short i = tiisg; i < DK4; i += NW) {
+            if (iq1 + j < args.ne01) {
+                sq4[j*DK4 + i] = (q4_t) q4[i];
+            } else {
+                sq4[j*DK4 + i] = 0;
+            }
+        }
+    }
+
+    // zero out
+    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+        const short j = jj*NSG + sgitg;
+
+        for (short i = tiisg; i < DV4; i += NW) {
+            so4[j*PV4 + i] = 0;
+        }
+
+        for (short i = tiisg; i < SH; i += NW) {
+            ss[j*SH + i] = 0.0f;
+        }
+    }
+
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+
+    float S[NQ] = { [0 ... NQ-1] = 0.0f };
+
+    {
+        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };
+
+        float slope = 1.0f;
+
+        // ALiBi
+        if (FC_flash_attn_ext_has_bias) {
+            const short h = iq2;
+
+            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
+            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
+
+            slope = pow(base, exph);
+        }
+
+        // loop over the KV cache
+        // each simdgroup handles blocks of Q rows and C columns
+        for (int ic0 = 0; ; ++ic0) {
+            int ic = ic0*C;
+            if (ic >= args.ne11) {
+                break;
+            }
+
+            // the last partial chunk uses the pad buffer as source
+            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
+                k    = pad;
+                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
+                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
+
+                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
+                const short ikv3 = iq3/(args.ne03/args.ne_12_3);
+
+                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
+                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;
+
+                if (!FC_flash_attn_ext_has_mask) {
+                    threadgroup half * sm = (threadgroup half *) (sm2);
+
+                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                        const short j = jj*NSG + sgitg;
+
+                        for (short i = tiisg; i < C; i += NW) {
+                            if (ic + i >= args.ne11) {
+                                sm[2*j*SH + i] = -MAXHALF;
+                            }
+                        }
+                    }
+                } else {
+                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                        const short j = jj*NSG + sgitg;
+
+                        pm2[jj] = (device const half2 *) ((device const half *) mask +
+                                (iq1 + j)*C +
+                                (iq2%args.ne32)*(C*args.ne31) +
+                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
+                    }
+                }
+
+                ic = 0;
+            }
+
+            char blk_cur = 1;
+
+            // read the mask into shared mem
+            if (FC_flash_attn_ext_has_mask) {
+                blk_cur = blk[ic0];
+
+                if (blk_cur == 0) {
+                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                        pm2[jj] += NW;
+                    }
+
+                    continue;
+                }
+
+                if (blk_cur == 1) {
+                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                        const short j = jj*NSG + sgitg;
+
+                        if (FC_flash_attn_ext_bc_mask) {
+                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
+                        } else {
+                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
+                        }
+
+                        pm2[jj] += NW;
+                    }
+                } else if (blk_cur == 2) {
+                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                        pm2[jj] += NW;
+                    }
+                }
+
+#if 0
+                // note: old -INF block optimization - obsoleted by pre-computing non-masked blocks
+
+                threadgroup_barrier(mem_flags::mem_threadgroup);
+
+                // used to detect blocks full of -INF
+                // skip only when the entire threadgroup is masked
+                half2 smax2(-MAXHALF/2, -MAXHALF/2);
+
+                FOR_UNROLL (short j = 0; j < Q; ++j) {
+                    smax2 = max(smax2, sm2[j*SH + tiisg]);
+                }
+
+                smax2 = simd_max(smax2);
+
+                if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {
+                    // this barrier is important
+                    threadgroup_barrier(mem_flags::mem_threadgroup);
+
+                    continue;
+                }
+#endif
+            }
+
+            // Q*K^T
+            // this is compile-time check, so it does not have runtime overhead
+            if (is_same<kd4x4_t, k4x4_t>::value) {
+                // we can read directly from global memory
+                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
+                threadgroup const q_t * pq = sq;
+                threadgroup       s_t * ps = ss;
+
+                pk += sgitg*(8*NS10);
+                ps += sgitg*(8*1);
+
+                static_assert((C/8) % NSG == 0, "");
+
+                constexpr short NC = (C/8)/NSG;
+
+                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
+                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);
+
+                    if (DK % 16 != 0) {
+                        k8x8_t mk;
+                        q8x8_t mq;
+
+                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
+                            simdgroup_barrier(mem_flags::mem_none);
+
+                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
+                            simdgroup_load(mq, pq + 8*i, DK);
+
+                            simdgroup_barrier(mem_flags::mem_none);
+
+                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
+                        }
+                    } else {
+                        k8x8_t mk[2];
+                        q8x8_t mq[2];
+
+                        // note: too much unroll can tank the performance for large heads
+                        #pragma unroll (MIN(DK8/2, 4*NSG))
+                        for (short i = 0; i < DK8/2; ++i) {
+                            simdgroup_barrier(mem_flags::mem_none);
+
+                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
+                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);
+
+                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
+                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);
+
+                            simdgroup_barrier(mem_flags::mem_none);
+
+                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
+                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
+                        }
+                    }
+
+                    simdgroup_store(mqk, ps, SH, 0, false);
+
+                    pk += 8*(NSG*NS10);
+                    ps += 8*(NSG);
+                }
+            } else {
+                // TODO: this is the quantized K cache branch - not optimized yet
+                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
+                    const short cc = ccc*NSG + sgitg;
+
+                    const short tx = tiisg%4;
+                    const short ty = tiisg/4;
+
+                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);
+
+                    for (short ii = 0; ii < DK16; ii += 4) {
+                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));
+
+                        if (DK16%4 == 0) {
+                            // the head is evenly divisible by 4*16 = 64, so no need for bound checks
+                            {
+                                k4x4_t tmp;
+                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
+                                sk4x4[4*ty + tx] = tmp;
+                            }
+
+                            simdgroup_barrier(mem_flags::mem_threadgroup);
+
+                            FOR_UNROLL (short k = 0; k < 4; ++k) {
+                                k8x8_t mk;
+                                q8x8_t mq;
+
+                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
+                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
+                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
+
+                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
+                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
+                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
+                            }
+                        } else {
+                            if (ii + tx < DK16) {
+                                k4x4_t tmp;
+                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
+                                sk4x4[4*ty + tx] = tmp;
+                            }
+
+                            simdgroup_barrier(mem_flags::mem_threadgroup);
+
+                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
+                                k8x8_t mk;
+                                q8x8_t mq;
+
+                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
+                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
+                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
+
+                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
+                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
+                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
+                            }
+                        }
+                    }
+
+                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
+                }
+            }
+
+            threadgroup_barrier(mem_flags::mem_threadgroup);
+
+            // online softmax
+            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                const short j = jj*NSG + sgitg;
+
+                const float m = M[jj];
+
+                // scale and apply the logitcap / mask
+                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;
+
+                if (FC_flash_attn_ext_has_scap) {
+                    s2 = args.logit_softcap*precise::tanh(s2);
+                }
+
+                // mqk = mqk + slope*mask
+                if (blk_cur != 2) {
+                    if (FC_flash_attn_ext_has_bias) {
+                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
+                    } else {
+                        s2 += s2_t(sm2[j*SH + tiisg]);
+                    }
+                }
+
+                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
+
+                const float  ms  = exp(m  - M[jj]);
+                const float2 vs2 = exp(s2 - M[jj]);
+
+                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);
+
+                // the P matrix from the paper (Q rows, C columns)
+                ss2[j*SH/2 + tiisg] = vs2;
+
+                if (DV4 % NW == 0) {
+                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
+                        const short i = ii*NW + tiisg;
+
+                        so4[j*PV4 + i] *= ms;
+                    }
+                } else {
+                    for (short i = tiisg; i < DV4; i += NW) {
+                        so4[j*PV4 + i] *= ms;
+                    }
+                }
+            }
+
+            threadgroup_barrier(mem_flags::mem_threadgroup);
+
+            // O = O + (Q*K^T)*V
+            {
+                // we can read directly from global memory
+                if (is_same<vd4x4_t, v4x4_t>::value) {
+                    static_assert(PV8 % NSG == 0, "");
+
+                    constexpr short NO = PV8/NSG;
+
+                    o8x8_t lo[NO];
+
+                    {
+                        auto sot = so + 8*sgitg;
+
+                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
+                            simdgroup_load(lo[ii], sot, PV, 0, false);
+
+                            sot += 8*NSG;
+                        }
+                    }
+
+                    {
+                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);
+
+                        pv += 8*sgitg;
+
+                        if (DV <= 64) {
+                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
+                                s8x8_t vs;
+                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);
+
+                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
+                                    v8x8_t mv[2];
+
+                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
+                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);
+
+                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
+                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
+                                }
+
+                                pv  += 8*NS20;
+                            }
+                        } else {
+                            constexpr short NC = (C/8)/2;
+
+                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
+                                s8x8_t vs[2];
+
+                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
+                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);
+
+                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
+                                    v8x8_t mv[4];
+
+                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
+                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
+                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
+                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
+
+                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
+                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
+                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
+                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
+                                }
+
+                                pv  += 2*8*NS20;
+                            }
+                        }
+                    }
+
+                    {
+                        auto sot = so + 8*sgitg;
+
+                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
+                            simdgroup_store(lo[ii], sot, PV, 0, false);
+
+                            sot += 8*NSG;
+                        }
+                    }
+                } else {
+                    // TODO: this is the quantized V cache branch - not optimized yet
+
+                    const short tx = tiisg%4;
+                    const short ty = tiisg/4;
+
+                    for (short cc = 0; cc < C/8; ++cc) {
+                        s8x8_t vs;
+                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);
+
+                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
+                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));
+
+                            if (DV16%4 == 0) {
+                                // no need for bound checks
+                                {
+                                    v4x4_t tmp;
+                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
+                                    sv4x4[4*ty + tx] = tmp;
+                                }
+
+                                simdgroup_barrier(mem_flags::mem_threadgroup);
+
+                                FOR_UNROLL (short k = 0; k < 4; ++k) {
+                                    v8x8_t mv[2];
+                                    o8x8_t lo[2];
+
+                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
+                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
+                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
+                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
+
+                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
+                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
+
+                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
+                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
+                                }
+                            } else {
+                                if (ii + tx < DV16) {
+                                    v4x4_t tmp;
+                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
+                                    sv4x4[4*ty + tx] = tmp;
+                                }
+
+                                simdgroup_barrier(mem_flags::mem_threadgroup);
+
+                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
+                                    v8x8_t mv[2];
+                                    o8x8_t lo[2];
+
+                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
+                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
+                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
+                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
+
+                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
+                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
+
+                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
+                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
+                                }
+                            }
+                        }
+                    }
+                }
+            }
+
+            threadgroup_barrier(mem_flags::mem_threadgroup);
+        }
+
+        if (FC_flash_attn_ext_has_sinks) {
+            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
+                const short j = jj*NSG + sgitg;
+
+                const float m = M[jj];
+                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;
+
+                M[jj] = simd_max(max(M[jj], s));
+
+                const float ms = exp(m - M[jj]);
+                const float vs = exp(s - M[jj]);
+
+                S[jj] = S[jj]*ms + simd_sum(vs);
+
+                for (short i = tiisg; i < DV4; i += NW) {
+                    so4[j*PV4 + i] *= ms;
+                }
+            }
+        }
+    }
+
+    // store to global memory
+    for (short jj = 0; jj < NQ; ++jj) {
+        const short j = jj*NSG + sgitg;
+        if (iq1 + j >= args.ne01) {
+            break;
+        }
+
+        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;
+
+        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];
+
+        if (DV4 % NW == 0) {
+            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
+                const short i = ii*NW + tiisg;
+
+                dst4[i] = (float4) so4[j*PV4 + i]*scale;
+            }
+        } else {
+            for (short i = tiisg; i < DV4; i += NW) {
+                dst4[i] = (float4) so4[j*PV4 + i]*scale;
+            }
+        }
+    }
+
+#undef NS10
+#undef NS20
+}
+
+template<
+    typename q_t,     // query types in shared memory
+    typename q4_t,
+    typename q8x8_t,
+    typename k_t,     // key types in shared memory
+    typename k4x4_t,
+    typename k8x8_t,
+    typename v_t,     // value types in shared memory
+    typename v4x4_t,
+    typename v8x8_t,
+    typename qk_t,    // Q*K types
+    typename qk8x8_t,
+    typename s_t,     // soft-max types
+    typename s2_t,
+    typename s8x8_t,
+    typename o_t,     // attention accumulation types
+    typename o4_t,
+    typename o8x8_t,
+    typename kd4x4_t, // key type in device memory
+    short nl_k,
+    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
+    typename vd4x4_t, // value type in device memory
+    short nl_v,
+    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
+    short DK,         // K head size
+    short DV,         // V head size
+    short Q  = OP_FLASH_ATTN_EXT_NQPSG, // queries per threadgroup
+    short C  = OP_FLASH_ATTN_EXT_NCPSG> // cache items per threadgroup
+kernel void kernel_flash_attn_ext(
+        constant ggml_metal_kargs_flash_attn_ext & args,
+        device const char * q,
+        device const char * k,
+        device const char * v,
+        device const char * mask,
+        device const char * sinks,
+        device const char * pad,
+        device const char * blk,
+        device       char * dst,
+        threadgroup  half * shmem_f16 [[threadgroup(0)]],
+        uint3   tgpig[[threadgroup_position_in_grid]],
+        ushort  tiisg[[thread_index_in_simdgroup]],
+        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
+#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
+#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
+    switch (FC_flash_attn_ext_nsg) {
+      // note: disabled cases to reduce library load time
+      //case 1: kernel_flash_attn_ext_impl<FWD_TMPL, 1>(FWD_ARGS); break;
+      //case 2: kernel_flash_attn_ext_impl<FWD_TMPL, 2>(FWD_ARGS); break;
+        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
+        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(FWD_ARGS); break;
+    }
+#undef FWD_TMPL
+#undef FWD_ARGS
+}
diff --git src/ggml-metal/kernels/fa_f16.metal src/ggml-metal/kernels/fa_f16.metal
new file mode 100644
index 00000000..f46eb2cd
--- /dev/null
+++ src/ggml-metal/kernels/fa_f16.metal
@@ -0,0 +1,75 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
+//       template to be able to explore different combinations
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_f16_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  32,  32>;
+template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
+template [[host_name("kernel_flash_attn_ext_f16_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  48,  48>;
+template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
+template [[host_name("kernel_flash_attn_ext_f16_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  72,  72>;
+template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
+template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
+template [[host_name("kernel_flash_attn_ext_f16_dk96_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  64>;
+template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
+template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
+template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
+template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
+template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
+template [[host_name("kernel_flash_attn_ext_f16_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  320, 256>;
+template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  512, 512>;
+template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;
+
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_bf16_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 32,  32>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 48,  48>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 72,  72>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  64>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 320, 256>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 512, 512>;
+template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
+#endif
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_f32.metal src/ggml-metal/kernels/fa_f32.metal
new file mode 100644
index 00000000..8d38d775
--- /dev/null
+++ src/ggml-metal/kernels/fa_f32.metal
@@ -0,0 +1,53 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_f32_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  32,  32>;
+template [[host_name("kernel_flash_attn_ext_f32_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  40,  40>;
+template [[host_name("kernel_flash_attn_ext_f32_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  48,  48>;
+template [[host_name("kernel_flash_attn_ext_f32_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  64,  64>;
+template [[host_name("kernel_flash_attn_ext_f32_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  72,  72>;
+template [[host_name("kernel_flash_attn_ext_f32_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  80,  80>;
+template [[host_name("kernel_flash_attn_ext_f32_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  96,  96>;
+template [[host_name("kernel_flash_attn_ext_f32_dk96_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  96,  64>;
+template [[host_name("kernel_flash_attn_ext_f32_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  112, 112>;
+template [[host_name("kernel_flash_attn_ext_f32_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  128, 128>;
+template [[host_name("kernel_flash_attn_ext_f32_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 192>;
+template [[host_name("kernel_flash_attn_ext_f32_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 128>;
+template [[host_name("kernel_flash_attn_ext_f32_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  256, 256>;
+template [[host_name("kernel_flash_attn_ext_f32_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  320, 256>;
+template [[host_name("kernel_flash_attn_ext_f32_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  512, 512>;
+template [[host_name("kernel_flash_attn_ext_f32_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  576, 512>;
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_q4_0.metal src/ggml-metal/kernels/fa_q4_0.metal
new file mode 100644
index 00000000..49bb4c13
--- /dev/null
+++ src/ggml-metal/kernels/fa_q4_0.metal
@@ -0,0 +1,53 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_q4_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 32,  32>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 40,  40>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 48,  48>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64,  64>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 72,  72>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80,  80>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  96>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  64>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112, 112>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128, 128>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 192>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 128>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256, 256>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 320, 256>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 512, 512>;
+template [[host_name("kernel_flash_attn_ext_q4_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 576, 512>;
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_q4_1.metal src/ggml-metal/kernels/fa_q4_1.metal
new file mode 100644
index 00000000..79797ffc
--- /dev/null
+++ src/ggml-metal/kernels/fa_q4_1.metal
@@ -0,0 +1,53 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_q4_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 32,  32>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 40,  40>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 48,  48>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64,  64>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 72,  72>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80,  80>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  96>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  64>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112, 112>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128, 128>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 192>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 128>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256, 256>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 320, 256>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 512, 512>;
+template [[host_name("kernel_flash_attn_ext_q4_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 576, 512>;
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_q5_0.metal src/ggml-metal/kernels/fa_q5_0.metal
new file mode 100644
index 00000000..93de514b
--- /dev/null
+++ src/ggml-metal/kernels/fa_q5_0.metal
@@ -0,0 +1,53 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_q5_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 32,  32>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 40,  40>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 48,  48>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64,  64>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 72,  72>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80,  80>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  96>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  64>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112, 112>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128, 128>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 192>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 128>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256, 256>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 320, 256>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 512, 512>;
+template [[host_name("kernel_flash_attn_ext_q5_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 576, 512>;
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_q5_1.metal src/ggml-metal/kernels/fa_q5_1.metal
new file mode 100644
index 00000000..3918c571
--- /dev/null
+++ src/ggml-metal/kernels/fa_q5_1.metal
@@ -0,0 +1,53 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_q5_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 32,  32>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 40,  40>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 48,  48>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64,  64>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 72,  72>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80,  80>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  96>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  64>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112, 112>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128, 128>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 192>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 128>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256, 256>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 320, 256>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 512, 512>;
+template [[host_name("kernel_flash_attn_ext_q5_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 576, 512>;
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_q8_0.metal src/ggml-metal/kernels/fa_q8_0.metal
new file mode 100644
index 00000000..146bbb31
--- /dev/null
+++ src/ggml-metal/kernels/fa_q8_0.metal
@@ -0,0 +1,53 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_common.metal"
+
+#define FA_TYPES \
+    half,   half4,     simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    half,   half4x4,   simdgroup_half8x8,  \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+#define FA_TYPES_BF \
+    bfloat, bfloat4,   simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
+    float,             simdgroup_float8x8,  \
+    float,  float2,    simdgroup_float8x8,  \
+    half,   half4,     simdgroup_half8x8
+    //float,  float4,    simdgroup_float8x8
+
+#define FA_TYPES_F32 \
+    half,   half4,     simdgroup_half8x8,  \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,  float4x4,  simdgroup_float8x8, \
+    float,             simdgroup_float8x8, \
+    float,  float2,    simdgroup_float8x8, \
+    float,  float4,    simdgroup_float8x8
+    //half,   half4,     simdgroup_half8x8
+
+typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;
+
+template [[host_name("kernel_flash_attn_ext_q8_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 32,  32>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 40,  40>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 48,  48>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64,  64>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 72,  72>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80,  80>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  96>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  64>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112, 112>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
+template [[host_name("kernel_flash_attn_ext_q8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;
+
+#undef FA_TYPES
+#undef FA_TYPES_BF
+#undef FA_TYPES_F32
diff --git src/ggml-metal/kernels/fa_vec_common.metal src/ggml-metal/kernels/fa_vec_common.metal
new file mode 100644
index 00000000..46270ba4
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_common.metal
@@ -0,0 +1,650 @@
+constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
+constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
+constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
+constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
+constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];
+
+//constant float FC_flash_attn_ext_vec_scale         [[function_constant(FC_FLASH_ATTN_EXT_VEC + 10)]];
+//constant float FC_flash_attn_ext_vec_max_bias      [[function_constant(FC_FLASH_ATTN_EXT_VEC + 11)]];
+//constant float FC_flash_attn_ext_vec_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT_VEC + 12)]];
+
+constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
+constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
+constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
+constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];
+constant bool    FC_flash_attn_ext_vec_has_sparse [[function_constant(FC_FLASH_ATTN_EXT_VEC + 5)]];
+template<
+    typename q4_t,  // query types in shared memory
+    typename k4_t,  // key types in shared memory
+    typename v4_t,  // value types in shared memory
+    typename qk_t,  // Q*K types
+    typename s_t,   // soft-max types
+    typename s4_t,
+    typename o4_t,  // attention accumulation types
+    typename kd4_t, // key type in device memory
+    short nl_k,
+    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
+    typename vd4_t, // value type in device memory
+    short nl_v,
+    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
+    short DK,       // K head size
+    short DV,       // V head size
+    short NE = 4,   // head elements per thread
+    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup
+    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup
+
+kernel void kernel_flash_attn_ext_vec(
+        constant ggml_metal_kargs_flash_attn_ext_vec & args,
+        device const char * q,
+        device const char * k,
+        device const char * v,
+        device const char * mask,
+        device const char * sinks,
+        device const char * pad,
+        device       char * dst,
+        device const char * idx,
+        threadgroup  half * shmem_f16 [[threadgroup(0)]],
+        uint3   tgpig[[threadgroup_position_in_grid]],
+        ushort  tiisg[[thread_index_in_simdgroup]],
+        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
+    static_assert(DK % 32 == 0, "DK must be divisible by 32");
+    static_assert(DV % 32 == 0, "DV must be divisible by 32");
+
+#define NWG  (FC_flash_attn_ext_vec_nwg)
+#define NSG  (FC_flash_attn_ext_vec_nsg)
+
+#define NS10 (FC_flash_attn_ext_vec_ns10)
+#define NS20 (FC_flash_attn_ext_vec_ns20)
+
+    const short iwg = tgpig[2]%NWG;
+
+    const ushort iq3 = tgpig[2]/NWG;
+    const ushort iq2 = tgpig[1];
+    const ushort iq1 = tgpig[0];
+
+    constexpr short DK4 = DK/4;
+    constexpr short DV4 = DV/4;
+
+    constexpr short PK  = PAD2(DK, 128);
+    constexpr short PK4 = PK/4;
+
+    constexpr short PV  = PAD2(DV, 128);
+    constexpr short PV4 = PV/4;
+
+    constexpr short NW  = N_SIMDWIDTH;
+    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
+    constexpr short SH  = 4*Q*C; // shared memory per simdgroup
+
+    const int SMEM_Q = Q*NSG*PK;
+    const int SMEM_S = NSG*SH;
+    const int SMEM_O = 2*NSG*Q*PV;
+    const int SMEM   = SMEM_Q + SMEM_S + SMEM_O;
+
+    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
+    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");
+
+    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) shmem_f16; // holds the query data
+    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 + SMEM_Q + sgitg*SH); // scratch buffer for attention
+    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 + SMEM_Q + sgitg*SH); // same as above but in s4_t
+    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 + SMEM_Q + sgitg*SH + 2*Q*C); // scratch buffer for mask
+    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + SMEM_Q + SMEM_S + 2*sgitg*Q*PV); // scratch buffer for the results
+
+    // sparse indices for the current block
+    threadgroup int * spidx = FC_flash_attn_ext_vec_has_sparse
+        ? (threadgroup int *) (shmem_f16 + SMEM) + sgitg*C
+        : nullptr;
+
+    // store the result for all queries in shared memory (the O matrix from the paper)
+    so4 += tiisg;
+
+    {
+        q += iq1*Q*args.nb01 + iq2*args.nb02 + iq3*args.nb03;
+
+        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
+        const short ikv3 = iq3/(args.ne03/args.ne_12_3);
+
+        k += ikv2*args.nb12 + ikv3*args.nb13;
+        v += ikv2*args.nb22 + ikv3*args.nb23;
+    }
+
+    // load Q query rows to shared memory
+    {
+        for (short qq = 0; qq < Q; ++qq) {
+            const int iq1_q = iq1*Q + qq;
+            device const float4 * q4 = (device const float4 *) ((device const char *) q + qq*args.nb01);
+            if (iq1_q < args.ne01) {
+                for (short i = tiisg; i < PK4; i += NW) {
+                    if (i < DK4) {
+                        sq4[qq*PK4 + i] = (q4_t) q4[i];
+                    } else {
+                        sq4[qq*PK4 + i] = (q4_t) 0.0f;
+                    }
+                }
+            } else {
+                for (short i = tiisg; i < PK4; i += NW) {
+                    sq4[qq*PK4 + i] = (q4_t) 0.0f;
+                }
+            }
+        }
+    }
+
+    // zero out so
+    for (short qq = 0; qq < Q; ++qq) {
+        for (short i = 0; i < DV4/NL; ++i) {
+            so4[qq*DV4 + i*NL] = (o4_t) 0.0f;
+        }
+    }
+
+    // zero out shared memory SH
+    for (short i = tiisg; i < SH/4; i += NW) {
+        ss4[i] = (s4_t) 0.0f;
+    }
+
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+
+    {
+        float S[Q];
+        float M[Q];
+        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+            S[qq] = 0.0f;
+            M[qq] = -FLT_MAX/2;
+        }
+
+        // thread indices inside the simdgroup
+        const short tx = tiisg%NL;
+        const short ty = tiisg/NL;
+
+        // pointer to the mask
+        device const half * pm_base = (device const half *) (mask + iq1*Q*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
+
+        // sparse indices: the list of finite mask entries per query row
+        // the sparse path requires Q == 1 (enforced by the host)
+        device const int * pidx = nullptr;
+        if (FC_flash_attn_ext_vec_has_sparse) {
+            pidx = (device const int *) idx +
+                ((int64_t)(iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*args.ne31*args.n_kv_max_padded + (iq1%args.ne31)*args.n_kv_max_padded;
+        }
+
+        float slope = 1.0f;
+
+        // ALiBi
+        if (FC_flash_attn_ext_vec_has_bias) {
+            const short h = iq2;
+
+            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
+            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
+
+            slope = pow(base, exph);
+        }
+
+        // loop over the KV cache
+        // each simdgroup handles blocks of Q rows and C columns
+        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
+            int ic = ic0*C;
+            if (ic >= args.ne11) {
+                break;
+            }
+
+            device const half * pm[Q];
+            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                // padded query rows clamp to row 0 of the mask to avoid OOB; their scores
+                // are forced to -inf below, so the values never affect the result.
+                pm[qq] = pm_base + ((iq1*Q + qq) < args.ne01 ? qq*(args.nb31/sizeof(half)) : -iq1*Q*(args.nb31/sizeof(half)));
+            }
+
+            // the last partial chunk uses the pad buffer as source
+            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
+                k    = pad;
+                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
+                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
+
+                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
+                const short ikv3 = iq3/(args.ne03/args.ne_12_3);
+
+                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
+                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;
+
+                if (!FC_flash_attn_ext_vec_has_mask) {
+                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                        if (ic + tiisg >= args.ne11) {
+                            sm[qq*C + tiisg] = -MAXHALF;
+                        }
+                    }
+                } else {
+                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                        pm[qq] = (device const half *) (mask) +
+                            (iq1*Q + qq)*C +
+                            (iq2%args.ne32)*(C*args.ne31) +
+                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
+                    }
+                }
+
+                ic = 0;
+            }
+
+            // load the sparse KV indices for the current block into shared memory
+            if (FC_flash_attn_ext_vec_has_sparse) {
+                FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
+                    const short i = ii*NW + tiisg;
+
+                    spidx[i] = pidx[ic + i];
+                }
+                simdgroup_barrier(mem_flags::mem_threadgroup);
+            }
+
+            if (FC_flash_attn_ext_vec_has_mask) {
+                if (FC_flash_attn_ext_vec_has_sparse) {
+                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                        const int i11 = spidx[tiisg];
+                        if ((iq1*Q + qq) < args.ne01 && i11 >= 0) {
+                            sm[qq*C + tiisg] = pm[qq][i11];
+                        } else {
+                            sm[qq*C + tiisg] = -MAXHALF;
+                        }
+                    }
+                } else {
+                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                        if ((iq1*Q + qq) < args.ne01) {
+                            sm[qq*C + tiisg] = pm[qq][ic + tiisg];
+                        } else {
+                            sm[qq*C + tiisg] = -MAXHALF;
+                        }
+                    }
+                }
+            } else {
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    if ((iq1*Q + qq) >= args.ne01) {
+                        sm[qq*C + tiisg] = -MAXHALF;
+                    }
+                }
+            }
+
+            // skip -INF mask
+            {
+                bool any_finite = false;
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    if (simd_max(sm[qq*C + tiisg]) > -MAXHALF) {
+                        any_finite = true;
+                    }
+                }
+                if (!any_finite) {
+                    continue;
+                }
+            }
+
+            // Q*K^T
+            {
+                device      const k4_t * pk4 = nullptr;
+
+                if (!FC_flash_attn_ext_vec_has_sparse) {
+                    pk4 = (device const k4_t *) (k + ic*args.nb11);
+
+                    pk4 += ty*NS10/4 + tx;
+                }
+
+                qk_t mqk[Q][C/NE];
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
+                        mqk[qq][cc] = 0.0f;
+                    }
+                }
+
+                // each simdgroup processes Q queries and NE (NW/NL) cache elements
+                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
+                    if (FC_flash_attn_ext_vec_has_sparse) {
+                        // the KV rows are gathered from the index list; -1 entries are padding
+                        const int i11 = spidx[NE*cc + ty];
+                        if (i11 >= 0) {
+                            if (is_same<kd4_t, k4_t>::value) {
+                                device const k4_t * pk4s = (device const k4_t *) (k + i11*args.nb11) + tx;
+                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
+                                    const k4_t k_elem = pk4s[ii*NL];
+                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                        mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
+                                    }
+                                }
+                            } else {
+                                device const kd4_t * pk = (device const kd4_t *) (k + i11*args.nb11);
+
+                                k4_t mk;
+
+                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
+                                    const short i = ii*NL + tx;
+
+                                    deq_k_t4(pk + i/nl_k, i%nl_k, mk);
+
+                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                        mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
+                                    }
+                                }
+                            }
+                        }
+                    } else if (is_same<kd4_t, k4_t>::value) {
+                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
+                            const k4_t k_elem = pk4[cc*NE*NS10/4 + ii*NL];
+                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
+                            }
+                        }
+                    } else {
+                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));
+
+                        k4_t mk;
+
+                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
+                            const short i = ii*NL + tx;
+
+                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);
+
+                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
+                            }
+                        }
+                    }
+
+                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                        if (NE == 1) {
+                            mqk[qq][cc] = simd_sum(mqk[qq][cc]);
+                        } else {
+                            // simdgroup reduce (NE = 4)
+                            // [ 0 ..  7] -> [ 0]
+                            // [ 8 .. 15] -> [ 8]
+                            // [16 .. 23] -> [16]
+                            // [24 .. 31] -> [24]
+                            if (NE <= 1) {
+                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
+                            }
+                            if (NE <= 2) {
+                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  8);
+                            }
+                            if (NE <= 4) {
+                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  4);
+                            }
+                            if (NE <= 8) {
+                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  2);
+                            }
+                            if (NE <= 16) {
+                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  1);
+                            }
+
+                            // broadcast
+                            mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL*ty);
+                        }
+                    }
+                }
+
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    if (FC_flash_attn_ext_vec_has_mask &&
+                       !FC_flash_attn_ext_vec_has_scap &&
+                       !FC_flash_attn_ext_vec_has_bias) {
+                        ss[qq*C + NE*tx + ty] = fma(mqk[qq][tx], args.scale, (qk_t) sm[qq*C + NE*tx + ty]);
+                    } else {
+                        mqk[qq][tx] *= args.scale;
+
+                        if (FC_flash_attn_ext_vec_has_scap) {
+                            mqk[qq][tx] = args.logit_softcap*precise::tanh(mqk[qq][tx]);
+                        }
+
+                        if (FC_flash_attn_ext_vec_has_bias) {
+                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty]*slope;
+                        } else {
+                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty];
+                        }
+
+                        ss[qq*C + NE*tx + ty] = mqk[qq][tx];
+                    }
+                }
+            }
+
+            simdgroup_barrier(mem_flags::mem_threadgroup);
+
+            // online softmax
+            {
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    const float m = M[qq];
+                    const float s = ss[qq*C + tiisg];
+
+                    M[qq] = simd_max(max(M[qq], s));
+
+                    const float ms = exp(m - M[qq]);
+                    const float vs = exp(s - M[qq]);
+
+                    S[qq] = S[qq]*ms + simd_sum(vs);
+
+                    // the P matrix from the paper (Q rows, C columns)
+                    ss[qq*C + tiisg] = vs;
+
+                    // O = diag(ms)*O
+                    if ((DV4/NL % NW == 0) || ty == 0) {
+                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                            so4[qq*DV4 + ii*NL] *= ms;
+                        }
+                    }
+                }
+            }
+
+            simdgroup_barrier(mem_flags::mem_threadgroup);
+
+            // O = O + (Q*K^T)*V
+            {
+                o4_t lo[Q][DV4/NL];
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                        lo[qq][ii] = 0.0f;
+                    }
+                }
+
+                if (FC_flash_attn_ext_vec_has_sparse) {
+                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
+                        // the KV rows are gathered from the index list; -1 entries are padding
+                        const int i11 = spidx[NE*cc + ty];
+                        if (i11 >= 0) {
+                            if (is_same<vd4_t, v4_t>::value) {
+                                device const v4_t * pv4 = (device const v4_t *) (v + i11*args.nb21);
+
+                                pv4 += tx;
+
+                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                                    const v4_t v_elem = pv4[ii*NL];
+                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                        lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
+                                    }
+                                }
+                            } else {
+                                device const vd4_t * pv4 = (device const vd4_t *) (v + i11*args.nb21);
+
+                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                                    const short i = ii*NL + tx;
+
+                                    v4_t mv;
+
+                                    deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);
+
+                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                        lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + cc*NE + ty]));
+                                    }
+                                }
+                            }
+                        }
+                    }
+                } else if (is_same<vd4_t, v4_t>::value) {
+                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);
+
+                    pv4 += ty*NS20/4 + tx;
+
+                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
+                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                            const v4_t v_elem = pv4[cc*NE*NS20/4 + ii*NL];
+                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
+                            }
+                        }
+                    }
+                } else {
+                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
+                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));
+
+                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                            const short i = ii*NL + tx;
+
+                            v4_t mv;
+                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);
+
+                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                                lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + NE*cc + ty]));
+                            }
+                        }
+                    }
+                }
+
+                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                        if (NE > 1) {
+                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
+                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
+                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
+                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
+                        }
+
+                        if (NE > 2) {
+                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  8);
+                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  8);
+                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  8);
+                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  8);
+                        }
+
+                        if (NE > 4) {
+                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  4);
+                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  4);
+                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  4);
+                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  4);
+                        }
+
+                        if (NE > 8) {
+                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  2);
+                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  2);
+                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  2);
+                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  2);
+                        }
+
+                        if (NE > 16) {
+                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  1);
+                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  1);
+                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  1);
+                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  1);
+                        }
+                    }
+                }
+
+                if ((DV4/NL % NW == 0) || ty == 0) {
+                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                            so4[qq*DV4 + ii*NL] += lo[qq][ii];
+                        }
+                    }
+                }
+            }
+        }
+
+        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
+            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                const float m = M[qq];
+                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;
+
+                M[qq] = simd_max(max(M[qq], s));
+
+                const float ms = exp(m - M[qq]);
+                const float vs = exp(s - M[qq]);
+
+                S[qq] = S[qq]*ms + simd_sum(vs);
+
+                if ((DV4/NL % NW == 0) || ty == 0) {
+                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
+                        so4[qq*DV4 + ii*NL] *= ms;
+                    }
+                }
+            }
+        }
+
+        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
+        if (tiisg == 0) {
+            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                ss[2*qq + 0] = (s_t) S[qq];
+                ss[2*qq + 1] = (s_t) M[qq];
+            }
+        }
+    }
+
+    so4 -= tiisg;
+
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+
+    // parallel reduce
+    for (short r = NSG/2; r > 0; r >>= 1) {
+        if (sgitg < r) {
+            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+                const float S0 = ss[                2*qq + 0];
+                const float S1 = ss[r*(SH/2) +      2*qq + 0];
+
+                const float M0 = ss[                2*qq + 1];
+                const float M1 = ss[r*(SH/2) +      2*qq + 1];
+
+                const float Mx  = max(M0, M1);
+
+                const float ms0 = exp(M0 - Mx);
+                const float ms1 = exp(M1 - Mx);
+
+                const float Sx  = S0*ms0 + S1*ms1;
+
+                if (tiisg == 0) {
+                    ss[2*qq + 0] = Sx;
+                    ss[2*qq + 1] = Mx;
+                }
+
+                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
+                for (short i = tiisg; i < DV4; i += NW) {
+                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
+                }
+            }
+        }
+
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+
+    // final rescale with 1/S and store to global memory
+    if (sgitg == 0) {
+        const int64_t nrows = args.ne3*args.ne2*args.ne1;
+
+        device float4 * dst4 = (device float4 *) dst;
+        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results
+
+        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
+            const int iq1_q = iq1*Q + qq;
+            if (iq1_q >= args.ne01) {
+                continue;
+            }
+
+            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1_q*args.ne1;
+
+            const float Sval = NWG == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;
+
+            // interleave the workgroup data
+            for (short i = tiisg; i < DV4; i += NW) {
+                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[qq*DV4 + i]*Sval;
+            }
+
+            // store S and M
+            if (NWG > 1) {
+                if (tiisg == 0) {
+                    dst1[rid*(2*NWG) + 2*iwg + 0] = ss[2*qq + 0];
+                    dst1[rid*(2*NWG) + 2*iwg + 1] = ss[2*qq + 1];
+                }
+            }
+        }
+    }
+
+#undef NWG
+#undef NSG
+#undef NS10
+#undef NS20
+}
diff --git src/ggml-metal/kernels/fa_vec_f16.metal src/ggml-metal/kernels/fa_vec_f16.metal
new file mode 100644
index 00000000..56b5adb8
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_f16.metal
@@ -0,0 +1,125 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 32, 32, 4>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 64, 64, 2>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 96, 4>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 64, 4>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 320, 256, 2>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 512, 512, 1>;
+#endif
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 4>;
+#if defined(GGML_METAL_HAS_BF16)
+template [[host_name("kernel_flash_attn_ext_vec_bf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
+#endif
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/fa_vec_f32.metal src/ggml-metal/kernels/fa_vec_f32.metal
new file mode 100644
index 00000000..1c6bed0c
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_f32.metal
@@ -0,0 +1,37 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_f32_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  576, 512, 2>;
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/fa_vec_q4_0.metal src/ggml-metal/kernels/fa_vec_q4_0.metal
new file mode 100644
index 00000000..7bb99ffd
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_q4_0.metal
@@ -0,0 +1,92 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 4>;
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/fa_vec_q4_1.metal src/ggml-metal/kernels/fa_vec_q4_1.metal
new file mode 100644
index 00000000..8a6b08bc
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_q4_1.metal
@@ -0,0 +1,92 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 4>;
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/fa_vec_q5_0.metal src/ggml-metal/kernels/fa_vec_q5_0.metal
new file mode 100644
index 00000000..1633c476
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_q5_0.metal
@@ -0,0 +1,92 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 4>;
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/fa_vec_q5_1.metal src/ggml-metal/kernels/fa_vec_q5_1.metal
new file mode 100644
index 00000000..2e7db017
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_q5_1.metal
@@ -0,0 +1,92 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 4>;
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/fa_vec_q8_0.metal src/ggml-metal/kernels/fa_vec_q8_0.metal
new file mode 100644
index 00000000..e9c6b152
--- /dev/null
+++ src/ggml-metal/kernels/fa_vec_q8_0.metal
@@ -0,0 +1,92 @@
+#include "common.h"
+#include "dequantize.h"
+#include "fa_vec_common.metal"
+
+#define FA_TYPES \
+           half4,  \
+           half4,  \
+           half4,  \
+    float,         \
+    float, float4, \
+           float4
+
+#define FA_TYPES_F32 \
+           half4,  \
+           float4, \
+           float4, \
+    float,         \
+    float, float4, \
+           float4
+
+typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;
+
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 1>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 2>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 4>;
+template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 4>;
+
+#undef FA_TYPES
+#undef FA_TYPES_F32
+
diff --git src/ggml-metal/kernels/misc.metal src/ggml-metal/kernels/misc.metal
index 279d69f8..d3b01978 100644
--- src/ggml-metal/kernels/misc.metal
+++ src/ggml-metal/kernels/misc.metal
@@ -429,6 +429,81 @@ kernel void kernel_fwht(
     }
 }
 
+// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
+// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
+// those up to NT go through threadgroup memory; the rest stay in registers.
+// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
+// TODO: try to unroll loops
+template<int N, int NT, typename src_t>
+kernel void kernel_fwht_tg(
+        constant ggml_metal_kargs_fwht & args,
+        device const src_t * src,
+        device float * dst,
+        uint3  tgpig[[threadgroup_position_in_grid]],
+        ushort sgitg[[simdgroup_index_in_threadgroup]],
+        ushort tiisg[[thread_index_in_simdgroup]],
+        ushort3  ntg[[threads_per_threadgroup]]) {
+
+    constexpr int NW = N_SIMDWIDTH;
+    constexpr int NE = N / NT;
+
+    threadgroup float shmem[N];
+
+    const float scale = 1.0f / sqrt((float) N);
+
+    const int64_t r = tgpig.x;
+    if (r >= args.nrows) {
+        return;
+    }
+
+    src += r * N;
+    dst += r * N;
+
+    const int tid = sgitg * NW + tiisg;
+
+    float reg[NE];
+    for (int i = 0; i < NE; i++) {
+        reg[i] = float(src[i*NT + tid])*scale;
+    }
+
+    for (int i = 1; i < NW; i *= 2) {
+        for (int j = 0; j < NE; j++) {
+            const float val  = reg[j];
+            const float val2 = simd_shuffle_xor(val, i);
+            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
+        }
+    }
+
+    for (int i = NW; i < NT; i *= 2) {
+        for (int j = 0; j < NE; j++) {
+            shmem[j*NT + tid] = reg[j];
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+        for (int j = 0; j < NE; j++) {
+            const float val  = reg[j];
+            const float val2 = shmem[j*NT + (tid ^ i)];
+            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
+        }
+        threadgroup_barrier(mem_flags::mem_threadgroup);
+    }
+
+    for (int i = NT; i < N; i *= 2) {
+        const int step = i / NT;
+        for (int j = 0; j < NE; j += (2 * step)) {
+            for (int k = 0; k < step; k++) {
+                const float x = reg[j + k ];
+                const float y = reg[j + k + step];
+                reg[j + k]        = x + y;
+                reg[j + k + step] = x - y;
+            }
+        }
+    }
+
+    for (int i = 0; i < NE; i++) {
+        dst[i*NT + tid] = reg[i];
+    }
+}
+
 typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
 typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;
 
@@ -442,6 +517,16 @@ template [[host_name("kernel_fwht_f16_128")]] kernel kernel_fwht_f16_t kernel_fw
 template [[host_name("kernel_fwht_f16_256")]] kernel kernel_fwht_f16_t kernel_fwht<256, half>;
 template [[host_name("kernel_fwht_f16_512")]] kernel kernel_fwht_f16_t kernel_fwht<512, half>;
 
+template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
+template [[host_name("kernel_fwht_f32_2048")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
+template [[host_name("kernel_fwht_f32_4096")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
+template [[host_name("kernel_fwht_f32_8192")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;
+
+template [[host_name("kernel_fwht_f16_1024")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
+template [[host_name("kernel_fwht_f16_2048")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
+template [[host_name("kernel_fwht_f16_4096")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
+template [[host_name("kernel_fwht_f16_8192")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;
+
 constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];
 
 kernel void kernel_dsv4_hc_comb_f32(
diff --git src/ggml-opencl/CMakeLists.txt src/ggml-opencl/CMakeLists.txt
index ff5e8ef4..97b862af 100644
--- src/ggml-opencl/CMakeLists.txt
+++ src/ggml-opencl/CMakeLists.txt
@@ -194,6 +194,7 @@ set(GGML_OPENCL_KERNELS
     gemv_noshuffle_q6_k_f32_32b_trans
     gemv_noshuffle_q5_k_f32
     gemm_noshuffle_q5_k_f32
+    gemv_noshuffle_q5_k_f32_32b_trans
     mul
     neg
     norm
diff --git src/ggml-opencl/ggml-opencl.cpp src/ggml-opencl/ggml-opencl.cpp
index 23982189..0b685e69 100644
--- src/ggml-opencl/ggml-opencl.cpp
+++ src/ggml-opencl/ggml-opencl.cpp
@@ -1264,6 +1264,9 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_gemv_noshuffle_q5_k_f32;
     cl_kernel kernel_gemv_noshuffle_q5_k_f32_mc3;  // multi-column (N=3) verify GEMV (spec/MTP)
     cl_kernel kernel_gemm_noshuffle_q5_k_f32;
+    cl_kernel kernel_gemm_noshuffle_q5_k_f32_32b_trans_ila_a8_bin;
+    cl_kernel kernel_gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8_bin;
+    cl_kernel kernel_gemv_noshuffle_q5_k_f32_32b_trans;
     cl_kernel kernel_gemv_noshuffle_q5_0_f32;
     cl_kernel kernel_gemm_noshuffle_q5_0_f32;
     cl_kernel kernel_gemm_noshuffle_q5_0_q8_1_dp4a = nullptr;  // dp4a (int8) dense q5_0 prefill GEMM
@@ -4455,6 +4458,55 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         }
     }
 
+    backend_ctx->kernel_gemv_noshuffle_q5_k_f32_32b_trans = nullptr;
+    backend_ctx->kernel_gemm_noshuffle_q5_k_f32_32b_trans_ila_a8_bin = nullptr;
+    backend_ctx->kernel_gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8_bin = nullptr;
+    if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E) {
+        {
+            std::string opts = std::string("-cl-std=") + opencl_c_std +
+                                           " -cl-mad-enable "
+                                           " -DSIMDGROUP_WIDTH=" +
+                                           std::to_string(backend_ctx->adreno_wave_size);
+#ifdef GGML_OPENCL_EMBED_KERNELS
+            const std::string kernel_src {
+                #include "gemv_noshuffle_q5_k_f32_32b_trans.cl.h"
+            };
+#else
+            const std::string kernel_src = read_file("gemv_noshuffle_q5_k_f32_32b_trans.cl");
+#endif
+            cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), opts);
+            CL_CHECK((backend_ctx->kernel_gemv_noshuffle_q5_k_f32_32b_trans =
+                clCreateKernel(prog, "gemv_noshuffle_q5_k_f32_32b_trans", &err), err));
+            CL_CHECK(clReleaseProgram(prog));
+            GGML_LOG_CONT(".");
+        }
+
+        if (use_adreno_bin_kernels(backend_ctx)) {
+            size_t bin_size = 0;
+            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q5_k_f32_32b_trans_ila_a8", &bin_size);
+            if (kernel_bin && bin_size > 0) {
+                cl_program bin_prog =
+                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);
+
+                CL_CHECK((backend_ctx->kernel_gemm_noshuffle_q5_k_f32_32b_trans_ila_a8_bin =
+                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q5_k_f32_32b_trans_ila_a8", &err), err));
+                CL_CHECK(clReleaseProgram(bin_prog));
+                GGML_LOG_CONT(".");
+            }
+
+            kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8", &bin_size);
+            if (kernel_bin && bin_size > 0) {
+                cl_program bin_prog =
+                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);
+
+                CL_CHECK((backend_ctx->kernel_gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8_bin =
+                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8", &err), err));
+                CL_CHECK(clReleaseProgram(bin_prog));
+                GGML_LOG_CONT(".");
+            }
+        }
+    }
+
     std::string CL_moe_compile_opts = std::string("-cl-std=") + opencl_c_std +
             " -cl-mad-enable "
             " -cl-fast-relaxed-math";
@@ -8749,6 +8801,20 @@ inline bool use_q4_k_bin_kernels(const ggml_backend_opencl_context *backend_ctx,
 #endif
 }
 
+inline bool use_q5_k_bin_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+    if (!backend_ctx->kernel_gemv_noshuffle_q5_k_f32_32b_trans ||
+        !backend_ctx->kernel_gemm_noshuffle_q5_k_f32_32b_trans_ila_a8_bin) {
+        return false;
+    }
+    return (tensor->ne[0] % 256 == 0) && (tensor->ne[1] % 64 == 0);
+#else
+    GGML_UNUSED(backend_ctx);
+    GGML_UNUSED(tensor);
+    return false;
+#endif
+}
+
 static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
     ggml_backend_opencl_device_context * dev_ctx     = (ggml_backend_opencl_device_context *)dev->context;
     ggml_backend_opencl_context *        backend_ctx = dev_ctx->backend_ctx;
@@ -11147,8 +11213,29 @@ static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer,
 
             GGML_ASSERT(K % 32 == 0);
 
-            // Transpose q, d, dm as ushort, qh as uchar
-            transpose_2d_as_16b(backend_ctx, extra->q,  extra->q,  size_q,  K/4,   M);
+            if (use_q5_k_bin_kernels(backend_ctx, tensor)) {
+                cl_int err;
+                cl_image_format wimg_fmt;
+                cl_image_desc   wimg_desc;
+
+                // transpose q as 32-bit words (M-first); qh/d/dm stay in their existing layout
+                // (both new ILA kernels read qh via the existing [K/8][M] uchar plane directly).
+                GGML_ASSERT(M % 64 == 0);
+                transpose_2d_as_32b(backend_ctx, extra->q, extra->q, size_q, K/8, M);
+
+                wimg_fmt = { CL_R, CL_UNSIGNED_INT32 };
+                memset(&wimg_desc, 0, sizeof(wimg_desc));
+                wimg_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+                wimg_desc.image_width = (size_t)M * K / 8;
+                wimg_desc.buffer      = extra->q;
+                CL_CHECK((extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &wimg_fmt, &wimg_desc, NULL, &err), err));
+
+                // Transpose s as uchar
+                transpose_2d_as_8b(backend_ctx, extra->s, extra->s, size_s, K/256*12, M, true, true);
+            } else {
+                // Transpose q as ushort
+                transpose_2d_as_16b(backend_ctx, extra->q, extra->q, size_q, K/4, M);
+            }
             transpose_2d_as_8b (backend_ctx, extra->qh, extra->qh, size_qh, K/8,   M);
             transpose_2d_as_16b(backend_ctx, extra->d,  extra->d,  size_d,  K/256, M);
             transpose_2d_as_16b(backend_ctx, extra->dm, extra->dm, size_dm, K/256, M);
@@ -12333,21 +12420,28 @@ static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer,
 
             size_t size_q  = extra->size_q;
             size_t size_qh = extra->size_qh;
+            size_t size_s  = extra->size_s;
             size_t size_d  = extra->size_d;
             size_t size_dm = extra->size_dm;
 
             static ggml_cl_buffer buf_trans_q;
             static ggml_cl_buffer buf_trans_qh;
+            static ggml_cl_buffer buf_trans_s;
             static ggml_cl_buffer buf_trans_d;
             static ggml_cl_buffer buf_trans_dm;
 
             buf_trans_q.allocate(backend_ctx->context, size_q);
             buf_trans_qh.allocate(backend_ctx->context, size_qh);
+            buf_trans_s.allocate(backend_ctx->context, size_s);
             buf_trans_d.allocate(backend_ctx->context, size_d);
             buf_trans_dm.allocate(backend_ctx->context, size_dm);
 
-            // Reverse transpose q, qh, d, dm
-            transpose_2d_as_16b(backend_ctx, extra->q,  buf_trans_q.buffer,  size_q,  M, K/4);
+            if (use_q5_k_bin_kernels(backend_ctx, tensor)) {
+                transpose_2d_as_32b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/8);
+                transpose_2d_as_8b (backend_ctx, extra->s,  buf_trans_s.buffer,  size_s,  M, K/256*12, true, true);
+            } else {
+                transpose_2d_as_16b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/4);
+            }
             transpose_2d_as_8b (backend_ctx, extra->qh, buf_trans_qh.buffer, size_qh, M, K/8);
             transpose_2d_as_16b(backend_ctx, extra->d,  buf_trans_d.buffer,  size_d,  M, K/256);
             transpose_2d_as_16b(backend_ctx, extra->dm, buf_trans_dm.buffer, size_dm, M, K/256);
@@ -12355,7 +12449,7 @@ static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer,
             cl_kernel kernel = backend_ctx->kernel_restore_block_q5_K_noshuffle;
             CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q.buffer));
             CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_qh.buffer));
-            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra->s));
+            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_trans_s.buffer));
             CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &buf_trans_d.buffer));
             CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &buf_trans_dm.buffer));
             CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &data_device));
@@ -22443,6 +22537,217 @@ static void ggml_cl_mul_mat_q6_K_f32_adreno(ggml_backend_t backend, const ggml_t
 #endif
 }
 
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+static void ggml_cl_mul_mat_q5_K_f32_adreno_ila(ggml_backend_t backend, const ggml_tensor * src0,
+                                                const ggml_tensor * src1, ggml_tensor * dst) {
+    GGML_ASSERT(src0);
+    GGML_ASSERT(src0->extra);
+    GGML_ASSERT(src1);
+    GGML_ASSERT(src1->extra);
+    GGML_ASSERT(dst);
+    GGML_ASSERT(dst->extra);
+
+    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
+
+    ggml_tensor_extra_cl_q5_K * extra0_q5_k = (ggml_tensor_extra_cl_q5_K *)src0->extra;
+    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
+    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
+
+    cl_ulong offset1 = extra1->offset + src1->view_offs;
+    cl_ulong offsetd = extrad->offset + dst->view_offs;
+
+    const int ne00 = src0->ne[0];
+    const int ne01 = src0->ne[1];
+
+    const int ne1 = dst->ne[1];
+
+    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);
+
+    cl_context context = backend_ctx->context;
+    cl_kernel kernel;
+
+    cl_int           err;
+    cl_buffer_region region;
+    cl_image_format  img_fmt;
+    cl_image_desc    img_desc;
+
+    const int M = ne01;
+    const int N = ne1;
+    const int K = ne00;
+
+    if (ne1 == 1) {
+        cl_mem b_sub_buf  = nullptr;
+        cl_mem b_img      = nullptr;
+
+        region.origin = offset1;
+        region.size   = (size_t)K * N * sizeof(float);
+        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+
+        img_fmt = { CL_RGBA, CL_FLOAT };
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = (size_t)K * N / 4;
+        img_desc.buffer      = b_sub_buf;
+        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+        kernel = backend_ctx->kernel_gemv_noshuffle_q5_k_f32_32b_trans;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q5_k->q_img));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q5_k->qh));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q5_k->d));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q5_k->dm));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extra0_q5_k->s));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &b_img));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem),   &extrad->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offsetd));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne00));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_int),   &ne01));
+
+        size_t local_work_size[3]  = { 64, 8, 1 };
+        size_t global_work_size[3] = { (size_t)ne01, 8, 1 };
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+
+        CL_CHECK(clReleaseMemObject(b_img));
+        CL_CHECK(clReleaseMemObject(b_sub_buf));
+    } else {
+        static const char * q5_k_bin_dp4a_env = getenv("GGML_OPENCL_Q5_K_BIN_DP4A");
+                     bool   q5_k_bin_dp4a_on  = q5_k_bin_dp4a_env
+                                                  ? (atoi(q5_k_bin_dp4a_env) != 0)
+                                                  : true;
+        // dot prod has to be available
+        q5_k_bin_dp4a_on = backend_ctx->has_integer_dot && q5_k_bin_dp4a_on;
+
+        if (q5_k_bin_dp4a_on && backend_ctx->kernel_gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8_bin) {
+            const int    dp4a_N_pad = CEIL_DIV(N, 32) * 32;
+            const size_t n_blocks   = (size_t)dp4a_N_pad * (K / 32);
+
+            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)dp4a_N_pad * K * sizeof(cl_char));
+            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
+            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));
+
+            cl_mem b_sub = nullptr;
+            region.origin = offset1;
+            region.size   = (size_t)K * N * sizeof(float);
+            CL_CHECK((b_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+
+            cl_int    tb = (cl_int)((size_t)N * (K / 32));
+            cl_kernel qk = backend_ctx->kernel_quant_a_q8_1;
+            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub));
+            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
+            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
+            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
+            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
+            size_t q_local[1]  = { 64 };
+            size_t q_global[1] = { (size_t)CEIL_DIV(tb, 64) * 64 };
+            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);
+
+            cl_mem d_sub = nullptr;
+            cl_mem d_img = nullptr;
+            region.origin = offsetd;
+            region.size   = (size_t)M * N * sizeof(float);
+            CL_CHECK((d_sub = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+
+            img_fmt = { CL_R, CL_FLOAT };
+            memset(&img_desc, 0, sizeof(img_desc));
+            img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+            img_desc.image_width = (size_t)M * N;
+            img_desc.buffer      = d_sub;
+            CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+            kernel = backend_ctx->kernel_gemm_noshuffle_q5_k_q8_1_dp4a_ila_a8_bin;
+
+            cl_uint k_arg = 0;
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q5_k->q_img));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q5_k->qh));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q5_k->d));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q5_k->dm));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q5_k->s));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &backend_ctx->prealloc_moe_qa.buffer));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &backend_ctx->prealloc_moe_da.buffer));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &backend_ctx->prealloc_moe_sa.buffer));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &d_img));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_uint), &ne00));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_uint), &ne01));
+            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_int),  &N));
+
+            size_t local_work_size_dp4a[3]  = { 64, 1, 1 };
+            size_t global_work_size_dp4a[3] = { 64, (size_t)(M / 64), (size_t)(dp4a_N_pad / 32) };
+            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size_dp4a, local_work_size_dp4a, dst);
+
+            CL_CHECK(clReleaseMemObject(b_sub));
+            CL_CHECK(clReleaseMemObject(d_img));
+            CL_CHECK(clReleaseMemObject(d_sub));
+            return;
+        }
+
+        const int gemm_tile_n = 64;
+        int N_pad = CEIL_DIV(N, gemm_tile_n) * gemm_tile_n;
+
+        cl_mem b_sub_buf = nullptr;
+        cl_mem b_padded  = nullptr;
+        cl_mem b_buf     = nullptr;
+        if (N_pad == N) {
+            region.origin = offset1;
+            region.size   = (size_t)K * N * sizeof(float);
+            CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+            b_buf = b_sub_buf;
+        } else {
+            CL_CHECK((b_padded = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)K * N_pad * sizeof(float), NULL, &err), err));
+            const float zero = 0.0f;
+            CL_CHECK(clEnqueueFillBuffer(backend_ctx->queue, b_padded, &zero, sizeof(zero), 0, (size_t)K * N_pad * sizeof(float), 0, NULL, NULL));
+            CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, extra1->data_device, b_padded, offset1, 0, (size_t)K * N * sizeof(float), 0, NULL, NULL));
+            b_buf = b_padded;
+        }
+
+        img_fmt = { CL_R, CL_FLOAT };
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = (size_t)K * N_pad;
+        img_desc.buffer      = b_buf;
+        cl_mem b_img;
+        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+        region.origin = offsetd;
+        region.size   = (size_t)M * N * sizeof(float);
+        cl_mem d_sub_buf;
+        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
+        img_fmt = { CL_R, CL_FLOAT };
+        memset(&img_desc, 0, sizeof(img_desc));
+        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+        img_desc.image_width = (size_t)M * N;
+        img_desc.buffer      = d_sub_buf;
+        cl_mem d_img;
+        CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));
+
+        kernel = backend_ctx->kernel_gemm_noshuffle_q5_k_f32_32b_trans_ila_a8_bin;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),  &extra0_q5_k->q_img));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),  &extra0_q5_k->qh));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),  &extra0_q5_k->d));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),  &extra0_q5_k->dm));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),  &extra0_q5_k->s));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),  &b_img));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem),  &d_img));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &ne00));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &ne01));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),     &N));
+
+        size_t local_work_size[3]  = { 64, 2, 2 };
+        size_t m_tiles = (size_t)CEIL_DIV(M, 64);
+        size_t global_work_size[3] = { 64, m_tiles, (size_t)CEIL_DIV(N_pad, gemm_tile_n) };
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+
+        CL_CHECK(clReleaseMemObject(b_img));
+        if (b_sub_buf) {
+            CL_CHECK(clReleaseMemObject(b_sub_buf));
+        }
+        if (b_padded) {
+            CL_CHECK(clReleaseMemObject(b_padded));
+        }
+        CL_CHECK(clReleaseMemObject(d_img));
+        CL_CHECK(clReleaseMemObject(d_sub_buf));
+    }
+}
+#endif // GGML_OPENCL_USE_ADRENO_KERNELS
+
 static void ggml_cl_mul_mat_q5_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
 #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
     GGML_ASSERT(src0);
@@ -22491,6 +22796,20 @@ static void ggml_cl_mul_mat_q5_K_f32_adreno(ggml_backend_t backend, const ggml_t
     static const bool q5k_mc3 = (getenv("GGML_OPENCL_Q5K_MC3") != nullptr);
     const bool use_q5k_mc3 = q5k_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);
 
+    const bool use_bin = use_q5_k_bin_kernels(backend_ctx, src0);
+
+    if (use_bin) {
+        if (use_q5k_mc3) {
+            static bool warned = false;
+            if (!warned) {
+                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q5K_MC3 is bypassed by Q5_K binary kernels\n");
+                warned = true;
+            }
+        }
+        ggml_cl_mul_mat_q5_K_f32_adreno_ila(backend, src0, src1, dst);
+        return;
+    }
+
     if (ne1 == 1 || use_q5k_mc3) {
         cl_mem q_img  = nullptr;
         cl_mem qh_img = nullptr;
diff --git src/ggml-opencl/kernels/gemv_noshuffle_q5_k_f32_32b_trans.cl src/ggml-opencl/kernels/gemv_noshuffle_q5_k_f32_32b_trans.cl
new file mode 100644
index 00000000..ecf15137
--- /dev/null
+++ src/ggml-opencl/kernels/gemv_noshuffle_q5_k_f32_32b_trans.cl
@@ -0,0 +1,141 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+#pragma OPENCL EXTENSION cl_khr_subgroups : enable
+#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
+
+#define QK_K 256
+#define K_SCALE_SIZE 12
+#define N_SIMDGROUP 8
+#define SIMDGROUP_WIDTH 64
+
+inline void get_scale_min_k4(
+    int j,
+    global const uchar * q,
+    uint stride,
+    uchar * d,
+    uchar * m
+) {
+    if (j < 4) {
+        *d = q[j*stride]     & 63;
+        *m = q[(j+4)*stride] & 63;
+    } else {
+        *d = (q[(j+4)*stride] & 0x0F) | ((q[(j-4)*stride] & 0xC0) >> 2);
+        *m = ((q[(j+4)*stride] >> 4) & 0x0F) | ((q[j*stride] & 0xC0) >> 2);
+    }
+}
+
+static inline float8 q5_k_to_fp32_packed8(ushort2 q4x8, uint qh_byte, float scale, float minv) {
+    float8 fp32x8;
+    fp32x8.s0 = (float)(( q4x8.s0        & 0x000F) | (((qh_byte >> 0) & 1) << 4)) * scale - minv;
+    fp32x8.s1 = (float)(((q4x8.s0 >> 4)  & 0x000F) | (((qh_byte >> 1) & 1) << 4)) * scale - minv;
+    fp32x8.s2 = (float)(((q4x8.s0 >> 8)  & 0x000F) | (((qh_byte >> 2) & 1) << 4)) * scale - minv;
+    fp32x8.s3 = (float)(((q4x8.s0 >> 12) & 0x000F) | (((qh_byte >> 3) & 1) << 4)) * scale - minv;
+    fp32x8.s4 = (float)(( q4x8.s1        & 0x000F) | (((qh_byte >> 4) & 1) << 4)) * scale - minv;
+    fp32x8.s5 = (float)(((q4x8.s1 >> 4)  & 0x000F) | (((qh_byte >> 5) & 1) << 4)) * scale - minv;
+    fp32x8.s6 = (float)(((q4x8.s1 >> 8)  & 0x000F) | (((qh_byte >> 6) & 1) << 4)) * scale - minv;
+    fp32x8.s7 = (float)(((q4x8.s1 >> 12) & 0x000F) | (((qh_byte >> 7) & 1) << 4)) * scale - minv;
+    return fp32x8;
+}
+
+__attribute__((qcom_reqd_sub_group_size("half")))
+__kernel void gemv_noshuffle_q5_k_f32_32b_trans(
+    read_only image1d_buffer_t src0_q,
+    __global uchar *        src0_qh,
+    __global half *         src0_d,
+    __global half *         src0_dm,
+    __global uchar *        src0_s,
+    __read_only image1d_buffer_t src1,
+    __global float *        dst,
+    ulong                   offsetd,
+    int                     ne00,
+    int                     ne01
+) {
+    uint i01  = get_global_id(0);
+    uint sgid = get_local_id(1);
+    uint slid = get_sub_group_local_id();
+
+    int num_subblocks = ne00 / 32;
+
+    __private float sum = 0.0f;
+
+    // Loop over sub-blocks of 32 elements, N_SIMDGROUP sub-blocks per iter.
+    for (uint ib = sgid; ib < num_subblocks; ib += N_SIMDGROUP) {
+        uint sb = ib / 8;
+        uint j  = ib % 8;
+
+        // Load d and dmin for this super-block.
+        half d_val  = src0_d[sb * ne01 + i01];
+        half dm_val = src0_dm[sb * ne01 + i01];
+
+        // Load sub-block scale and min. s is transposed [nb][12][M]; stride ne01 per code.
+        global const uchar * sc = src0_s + sb * K_SCALE_SIZE * ne01 + i01;
+        uchar sv, mn;
+        get_scale_min_k4(j, sc, ne01, &sv, &mn);
+
+        float scale = (float)d_val * (float)sv;
+        float minv  = (float)dm_val * (float)mn;
+
+        // Load 4 uints of quants (32 nibbles = 32 elements), column-major stride ne01.
+        uint q_base = ib * ne01 * 4 + i01;
+
+        uint4 regQ;
+        regQ.s0 = read_imageui(src0_q, q_base).x;
+        regQ.s1 = read_imageui(src0_q, q_base + ne01).x;
+        regQ.s2 = read_imageui(src0_q, q_base + ne01 * 2).x;
+        regQ.s3 = read_imageui(src0_q, q_base + ne01 * 3).x;
+
+        uint qh_grp = ib * 4;
+        uint qh_word = (uint)src0_qh[(qh_grp + 0) * ne01 + i01]
+                      | ((uint)src0_qh[(qh_grp + 1) * ne01 + i01] << 8)
+                      | ((uint)src0_qh[(qh_grp + 2) * ne01 + i01] << 16)
+                      | ((uint)src0_qh[(qh_grp + 3) * ne01 + i01] << 24);
+
+        // Load activations: 32 floats = 8 float4s.
+        uint y_offset = ib * 8;
+
+        float4 y_local = (slid < 8) ? read_imagef(src1, (y_offset + slid)) : (float4)0.0f;
+        float4 y0 = sub_group_broadcast(y_local, 0);
+        float4 y1 = sub_group_broadcast(y_local, 1);
+        float4 y2 = sub_group_broadcast(y_local, 2);
+        float4 y3 = sub_group_broadcast(y_local, 3);
+        float4 y4 = sub_group_broadcast(y_local, 4);
+        float4 y5 = sub_group_broadcast(y_local, 5);
+        float4 y6 = sub_group_broadcast(y_local, 6);
+        float4 y7 = sub_group_broadcast(y_local, 7);
+
+        float8 fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s0), qh_word & 0xFF, scale, minv);
+        float4 acc = y0 * fp32x8.lo;
+        acc += y1 * fp32x8.hi;
+
+        fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s1), (qh_word >> 8) & 0xFF, scale, minv);
+        acc += y2 * fp32x8.lo;
+        acc += y3 * fp32x8.hi;
+
+        fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s2), (qh_word >> 16) & 0xFF, scale, minv);
+        acc += y4 * fp32x8.lo;
+        acc += y5 * fp32x8.hi;
+
+        fp32x8 = q5_k_to_fp32_packed8(as_ushort2(regQ.s3), (qh_word >> 24) & 0xFF, scale, minv);
+        acc += y6 * fp32x8.lo;
+        acc += y7 * fp32x8.hi;
+
+        sum += ((acc.s0 + acc.s1) + (acc.s2 + acc.s3));
+    }
+
+    // reduction in local memory over N_SIMDGROUP subgroups
+    __local float reduceLM[SIMDGROUP_WIDTH * (N_SIMDGROUP - 1)];
+    if (sgid > 0) {
+        reduceLM[SIMDGROUP_WIDTH * (sgid - 1) + slid] = sum;
+    }
+    barrier(CLK_LOCAL_MEM_FENCE);
+    if (sgid == 0) {
+        for (uint i = 0; i < N_SIMDGROUP - 1; ++i) {
+            sum += reduceLM[SIMDGROUP_WIDTH * i + slid];
+        }
+    }
+
+    // 1 output per thread in subgroup 0
+    if (sgid == 0) {
+        dst = dst + (offsetd >> 2);
+        dst[i01] = sum;
+    }
+}
diff --git src/ggml-rpc/ggml-rpc.cpp src/ggml-rpc/ggml-rpc.cpp
index c24caad7..353b79b0 100644
--- src/ggml-rpc/ggml-rpc.cpp
+++ src/ggml-rpc/ggml-rpc.cpp
@@ -858,6 +858,11 @@ static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_ty
     if (rpc_get) {
         ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
 
+        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
+        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
+        // TODO: change rpc_tensor nb to 64-bit int
+        const size_t min_size = ggml_nbytes(tensor);
+
         // Cache key for calls to read the alloc_size.
         // We deliberately exclude src tensor dimensions from the key because:
         // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
@@ -871,6 +876,7 @@ static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_ty
             uint32_t op;
             int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
             uint32_t ne[GGML_MAX_DIMS];
+            uint64_t nb[GGML_MAX_DIMS];
         };
 
         alloc_size_cache_key key = {};
@@ -880,6 +886,7 @@ static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_ty
         memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
         for (int i = 0; i < GGML_MAX_DIMS; i++) {
             key.ne[i] = (uint32_t)tensor->ne[i];
+            key.nb[i] = (uint64_t)tensor->nb[i];
         }
 
         uint64_t cache_hash = fnv_hash((const uint8_t *)&key, sizeof(key));
@@ -893,7 +900,7 @@ static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_ty
             std::lock_guard<std::mutex> lock(cache_mutex);
             auto it = cache.find(cache_hash);
             if (it != cache.end()) {
-                return it->second;
+                return std::max<size_t>(it->second, min_size);
             }
         }
 
@@ -915,7 +922,7 @@ static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_ty
             cache[cache_hash] = response.alloc_size;
         }
 
-        return response.alloc_size;
+        return std::max<size_t>(response.alloc_size, min_size);
     }
 
     return ggml_nbytes(tensor);
diff --git src/ggml-sycl/fattn-sparse.cpp src/ggml-sycl/fattn-sparse.cpp
new file mode 100644
index 00000000..df283709
--- /dev/null
+++ src/ggml-sycl/fattn-sparse.cpp
@@ -0,0 +1,267 @@
+#include "fattn.hpp"
+#include "fattn-sparse.hpp"
+
+#include <cstdint>
+#include <cstdio>
+#include <cstring>
+
+static constexpr int64_t SPARSE_FA_PAD       = 256;
+static constexpr int64_t SPARSE_FA_MIN_RATIO = 2;
+
+extern int g_ggml_sycl_enable_sparse_fa;
+extern int g_ggml_sycl_debug_sparse_fa;
+extern int g_ggml_sycl_sparse_fa_margin;
+
+static int sparse_fa_enabled(void) {
+    return g_ggml_sycl_enable_sparse_fa;
+}
+
+static int sparse_fa_debug(void) {
+    return g_ggml_sycl_debug_sparse_fa;
+}
+
+// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
+static int sparse_fa_margin(void) {
+    return g_ggml_sycl_sparse_fa_margin;
+}
+
+// Unordered output is fine: softmax over the selected set is permutation invariant.
+static void sparse_fa_compact_mask(sycl::queue * stream,
+                                   const sycl::half * __restrict__ mask,
+                                   int32_t * __restrict__ indices,
+                                   int32_t * __restrict__ count,
+                                   const int64_t n_kv,
+                                   const int64_t n_kv_g) {
+    constexpr size_t WG = 256;
+    const size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);
+
+    stream->parallel_for(
+        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
+        [=](sycl::nd_item<1> item) {
+            const int64_t i = (int64_t) item.get_global_id(0);
+            if (i >= n_kv || !sycl::isfinite((float) mask[i])) {
+                return;
+            }
+
+            sycl::atomic_ref<int32_t,
+                             sycl::memory_order::relaxed,
+                             sycl::memory_scope::device,
+                             sycl::access::address_space::global_space> ctr(*count);
+
+            const int32_t pos = ctr.fetch_add(1);
+            if (pos < (int32_t) n_kv_g) {
+                indices[pos] = (int32_t) i;
+            }
+        });
+}
+
+// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
+// a plain byte copy and needs no per-type code. Padding slots are zeroed.
+static void sparse_fa_gather_rows(sycl::queue * stream,
+                                  const uint8_t * __restrict__ src,
+                                  uint8_t * __restrict__ dst,
+                                  const int32_t * __restrict__ indices,
+                                  const int32_t * __restrict__ count,
+                                  const size_t row_size,
+                                  const size_t src_nb1,
+                                  const size_t src_nb2,
+                                  const int64_t n_kv_g,
+                                  const int64_t n_head) {
+    GGML_ASSERT(row_size % sizeof(uint32_t) == 0);
+    const size_t words = row_size / sizeof(uint32_t);
+
+    stream->parallel_for(
+        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, words),
+        [=](sycl::id<3> id) {
+            const int64_t h    = (int64_t) id[0];
+            const int64_t slot = (int64_t) id[1];
+            const size_t  w    = id[2];
+
+            uint32_t * dst_row =
+                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);
+
+            if (slot >= (int64_t) *count) {
+                dst_row[w] = 0;
+                return;
+            }
+
+            const uint32_t * src_row =
+                (const uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
+                                    (size_t) h * src_nb2);
+            dst_row[w] = src_row[w];
+        });
+}
+
+static void sparse_fa_gather_mask(sycl::queue * stream,
+                                  const sycl::half * __restrict__ mask,
+                                  sycl::half * __restrict__ mask_g,
+                                  const int32_t * __restrict__ indices,
+                                  const int32_t * __restrict__ count,
+                                  const int64_t n_kv_g,
+                                  const int64_t n_rows,
+                                  const size_t mask_s1) {
+    stream->parallel_for(
+        sycl::range<2>((size_t) n_rows, (size_t) n_kv_g),
+        [=](sycl::id<2> id) {
+            const int64_t r    = (int64_t) id[0];
+            const int64_t slot = (int64_t) id[1];
+
+            sycl::half v = sycl::half(-INFINITY);
+            if (slot < (int64_t) *count) {
+                v = mask[(size_t) r * mask_s1 + (size_t) indices[slot]];
+            }
+            mask_g[(size_t) r * n_kv_g + slot] = v;
+        });
+}
+
+static bool sparse_fa_applicable(const ggml_tensor * dst, int64_t & n_kv_g_out) {
+    const ggml_tensor * Q    = dst->src[0];
+    const ggml_tensor * K    = dst->src[1];
+    const ggml_tensor * V    = dst->src[2];
+    const ggml_tensor * mask = dst->src[3];
+
+    if (!Q || !K || !V || !mask) {
+        return false;
+    }
+
+    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
+    if (n_kv_max <= 0) {
+        return false;
+    }
+
+    float max_bias      = 0.0f;
+    float logit_softcap = 0.0f;
+    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
+    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
+    if (max_bias != 0.0f || logit_softcap != 0.0f) {
+        return false;
+    }
+
+    // single-token decode only; prefill amortises the scan already
+    if (Q->ne[1] != 1) {
+        return false;
+    }
+    if (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
+        return false;
+    }
+    if (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
+        return false;
+    }
+    if (K->ne[2] != V->ne[2]) {
+        return false;
+    }
+
+    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
+    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
+        return false;
+    }
+
+    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
+    const size_t v_row = ggml_row_size(V->type, V->ne[0]);
+    if (k_row % sizeof(uint32_t) || v_row % sizeof(uint32_t)) {
+        return false;
+    }
+
+    const int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
+    if (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
+        return false;
+    }
+
+    n_kv_g_out = n_kv_g;
+    return true;
+}
+
+bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
+    int64_t n_kv_g = 0;
+    if (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
+        return false;
+    }
+
+    ggml_tensor * K    = dst->src[1];
+    ggml_tensor * V    = dst->src[2];
+    ggml_tensor * mask = dst->src[3];
+
+    const int64_t n_kv     = K->ne[1];
+    const int64_t n_head_k = K->ne[2];
+    const int64_t n_rows_m = mask->ne[1];
+
+    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
+    const size_t v_row = ggml_row_size(V->type, V->ne[0]);
+
+    dpct::queue_ptr stream = ctx.stream();
+
+    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
+    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
+    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * k_row);
+    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * v_row);
+    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_rows_m * n_kv_g);
+
+    int32_t *    d_idx  = idx_alloc.get();
+    int32_t *    d_cnt  = cnt_alloc.get();
+    uint8_t *    d_K    = k_alloc.get();
+    uint8_t *    d_V    = v_alloc.get();
+    sycl::half * d_mask = m_alloc.get();
+
+    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(d_cnt, 0, sizeof(int32_t))));
+
+    sparse_fa_compact_mask(stream, (const sycl::half *) mask->data,
+                           d_idx, d_cnt, n_kv, n_kv_g);
+
+    sparse_fa_gather_rows(stream, (const uint8_t *) K->data, d_K, d_idx, d_cnt,
+                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);
+
+    sparse_fa_gather_rows(stream, (const uint8_t *) V->data, d_V, d_idx, d_cnt,
+                          v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);
+
+    sparse_fa_gather_mask(stream, (const sycl::half *) mask->data, d_mask,
+                          d_idx, d_cnt, n_kv_g, n_rows_m,
+                          mask->nb[1] / sizeof(sycl::half));
+
+    if (sparse_fa_debug()) {
+        int32_t h_cnt = 0;
+        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, sizeof(int32_t))));
+        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
+        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
+                (long long) n_kv, ggml_get_op_params_i32(dst, 4),
+                (long long) n_kv_g, (int) h_cnt,
+                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
+    }
+
+    // shallow copies retargeted at the gathered buffers; kernels are unchanged
+    ggml_tensor K_g = *K;
+    K_g.data      = d_K;
+    K_g.ne[1]     = n_kv_g;
+    K_g.nb[1]     = k_row;
+    K_g.nb[2]     = (size_t) n_kv_g * k_row;
+    K_g.nb[3]     = (size_t) n_head_k * n_kv_g * k_row;
+    K_g.view_src  = nullptr;
+    K_g.view_offs = 0;
+
+    ggml_tensor V_g = *V;
+    V_g.data      = d_V;
+    V_g.ne[1]     = n_kv_g;
+    V_g.nb[1]     = v_row;
+    V_g.nb[2]     = (size_t) n_kv_g * v_row;
+    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_row;
+    V_g.view_src  = nullptr;
+    V_g.view_offs = 0;
+
+    ggml_tensor M_g = *mask;
+    M_g.data      = d_mask;
+    M_g.ne[0]     = n_kv_g;
+    M_g.nb[1]     = (size_t) n_kv_g * sizeof(sycl::half);
+    M_g.nb[2]     = M_g.nb[1] * mask->ne[1];
+    M_g.nb[3]     = M_g.nb[2];
+    M_g.view_src  = nullptr;
+    M_g.view_offs = 0;
+
+    ggml_tensor dst_g = *dst;
+    dst_g.src[1] = &K_g;
+    dst_g.src[2] = &V_g;
+    dst_g.src[3] = &M_g;
+    dst_g.op_params[4] = 0;   // avoid re-entering this path
+
+    ggml_sycl_flash_attn_ext(ctx, &dst_g);
+
+    return true;
+}
diff --git src/ggml-sycl/fattn-sparse.hpp src/ggml-sycl/fattn-sparse.hpp
new file mode 100644
index 00000000..98b06bdd
--- /dev/null
+++ src/ggml-sycl/fattn-sparse.hpp
@@ -0,0 +1,10 @@
+#ifndef GGML_SYCL_FATTN_SPARSE_HPP
+#define GGML_SYCL_FATTN_SPARSE_HPP
+
+#include "common.hpp"
+
+// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
+// kernels onto them. Returns false if the caller should use the dense path.
+bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
+
+#endif // GGML_SYCL_FATTN_SPARSE_HPP
diff --git src/ggml-sycl/fattn.cpp src/ggml-sycl/fattn.cpp
index 394cda59..541ae8a8 100644
--- src/ggml-sycl/fattn.cpp
+++ src/ggml-sycl/fattn.cpp
@@ -19,7 +19,7 @@
 #include "fattn-vec.hpp"
 #include "fattn.hpp"
 #include "fattn-onednn.hpp"
-
+#include "fattn-sparse.hpp"
 
 #define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
     {                                                                                                            \
@@ -276,6 +276,11 @@ static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const
 void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
     ggml_sycl_set_device(ctx.device);
 
+    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
+    if (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
+        return;
+    }
+
     // n_kv watchdog: log when n_kv differs from the last FA call with
     // the same D — helps detect cache-truncation issues.
     static int nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index e13ec852..99ebdb3e 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -113,6 +113,9 @@ int g_ggml_sycl_usm_system = 0;
 int g_ggml_sycl_enable_host_pinned_mem = 1;
 int g_ggml_sycl_host_pinned_mem_2g = 0;
 int g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
+int g_ggml_sycl_enable_sparse_fa = 0;
+int g_ggml_sycl_debug_sparse_fa = 0;
+int g_ggml_sycl_sparse_fa_margin = 256;
 
 static ggml_sycl_device_info ggml_sycl_init() {
     GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
@@ -384,6 +387,10 @@ static void ggml_check_sycl() try {
         g_ggml_sycl_host_pinned_mem_2g =
             ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;
 
+        g_ggml_sycl_enable_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA", 0);
+        g_ggml_sycl_debug_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_DEBUG", 0);
+        g_ggml_sycl_sparse_fa_margin = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_MARGIN", 256);
+
         GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");
 
         GGML_LOG_INFO("Build with Macros:\n");
@@ -422,6 +429,7 @@ static void ggml_check_sycl() try {
         GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
 #endif
 
+        //Print the running environment variables for SYCL backend
         GGML_LOG_INFO("Running with Environment Variables:\n");
         GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
         GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);
@@ -491,6 +499,10 @@ static void ggml_check_sycl() try {
         GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);
         GGML_LOG_INFO("  GGML_SYCL_HOST_PINNED_MEM_2G: %d\n", g_ggml_sycl_host_pinned_mem_2g);
 
+        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA: %d\n", g_ggml_sycl_enable_sparse_fa);
+        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_DEBUG: %d\n", g_ggml_sycl_debug_sparse_fa);
+        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_MARGIN: %d\n", g_ggml_sycl_sparse_fa_margin);
+
 /* NOT REMOVE, keep it for next optimize for XMX.
 #if defined(SYCL_USE_XMX)
         fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
diff --git src/ggml-vulkan/ggml-vulkan-types.h src/ggml-vulkan/ggml-vulkan-types.h
index d12cc007..252359bf 100644
--- src/ggml-vulkan/ggml-vulkan-types.h
+++ src/ggml-vulkan/ggml-vulkan-types.h
@@ -386,6 +386,7 @@ enum vk_device_architecture {
     AMD_RDNA1,
     AMD_RDNA2,
     AMD_RDNA3,
+    AMD_RDNA4,
     INTEL_XE1,
     INTEL_XE2,
     NVIDIA_PRE_TURING,
diff --git src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan.cpp
index ded65419..ce40baec 100644
--- src/ggml-vulkan/ggml-vulkan.cpp
+++ src/ggml-vulkan/ggml-vulkan.cpp
@@ -14,6 +14,7 @@ static vk_device_architecture get_device_architecture(const vk::PhysicalDevice&
         bool amd_shader_core_properties = false;
         bool integer_dot_product = false;
         bool subgroup_size_control = false;
+        bool shader_float8 = false;
 
         for (const auto& properties : ext_props) {
             if (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
@@ -22,6 +23,8 @@ static vk_device_architecture get_device_architecture(const vk::PhysicalDevice&
                 integer_dot_product = true;
             } else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                 subgroup_size_control = true;
+            } else if (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
+                shader_float8 = true;
             }
         }
 
@@ -48,6 +51,9 @@ static vk_device_architecture get_device_architecture(const vk::PhysicalDevice&
             if (shader_core_props_amd.wavefrontsPerSimd == 20) {
                 return vk_device_architecture::AMD_RDNA1;
             }
+            if (shader_float8) {
+                return vk_device_architecture::AMD_RDNA4;
+            }
             if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
                 return vk_device_architecture::AMD_RDNA3;
             }
@@ -1497,6 +1503,8 @@ static bool ggml_vk_matmul_int_shmem_support(const vk_device& device, const std:
         case GGML_TYPE_Q8_0:    block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + dm
         case GGML_TYPE_IQ4_XS:  block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
         case GGML_TYPE_MXFP4:   block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
+        case GGML_TYPE_IQ4_NL:  block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
+        case GGML_TYPE_NVFP4:   block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 break; // qs[8] + d_scales(vec2)
         case GGML_TYPE_Q2_K:    block_a_size = std430_size({{ 8, 4}, {2, 2}, {fp2_size, fp2_align}});         break; // qs[2] + scales(u8vec2) + dm(vec2)
         case GGML_TYPE_Q3_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[4] + d_scales(vec2)
         case GGML_TYPE_Q4_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[4] + dm(vec2)
@@ -1534,6 +1542,66 @@ static bool ggml_vk_matmul_int_shmem_support(const vk_device& device, const std:
     return supported;
 }
 
+static bool ggml_vk_matmul_cm1_int_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {
+
+    bool kscales2 = false;    // two scale sets per block
+    bool has_dm   = false;    // d+m as vec2 + b-side sum
+    bool has_kvalues = false;
+    switch (src0_type) {
+        case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0:
+            break;
+        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1:
+        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
+            has_dm = true;                          break;
+        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_MXFP4:
+            has_kvalues = true;                     break;
+        case GGML_TYPE_Q3_K: case GGML_TYPE_Q6_K:
+            kscales2 = true;                        break;
+        case GGML_TYPE_NVFP4:
+            kscales2 = true; has_kvalues = true;    break;
+        default:
+            return false;
+    }
+
+    const uint32_t BLOCK_SIZE = warptile[0];
+    const uint32_t BM         = warptile[1];
+    const uint32_t BN         = warptile[2];
+    const uint32_t WARP       = warptile[10];
+
+    const uint32_t BK      = 32;
+    const uint32_t BK_STEP = mul_mat_id ? 2u : 4u;
+    const uint32_t QPITCH  = BK_STEP * (BK / 4u) + 4u;
+    const uint32_t KSCALES = kscales2 ? 2u : 1u;
+
+    uint32_t total = 0;
+    total += BM * QPITCH * (uint32_t)sizeof(uint32_t);   // buf_a_qs
+    total += BN * QPITCH * (uint32_t)sizeof(uint32_t);   // buf_b_qs
+    total += has_dm ? (BM * BK_STEP * 2u * (uint32_t)sizeof(float))   // buf_a_dm (vec2)
+                    : (BM * BK_STEP * KSCALES * (uint32_t)sizeof(float)); // buf_a_d
+    total += BN * BK_STEP * (uint32_t)sizeof(float);     // buf_b_d
+    if (has_dm) {
+        total += BN * BK_STEP * (uint32_t)sizeof(float); // buf_b_s
+    }
+    if (has_kvalues) {
+        total += 16u * (uint32_t)sizeof(int8_t);         // cm1_kvalues[16]
+    }
+    if (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
+        total += 128u * (uint32_t)sizeof(float);         // ue4m3_fp32_lut[128]
+    }
+    if (mul_mat_id) {
+        total += BN * 2u * (uint32_t)sizeof(uint16_t);   // row_ids[BN] (u16vec2)
+        const uint32_t num_warps = BLOCK_SIZE / std::max(WARP, 1u);
+        total += num_warps * 4u * (uint32_t)sizeof(uint32_t); // ballots_sh[NUM_WARPS] (uvec4)
+    }
+
+    const bool supported = total <= device->properties.limits.maxComputeSharedMemorySize;
+
+    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
+                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << total << ", supported=" << supported);
+
+    return supported;
+}
+
 static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
     {"soft_max", 64}, {"im2col", 64},
     {"argmax", 64}, {"mul_mat_vec", 64},
@@ -1637,6 +1705,8 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
                           l_warptile_id, m_warptile_id, s_warptile_id,
                           l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                           l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
+                          l_warptile_mmq_cm1_int, m_warptile_mmq_cm1_int, s_warptile_mmq_cm1_int,
+                          l_warptile_mmq_cm1_int_k, m_warptile_mmq_cm1_int_k, s_warptile_mmq_cm1_int_k,
                           l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                           l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                           l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
@@ -1645,10 +1715,17 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
     std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                             l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                             l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
+                            l_mmq_cm1_wg_denoms_k, m_mmq_cm1_wg_denoms_k, s_mmq_cm1_wg_denoms_k,
                             l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;
 
     uint32_t l_align, m_align, s_align;
 
+    // RDNA3.5 preferred wave32 here
+    const bool cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
+                                device->subgroup_size_control &&
+                                device->subgroup_min_size <= 32 && device->subgroup_max_size >= 32;
+    const uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_size;
+
     vk_pipeline wait_pipeline;
     CompileTask claimed_task {};
     bool has_claimed_task = false;
@@ -1706,6 +1783,10 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
         const uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
         const uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;
 
+        const uint32_t itm = device->coopmat_int_m;
+        const uint32_t itn = device->coopmat_int_n;
+        const uint32_t itk = device->coopmat_int_k;
+
         const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;
 
         l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
@@ -1721,6 +1802,22 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
         m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
         s_warptile_mmq_int = { subgroup_size_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_size_8 };
 
+        const auto cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
+            return cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
+        };
+
+        l_warptile_mmq_cm1_int = { cm1_bs(128, 128), 128, 128, 32, std::min(cm1_sg, 128u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
+        m_warptile_mmq_cm1_int = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
+        s_warptile_mmq_cm1_int = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
+
+        l_warptile_mmq_cm1_int_k = { cm1_bs( 64, 128),  64, 128, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
+        m_warptile_mmq_cm1_int_k = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
+        s_warptile_mmq_cm1_int_k = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
+
+        l_mmq_cm1_wg_denoms_k = { l_warptile_mmq_cm1_int_k[1], l_warptile_mmq_cm1_int_k[2], 1 };
+        m_mmq_cm1_wg_denoms_k = { m_warptile_mmq_cm1_int_k[1], m_warptile_mmq_cm1_int_k[2], 1 };
+        s_mmq_cm1_wg_denoms_k = { s_warptile_mmq_cm1_int_k[1], s_warptile_mmq_cm1_int_k[2], 1 };
+
         // K-quants use even more registers, mitigate by setting WMITER to 1
         l_warptile_mmq_int_k = { 128,               128, 128, 32, mm_warp_8 * 2, 64, 1, 4, 4, 1, mm_warp_8 };
         m_warptile_mmq_int_k = { 128,                64,  64, 32, mm_warp_8,     32, 1, 2, 2, 1, mm_warp_8 };
@@ -1777,6 +1874,9 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
             }
         }
 
+        const bool use_cm1_int = device->coopmat_int_support &&
+                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
+
         for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
             ggml_type t = (ggml_type)i;
             // Disable medium and large matrix multiplication if not enough shared memory is available
@@ -1806,35 +1906,50 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
 
             // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
             // K-quants and IQ3_S use the _int_k warptiles, others use _int.
+            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
             const bool is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                      t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                      t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
-            const auto & s_int   = is_k_quant ? s_warptile_mmq_int_k   : s_warptile_mmq_int;
-            const auto & m_int   = is_k_quant ? m_warptile_mmq_int_k   : m_warptile_mmq_int;
-            const auto & l_int   = is_k_quant ? l_warptile_mmq_int_k   : l_warptile_mmq_int;
-            const auto & s_intid = is_k_quant ? s_warptile_mmqid_int_k : s_warptile_mmqid_int;
-            const auto & m_intid = is_k_quant ? m_warptile_mmqid_int_k : m_warptile_mmqid_int;
-            const auto & l_intid = is_k_quant ? l_warptile_mmqid_int_k : l_warptile_mmqid_int;
-
-            if (!ggml_vk_matmul_int_shmem_support(device, s_int, false, t)) {
+            const bool cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
+                                     t == GGML_TYPE_NVFP4);
+
+            const auto & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
+                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
+            const auto & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
+                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
+            const auto & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
+                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
+            const auto & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
+                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
+            const auto & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
+                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
+            const auto & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
+                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);
+
+            const auto int_shmem_support = [&](const std::vector<uint32_t>& wt, bool id) {
+                return use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
+                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
+            };
+
+            if (!int_shmem_support(s_int, false)) {
                 device->mul_mat_s_int[i] = false;
                 device->mul_mat_m_int[i] = false;
                 device->mul_mat_l_int[i] = false;
-            } else if (!ggml_vk_matmul_int_shmem_support(device, m_int, false, t)) {
+            } else if (!int_shmem_support(m_int, false)) {
                 device->mul_mat_m_int[i] = false;
                 device->mul_mat_l_int[i] = false;
-            } else if (!ggml_vk_matmul_int_shmem_support(device, l_int, false, t)) {
+            } else if (!int_shmem_support(l_int, false)) {
                 device->mul_mat_l_int[i] = false;
             }
 
-            if (!ggml_vk_matmul_int_shmem_support(device, s_intid, true, t)) {
+            if (!int_shmem_support(s_intid, true)) {
                 device->mul_mat_id_s_int[i] = false;
                 device->mul_mat_id_m_int[i] = false;
                 device->mul_mat_id_l_int[i] = false;
-            } else if (!ggml_vk_matmul_int_shmem_support(device, m_intid, true, t)) {
+            } else if (!int_shmem_support(m_intid, true)) {
                 device->mul_mat_id_m_int[i] = false;
                 device->mul_mat_id_l_int[i] = false;
-            } else if (!ggml_vk_matmul_int_shmem_support(device, l_intid, true, t)) {
+            } else if (!int_shmem_support(l_intid, true)) {
                 device->mul_mat_id_l_int[i] = false;
             }
         }
@@ -2283,6 +2398,29 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
             auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id);
             if (!tc.empty()) create_mm_pipelines(key, tc, name, len, data, pc_size, pc, qs, false, true, 0, true, cm1_pin);
         };
+        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
+        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
+        auto cm1_create_mmq = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
+                                  const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc) {
+            spec_fn_t identity = [](const std::vector<uint32_t>& wt, bool) { return wt; };
+            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id, true);
+            if (!tc.empty()) create_mm_pipelines(key, tc, name, len, data, pc_size, pc, identity, false, false, 0, false, true);
+        };
+
+        std::vector<vk_tile_config> tc_mmq_cm1_int = {
+            {s_warptile_mmq_cm1_int, s_mmq_wg_denoms, s_align},
+            {m_warptile_mmq_cm1_int, m_mmq_wg_denoms, m_align},
+            {l_warptile_mmq_cm1_int, l_mmq_wg_denoms, l_align},
+        };
+        std::vector<vk_tile_config> tc_mmq_cm1_int_k = {
+            {s_warptile_mmq_cm1_int_k, s_mmq_cm1_wg_denoms_k, s_align},
+            {m_warptile_mmq_cm1_int_k, m_mmq_cm1_wg_denoms_k, m_align},
+            {l_warptile_mmq_cm1_int_k, l_mmq_cm1_wg_denoms_k, l_align},
+        };
+
+        // Some quants are not performant on RDNA4, those fall back to FP16 matmul
+        const bool rdna3 = device->architecture == AMD_RDNA3;
+        const bool rdna4 = device->architecture == AMD_RDNA4;
 
         cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_data,     sizeof(vk_mat_mat_push_constants), 3);
         cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_data,     sizeof(vk_mat_mat_push_constants), 3);
@@ -2340,6 +2478,22 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
         }
 #undef X_CM1
 
+        if (device->coopmat_int_support && (rdna3 || rdna4)) {
+            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
+            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
+            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
+            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
+            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
+            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
+            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
+            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_push_constants), 3);
+            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
+            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
+            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
+            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
+            if (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_push_constants), 3); }
+        }
+
         GGML_ASSERT(device->subgroup_ballot);
 
         cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
@@ -2396,6 +2550,22 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
             FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
         }
 #undef X_CM1_ID
+
+        if (device->coopmat_int_support && (rdna3 || rdna4)) {
+            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
+            if (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); }
+        }
     } else
 #endif  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
     {
@@ -3000,6 +3170,7 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
     ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_data, "main", 3, sizeof(vk_op_flash_attn_split_k_reduce_push_constants), {1, device->subgroup_size, 1}, {device->subgroup_size}, 1, true);
 
+#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
     if (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
         auto upper_power_of_2 = [&](uint32_t in) {
             GGML_ASSERT(in != 0);
@@ -3039,6 +3210,7 @@ void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
             ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_data, "main", 5, sizeof(vk_fa_xe_opt_push_constants), { 1, 1, 1 }, { group_sz_ph2, gqa_ratio, head_dim_pv, out_per_wg_ph2, xe_native_sub_group_size, split_p_per_iter_ph2, split_p_chunk, out_dim_per_wg }, 1, false, true, xe_native_sub_group_size);
         }
     }
+#endif
 
     for (auto &it : device->pipeline_fa_mask_opt) {
         auto BrBc = it.first;
@@ -6114,27 +6286,34 @@ static void ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& sub
     // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
     const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                               !ggml_vk_dim01_contiguous(src0);
-    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
-                              // coopmat1: force f32->f16 conversion so the f16 B-type quant pipeline is used.
-                              (ctx->device->coopmat_support && !ctx->device->coopmat2 &&
-                               ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32) ||
-                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
-                              !ggml_vk_dim01_contiguous(src1);
-
     // If src0 is BF16, try to use a BF16 x BF16 multiply
     ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
 
-    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;
-
-    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0;
+    // Prefer the int8 MMQ path (quantize src1 to q8_1) whenever a matching pipeline exists.
+    // The pipeline lookup returns nullptr for types without a q8_1 pipeline (e.g. RDNA4-skipped
+    // quants), in which case coopmat1 falls back to the f16 B-type quant matmul below.
+    bool quantize_y = (ctx->device->integer_dot_product || ctx->device->coopmat_int_support) &&
+                      src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
 
     // Check for mmq first
     const std::vector<vk_matmul_pipeline_pair>* mmp_map = quantize_y ? ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, GGML_TYPE_Q8_1, (ggml_prec)dst->op_params[0]) : nullptr;
+    if (mmp_map == nullptr) {
+        quantize_y = false;
+    }
+
+    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
+                              // coopmat1: force f32->f16 conversion so the f16 B-type quant pipeline is
+                              // used, but only when the int8 MMQ path above is not taken.
+                              (ctx->device->coopmat_support && !ctx->device->coopmat2 && !quantize_y &&
+                               ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32) ||
+                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
+                              !ggml_vk_dim01_contiguous(src1);
+
+    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;
 
     if (mmp_map == nullptr) {
         // Fall back to f16 dequant mul mat
         mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);
-        quantize_y = false;
     }
 
     const bool qx_needs_dequant = mmp_map == nullptr || x_non_contig;
@@ -8042,10 +8221,11 @@ void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const
         split_k = CEIL_DIV(KV, split_kv);
         xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
         if (xe_fa_opt) {
-            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
             const uint32_t split_p_size = 32;
             const size_t max_dim = (nek1 + split_p_size - 1) / split_p_size;
             const size_t p_dim = max_dim * split_p_size;
+#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
+            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
             auto& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
             auto it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1 });
             if (it != pipelines.end()) {
@@ -8053,18 +8233,23 @@ void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const
             } else {
                 pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
             }
+#endif
+            if (xe_fa_pipeline_dual_phases.first == nullptr || xe_fa_pipeline_dual_phases.second == nullptr) {
+                xe_fa_opt = false;
+                fa_copy_qstate = false;
+            } else {
+                size_p = neq1 * neq2 * p_dim * neq3 * sizeof(ggml_fp16_t);
+                size_group_max = neq1 * neq2 * max_dim * neq3 * sizeof(float);
+                size_t temp_size = ggml_nelements(q) * sizeof(ggml_fp16_t) + size_p + size_group_max;
+                fa_copy_qstate = true;
+                if (ctx->prealloc_size_x < temp_size) {
+                    ctx->prealloc_size_x = temp_size;
+                    ggml_vk_preallocate_buffers(ctx, subctx);
+                }
 
-            size_p = neq1 * neq2 * p_dim * neq3 * sizeof(ggml_fp16_t);
-            size_group_max = neq1 * neq2 * max_dim * neq3 * sizeof(float);
-            size_t temp_size = ggml_nelements(q) * sizeof(ggml_fp16_t) + size_p + size_group_max;
-            fa_copy_qstate = true;
-            if (ctx->prealloc_size_x < temp_size) {
-                ctx->prealloc_size_x = temp_size;
-                ggml_vk_preallocate_buffers(ctx, subctx);
-            }
-
-            if (ctx->prealloc_x_need_sync) {
-                ggml_vk_sync_buffers(ctx, subctx);
+                if (ctx->prealloc_x_need_sync) {
+                    ggml_vk_sync_buffers(ctx, subctx);
+                }
             }
         }
     }
@@ -15975,7 +16160,7 @@ bool ggml_vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties&
     case VK_VENDOR_ID_AMD:
         if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
             // Workaround for AMD proprietary driver reporting support on all GPUs
-            return arch == vk_device_architecture::AMD_RDNA3;
+            return arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
         }
         return true;
     case VK_VENDOR_ID_QUALCOMM:
diff --git src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1.comp src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1.comp
new file mode 100644
index 00000000..7cab9a11
--- /dev/null
+++ src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1.comp
@@ -0,0 +1,510 @@
+#version 450
+
+#extension GL_EXT_control_flow_attributes : enable
+#extension GL_EXT_shader_16bit_storage : require
+#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
+#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
+
+#extension GL_KHR_shader_subgroup_basic : require
+#extension GL_KHR_cooperative_matrix : require
+#extension GL_KHR_memory_scope_semantics : enable
+
+#if defined(MUL_MAT_ID_USE_SUBGROUPS)
+#extension GL_KHR_shader_subgroup_ballot : enable
+#endif
+
+#ifdef MUL_MAT_ID
+#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
+#endif
+
+#include "types.glsl"
+
+#if defined(DATA_A_Q3_K) || defined(DATA_A_Q6_K) || defined(DATA_A_NVFP4)
+#define KSCALES 2
+#else
+#define KSCALES 1
+#endif
+
+layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
+
+layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
+#if defined(A_TYPE_PACKED16)
+layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
+#endif
+#if defined(A_TYPE_PACKED32)
+layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
+#endif
+layout (binding = 1) readonly buffer B {block_q8_1_x4_packed128 data_b[];};
+layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};
+
+#ifdef MUL_MAT_ID
+layout (binding = 3) readonly buffer IDS {int data_ids[];};
+layout (binding = 4) readonly buffer Counts {int data_expert_count[];};
+#endif
+
+layout (push_constant) uniform parameter
+{
+    uint M;
+    uint N;
+    uint K;
+    uint stride_a;
+    uint stride_b;
+    uint stride_d;
+
+    uint batch_stride_a;
+    uint batch_stride_b;
+    uint batch_stride_d;
+
+#ifdef MUL_MAT_ID
+    uint nei0;
+    uint nei1;
+    uint nbi1;
+    uint ne11;
+    uint n_experts;
+    uint hoist_row_ids;
+#else
+    uint base_work_group_z;
+    uint num_batches;
+    uint k_split;
+    uint ne02;
+    uint ne12;
+    uint broadcast2;
+    uint broadcast3;
+#endif
+} p;
+
+layout (constant_id = 0) const uint BLOCK_SIZE = 256;
+layout (constant_id = 1) const uint BM = 128;
+layout (constant_id = 2) const uint BN = 128;
+// layout (constant_id = 3) const uint BK = 32;
+layout (constant_id = 4) const uint WM = 64;
+layout (constant_id = 5) const uint WN = 32;
+layout (constant_id = 7) const uint TM = 16;
+layout (constant_id = 8) const uint TN = 16;
+layout (constant_id = 9) const uint TK = 16;
+layout (constant_id = 10) const uint WARP = 32;
+layout (constant_id = 11) const uint DEVICE_ARCH = 0; // vk_device_architecture (ggml-vulkan.cpp)
+#define VK_ARCH_AMD_RDNA4 5u
+
+#define BK 32
+#ifdef MUL_MAT_ID
+#define BK_STEP 2
+#else
+#define BK_STEP 4
+#endif
+#define GROUP_A_BUDGET (16u * 1024u * 1024u)
+
+const uint QPITCH = BK_STEP * (BK / 4) + 4;
+
+shared uint32_t buf_a_qs[BM * QPITCH];
+#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
+shared vec2 buf_a_dm[BM * BK_STEP];   // .x = d, .y = m
+#else
+shared float buf_a_d[BM * BK_STEP * KSCALES];
+#endif
+
+shared uint32_t buf_b_qs[BN * QPITCH];
+shared float buf_b_d[BN * BK_STEP];
+
+#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
+shared float buf_b_s[BN * BK_STEP];
+#endif
+
+#if defined(DATA_A_IQ4_NL) || defined(DATA_A_IQ4_XS) || defined(DATA_A_MXFP4) || defined(DATA_A_NVFP4)
+shared int8_t cm1_kvalues[16];
+#endif
+
+#if defined(DATA_A_QUANT_K) || defined(DATA_A_IQ4_XS) || defined(DATA_A_NVFP4)
+#define LOAD_VEC_A 8
+#else
+#define LOAD_VEC_A (4 * QUANT_R)
+#endif
+#define LOAD_VEC_B 16
+
+const uint CM_ELEMS = (TM * TN) / WARP;
+#define ACC_BIAS_BITS 0x4B400000
+#define ACC_BIAS_F    12582912.0f
+const bool USE_MAGIC_BIAS = WARP != 32;
+
+// Accumulator row for element e: RDNA4 blocked, RDNA3/3.5 interleaved.
+uint cm_elem_row(uint e) {
+    const uint row_half = gl_SubgroupInvocationID / TN;
+    return (DEVICE_ARCH == VK_ARCH_AMD_RDNA4) ? (e + row_half * CM_ELEMS) : (row_half + 2u * e);
+}
+
+// min_term = asymmetric-quant min*b_sum correction (0 for symmetric types).
+ACC_TYPE cm1_accumulate(ACC_TYPE prev, int acc_e, float scale_a, float nbias_a, float scale_b, float min_term) {
+    if (USE_MAGIC_BIAS) {
+        const float t = fma(intBitsToFloat(acc_e), scale_a, nbias_a);
+        return ACC_TYPE(fma(t, scale_b, float(prev) + min_term));
+    }
+    return prev + ACC_TYPE(fma(float(acc_e) * scale_a, scale_b, min_term));
+}
+
+#ifdef MUL_MAT_ID
+#define NUM_WARPS (BLOCK_SIZE / WARP)
+#include "mul_mm_id_funcs.glsl"
+#endif
+
+#include "mul_mmq_cm1_funcs.glsl"
+
+void main() {
+#if defined(DATA_A_IQ4_NL) || defined(DATA_A_IQ4_XS)
+    if (gl_LocalInvocationIndex < 16u) {
+        cm1_kvalues[gl_LocalInvocationIndex] = kvalues_iq4nl_const[gl_LocalInvocationIndex];
+    }
+    barrier();
+#elif defined(DATA_A_MXFP4)
+    if (gl_LocalInvocationIndex < 16u) {
+        cm1_kvalues[gl_LocalInvocationIndex] = kvalues_mxfp4_const[gl_LocalInvocationIndex];
+    }
+    barrier();
+#elif defined(DATA_A_NVFP4)
+    if (gl_LocalInvocationIndex < 16u) {
+        cm1_kvalues[gl_LocalInvocationIndex] = kvalues_mxfp4_const[gl_LocalInvocationIndex];
+    }
+#if !defined(USE_OCP_FP4)
+    for (uint i = gl_LocalInvocationIndex; i < 128u; i += BLOCK_SIZE) {
+        ue4m3_fp32_lut[i] = ue4m3_to_fp32_build(i);
+    }
+#endif
+    barrier();
+#endif
+
+    const uint blocks_m = (p.M + BM - 1) / BM;
+    const uint ik = gl_WorkGroupID.x / blocks_m;
+
+#ifdef MUL_MAT_ID
+    const uint ic = gl_WorkGroupID.y;
+    const uint ir = gl_WorkGroupID.x % blocks_m;
+    const uint expert_idx = gl_WorkGroupID.z;
+    if (ic * BN >= data_expert_count[expert_idx]) {
+        return;
+    }
+#else
+    // L2-friendly workgroup scheduling
+    const uint blocks_n = (p.N + BN - 1) / BN;
+#if defined(DATA_A_IQ4_XS)
+    const uint a_panel_bytes = (BM * p.K) / 2 + (BM * p.K) / 32;
+#else
+    const uint a_panel_bytes = BM * p.K + (BM * p.K) / 16;
+#endif
+    const uint group_m = clamp(GROUP_A_BUDGET / max(a_panel_bytes, 1u), 1u, min(blocks_m, 32u));
+    const uint tiles_per_group = group_m * blocks_n;
+    const uint lin = gl_WorkGroupID.y * blocks_m + (gl_WorkGroupID.x % blocks_m);
+    const uint group_id = lin / tiles_per_group;
+    const uint first_m = group_id * group_m;
+    const uint gsize = min(blocks_m - first_m, group_m);
+    const uint in_group = lin - group_id * tiles_per_group;
+    const uint ir = first_m + in_group % gsize;
+    const uint ic = in_group / gsize;
+#endif
+
+#ifndef MUL_MAT_ID
+    const uint batch_idx = gl_WorkGroupID.z + p.base_work_group_z;
+
+    const uint i13 = batch_idx / p.ne12;
+    const uint i12 = batch_idx % p.ne12;
+
+    const uint i03 = i13 / p.broadcast3;
+    const uint i02 = i12 / p.broadcast2;
+
+    const uint batch_idx_a = i03 * p.ne02 + i02;
+#endif
+
+    const uint warp_i = gl_SubgroupID;
+
+    const uint cms_per_row = WM / TM;
+    const uint cms_per_col = WN / TN;
+
+    const uint warp_r = warp_i % (BM / WM);
+    const uint warp_c = warp_i / (BM / WM);
+
+    const uint elem_col0 = gl_SubgroupInvocationID % TN;
+
+    const uint loadr_a = gl_LocalInvocationID.x % (BK / LOAD_VEC_A);
+    const uint loadc_a = gl_LocalInvocationID.x / (BK / LOAD_VEC_A);
+    const uint loadr_b = gl_LocalInvocationID.x % (BK / LOAD_VEC_B);
+    const uint loadc_b = gl_LocalInvocationID.x / (BK / LOAD_VEC_B);
+
+    const uint loadstride_a = BLOCK_SIZE * LOAD_VEC_A / BK;
+    const uint loadstride_b = BLOCK_SIZE * LOAD_VEC_B / BK;
+
+#ifdef MUL_MAT_ID
+    if (p.hoist_row_ids != 0) {
+        load_row_ids_hoisted(expert_idx, ic);
+    } else {
+#ifdef MUL_MAT_ID_USE_SUBGROUPS
+        if (bitCount(p.nei0) == 1) {
+            load_row_ids(expert_idx, true, ic);
+        } else {
+            load_row_ids(expert_idx, false, ic);
+        }
+#else
+        _ne1 = 0;
+        for (uint ii1 = 0; ii1 < p.nei1 && _ne1 < (ic + 1) * BN; ii1++) {
+            for (uint ii0 = 0; ii0 < p.nei0 && _ne1 < (ic + 1) * BN; ii0++) {
+                if (data_ids[ii1*p.nbi1 + ii0] == expert_idx) {
+                    if (_ne1 >= ic * BN) {
+                        row_ids[_ne1 - ic * BN] = u16vec2(ii0, ii1);
+                    }
+                    _ne1++;
+                }
+            }
+        }
+
+        barrier();
+#endif
+    }
+
+    if (ic * BN >= _ne1) return;
+#endif
+
+#ifdef MUL_MAT_ID
+    const uint start_k = 0;
+    const uint end_k = p.K;
+#else
+    const uint start_k = ik * p.k_split;
+    const uint end_k = min(p.K, (ik + 1) * p.k_split);
+#endif
+
+    uint pos_a_ib =
+#ifdef MUL_MAT_ID
+        expert_idx * (p.batch_stride_a / BK) +
+#else
+        batch_idx_a * (p.batch_stride_a / BK) +
+#endif
+        (ir * BM * p.stride_a + start_k) / BK;
+#ifdef MUL_MAT_ID
+    uint pos_b_ib = 0;
+#else
+    uint pos_b_ib = (batch_idx * p.batch_stride_b + ic * BN * p.stride_b + start_k) / BK;
+#endif
+
+    ACC_TYPE sums[cms_per_row * cms_per_col * CM_ELEMS];
+    [[unroll]] for (uint i = 0; i < cms_per_row * cms_per_col * CM_ELEMS; i++) {
+        sums[i] = ACC_TYPE(0.0);
+    }
+
+    // Double-buffering: prefetch registers
+    const uint A_LOADS = (BM + loadstride_a - 1) / loadstride_a;
+    const uint B_LOADS = (BN + loadstride_b - 1) / loadstride_b;
+
+    block_a_prefetch pre_a[A_LOADS * BK_STEP];
+    block_b_prefetch pre_b[B_LOADS * BK_STEP];
+
+    if (start_k < end_k) {
+        PREFETCH_BLOCK(start_k)
+    }
+
+    const uint a_row0 = warp_r * WM;
+    const uint b_col0 = warp_c * WN;
+#ifdef MUL_MAT_ID
+    const bool active_col_tile = ic * BN + b_col0 < _ne1;
+#else
+    const bool active_col_tile = ic * BN + b_col0 < p.N;
+#endif
+
+    barrier();
+
+    for (uint block = start_k; block < end_k; block += BK * BK_STEP) {
+        STORE_BLOCK_TO_LDS(block)
+
+        barrier();
+
+        pos_a_ib += BK_STEP;
+        pos_b_ib += BK_STEP;
+
+        const uint next_block = block + BK * BK_STEP;
+        if (next_block < end_k) {
+            PREFETCH_BLOCK(next_block)
+        }
+
+        if (active_col_tile) {
+        [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {
+            const uint K_SUB = BK / TK;
+
+#if KSCALES == 2
+            [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+                    coopmat<int8_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a;
+                    coopMatLoad(cache_a, buf_a_qs, (a_row0 + r * TM) * QPITCH + ks * (BK / 4) + h * (TK / 4), QPITCH, gl_CooperativeMatrixLayoutRowMajor);
+
+                    float scale_a[CM_ELEMS];
+                    float nbias_a[CM_ELEMS];
+                    [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                        scale_a[e] = buf_a_d[(ks * KSCALES + h) * BM + a_row0 + r * TM + cm_elem_row(e)];
+                        if (USE_MAGIC_BIAS) {
+                            nbias_a[e] = -ACC_BIAS_F * scale_a[e];
+                        }
+                    }
+
+                    [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                        coopmat<int8_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b;
+                        coopMatLoad(cache_b, buf_b_qs, (b_col0 + c * TN) * QPITCH + ks * (BK / 4) + h * (TK / 4), QPITCH, gl_CooperativeMatrixLayoutColumnMajor);
+
+                        const float scale_b_v = buf_b_d[ks * BN + b_col0 + c * TN + elem_col0];
+
+                        coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc =
+                            coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(
+                                USE_MAGIC_BIAS ? ACC_BIAS_BITS : 0);
+                        acc = coopMatMulAdd(cache_a, cache_b, acc);
+
+                        const uint tile_idx = r * cms_per_col + c;
+                        [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                            sums[tile_idx * CM_ELEMS + e] = cm1_accumulate(
+                                sums[tile_idx * CM_ELEMS + e], int(acc[e]),
+                                scale_a[e], nbias_a[e], scale_b_v, 0.0);
+                        }
+                    }
+                }
+            }
+#elif defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
+            // Preload all A/B fragments up front (ILP).
+            coopmat<int8_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a[cms_per_row * K_SUB];
+            coopmat<int8_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b[cms_per_col * K_SUB];
+
+            [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+                [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                    coopMatLoad(cache_a[r * K_SUB + h], buf_a_qs, (a_row0 + r * TM) * QPITCH + ks * (BK / 4) + h * (TK / 4), QPITCH, gl_CooperativeMatrixLayoutRowMajor);
+                }
+            }
+            [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                    coopMatLoad(cache_b[c * K_SUB + h], buf_b_qs, (b_col0 + c * TN) * QPITCH + ks * (BK / 4) + h * (TK / 4), QPITCH, gl_CooperativeMatrixLayoutColumnMajor);
+                }
+            }
+
+            float scale_b[cms_per_col];
+            float bs[cms_per_col];
+            [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                scale_b[c] = buf_b_d[ks * BN + b_col0 + c * TN + elem_col0];
+                bs[c] = float(buf_b_s[ks * BN + b_col0 + c * TN + elem_col0]);
+            }
+
+            [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+                float scale_a[CM_ELEMS];
+                float nbias_a[CM_ELEMS];
+                float ma[CM_ELEMS];
+                [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                    vec2 dm = buf_a_dm[ks * BM + a_row0 + r * TM + cm_elem_row(e)];
+                    scale_a[e] = dm.x;
+                    if (USE_MAGIC_BIAS) {
+                        nbias_a[e] = -ACC_BIAS_F * scale_a[e];
+                    }
+                    ma[e] = dm.y;
+                }
+
+                [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                    coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc =
+                        coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(
+                            USE_MAGIC_BIAS ? ACC_BIAS_BITS : 0);
+
+                    [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                        acc = coopMatMulAdd(cache_a[r * K_SUB + h], cache_b[c * K_SUB + h], acc);
+                    }
+
+                    const uint tile_idx = r * cms_per_col + c;
+                    [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                        sums[tile_idx * CM_ELEMS + e] = cm1_accumulate(
+                            sums[tile_idx * CM_ELEMS + e], int(acc[e]),
+                            scale_a[e], nbias_a[e], scale_b[c], ma[e] * bs[c]);
+                    }
+                }
+            }
+#else
+            // Preload all A/B fragments up front (ILP).
+            coopmat<int8_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a[cms_per_row * K_SUB];
+            coopmat<int8_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b[cms_per_col * K_SUB];
+
+            [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+                [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                    coopMatLoad(cache_a[r * K_SUB + h], buf_a_qs, (a_row0 + r * TM) * QPITCH + ks * (BK / 4) + h * (TK / 4), QPITCH, gl_CooperativeMatrixLayoutRowMajor);
+                }
+            }
+            [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                    coopMatLoad(cache_b[c * K_SUB + h], buf_b_qs, (b_col0 + c * TN) * QPITCH + ks * (BK / 4) + h * (TK / 4), QPITCH, gl_CooperativeMatrixLayoutColumnMajor);
+                }
+            }
+
+            float scale_b[cms_per_col];
+            [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                scale_b[c] = buf_b_d[ks * BN + b_col0 + c * TN + elem_col0];
+            }
+
+            [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+                float scale_a[CM_ELEMS];
+                float nbias_a[CM_ELEMS];
+                [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                    scale_a[e] = buf_a_d[ks * BM + a_row0 + r * TM + cm_elem_row(e)];
+                    if (USE_MAGIC_BIAS) {
+                        nbias_a[e] = -ACC_BIAS_F * scale_a[e];
+                    }
+                }
+
+                [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+                    coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> acc =
+                        coopmat<int32_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(
+                            USE_MAGIC_BIAS ? ACC_BIAS_BITS : 0);
+
+                    [[unroll]] for (uint h = 0; h < K_SUB; h++) {
+                        acc = coopMatMulAdd(cache_a[r * K_SUB + h], cache_b[c * K_SUB + h], acc);
+                    }
+
+                    const uint tile_idx = r * cms_per_col + c;
+                    [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                        sums[tile_idx * CM_ELEMS + e] = cm1_accumulate(
+                            sums[tile_idx * CM_ELEMS + e], int(acc[e]),
+                            scale_a[e], nbias_a[e], scale_b[c], 0.0);
+                    }
+                }
+            }
+#endif // KSCALES
+        }
+        }
+
+        barrier();
+    }
+
+#undef PREFETCH_BLOCK
+#undef STORE_BLOCK_TO_LDS
+#undef B_IB_CALC
+
+    const uint dr = ir * BM + a_row0;
+    const uint dc = ic * BN + b_col0;
+
+#ifdef MUL_MAT_ID
+    [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+        [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+            const uint tile_idx = r * cms_per_col + c;
+            [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                const uint col_i = dc + c * TN + elem_col0;
+                if (col_i >= _ne1) continue;
+
+                const uint row_g = dr + r * TM + cm_elem_row(e);
+                if (row_g >= p.M) continue;
+
+                const u16vec2 row_idx = row_ids[col_i - ic * BN];
+                const uint store_offset = row_idx.y * p.batch_stride_d + row_idx.x * p.stride_d + row_g;
+                data_d[store_offset] = D_TYPE(sums[tile_idx * CM_ELEMS + e]);
+            }
+        }
+    }
+#else
+    const uint offsets = batch_idx * p.batch_stride_d + ik * p.batch_stride_d * p.num_batches;
+
+    [[unroll]] for (uint r = 0; r < cms_per_row; r++) {
+        [[unroll]] for (uint c = 0; c < cms_per_col; c++) {
+            const uint tile_idx = r * cms_per_col + c;
+            [[unroll]] for (uint e = 0; e < CM_ELEMS; e++) {
+                const uint row_g = dr + r * TM + cm_elem_row(e);
+                const uint col_g = dc + c * TN + elem_col0;
+                if (row_g < p.M && col_g < p.N) {
+                    data_d[offsets + col_g * p.stride_d + row_g] = D_TYPE(sums[tile_idx * CM_ELEMS + e]);
+                }
+            }
+        }
+    }
+#endif // MUL_MAT_ID
+}
diff --git src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1_funcs.glsl src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1_funcs.glsl
new file mode 100644
index 00000000..1760c138
--- /dev/null
+++ src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1_funcs.glsl
@@ -0,0 +1,594 @@
+// Per-quant-type data structures and functions for the cm1 int8 coopmat path.
+// Each quant type defines:
+//   struct block_a_prefetch  — register data for one A-block per thread
+//   block_a_load()           — load from global memory into a block_a_prefetch
+//   block_a_to_shmem()       — unpack and write to shared memory
+
+#if defined(DATA_A_Q4_0)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    float16_t d;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
+                             data_a_packed16[ib].qs[loadr * 2 + 1]));
+    blk.d = data_a_packed16[ib].d;
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
+    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
+    lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;
+    hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
+
+    if (loadr == 0) {
+        buf_a_d[ks * BM + buf_ib] = float(blk.d);
+    }
+}
+
+#elif defined(DATA_A_Q4_1)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    f16vec2 dm;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = data_a_packed32[ib].qs[loadr];
+    blk.dm = data_a_packed32[ib].dm;
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    // Store raw unsigned nibbles; the -8 offset is absorbed by the min term.
+    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
+    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
+
+    if (loadr == 0) {
+        buf_a_dm[ks * BM + buf_ib] = vec2(float(blk.dm.x), float(blk.dm.y));
+    }
+}
+
+#elif defined(DATA_A_Q5_0)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    float16_t d;
+    uint32_t qh;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
+                             data_a_packed16[ib].qs[loadr * 2 + 1]));
+    blk.d = data_a_packed16[ib].d;
+    blk.qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
+    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
+    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
+    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
+    lo4 = ((lo4 | 0x80808080) - 0x10101010) ^ 0x80808080;
+    hi4 = ((hi4 | 0x80808080) - 0x10101010) ^ 0x80808080;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
+
+    if (loadr == 0) {
+        buf_a_d[ks * BM + buf_ib] = float(blk.d);
+    }
+}
+
+#elif defined(DATA_A_Q5_1)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    f16vec2 dm;
+    uint32_t qh;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = data_a_packed32[ib].qs[loadr];
+    blk.dm = data_a_packed32[ib].dm;
+    blk.qh = data_a_packed32[ib].qh;
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    // Store raw unsigned 5-bit values; the -16 offset is absorbed by the min term.
+    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
+    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
+    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
+    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
+
+    if (loadr == 0) {
+        buf_a_dm[ks * BM + buf_ib] = vec2(float(blk.dm.x), float(blk.dm.y));
+    }
+}
+
+#elif defined(DATA_A_Q8_0)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    float16_t d;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
+                             data_a_packed16[ib].qs[loadr * 2 + 1]));
+    blk.d = data_a_packed16[ib].d;
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr] = blk.qs;
+
+    if (loadr == 0) {
+        buf_a_d[ks * BM + buf_ib] = float(blk.d);
+    }
+}
+
+#elif defined(DATA_A_IQ4_NL)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    float16_t d;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
+                             data_a_packed16[ib].qs[loadr * 2 + 1]));
+    blk.d = data_a_packed16[ib].d;
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
+    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
+        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
+                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
+        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
+                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));
+
+    if (loadr == 0) {
+        buf_a_d[ks * BM + buf_ib] = float(blk.d);
+    }
+}
+
+#elif defined(DATA_A_IQ4_XS)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    float d;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    const uint ib_k = ib / 8;
+    const uint ib32 = ib % 8;
+    blk.qs = data_a_packed32[ib_k].qs[4 * ib32 + loadr];
+    blk.d = 0.0;
+    if (loadr == 0) {
+        const uint sl = (data_a_packed32[ib_k].scales_l >> (4 * ib32)) & 0xF;
+        const uint sh = (data_a_packed32[ib_k].scales_h >> (2 * ib32)) & 3;
+        blk.d = float(data_a_packed32[ib_k].d) * float(int(sl | (sh << 4)) - 32);
+    }
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
+    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
+        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
+                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
+        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
+                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));
+
+    if (loadr == 0) {
+        buf_a_d[ks * BM + buf_ib] = blk.d;
+    }
+}
+
+#elif defined(DATA_A_MXFP4)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    uint8_t e;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    blk.qs = pack32(u8vec4(data_a[ib].qs[loadr * 4],
+                            data_a[ib].qs[loadr * 4 + 1],
+                            data_a[ib].qs[loadr * 4 + 2],
+                            data_a[ib].qs[loadr * 4 + 3]));
+    blk.e = data_a[ib].e;
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
+    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
+        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
+                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
+        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
+                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));
+
+    if (loadr == 0) {
+        buf_a_d[ks * BM + buf_ib] = e8m0_to_fp32(blk.e) * 0.5;
+    }
+}
+
+// LOAD_VEC_A=8 for k-quants and NVFP4: loadr has 4 positions, each writes 2 uint32
+
+#elif defined(DATA_A_Q4_K)
+
+struct block_a_prefetch {
+    uint32_t qs0;
+    uint32_t qs1;
+    uint ib;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    const uint ib_k = ib / 8;
+    const uint sub = ib % 8;
+    const uint qs_base = (sub >> 1) * 8;
+
+    uint32_t raw0 = data_a_packed32[ib_k].qs[qs_base + loadr * 2];
+    uint32_t raw1 = data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1];
+    if ((sub & 1u) != 0u) {
+        blk.qs0 = (raw0 >> 4) & 0x0F0F0F0F;
+        blk.qs1 = (raw1 >> 4) & 0x0F0F0F0F;
+    } else {
+        blk.qs0 = raw0 & 0x0F0F0F0F;
+        blk.qs1 = raw1 & 0x0F0F0F0F;
+    }
+    blk.ib = ib;
+
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    // Store raw unsigned nibbles (blk.qs already masked); no -8 recentering needed.
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = blk.qs0;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = blk.qs1;
+
+    if (loadr == 0) {
+        const uint ib_k = blk.ib / 8;
+        const uint sub = blk.ib % 8;
+        const uint j = sub & 3u;
+        const uint s_j  = uint(data_a[ib_k].scales[j]);
+        const uint s_j4 = uint(data_a[ib_k].scales[j + 4]);
+        const uint s_j8 = uint(data_a[ib_k].scales[j + 8]);
+        const uint sc_val = (sub < 4) ? (s_j  & 0x3Fu) : ((s_j8 & 0x0Fu) | ((s_j  >> 6) << 4));
+        const uint mn_val = (sub < 4) ? (s_j4 & 0x3Fu) : ((s_j8 >> 4)    | ((s_j4 >> 6) << 4));
+        vec2 dm = vec2(data_a_packed32[ib_k].dm);
+        float d_scaled = dm.x * float(sc_val);
+        buf_a_dm[ks * BM + buf_ib] = vec2(d_scaled, -(dm.y * float(mn_val)));
+    }
+}
+
+#elif defined(DATA_A_Q5_K)
+
+struct block_a_prefetch {
+    uint32_t qs0;
+    uint32_t qs1;
+    uint32_t qh0;
+    uint32_t qh1;
+    uint ib;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    const uint ib_k = ib / 8;
+    const uint sub = ib % 8;
+    const uint qs_base = (sub >> 1) * 8;
+
+    uint32_t raw0 = data_a_packed32[ib_k].qs[qs_base + loadr * 2];
+    uint32_t raw1 = data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1];
+    if ((sub & 1u) != 0u) {
+        blk.qs0 = (raw0 >> 4) & 0x0F0F0F0F;
+        blk.qs1 = (raw1 >> 4) & 0x0F0F0F0F;
+    } else {
+        blk.qs0 = raw0 & 0x0F0F0F0F;
+        blk.qs1 = raw1 & 0x0F0F0F0F;
+    }
+    blk.qh0 = ((data_a_packed32[ib_k].qh[loadr * 2    ] >> sub) & 0x01010101) << 4;
+    blk.qh1 = ((data_a_packed32[ib_k].qh[loadr * 2 + 1] >> sub) & 0x01010101) << 4;
+    blk.ib = ib;
+
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    // Store raw unsigned 5-bit values (qs nibble | qh bit); no -16 recentering needed.
+    uint32_t v0 = blk.qs0 | blk.qh0;
+    uint32_t v1 = blk.qs1 | blk.qh1;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;
+
+    if (loadr == 0) {
+        const uint ib_k = blk.ib / 8;
+        const uint sub = blk.ib % 8;
+        const uint j = sub & 3u;
+        const uint s_j  = uint(data_a[ib_k].scales[j]);
+        const uint s_j4 = uint(data_a[ib_k].scales[j + 4]);
+        const uint s_j8 = uint(data_a[ib_k].scales[j + 8]);
+        const uint sc_val = (sub < 4) ? (s_j  & 0x3Fu) : ((s_j8 & 0x0Fu) | ((s_j  >> 6) << 4));
+        const uint mn_val = (sub < 4) ? (s_j4 & 0x3Fu) : ((s_j8 >> 4)    | ((s_j4 >> 6) << 4));
+        vec2 dm = vec2(data_a_packed32[ib_k].dm);
+        float d_scaled = dm.x * float(sc_val);
+        buf_a_dm[ks * BM + buf_ib] = vec2(d_scaled, -(dm.y * float(mn_val)));
+    }
+}
+
+#elif defined(DATA_A_Q6_K)
+
+struct block_a_prefetch {
+    uint32_t qs0;
+    uint32_t qs1;
+    uint ib;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    const uint ib_k = ib / 8;
+    const uint sub = ib % 8;
+    const uint g = sub / 4;
+    const uint j = sub % 4;
+
+    const uint ql_u16 = g * 32 + (j & 1) * 16 + loadr * 4;
+    const uint qh_u16 = g * 16 + loadr * 4;
+    const uint qh_shift = j * 2;
+
+    uint32_t ql0 = pack32(u16vec2(data_a_packed16[ib_k].ql[ql_u16    ],
+                                   data_a_packed16[ib_k].ql[ql_u16 + 1]));
+    uint32_t ql1 = pack32(u16vec2(data_a_packed16[ib_k].ql[ql_u16 + 2],
+                                   data_a_packed16[ib_k].ql[ql_u16 + 3]));
+    if (j >= 2) {
+        ql0 = (ql0 >> 4) & 0x0F0F0F0F;
+        ql1 = (ql1 >> 4) & 0x0F0F0F0F;
+    } else {
+        ql0 = ql0 & 0x0F0F0F0F;
+        ql1 = ql1 & 0x0F0F0F0F;
+    }
+
+    uint32_t qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_u16    ],
+                                   data_a_packed16[ib_k].qh[qh_u16 + 1]));
+    uint32_t qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_u16 + 2],
+                                   data_a_packed16[ib_k].qh[qh_u16 + 3]));
+
+    blk.qs0 = ql0 | (((qh0 >> qh_shift) & 0x03030303) << 4);
+    blk.qs1 = ql1 | (((qh1 >> qh_shift) & 0x03030303) << 4);
+    blk.ib = ib;
+
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x20202020) ^ 0x80808080;
+    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x20202020) ^ 0x80808080;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;
+
+    if (loadr == 0) {
+        const uint ib_k = blk.ib / 8;
+        const uint sub = blk.ib % 8;
+        i8vec2 sc = unpack8(int32_t(int16_t(data_a_packed16[ib_k].scales[sub]))).xy;
+        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = float(data_a_packed16[ib_k].d) * float(sc.x);
+        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = float(data_a_packed16[ib_k].d) * float(sc.y);
+    }
+}
+
+#elif defined(DATA_A_Q3_K)
+
+struct block_a_prefetch {
+    uint32_t qs0;
+    uint32_t qs1;
+    uint ib;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    const uint ib_k = ib / 8;
+    const uint sub = ib % 8;
+    const uint g = sub / 4;
+    const uint j = sub % 4;
+    const uint qs_shift = j * 2;
+    const uint hm_bit = j + g * 4;
+
+    const uint qs_u16 = g * 16 + loadr * 4;
+    uint32_t qs0 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_u16    ],
+                                   data_a_packed16[ib_k].qs[qs_u16 + 1]));
+    uint32_t qs1 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_u16 + 2],
+                                   data_a_packed16[ib_k].qs[qs_u16 + 3]));
+
+    const uint hm_u16 = loadr * 4;
+    uint32_t hm0 = pack32(u16vec2(data_a_packed16[ib_k].hmask[hm_u16    ],
+                                   data_a_packed16[ib_k].hmask[hm_u16 + 1]));
+    uint32_t hm1 = pack32(u16vec2(data_a_packed16[ib_k].hmask[hm_u16 + 2],
+                                   data_a_packed16[ib_k].hmask[hm_u16 + 3]));
+
+    blk.qs0 = ((qs0 >> qs_shift) & 0x03030303) | (((hm0 >> hm_bit) & 0x01010101) << 2);
+    blk.qs1 = ((qs1 >> qs_shift) & 0x03030303) | (((hm1 >> hm_bit) & 0x01010101) << 2);
+    blk.ib = ib;
+
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x04040404) ^ 0x80808080;
+    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x04040404) ^ 0x80808080;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;
+
+    if (loadr == 0) {
+        const uint ib_k = blk.ib / 8;
+        const uint sub = blk.ib % 8;
+        const uint is = sub * 2;
+        uint lo = uint(data_a_packed16[ib_k].scales[(is % 8) / 2]);
+        lo = (lo >> (4 * (is / 8))) & 0x0F0Fu;
+        uint hi = uint(data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2]);
+        hi = (hi >> (2 * (is / 4))) & 0x0303u;
+        uint combined = lo | (hi << 4);
+        i8vec2 sc = unpack8(int32_t(combined)).xy;
+        float d = float(data_a_packed16[ib_k].d);
+        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = d * float(int(sc.x) - 32);
+        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = d * float(int(sc.y) - 32);
+    }
+}
+
+#elif defined(DATA_A_NVFP4)
+
+struct block_a_prefetch {
+    uint32_t qs;
+    uint8_t d0;
+    uint8_t d1;
+};
+
+block_a_prefetch block_a_load(uint ib, uint loadr) {
+    block_a_prefetch blk;
+    const uint ib_k = ib / 2;
+    const uint ihalf = ib % 2;
+    const uint sub = ihalf * 2 + (loadr >> 1);
+    const uint byte_group = loadr & 1u;
+
+    blk.qs = pack32(u8vec4(data_a[ib_k].qs[sub * 8 + byte_group * 4],
+                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 1],
+                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 2],
+                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 3]));
+    blk.d0 = data_a[ib_k].d[ihalf * 2];
+    blk.d1 = data_a[ib_k].d[ihalf * 2 + 1];
+
+    return blk;
+}
+
+void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
+    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
+    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
+    const uint sub_base = (loadr >> 1) * 4;
+    const uint byte_group = loadr & 1u;
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + byte_group] =
+        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
+                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
+    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + 2 + byte_group] =
+        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
+                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));
+
+    if (loadr == 0) {
+        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = ue4m3_to_fp32(blk.d0) * 0.5;
+        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = ue4m3_to_fp32(blk.d1) * 0.5;
+    }
+}
+
+#endif
+
+// ===== B-side: load and store =====
+
+struct block_b_prefetch {
+    ivec4 qs;
+    float16_t d;
+#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
+    float16_t s;
+#endif
+};
+
+block_b_prefetch block_b_load(uint ib_outer, uint ib_inner, uint loadr) {
+    block_b_prefetch blk;
+    blk.qs = data_b[ib_outer].qs[ib_inner * 2 + loadr];
+    blk.d = data_b[ib_outer].ds[ib_inner].x;
+#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
+    blk.s = data_b[ib_outer].ds[ib_inner].y;
+#endif
+    return blk;
+}
+
+void block_b_to_shmem(block_b_prefetch blk, uint buf_ib, uint ks, uint loadr, bool in_bounds) {
+    const ivec4 v = in_bounds ? blk.qs : ivec4(0);
+    const uint base = buf_ib * QPITCH + ks * (BK / 4) + loadr * 4;
+    buf_b_qs[base    ] = v.x;
+    buf_b_qs[base + 1] = v.y;
+    buf_b_qs[base + 2] = v.z;
+    buf_b_qs[base + 3] = v.w;
+    if (loadr == 0) {
+        buf_b_d[ks * BN + buf_ib] = in_bounds ? float(blk.d) : 0.0f;
+#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
+        buf_b_s[ks * BN + buf_ib] = in_bounds ? float(blk.s) : 0.0f;
+#endif
+    }
+}
+
+// ===== Framework macros =====
+
+#ifdef MUL_MAT_ID
+#define B_IB_CALC                                                                               \
+            const u16vec2 row_idx = row_ids[buf_ib];                                            \
+            const uint ib = pos_b_ib + row_idx.y * p.batch_stride_b / BK                        \
+                          + (row_idx.x % p.ne11) * p.stride_b / BK;
+#else
+#define B_IB_CALC                                                                               \
+            const uint ib = pos_b_ib + buf_ib * p.stride_b / BK;
+#endif
+
+#define PREFETCH_BLOCK(blk)                                                                     \
+    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
+        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
+        if (buf_ib < BM) {                                                                      \
+            const uint ib = pos_a_ib + buf_ib * p.stride_a / BK;                                \
+            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
+                pre_a[li * BK_STEP + ks] = block_a_load(ib + ks, loadr_a);                      \
+            }                                                                                   \
+        }                                                                                       \
+    }                                                                                           \
+    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
+        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
+        if (buf_ib < BN) {                                                                      \
+            B_IB_CALC                                                                           \
+            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
+                const uint ib_k = ((blk) + ks * BK < end_k) ? (ib + ks) : ib;                   \
+                pre_b[li * BK_STEP + ks] = block_b_load(ib_k / 4, ib_k % 4, loadr_b);          \
+            }                                                                                   \
+        }                                                                                       \
+    }
+
+#define STORE_BLOCK_TO_LDS(blk)                                                                 \
+    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
+        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
+        if (buf_ib < BM) {                                                                      \
+            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
+                block_a_to_shmem(pre_a[li * BK_STEP + ks], buf_ib, ks, loadr_a);                \
+            }                                                                                   \
+        }                                                                                       \
+    }                                                                                           \
+    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
+        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
+        if (buf_ib < BN) {                                                                      \
+            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
+                const bool in_bounds = (blk) + ks * BK < end_k;                                 \
+                block_b_to_shmem(pre_b[li * BK_STEP + ks], buf_ib, ks, loadr_b, in_bounds);     \
+            }                                                                                   \
+        }                                                                                       \
+    }
diff --git src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
index 12f9b3f5..e7e303e5 100644
--- src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
+++ src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
@@ -480,8 +480,9 @@ void matmul_shaders(bool fp16, MatMulIdType matmul_id_type, bool coopmat, bool c
         base_dict["FLOAT16"] = "1";
     }
 
-    base_dict["ACC_TYPE"  ] = f16acc ? "float16_t" : "float";
-    base_dict["ACC_TYPEV2"] = f16acc ? "f16vec2"   : "vec2";
+    base_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
+    base_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
+    base_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
     if (f16acc) {
         base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
     }
@@ -629,6 +630,11 @@ void matmul_shaders(bool fp16, MatMulIdType matmul_id_type, bool coopmat, bool c
         }
 #endif
 
+        if (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
+                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
+            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_key, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
+        }
+
         if (is_lut_quant(tname)) {
             std::string lva = lut_load_vec_a(tname);
 
@@ -923,8 +929,10 @@ void process_shaders() {
 
     string_to_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});
 
+#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
     string_to_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, true, true, false, false);
     string_to_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, true, true, false, false);
+#endif
 
     string_to_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
     string_to_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});
