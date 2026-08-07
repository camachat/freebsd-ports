diff --git src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal-ops.cpp
index c5d7619c..6d324056 100644
--- src/ggml-metal/ggml-metal-ops.cpp
+++ src/ggml-metal/ggml-metal-ops.cpp
@@ -3816,7 +3816,7 @@ int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
     }
 
     nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
-    nth = std::min(nth, args.ne00_t);
+    nth = std::min(nth, (args.ne00_t + 31)/32*32);
 
     const size_t smem = pipeline.smem;
 
diff --git src/ggml-sycl/ssm_conv.cpp src/ggml-sycl/ssm_conv.cpp
index e5522358..3eafa1a6 100644
--- src/ggml-sycl/ssm_conv.cpp
+++ src/ggml-sycl/ssm_conv.cpp
@@ -36,9 +36,13 @@ static void kernel_ssm_conv(
                     return;
                 }
 
-                const int channel = static_cast<int>(idx % d_inner);
-                const int token   = static_cast<int>((idx / d_inner) % n_t);
-                const int seq     = static_cast<int>(idx / (static_cast<size_t>(d_inner) * static_cast<size_t>(n_t)));
+                // src has the tokens of one channel contiguous, dst has the channels of one
+                // token contiguous, so either the loads or the store must be strided. Indexing
+                // token-fastest coalesces the d_conv loads, which measured faster except for
+                // short, cache-resident rows.
+                const int token   = static_cast<int>(idx % n_t);
+                const int channel = static_cast<int>((idx / n_t) % d_inner);
+                const int seq     = static_cast<int>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));
 
                 const float *s = src_data
                     + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
