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
