diff --git src/ggml-metal/ggml-metal-tuning.cpp src/ggml-metal/ggml-metal-tuning.cpp
index 7de01fac..8cdc55a0 100644
--- src/ggml-metal/ggml-metal-tuning.cpp
+++ src/ggml-metal/ggml-metal-tuning.cpp
@@ -1525,6 +1525,178 @@ constexpr fa_vec_entry_t fa_vec_tuned_table[] = {
     { { GGML_METAL_DEVICE_M3, GGML_TYPE_F16, 576, 512, 2, 1 }, { 4, 2 } },
     { { GGML_METAL_DEVICE_M3, GGML_TYPE_F16, 576, 512, 2, 2 }, { 4, 2 } },
     { { GGML_METAL_DEVICE_M3, GGML_TYPE_F16, 576, 512, 3, 1 }, { 4, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 32, 32, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 32, 32, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 32, 32, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 32, 32, 2, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 32, 32, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, 1, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 64, 64, 3, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 96, 96, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 96, 96, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 96, 96, 2, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 128, 128, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 128, 128, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 128, 128, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 128, 128, 2, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 128, 128, 3, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, 2, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, 3, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, 3, 2 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 192, 3, 3 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 128, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 128, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 128, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 192, 128, 2, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 256, 256, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 256, 256, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 320, 256, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 320, 256, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 320, 256, 3, 4 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 512, 512, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 512, 512, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 576, 512, 2, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 576, 512, 3, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 576, 512, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_0, 576, 512, 1, 4 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 32, 32, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 32, 32, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 32, 32, 1, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 32, 32, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 32, 32, 2, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 32, 32, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 64, 64, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 64, 64, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 64, 64, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 64, 64, 1, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 64, 64, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 64, 64, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 96, 96, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 96, 96, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 96, 96, 1, 3 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 96, 96, 2, 2 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, 1, 2 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, 2, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, 3, 1 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, 3, 2 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 128, 128, 3, 3 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 192, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 192, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 192, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 192, 2, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 192, 3, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 128, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 192, 128, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 256, 256, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 256, 256, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 320, 256, 2, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 320, 256, 3, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 320, 256, 1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 320, 256, 2, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 320, 256, 3, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 320, 256, 3, 3 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 512, 512, 2, 0 }, { 4, 1 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 512, 512, 2, 4 }, { 4, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 512, 512, 3, 1 }, { 4, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 512, 512, 3, 2 }, { 4, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 576, 512, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 576, 512, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q4_1, 576, 512, 1, 4 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 32, 32, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 32, 32, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 32, 32, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 32, 32, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 64, 64, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 64, 64, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 64, 64, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 64, 64, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 64, 64, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 96, 96, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 96, 96, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 96, 96, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 96, 96, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 128, 128, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 128, 128, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 128, 128, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 128, 128, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 128, 128, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 192, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 192, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 128, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 128, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 128, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 128, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 128, 2, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 192, 128, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 256, 256, -1, 0 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 256, 256, -1, 1 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 256, 256, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 256, 256, 2, 2 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 256, 256, 3, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 320, 256, 1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 320, 256, -1, 1 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 320, 256, 2, 2 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 320, 256, 3, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 512, 512, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 512, 512, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 512, 512, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 576, 512, 2, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 576, 512, 3, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 576, 512, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 576, 512, 1, 1 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_0, 576, 512, 1, 4 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 32, 32, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 32, 32, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 32, 32, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 32, 32, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 64, 64, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 64, 64, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 64, 64, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 64, 64, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 64, 64, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 96, 96, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 96, 96, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 96, 96, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 96, 96, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 128, 128, 3, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 128, 128, -1, 1 }, { 4, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 128, 128, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 128, 128, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 128, 128, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 192, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 192, -1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, 1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, 2, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, -1, 1 }, { 4, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, 1, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, 1, 4 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, 2, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 192, 128, 3, 1 }, { 2, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 256, 256, -1, 0 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 256, 256, -1, 1 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 256, 256, 3, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 320, 256, 3, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 320, 256, -1, 1 }, { 2, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 320, 256, 1, 2 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 320, 256, 2, 2 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 320, 256, 3, 2 }, { 1, 2 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 512, 512, -1, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 512, 512, -1, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 576, 512, 2, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 576, 512, 3, 0 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 576, 512, 2, 4 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 576, 512, 3, 1 }, { 1, 4 } },
+    { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q5_1, 576, 512, 3, 3 }, { 1, 4 } },
     { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q8_0, 32, 32, -1, 1 }, { 4, 4 } },
     { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q8_0, 32, 32, 1, 1 }, { 2, 4 } },
     { { GGML_METAL_DEVICE_M3, GGML_TYPE_Q8_0, 32, 32, 2, 1 }, { 2, 4 } },
diff --git src/ggml-opencl/CMakeLists.txt src/ggml-opencl/CMakeLists.txt
index 8a1b6b96..37e565ef 100644
--- src/ggml-opencl/CMakeLists.txt
+++ src/ggml-opencl/CMakeLists.txt
@@ -222,6 +222,7 @@ set(GGML_OPENCL_KERNELS
     exp
     expm1
     abs
+    unary_ext
     softplus
     pad
     repeat
@@ -238,7 +239,7 @@ set(GGML_OPENCL_KERNELS
 )
 
 if (GGML_OPENCL_USE_ADRENO_KERNELS)
-    list(APPEND GGML_OPENCL_KERNELS gemm_xmem_f16_f32_os8)
+    list(APPEND GGML_OPENCL_KERNELS gemm_xmem_f16_f32_os8 sdpa_xmem_f32_f16_os8)
 endif ()
 
 foreach (K ${GGML_OPENCL_KERNELS})
diff --git src/ggml-opencl/ggml-opencl.cpp src/ggml-opencl/ggml-opencl.cpp
index 12465a51..ad4a995a 100644
--- src/ggml-opencl/ggml-opencl.cpp
+++ src/ggml-opencl/ggml-opencl.cpp
@@ -417,6 +417,10 @@ static void populateProfilingInfo(
 
 struct ggml_backend_opencl_context;
 
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+static void ggml_cl_adreno_xmem_attn_release_scratch(ggml_backend_opencl_context * backend_ctx);
+#endif
+
 // backend device context
 struct ggml_backend_opencl_device_context {
     cl_platform_id platform;
@@ -537,6 +541,54 @@ struct ggml_opencl_fa_kernels {
     std::set<std::pair<int, std::pair<int, int>>> variant_attempted;
 };
 
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+struct ggml_cl_adreno_xmem_attn_scratch {
+    cl_mem q_img = nullptr;
+    cl_mem k_img = nullptr;
+    cl_mem v_img = nullptr;
+    cl_mem out_img = nullptr;
+    cl_mem k_transpose_buf = nullptr;
+    cl_mem k_transpose_img1d = nullptr;
+    cl_mem k_packed_buf = nullptr;
+    cl_mem v_packed_buf = nullptr;
+    cl_mem score_buf = nullptr;
+    cl_mem prob_buf = nullptr;
+    cl_mem score_img1d = nullptr;
+    cl_mem prob_img1d = nullptr;
+    cl_mem softmax_stats_img2d = nullptr;
+    cl_mem xmem_qk = nullptr;
+    cl_mem xmem_pv = nullptr;
+
+    int n_q = 0;
+    int n_kv = 0;
+    int n_kv_padded = 0;
+    int d_head_q = 0;
+    int d_head_v = 0;
+    int q_width = 0;
+    int kv_heads_total = 0;
+};
+
+struct ggml_cl_adreno_xmem_attn_state {
+    bool compiled = false;
+    bool logged = false;
+
+    cl_kernel kernel_q_f32_to_img_scaled = nullptr;
+    cl_kernel kernel_kv_f32_to_img_gqa = nullptr;
+    cl_kernel kernel_kv_f16_to_img_gqa = nullptr;
+    cl_kernel kernel_img_to_f32 = nullptr;
+    cl_kernel kernel_k_gather = nullptr;
+    cl_kernel kernel_pack_k = nullptr;
+    cl_kernel kernel_qk_gemm = nullptr;
+    cl_kernel kernel_softmax_reduce_basic = nullptr;
+    cl_kernel kernel_softmax_apply_basic = nullptr;
+    cl_kernel kernel_mask_scores = nullptr;
+    cl_kernel kernel_pack_v = nullptr;
+    cl_kernel kernel_pv_gemm = nullptr;
+
+    ggml_cl_adreno_xmem_attn_scratch scratch;
+};
+#endif
+
 // backend context
 struct ggml_backend_opencl_context {
     int ref_count;
@@ -762,6 +814,9 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_soft_max, kernel_soft_max_4;
     cl_kernel kernel_soft_max_f16, kernel_soft_max_4_f16;
     ggml_opencl_fa_kernels fa;
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+    ggml_cl_adreno_xmem_attn_state adreno_xmem_attn;
+#endif
     cl_kernel kernel_get_rows_f32, kernel_get_rows_f16, kernel_get_rows_q4_0;
     cl_kernel kernel_set_rows_f32_i64, kernel_set_rows_f32_i32, kernel_set_rows_f16_i64, kernel_set_rows_f16_i32;
     cl_kernel kernel_set_rows_q8_0_i64, kernel_set_rows_q8_0_i32;
@@ -771,6 +826,7 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_rope_norm_f32, kernel_rope_norm_f16, kernel_rope_neox_f32, kernel_rope_neox_f16;
     cl_kernel kernel_rope_multi_f32, kernel_rope_multi_f16, kernel_rope_vision_f32, kernel_rope_vision_f16;
     cl_kernel kernel_cpy_f16_f16, kernel_cpy_f16_f32, kernel_cpy_f32_f16, kernel_cpy_f32_f32, kernel_cpy_f32_f32_pack, kernel_cpy_i32_i32;
+    cl_kernel kernel_cpy_f32_f32_flat = nullptr;
     cl_kernel kernel_mul_mat_f32_f32;
     cl_kernel kernel_mul_mat_f16_f16;
     cl_kernel kernel_mul_mat_f16_f32_1row;
@@ -877,11 +933,20 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_expm1_f16, kernel_expm1_f16_4, kernel_expm1_f16_nc;
     cl_kernel kernel_abs_f32, kernel_abs_f32_4, kernel_abs_f32_nc;
     cl_kernel kernel_abs_f16, kernel_abs_f16_4, kernel_abs_f16_nc;
+    cl_kernel kernel_sgn_f32, kernel_sgn_f32_4, kernel_sgn_f32_nc, kernel_sgn_f16, kernel_sgn_f16_4, kernel_sgn_f16_nc;
+    cl_kernel kernel_step_f32, kernel_step_f32_4, kernel_step_f32_nc, kernel_step_f16, kernel_step_f16_4, kernel_step_f16_nc;
+    cl_kernel kernel_elu_f32, kernel_elu_f32_4, kernel_elu_f32_nc, kernel_elu_f16, kernel_elu_f16_4, kernel_elu_f16_nc;
+    cl_kernel kernel_hardswish_f32, kernel_hardswish_f32_4, kernel_hardswish_f32_nc, kernel_hardswish_f16, kernel_hardswish_f16_4, kernel_hardswish_f16_nc;
+    cl_kernel kernel_hardsigmoid_f32, kernel_hardsigmoid_f32_4, kernel_hardsigmoid_f32_nc, kernel_hardsigmoid_f16, kernel_hardsigmoid_f16_4, kernel_hardsigmoid_f16_nc;
+    cl_kernel kernel_floor_f32, kernel_floor_f32_4, kernel_floor_f32_nc, kernel_floor_f16, kernel_floor_f16_4, kernel_floor_f16_nc;
+    cl_kernel kernel_ceil_f32, kernel_ceil_f32_4, kernel_ceil_f32_nc, kernel_ceil_f16, kernel_ceil_f16_4, kernel_ceil_f16_nc;
+    cl_kernel kernel_round_f32, kernel_round_f32_4, kernel_round_f32_nc, kernel_round_f16, kernel_round_f16_4, kernel_round_f16_nc;
+    cl_kernel kernel_trunc_f32, kernel_trunc_f32_4, kernel_trunc_f32_nc, kernel_trunc_f16, kernel_trunc_f16_4, kernel_trunc_f16_nc;
     cl_kernel kernel_softplus_f32, kernel_softplus_f32_4, kernel_softplus_f32_nc;
     cl_kernel kernel_softplus_f16, kernel_softplus_f16_4, kernel_softplus_f16_nc;
     cl_kernel kernel_upscale;
     cl_kernel kernel_upscale_bilinear;
-    cl_kernel kernel_concat_f32, kernel_concat_f32_pack;
+    cl_kernel kernel_concat_b1, kernel_concat_b2, kernel_concat_b4, kernel_concat_b8, kernel_concat_b4_pack;
     cl_kernel kernel_conv_2d_f16;
     cl_kernel kernel_conv_2d_f32;
     cl_kernel kernel_conv_2d_f16_f32;
@@ -1176,6 +1241,9 @@ struct ggml_backend_opencl_context {
                 if (kv.second.image) { CL_CHECK(clReleaseMemObject(kv.second.image)); }
             }
             dequant_f16_pool.clear();
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+            ggml_cl_adreno_xmem_attn_release_scratch(this);
+#endif
         }
     }
 };
@@ -1479,6 +1547,13 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         CL_CHECK((backend_ctx->kernel_cpy_f32_f16 = clCreateKernel(prog, "kernel_cpy_f32_f16", &err), err));
         CL_CHECK((backend_ctx->kernel_cpy_f32_f32 = clCreateKernel(prog, "kernel_cpy_f32_f32", &err), err));
         CL_CHECK((backend_ctx->kernel_cpy_f32_f32_pack = clCreateKernel(prog, "kernel_cpy_f32_f32_pack", &err), err));
+        {   // optional: without it ggml_cl_cpy keeps the row-mapped kernel
+            cl_int err_flat = CL_SUCCESS;
+            cl_kernel k = clCreateKernel(prog, "kernel_cpy_f32_f32_flat", &err_flat);
+            if (err_flat == CL_SUCCESS) {
+                backend_ctx->kernel_cpy_f32_f32_flat = k;
+            }
+        }
         CL_CHECK((backend_ctx->kernel_cpy_i32_i32 = clCreateKernel(prog, "kernel_cpy_i32_i32", &err), err));
         GGML_LOG_CONT(".");
     }
@@ -2343,6 +2418,49 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
     }
 #endif // GGML_OPENCL_USE_ADRENO_KERNELS
 
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+    // Adreno xmem SDPA
+    if (backend_ctx->gpu_family == GPU_FAMILY::ADRENO) {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "sdpa_xmem_f32_f16_os8.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("sdpa_xmem_f32_f16_os8.cl");
+#endif
+        cl_program program = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
+
+        auto & xmem_attn = backend_ctx->adreno_xmem_attn;
+        CL_CHECK((xmem_attn.kernel_q_f32_to_img_scaled =
+            clCreateKernel(program, "adreno_xmem_attn_q_f32_to_img_scaled", &err), err));
+        CL_CHECK((xmem_attn.kernel_kv_f32_to_img_gqa =
+            clCreateKernel(program, "adreno_xmem_attn_kv_f32_to_img_gqa", &err), err));
+        CL_CHECK((xmem_attn.kernel_kv_f16_to_img_gqa =
+            clCreateKernel(program, "adreno_xmem_attn_kv_f16_to_img_gqa", &err), err));
+        CL_CHECK((xmem_attn.kernel_img_to_f32 =
+            clCreateKernel(program, "adreno_xmem_attn_img_to_f32", &err), err));
+        CL_CHECK((xmem_attn.kernel_k_gather =
+            clCreateKernel(program, "adreno_xmem_attn_k_gather", &err), err));
+        CL_CHECK((xmem_attn.kernel_pack_k =
+            clCreateKernel(program, "adreno_xmem_attn_pack_k", &err), err));
+        CL_CHECK((xmem_attn.kernel_qk_gemm =
+            clCreateKernel(program, "adreno_xmem_attn_qk_gemm", &err), err));
+        CL_CHECK((xmem_attn.kernel_softmax_reduce_basic =
+            clCreateKernel(program, "adreno_xmem_attn_softmax_reduce_basic", &err), err));
+        CL_CHECK((xmem_attn.kernel_softmax_apply_basic =
+            clCreateKernel(program, "adreno_xmem_attn_softmax_apply_basic", &err), err));
+        CL_CHECK((xmem_attn.kernel_mask_scores =
+            clCreateKernel(program, "adreno_xmem_attn_mask_scores", &err), err));
+        CL_CHECK((xmem_attn.kernel_pack_v =
+            clCreateKernel(program, "adreno_xmem_attn_pack_v", &err), err));
+        CL_CHECK((xmem_attn.kernel_pv_gemm =
+            clCreateKernel(program, "adreno_xmem_attn_pv_gemm", &err), err));
+        CL_CHECK(clReleaseProgram(program));
+        xmem_attn.compiled = true;
+        GGML_LOG_CONT(".");
+    }
+#endif // GGML_OPENCL_USE_ADRENO_KERNELS
+
     // mul_mm_f32_f32_l4_lm
     {
 #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -3082,6 +3200,38 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         GGML_LOG_CONT(".");
     }
 
+    // unary_ext (sgn, step, elu, hardswish, hardsigmoid, floor, ceil, round, trunc)
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "unary_ext.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("unary_ext.cl");
+#endif
+        cl_program prog =
+            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
+#define CL_UNARY_EXT_K(op) \
+        CL_CHECK((backend_ctx->kernel_##op##_f32    = clCreateKernel(prog, "kernel_" #op "_f32",    &err), err)); \
+        CL_CHECK((backend_ctx->kernel_##op##_f32_4  = clCreateKernel(prog, "kernel_" #op "_f32_4",  &err), err)); \
+        CL_CHECK((backend_ctx->kernel_##op##_f32_nc = clCreateKernel(prog, "kernel_" #op "_f32_nc", &err), err)); \
+        CL_CHECK((backend_ctx->kernel_##op##_f16    = clCreateKernel(prog, "kernel_" #op "_f16",    &err), err)); \
+        CL_CHECK((backend_ctx->kernel_##op##_f16_4  = clCreateKernel(prog, "kernel_" #op "_f16_4",  &err), err)); \
+        CL_CHECK((backend_ctx->kernel_##op##_f16_nc = clCreateKernel(prog, "kernel_" #op "_f16_nc", &err), err));
+        CL_UNARY_EXT_K(sgn)
+        CL_UNARY_EXT_K(step)
+        CL_UNARY_EXT_K(elu)
+        CL_UNARY_EXT_K(hardswish)
+        CL_UNARY_EXT_K(hardsigmoid)
+        CL_UNARY_EXT_K(floor)
+        CL_UNARY_EXT_K(ceil)
+        CL_UNARY_EXT_K(round)
+        CL_UNARY_EXT_K(trunc)
+#undef CL_UNARY_EXT_K
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
     // softplus
     {
 #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -3146,8 +3296,11 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
 #endif
         cl_program prog =
             build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
-        CL_CHECK((backend_ctx->kernel_concat_f32 = clCreateKernel(prog, "kernel_concat_f32", &err), err));
-        CL_CHECK((backend_ctx->kernel_concat_f32_pack = clCreateKernel(prog, "kernel_concat_f32_pack", &err), err));
+        CL_CHECK((backend_ctx->kernel_concat_b1 = clCreateKernel(prog, "kernel_concat_b1", &err), err));
+        CL_CHECK((backend_ctx->kernel_concat_b2 = clCreateKernel(prog, "kernel_concat_b2", &err), err));
+        CL_CHECK((backend_ctx->kernel_concat_b4 = clCreateKernel(prog, "kernel_concat_b4", &err), err));
+        CL_CHECK((backend_ctx->kernel_concat_b8 = clCreateKernel(prog, "kernel_concat_b8", &err), err));
+        CL_CHECK((backend_ctx->kernel_concat_b4_pack = clCreateKernel(prog, "kernel_concat_b4_pack", &err), err));
         CL_CHECK(clReleaseProgram(prog));
         GGML_LOG_CONT(".");
     }
@@ -8351,6 +8504,15 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
                 case GGML_UNARY_OP_EXPM1:
                     return op->src[0]->type == GGML_TYPE_F32;
                 case GGML_UNARY_OP_ABS:
+                case GGML_UNARY_OP_SGN:
+                case GGML_UNARY_OP_STEP:
+                case GGML_UNARY_OP_ELU:
+                case GGML_UNARY_OP_HARDSWISH:
+                case GGML_UNARY_OP_HARDSIGMOID:
+                case GGML_UNARY_OP_FLOOR:
+                case GGML_UNARY_OP_CEIL:
+                case GGML_UNARY_OP_ROUND:
+                case GGML_UNARY_OP_TRUNC:
                     return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
                 case GGML_UNARY_OP_SOFTPLUS:
                     return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
@@ -8430,7 +8592,13 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
                 return S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128;
             }
         case GGML_OP_CONCAT:
-            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
+            {
+                const ggml_type t = op->src[0]->type;
+                return op->src[1]->type == t && op->type == t &&
+                       !ggml_is_quantized(t) && ggml_blck_size(t) == 1 &&
+                       (ggml_type_size(t) == 1 || ggml_type_size(t) == 2 ||
+                        ggml_type_size(t) == 4 || ggml_type_size(t) == 8);
+            }
         case GGML_OP_TIMESTEP_EMBEDDING:
             return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
         case GGML_OP_GROUP_NORM:
@@ -15039,6 +15207,97 @@ static void ggml_cl_abs(ggml_backend_t backend, const ggml_tensor * src0, const
     }
 }
 
+// Shared driver for the extended unary ops (unary_ext.cl), same selection as
+// ggml_cl_abs: contiguous picks the vec4 kernel when the element count is a
+// multiple of 4 (else scalar); non-contiguous uses the stride-addressed kernel.
+static void ggml_cl_unary_ext(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst,
+                              cl_kernel k_f32, cl_kernel k_f32_4, cl_kernel k_f32_nc,
+                              cl_kernel k_f16, cl_kernel k_f16_4, cl_kernel k_f16_nc) {
+    GGML_ASSERT(src0);
+    GGML_ASSERT(src0->extra);
+    GGML_ASSERT(dst);
+    GGML_ASSERT(dst->extra);
+
+    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
+
+    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
+    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
+
+    cl_ulong offset0 = extra0->offset + src0->view_offs;
+    cl_ulong offsetd = extrad->offset + dst->view_offs;
+
+    const int     ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
+    const cl_ulong nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
+    const cl_ulong nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
+
+    const bool is_f16 = (src0->type == GGML_TYPE_F16);
+    cl_kernel kernel;
+
+    if (ggml_is_contiguous(src0)) {
+        int n = ggml_nelements(dst);
+        if (n % 4 == 0) {
+            kernel = is_f16 ? k_f16_4 : k_f32_4;
+            n /= 4;
+        } else {
+            kernel = is_f16 ? k_f16 : k_f32;
+        }
+
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
+
+        size_t global_work_size[] = {(size_t)n, 1, 1};
+        size_t local_work_size[]  = {64, 1, 1};
+        size_t * local_work_size_ptr = local_work_size;
+        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
+            local_work_size_ptr = nullptr;
+        }
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
+    } else {
+        kernel = is_f16 ? k_f16_nc : k_f32_nc;
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
+        int nth = 64;
+        size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
+        size_t local_work_size[]  = {(size_t)nth, 1, 1};
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+    }
+}
+
+#define GGML_CL_UNARY_EXT_WRAP(FN, OP)                                                                 \
+static void FN(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) { \
+    UNUSED(src1);                                                                                      \
+    ggml_backend_opencl_context *c = (ggml_backend_opencl_context *)backend->context;                  \
+    ggml_cl_unary_ext(backend, src0, dst, c->kernel_##OP##_f32, c->kernel_##OP##_f32_4, c->kernel_##OP##_f32_nc, \
+                      c->kernel_##OP##_f16, c->kernel_##OP##_f16_4, c->kernel_##OP##_f16_nc);           \
+}
+
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_sgn,         sgn)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_step,        step)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_elu,         elu)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_hardswish,   hardswish)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_hardsigmoid, hardsigmoid)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_floor,       floor)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_ceil,        ceil)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_round,       round)
+GGML_CL_UNARY_EXT_WRAP(ggml_cl_trunc,       trunc)
+
+#undef GGML_CL_UNARY_EXT_WRAP
+
 static void ggml_cl_softplus(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
     GGML_ASSERT(src0);
     GGML_ASSERT(src0->extra);
@@ -15416,9 +15675,8 @@ static void ggml_cl_concat(ggml_backend_t backend, const ggml_tensor * src0, con
     GGML_ASSERT(src1->extra);
     GGML_ASSERT(dst);
     GGML_ASSERT(dst->extra);
-    GGML_ASSERT(src0->type == GGML_TYPE_F32);
-    GGML_ASSERT(src1->type == GGML_TYPE_F32);
-    GGML_ASSERT(dst->type == GGML_TYPE_F32);
+    GGML_ASSERT(src0->type == src1->type);
+    GGML_ASSERT(src0->type == dst->type);
 
     ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
 
@@ -15460,9 +15718,21 @@ static void ggml_cl_concat(ggml_backend_t backend, const ggml_tensor * src0, con
 
     int nth = MIN(64, ne0);
 
-    const bool concat_pack = (dim == 0 && ne0 < 32);
-    cl_kernel kernel = concat_pack ? backend_ctx->kernel_concat_f32_pack
-                                   : backend_ctx->kernel_concat_f32;
+    const size_t ts = ggml_type_size(dst->type);
+    // the pack kernel copies 4-byte elements, so it is only valid for those.
+    const bool concat_pack = (dim == 0 && ne0 < 32 && ts == 4);
+    cl_kernel kernel;
+    if (concat_pack) {
+        kernel = backend_ctx->kernel_concat_b4_pack;
+    } else {
+        switch (ts) {
+            case 1:  kernel = backend_ctx->kernel_concat_b1; break;
+            case 2:  kernel = backend_ctx->kernel_concat_b2; break;
+            case 4:  kernel = backend_ctx->kernel_concat_b4; break;
+            case 8:  kernel = backend_ctx->kernel_concat_b8; break;
+            default: GGML_ABORT("unsupported concat element size: %zu", ts);
+        }
+    }
 
     CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
     CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
@@ -15927,6 +16197,581 @@ static constexpr int FD_MAX_N_Q_MULTI = 8;
 static constexpr int FD_MQ_KV_PER_SPLIT = 256;
 static constexpr int FD_MQ_MAX_SPLITS   = 128;
 
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+struct ggml_cl_adreno_xmem_attn_schedule {
+    int qk_lws0 = 256;
+    int qk_lws2 = 1;
+    int softmax_reduce_lws0 = 256;
+    int softmax_apply_lws0 = 64;
+    int softmax_apply_lws2 = 4;
+    int pv_lws0 = 64;
+    int pv_lws2 = 4;
+};
+
+static inline size_t ggml_cl_round_up(size_t x, size_t a) {
+    return ((x + a - 1) / a) * a;
+}
+
+static inline int ggml_cl_round_up_div(int x, int y) {
+    return (x + y - 1) / y;
+}
+
+static inline void ggml_cl_set_arg_int4(cl_kernel kernel, cl_uint index, int x, int y, int z, int w) {
+    struct { int x, y, z, w; } value { x, y, z, w };
+    CL_CHECK(clSetKernelArg(kernel, index, sizeof(value), &value));
+}
+
+static cl_mem ggml_cl_make_image2d_half4(cl_context context, cl_mem_flags flags, size_t width, size_t height) {
+    cl_int err = CL_SUCCESS;
+    cl_image_format format = { CL_RGBA, CL_HALF_FLOAT };
+    cl_image_desc desc = {};
+    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
+    desc.image_width = width;
+    desc.image_height = height;
+    cl_mem image = clCreateImage(context, flags, &format, &desc, nullptr, &err);
+    CL_CHECK(err);
+    return image;
+}
+
+static cl_mem ggml_cl_make_image1d_buffer_half4(cl_context context, cl_mem_flags flags, size_t width, cl_mem backing_buffer) {
+    cl_int err = CL_SUCCESS;
+    cl_image_format format = { CL_RGBA, CL_HALF_FLOAT };
+    cl_image_desc desc = {};
+    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
+    desc.image_width = width;
+    desc.buffer = backing_buffer;
+    cl_mem image = clCreateImage(context, flags, &format, &desc, nullptr, &err);
+    CL_CHECK(err);
+    return image;
+}
+
+static void ggml_cl_release_mem(cl_mem & mem) {
+    if (mem != nullptr) {
+        CL_CHECK(clReleaseMemObject(mem));
+        mem = nullptr;
+    }
+}
+
+static void ggml_cl_adreno_xmem_attn_release_scratch(ggml_backend_opencl_context * backend_ctx) {
+    auto & s = backend_ctx->adreno_xmem_attn.scratch;
+    ggml_cl_release_mem(s.q_img);
+    ggml_cl_release_mem(s.k_img);
+    ggml_cl_release_mem(s.v_img);
+    ggml_cl_release_mem(s.out_img);
+    ggml_cl_release_mem(s.k_transpose_img1d);
+    ggml_cl_release_mem(s.k_transpose_buf);
+    ggml_cl_release_mem(s.k_packed_buf);
+    ggml_cl_release_mem(s.v_packed_buf);
+    ggml_cl_release_mem(s.score_img1d);
+    ggml_cl_release_mem(s.prob_img1d);
+    ggml_cl_release_mem(s.score_buf);
+    ggml_cl_release_mem(s.prob_buf);
+    ggml_cl_release_mem(s.softmax_stats_img2d);
+    ggml_cl_release_mem(s.xmem_qk);
+    ggml_cl_release_mem(s.xmem_pv);
+    s = {};
+}
+
+static ggml_cl_adreno_xmem_attn_schedule ggml_cl_adreno_xmem_attn_select_schedule(
+        const ggml_backend_opencl_context * backend_ctx,
+        int n_q,
+        int n_kv,
+        int heads_total,
+        int q_width,
+        int gqa_ratio) {
+    const bool big_h = heads_total >= 8;
+    ggml_cl_adreno_xmem_attn_schedule sched;
+
+    if (gqa_ratio == 1) {
+        if (n_q >= 512) { sched.qk_lws0 = 512; }
+        else if (n_q >= 256) { sched.qk_lws0 = 128; }
+        else { sched.qk_lws0 = 64; }
+        sched.qk_lws2 = (big_h && n_q >= 512) ? 2 : 1;
+    } else {
+        if (q_width >= 2048) { sched.qk_lws0 = 512; }
+        else if (q_width >= 256) { sched.qk_lws0 = 128; }
+        else { sched.qk_lws0 = 64; }
+        sched.qk_lws2 = MIN(8, (int) backend_ctx->max_workgroup_size / sched.qk_lws0);
+    }
+
+    if (n_kv >= 2048) { sched.softmax_reduce_lws0 = 1024; }
+    else if (n_kv >= 512) { sched.softmax_reduce_lws0 = big_h ? 256 : 512; }
+    else { sched.softmax_reduce_lws0 = 256; }
+
+    if (n_kv < 256) { sched.softmax_apply_lws0 = 64; }
+    else { sched.softmax_apply_lws0 = big_h ? 128 : 64; }
+    sched.softmax_apply_lws2 = n_kv >= 512 ? 8 : 4;
+
+    if (n_q < 256) { sched.pv_lws0 = 64; }
+    else { sched.pv_lws0 = big_h ? 128 : 64; }
+    sched.pv_lws2 = big_h ? 8 : (n_q <= 256 ? 8 : 4);
+
+    const int max_wg = (int) backend_ctx->max_workgroup_size;
+    auto fix = [&](int & l0, int & l2) {
+        while (l0 * l2 > max_wg) {
+            if (l2 > 1) { l2 /= 2; }
+            else if (l0 > 32) { l0 /= 2; }
+            else { break; }
+        }
+    };
+    fix(sched.qk_lws0, sched.qk_lws2);
+    fix(sched.softmax_apply_lws0, sched.softmax_apply_lws2);
+    fix(sched.pv_lws0, sched.pv_lws2);
+    while (sched.softmax_reduce_lws0 > max_wg) {
+        sched.softmax_reduce_lws0 /= 2;
+    }
+
+    return sched;
+}
+
+static bool ggml_cl_adreno_xmem_attn_prepare(
+        ggml_backend_opencl_context * backend_ctx,
+        int n_q,
+        int n_kv,
+        int d_head_q,
+        int d_head_v,
+        int n_head,
+        int n_head_kv,
+        int n_batch) {
+    auto & s = backend_ctx->adreno_xmem_attn.scratch;
+    const int gqa_ratio = n_head / n_head_kv;
+    const int q_width = n_q * gqa_ratio;
+    const int kv_heads_total = n_head_kv * n_batch;
+    const int n_kv_padded = (int) ggml_cl_round_up((size_t) n_kv, 32);
+    if (s.q_img != nullptr &&
+            s.n_q == n_q &&
+            s.n_kv == n_kv &&
+            s.n_kv_padded == n_kv_padded &&
+            s.d_head_q == d_head_q &&
+            s.d_head_v == d_head_v &&
+            s.q_width == q_width &&
+            s.kv_heads_total == kv_heads_total) {
+        return true;
+    }
+
+    ggml_cl_adreno_xmem_attn_release_scratch(backend_ctx);
+
+    const int qpack = d_head_q / 4;
+    const int vpack = d_head_v / 4;
+    const int npack = n_kv_padded / 4;
+    const size_t q_img_h = (size_t) kv_heads_total * qpack;
+    const size_t v_img_h = (size_t) kv_heads_total * vpack;
+
+    s.q_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) q_width, q_img_h);
+    s.k_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) n_kv_padded, q_img_h);
+    s.v_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) n_kv_padded, v_img_h);
+    s.out_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) q_width, v_img_h);
+
+    const size_t k_transpose_half4_elems = (size_t) npack * kv_heads_total * d_head_q;
+    s.k_transpose_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, k_transpose_half4_elems * sizeof(uint16_t) * 4, nullptr, nullptr);
+    GGML_ASSERT(s.k_transpose_buf != nullptr);
+    s.k_transpose_img1d = ggml_cl_make_image1d_buffer_half4(backend_ctx->context, CL_MEM_READ_ONLY, k_transpose_half4_elems, s.k_transpose_buf);
+
+    const size_t k_groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_q, 16);
+    const size_t v_groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_v, 16);
+    const size_t k_packed_half4_elems = (size_t) n_kv_padded * k_groups16 * 4;
+    const size_t v_packed_half4_elems = (size_t) n_kv_padded * v_groups16 * 4;
+    s.k_packed_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, k_packed_half4_elems * sizeof(uint16_t) * 4, nullptr, nullptr);
+    s.v_packed_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, v_packed_half4_elems * sizeof(uint16_t) * 4, nullptr, nullptr);
+    GGML_ASSERT(s.k_packed_buf != nullptr && s.v_packed_buf != nullptr);
+
+    const size_t score_half4_elems = (size_t) npack * kv_heads_total * q_width;
+    const size_t score_bytes = score_half4_elems * sizeof(uint16_t) * 4;
+    s.score_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, score_bytes, nullptr, nullptr);
+    s.prob_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, score_bytes, nullptr, nullptr);
+    GGML_ASSERT(s.score_buf != nullptr && s.prob_buf != nullptr);
+    s.score_img1d = ggml_cl_make_image1d_buffer_half4(backend_ctx->context, CL_MEM_READ_ONLY, score_half4_elems, s.score_buf);
+    s.prob_img1d = ggml_cl_make_image1d_buffer_half4(backend_ctx->context, CL_MEM_READ_ONLY, score_half4_elems, s.prob_buf);
+    s.softmax_stats_img2d = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE,
+                                                       (size_t) q_width, (size_t) kv_heads_total);
+    s.xmem_qk = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY, 6144, nullptr, nullptr);
+    s.xmem_pv = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY, 6144, nullptr, nullptr);
+    GGML_ASSERT(s.softmax_stats_img2d != nullptr && s.xmem_qk != nullptr && s.xmem_pv != nullptr);
+
+    s.n_q = n_q;
+    s.n_kv = n_kv;
+    s.n_kv_padded = n_kv_padded;
+    s.d_head_q = d_head_q;
+    s.d_head_v = d_head_v;
+    s.q_width = q_width;
+    s.kv_heads_total = kv_heads_total;
+    return true;
+}
+
+static bool ggml_cl_adreno_xmem_attn_can_use(
+        const ggml_backend_opencl_context * backend_ctx,
+        const ggml_tensor * q,
+        const ggml_tensor * k,
+        const ggml_tensor * dst) {
+    static const char * xmem_sdpa_env = getenv("GGML_OPENCL_XMEM_SDPA");
+    if (xmem_sdpa_env == nullptr || xmem_sdpa_env[0] == '0') {
+        return false;
+    }
+
+    const ggml_tensor * v = dst->src[2];
+    const ggml_tensor * mask = dst->src[3];
+    const ggml_tensor * sinks = dst->src[4];
+
+    if (!backend_ctx->adreno_xmem_attn.compiled || backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
+        return false;
+    }
+    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
+        (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) ||
+        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32)) {
+        return false;
+    }
+    if (sinks != nullptr) {
+        return false;
+    }
+    if (q->nb[0] != ggml_type_size(q->type) || k->nb[0] != ggml_type_size(k->type) ||
+        v->nb[0] != ggml_type_size(v->type) || dst->nb[0] != ggml_type_size(dst->type)) {
+        return false;
+    }
+    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(ggml_fp16_t))) {
+        return false;
+    }
+
+    const int n_q = q->ne[1];
+    const int n_kv = k->ne[1];
+    const int d_head_q = q->ne[0];
+    const int d_head_v = v->ne[0];
+    const int n_head = q->ne[2];
+    const int n_head_kv = k->ne[2];
+    const int n_batch = q->ne[3];
+
+    if (n_q <= 1 || n_kv <= 0 || n_kv > 8192) {
+        return false;
+    }
+    if (d_head_q != k->ne[0] || d_head_v != v->ne[0] || k->ne[1] != v->ne[1] || k->ne[3] != v->ne[3]) {
+        return false;
+    }
+    if (q->ne[3] != k->ne[3]) {
+        return false;
+    }
+    if (n_head_kv <= 0 || n_head % n_head_kv != 0 || k->ne[2] != v->ne[2]) {
+        return false;
+    }
+    if (dst->ne[0] != d_head_v || dst->ne[1] != n_head || dst->ne[2] != n_q || dst->ne[3] != n_batch) {
+        return false;
+    }
+    if ((d_head_q % 8) != 0 || (d_head_v % 32) != 0) {
+        return false;
+    }
+    if (mask != nullptr &&
+        (mask->ne[0] < n_kv || mask->ne[1] < n_q || mask->ne[2] <= 0 || mask->ne[3] <= 0)) {
+        return false;
+    }
+
+    float params[3];
+    memcpy(params, dst->op_params, sizeof(params));
+    if (params[1] != 0.0f || params[2] != 0.0f) {
+        return false;
+    }
+
+    const int gqa_ratio = n_head / n_head_kv;
+    const int q_width = n_q * gqa_ratio;
+    const int kv_heads_total = n_head_kv * n_batch;
+    const int n_kv_padded = (int) ggml_cl_round_up((size_t) n_kv, 32);
+    const int qpack = d_head_q / 4;
+    const int vpack = d_head_v / 4;
+    const int npack = n_kv_padded / 4;
+
+    if ((size_t) q_width > backend_ctx->image2d_max_width ||
+        (size_t) n_kv_padded > backend_ctx->image2d_max_width) {
+        return false;
+    }
+    if ((size_t) kv_heads_total * (size_t) qpack > backend_ctx->image2d_max_height ||
+        (size_t) kv_heads_total * (size_t) vpack > backend_ctx->image2d_max_height) {
+        return false;
+    }
+    if ((size_t) npack * (size_t) kv_heads_total * (size_t) d_head_q > backend_ctx->image_max_buffer_size ||
+        (size_t) npack * (size_t) kv_heads_total * (size_t) q_width > backend_ctx->image_max_buffer_size) {
+        return false;
+    }
+
+    return true;
+}
+
+static void ggml_cl_adreno_xmem_attn_run(
+        ggml_backend_t backend,
+        const ggml_tensor * q,
+        const ggml_tensor * k,
+        ggml_tensor * dst) {
+    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
+    auto & xstate = backend_ctx->adreno_xmem_attn;
+    auto & s = xstate.scratch;
+    if (!xstate.logged) {
+        GGML_LOG_INFO("ggml_opencl: using Adreno xmem attention path\n");
+        xstate.logged = true;
+    }
+
+    const ggml_tensor * v = dst->src[2];
+    const ggml_tensor * mask = dst->src[3];
+
+    ggml_tensor_extra_cl * extra_q = (ggml_tensor_extra_cl *) q->extra;
+    ggml_tensor_extra_cl * extra_k = (ggml_tensor_extra_cl *) k->extra;
+    ggml_tensor_extra_cl * extra_v = (ggml_tensor_extra_cl *) v->extra;
+    ggml_tensor_extra_cl * extra_o = (ggml_tensor_extra_cl *) dst->extra;
+    ggml_tensor_extra_cl * extra_mask = mask ? (ggml_tensor_extra_cl *) mask->extra : nullptr;
+
+    const cl_ulong offset_q = extra_q->offset + q->view_offs;
+    const cl_ulong offset_k = extra_k->offset + k->view_offs;
+    const cl_ulong offset_v = extra_v->offset + v->view_offs;
+    const cl_ulong offset_o = extra_o->offset + dst->view_offs;
+    const cl_ulong offset_mask = extra_mask ? extra_mask->offset + mask->view_offs : 0;
+
+    const int n_q = q->ne[1];
+    const int n_kv = k->ne[1];
+    const int d_head_q = q->ne[0];
+    const int d_head_v = v->ne[0];
+    const int n_head = q->ne[2];
+    const int n_head_kv = k->ne[2];
+    const int n_batch = q->ne[3];
+    const int heads_total = n_head * n_batch;
+    const int gqa_ratio = n_head / n_head_kv;
+    const int q_width = n_q * gqa_ratio;
+    const int kv_heads_total = n_head_kv * n_batch;
+    const int n_kv_padded = (int) ggml_cl_round_up((size_t) n_kv, 32);
+    const int qpack = d_head_q / 4;
+    const int opack = d_head_v / 4;
+    const int npack = n_kv_padded / 4;
+    const float scale = ((const float *) dst->op_params)[0];
+
+    GGML_ASSERT(ggml_cl_adreno_xmem_attn_prepare(
+        backend_ctx, n_q, n_kv, d_head_q, d_head_v, n_head, n_head_kv, n_batch));
+    const ggml_cl_adreno_xmem_attn_schedule sched =
+        ggml_cl_adreno_xmem_attn_select_schedule(
+            backend_ctx, n_q, n_kv_padded, heads_total, q_width, gqa_ratio);
+
+    {
+        size_t gws[3] = {ggml_cl_round_up((size_t) n_q, 8), (size_t) heads_total, (size_t) qpack};
+        size_t lws[3] = {8, 1, (size_t) ((qpack <= 32) ? qpack : 1)};
+        cl_kernel kernel = xstate.kernel_q_f32_to_img_scaled;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_q->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_q));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.q_img));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float),    &scale));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &d_head_q));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_q));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_head_kv));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),      &n_batch));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &q->nb[1]));
+        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &q->nb[2]));
+        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &q->nb[3]));
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+
+    {
+        size_t gws[3] = {(size_t) n_kv_padded, (size_t) kv_heads_total, (size_t) qpack};
+        size_t lws[3] = {8, 1, (size_t) ((qpack <= 32) ? qpack : 1)};
+        cl_kernel kernel = k->type == GGML_TYPE_F16 ?
+            xstate.kernel_kv_f16_to_img_gqa : xstate.kernel_kv_f32_to_img_gqa;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_k->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_k));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.k_img));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),      &d_head_q));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n_kv));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_kv_padded));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head_kv));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_batch));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &k->nb[1]));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &k->nb[2]));
+        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &k->nb[3]));
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+
+    {
+        size_t gws[3] = {(size_t) n_kv_padded, (size_t) kv_heads_total, (size_t) opack};
+        size_t lws[3] = {8, 1, (size_t) ((opack <= 32) ? opack : 1)};
+        cl_kernel kernel = v->type == GGML_TYPE_F16 ?
+            xstate.kernel_kv_f16_to_img_gqa : xstate.kernel_kv_f32_to_img_gqa;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_v->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_v));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.v_img));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),      &d_head_v));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n_kv));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_kv_padded));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head_kv));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_batch));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &v->nb[1]));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &v->nb[2]));
+        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &v->nb[3]));
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+
+    {
+        size_t gws[3] = {(size_t) d_head_q, (size_t) kv_heads_total, (size_t) npack};
+        size_t lws[3] = {(size_t) MIN(64, d_head_q), (size_t) (kv_heads_total >= 2 ? 2 : 1), (size_t) MIN(8, npack)};
+        if (lws[0] * lws[1] * lws[2] > backend_ctx->max_workgroup_size) {
+            lws[1] = 1;
+        }
+        cl_kernel kernel = xstate.kernel_k_gather;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.k_transpose_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.k_img));
+        ggml_cl_set_arg_int4(kernel, 2, n_kv_padded, kv_heads_total, npack, d_head_q);
+        ggml_cl_set_arg_int4(kernel, 3, qpack, 0, 0, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+    {
+        const size_t groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_q, 16);
+        const size_t packed_linear = (size_t) n_kv_padded * groups16;
+        const size_t lws0 = MIN((size_t) 1024, backend_ctx->max_workgroup_size);
+        size_t gws[3] = {ggml_cl_round_up(packed_linear, lws0), 1, 1};
+        size_t lws[3] = {lws0, 1, 1};
+        cl_kernel kernel = xstate.kernel_pack_k;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.k_packed_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.k_transpose_img1d));
+        ggml_cl_set_arg_int4(kernel, 2, 8, (int) packed_linear, qpack, d_head_q);
+        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, kv_heads_total, kv_heads_total, npack);
+        ggml_cl_set_arg_int4(kernel, 4, d_head_q, 0, 0, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+
+    {
+        size_t lws[3] = {(size_t) sched.qk_lws0, 1, (size_t) sched.qk_lws2};
+        const int slices_per_group = sched.qk_lws2 * 8;
+        const size_t groups_z = (size_t) ggml_cl_round_up_div(npack, slices_per_group);
+        const size_t groups_x = (size_t) ggml_cl_round_up_div(q_width, sched.qk_lws0);
+        size_t gws[3] = {
+            lws[0] * groups_z,
+            groups_x,
+            (size_t) kv_heads_total * lws[2],
+        };
+
+        cl_kernel kernel = xstate.kernel_qk_gemm;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.score_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.k_packed_buf));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &s.xmem_qk));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &s.q_img));
+        ggml_cl_set_arg_int4(kernel, 4, kv_heads_total, npack, q_width, 32);
+        ggml_cl_set_arg_int4(kernel, 5, qpack, 0, 0, kv_heads_total);
+        ggml_cl_set_arg_int4(kernel, 6, qpack, 1, 1, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+    cl_mem softmax_input_img = s.score_img1d;
+    cl_mem softmax_output_buf = s.prob_buf;
+    cl_mem pv_prob_img = s.prob_img1d;
+
+    if (mask != nullptr) {
+        const cl_ulong mask_nb1 = mask->nb[1];
+        const cl_ulong mask_nb2 = mask->nb[2];
+        const cl_ulong mask_nb3 = mask->nb[3];
+        const int mask_ne2 = mask->ne[2];
+        const int mask_ne3 = mask->ne[3];
+        size_t lws[3] = {(size_t) sched.softmax_apply_lws0, 1, (size_t) sched.softmax_apply_lws2};
+        size_t gws[3] = {
+            ggml_cl_round_up((size_t) q_width, lws[0]),
+            (size_t) kv_heads_total,
+            ggml_cl_round_up((size_t) npack, lws[2]),
+        };
+        cl_kernel kernel = xstate.kernel_mask_scores;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.prob_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.score_img1d));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra_mask->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset_mask));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &q_width));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &n_q));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &n_kv));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &n_kv_padded));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &kv_heads_total));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &n_head));
+        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), &n_head_kv));
+        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &mask_nb1));
+        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &mask_nb2));
+        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &mask_nb3));
+        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int), &mask_ne2));
+        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int), &mask_ne3));
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+
+        softmax_input_img = s.prob_img1d;
+        softmax_output_buf = s.score_buf;
+        pv_prob_img = s.score_img1d;
+    }
+
+    {
+        size_t lws[3] = {(size_t) sched.softmax_reduce_lws0, 1, 1};
+        size_t gws[3] = {ggml_cl_round_up((size_t) q_width, lws[0]), (size_t) kv_heads_total, 1};
+        cl_kernel kernel = xstate.kernel_softmax_reduce_basic;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &softmax_input_img));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.softmax_stats_img2d));
+        ggml_cl_set_arg_int4(kernel, 2, kv_heads_total, 1, q_width, n_kv);
+        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, q_width, 0, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+    {
+        size_t lws[3] = {(size_t) sched.softmax_apply_lws0, 1, (size_t) sched.softmax_apply_lws2};
+        size_t gws[3] = {
+            ggml_cl_round_up((size_t) q_width, lws[0]),
+            (size_t) kv_heads_total,
+            ggml_cl_round_up((size_t) npack, lws[2]),
+        };
+        cl_kernel kernel = xstate.kernel_softmax_apply_basic;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &softmax_output_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &softmax_input_img));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &s.softmax_stats_img2d));
+        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, npack, q_width, 1);
+        ggml_cl_set_arg_int4(kernel, 4, kv_heads_total, q_width, n_kv, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+    {
+        const size_t groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_v, 16);
+        const size_t packed_linear = (size_t) n_kv_padded * groups16;
+        const size_t lws0 = MIN((size_t) 1024, backend_ctx->max_workgroup_size);
+        size_t gws[3] = {ggml_cl_round_up(packed_linear, lws0), 1, 1};
+        size_t lws[3] = {lws0, 1, 1};
+        cl_kernel kernel = xstate.kernel_pack_v;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.v_packed_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.v_img));
+        ggml_cl_set_arg_int4(kernel, 2, 8, (int) packed_linear, npack, n_kv_padded);
+        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, kv_heads_total, opack, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+
+    {
+        size_t lws[3] = {(size_t) sched.pv_lws0, 1, (size_t) sched.pv_lws2};
+        const int blocks = ggml_cl_round_up_div(opack, 8);
+        const size_t groups_z = (size_t) ggml_cl_round_up_div(blocks, sched.pv_lws2);
+        const size_t groups_x = (size_t) ggml_cl_round_up_div(q_width, sched.pv_lws0);
+        size_t gws[3] = {
+            lws[0] * groups_z,
+            groups_x,
+            (size_t) kv_heads_total * lws[2],
+        };
+
+        cl_kernel kernel = xstate.kernel_pv_gemm;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.v_packed_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.xmem_pv));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &pv_prob_img));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &s.out_img));
+        ggml_cl_set_arg_int4(kernel, 4, kv_heads_total, opack, q_width, 32);
+        ggml_cl_set_arg_int4(kernel, 5, npack, 0, 0, kv_heads_total);
+        ggml_cl_set_arg_int4(kernel, 6, kv_heads_total * q_width, npack, q_width, 1);
+        ggml_cl_set_arg_int4(kernel, 7, 1, 0, 0, 0);
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+
+    {
+        size_t gws[3] = {ggml_cl_round_up((size_t) n_q, 8), (size_t) heads_total, (size_t) opack};
+        size_t lws[3] = {8, 1, (size_t) ((opack <= 32) ? opack : 1)};
+        cl_kernel kernel = xstate.kernel_img_to_f32;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_o->data_device));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_o));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.out_img));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),      &d_head_v));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n_q));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_head));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head_kv));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_batch));
+        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &dst->nb[1]));
+        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &dst->nb[2]));
+        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &dst->nb[3]));
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
+    }
+}
+
+#endif // GGML_OPENCL_USE_ADRENO_KERNELS
+
 static void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, const ggml_tensor * k, ggml_tensor * dst) {
     const ggml_tensor * v = dst->src[2];
     const ggml_tensor * mask = dst->src[3];
@@ -15954,6 +16799,13 @@ static void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, co
     const int n_head_kv = k->ne[2];
     const int n_batch = q->ne[3];
 
+#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
+    if (ggml_cl_adreno_xmem_attn_can_use(backend_ctx, q, k, dst)) {
+        ggml_cl_adreno_xmem_attn_run(backend, q, k, dst);
+        return;
+    }
+#endif
+
     // DK=512 (Gemma-4 global layers) runs decode-only (q1 / q1_split) on
     // Adreno - it never uses the BM-tile path, and the prepass + split-tile
     // programs OOM the compiler at DK=512; supports_op only admits
@@ -25338,6 +26190,38 @@ static void ggml_cl_cpy(ggml_backend_t backend, const ggml_tensor * src0, const
     cl_ulong offset0 = extra0->offset + src0->view_offs;
     cl_ulong offset1 = extra1->offset + src1->view_offs;
 
+    // A contiguous f32 -> f32 copy is a linear move. The kernel below maps one workgroup to
+    // each row, so a tensor with few long rows runs on a single compute unit; dispatch those
+    // over the whole device instead. GGML_OPENCL_CPY_FLAT=0 restores the row-mapped path.
+    static const bool cpy_flat_on = []{
+        const char * e = getenv("GGML_OPENCL_CPY_FLAT");
+        return !(e && e[0] == '0');
+    }();
+    if (cpy_flat_on && backend_ctx->kernel_cpy_f32_f32_flat != nullptr &&
+        src0t == GGML_TYPE_F32 && src1t == GGML_TYPE_F32 &&
+        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
+        ggml_nelements(src0) == ggml_nelements(src1)) {
+        cl_kernel k = backend_ctx->kernel_cpy_f32_f32_flat;
+        const cl_ulong nelem = (cl_ulong) ggml_nelements(src0);
+        const cl_ulong n4    = nelem / 4;
+
+        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem),   &extra0->data_device));
+        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_ulong), &offset0));
+        CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem),   &extra1->data_device));
+        CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_ulong), &offset1));
+        CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_ulong), &nelem));
+        CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_ulong), &n4));
+
+        // one work item per float4, plus one for the trailing scalars
+        const size_t items = (size_t) n4 + ((nelem % 4) ? 1 : 0);
+        const size_t lsz   = MIN((size_t) 64, backend_ctx->max_workgroup_size);
+        size_t global_work_size[] = { ((items + lsz - 1) / lsz) * lsz, 1, 1 };
+        size_t local_work_size[]  = { lsz, 1, 1 };
+
+        backend_ctx->enqueue_ndrange_kernel(k, 1, global_work_size, local_work_size, src1);
+        return;
+    }
+
     cl_kernel kernel;
 
     switch (src0t) {
@@ -26771,6 +27655,42 @@ bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor
                     }
                     func = ggml_cl_abs;
                     break;
+                case GGML_UNARY_OP_SGN:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_sgn;
+                    break;
+                case GGML_UNARY_OP_STEP:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_step;
+                    break;
+                case GGML_UNARY_OP_ELU:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_elu;
+                    break;
+                case GGML_UNARY_OP_HARDSWISH:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_hardswish;
+                    break;
+                case GGML_UNARY_OP_HARDSIGMOID:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_hardsigmoid;
+                    break;
+                case GGML_UNARY_OP_FLOOR:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_floor;
+                    break;
+                case GGML_UNARY_OP_CEIL:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_ceil;
+                    break;
+                case GGML_UNARY_OP_ROUND:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_round;
+                    break;
+                case GGML_UNARY_OP_TRUNC:
+                    if (!any_on_device) { return false; }
+                    func = ggml_cl_trunc;
+                    break;
                 case GGML_UNARY_OP_SOFTPLUS:
                     if (!any_on_device) {
                         return false;
diff --git src/ggml-opencl/kernels/concat.cl src/ggml-opencl/kernels/concat.cl
index 2fbd7851..8ecf7466 100644
--- src/ggml-opencl/kernels/concat.cl
+++ src/ggml-opencl/kernels/concat.cl
@@ -1,56 +1,66 @@
-kernel void kernel_concat_f32(
-    global  const char * src0,
-    ulong                offset0,
-    global  const char * src1,
-    ulong                offset1,
-    global        char * dst,
-    ulong                offsetd,
-    int             ne00,
-    int             ne01,
-    int             ne02,
-    int             ne03,
-    ulong           nb00,
-    ulong           nb01,
-    ulong           nb02,
-    ulong           nb03,
-    ulong           nb10,
-    ulong           nb11,
-    ulong           nb12,
-    ulong           nb13,
-    int             ne0,
-    ulong           nb0,
-    ulong           nb1,
-    ulong           nb2,
-    ulong           nb3,
-    int             dim
-) {
-    src0 = src0 + offset0;
-    src1 = src1 + offset1;
-    dst  = dst  + offsetd;
-
-    const int i3 = get_group_id(2);
-    const int i2 = get_group_id(1);
-    const int i1 = get_group_id(0);
-
-    int o[4] = {0, 0, 0, 0};
-    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));
-
-    global const float * x;
-
-    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
-        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
-            x = (global const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
-        } else {
-            x = (global const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
-        }
-
-        global float * y = (global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
+// concat is a pure copy, so the kernels are keyed by element byte size
+// (1/2/4/8) rather than logical type, matching the CUDA backend.
 
-        *y = *x;
-    }
+#define KERNEL_CONCAT(SUFFIX, T)                                                     \
+kernel void kernel_concat_##SUFFIX(                                                  \
+    global  const char * src0,                                                       \
+    ulong                offset0,                                                    \
+    global  const char * src1,                                                       \
+    ulong                offset1,                                                    \
+    global        char * dst,                                                        \
+    ulong                offsetd,                                                    \
+    int             ne00,                                                            \
+    int             ne01,                                                            \
+    int             ne02,                                                            \
+    int             ne03,                                                            \
+    ulong           nb00,                                                            \
+    ulong           nb01,                                                            \
+    ulong           nb02,                                                            \
+    ulong           nb03,                                                            \
+    ulong           nb10,                                                            \
+    ulong           nb11,                                                            \
+    ulong           nb12,                                                            \
+    ulong           nb13,                                                            \
+    int             ne0,                                                             \
+    ulong           nb0,                                                             \
+    ulong           nb1,                                                             \
+    ulong           nb2,                                                             \
+    ulong           nb3,                                                             \
+    int             dim                                                              \
+) {                                                                                  \
+    src0 = src0 + offset0;                                                           \
+    src1 = src1 + offset1;                                                           \
+    dst  = dst  + offsetd;                                                           \
+                                                                                     \
+    const int i3 = get_group_id(2);                                                  \
+    const int i2 = get_group_id(1);                                                  \
+    const int i1 = get_group_id(0);                                                  \
+                                                                                     \
+    int o[4] = {0, 0, 0, 0};                                                         \
+    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));         \
+                                                                                     \
+    global const T * x;                                                              \
+                                                                                     \
+    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {              \
+        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {                      \
+            x = (global const T *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00); \
+        } else {                                                                     \
+            x = (global const T *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10); \
+        }                                                                            \
+                                                                                     \
+        global T * y = (global T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);        \
+                                                                                     \
+        *y = *x;                                                                     \
+    }                                                                                \
 }
 
-kernel void kernel_concat_f32_pack(
+KERNEL_CONCAT(b1, char)
+KERNEL_CONCAT(b2, short)
+KERNEL_CONCAT(b4, int)
+KERNEL_CONCAT(b8, long)
+
+// packed variant for the common dim==0, small-ne0 case (4-byte elements only).
+kernel void kernel_concat_b4_pack(
     global  const char * src0,
     ulong                offset0,
     global  const char * src1,
@@ -104,14 +114,14 @@ kernel void kernel_concat_f32_pack(
     o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));
 
     for (int i0 = lane; i0 < ne0; i0 += tpr) {
-        global const float * x;
+        global const int * x;
         if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
-            x = (global const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
+            x = (global const int *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
         } else {
-            x = (global const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
+            x = (global const int *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
         }
 
-        global float * y = (global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
+        global int * y = (global int *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
 
         *y = *x;
     }
diff --git src/ggml-opencl/kernels/cpy.cl src/ggml-opencl/kernels/cpy.cl
index adbd2e76..e875bfaf 100644
--- src/ggml-opencl/kernels/cpy.cl
+++ src/ggml-opencl/kernels/cpy.cl
@@ -286,3 +286,28 @@ kernel void kernel_cpy_i32_i32(
         dst_data[i00] = src[0];
     }
 }
+
+// Contiguous f32 copy, one work item per float4 over the whole tensor. The kernels above map
+// one workgroup to each row, which leaves a tensor with few long rows on a single compute unit.
+// vload4/vstore4 rather than a float4 cast: these buffers carry an arbitrary 4-byte view offset.
+kernel void kernel_cpy_f32_f32_flat(
+        global float * src0,
+        ulong offset0,
+        global float * dst,
+        ulong offsetd,
+        ulong ne,
+        ulong n4
+) {
+    src0 = (global float*)((global char*)src0 + offset0);
+    dst  = (global float*)((global char*)dst  + offsetd);
+
+    const ulong i = get_global_id(0);
+
+    if (i < n4) {
+        vstore4(vload4(i, src0), i, dst);
+    } else if (i == n4) {
+        for (ulong t = n4 * 4; t < ne; ++t) {
+            dst[t] = src0[t];
+        }
+    }
+}
diff --git src/ggml-opencl/kernels/sdpa_xmem_f32_f16_os8.cl src/ggml-opencl/kernels/sdpa_xmem_f32_f16_os8.cl
new file mode 100644
index 00000000..26f0fbd5
--- /dev/null
+++ src/ggml-opencl/kernels/sdpa_xmem_f32_f16_os8.cl
@@ -0,0 +1,871 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load : enable
+#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load : enable
+
+#define bool2 uchar2
+#define bool3 uchar3
+#define bool4 uchar4
+
+__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
+__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
+
+__kernel void adreno_xmem_attn_q_f32_to_img_scaled(const global void *  src_void,
+                                                   ulong                src_offset,
+                                                   write_only image2d_t dst_image2d,
+                                                   const float          scale,
+                                                   const int            d_head,
+                                                   const int            n_q,
+                                                   const int            n_head,
+                                                   const int            n_head_kv,
+                                                   const int            n_batch,
+                                                   const ulong          src_nb1,
+                                                   const ulong          src_nb2,
+                                                   const ulong          src_nb3) {
+    const int x      = get_global_id(0);
+    const int flat_h = get_global_id(1);
+    const int d      = get_global_id(2);
+
+    const int heads_total = n_head * n_batch;
+    const int kpack       = d_head / 4;
+
+    if (x >= n_q || flat_h >= heads_total || d >= kpack) {
+        return;
+    }
+
+    const int batch = flat_h / n_head;
+    const int head  = flat_h % n_head;
+    const int gqa   = n_head / n_head_kv;
+    const int head_kv = head / gqa;
+    const int head_group = head - head_kv * gqa;
+    const int compact_h = batch * n_head_kv + head_kv;
+    const int compact_x = head_group * n_q + x;
+    const int c     = d * 4;
+
+    const global char *  src_base = (const global char *) src_void + src_offset;
+    const global float * row_ptr  = (const global float *) (src_base + batch * src_nb3 + head * src_nb2 + x * src_nb1);
+
+    half4 out = (half4) (0.0h);
+    out.x     = convert_half(row_ptr[c + 0] * scale);
+    if (c + 1 < d_head) {
+        out.y = convert_half(row_ptr[c + 1] * scale);
+    }
+    if (c + 2 < d_head) {
+        out.z = convert_half(row_ptr[c + 2] * scale);
+    }
+    if (c + 3 < d_head) {
+        out.w = convert_half(row_ptr[c + 3] * scale);
+    }
+
+    write_imageh(dst_image2d, (int2) (compact_x, compact_h * kpack + d), out);
+}
+
+__kernel void adreno_xmem_attn_kv_f32_to_img_gqa(const global void *  src_void,
+                                                 ulong                src_offset,
+                                                 write_only image2d_t dst_image2d,
+                                                 const int            d_head,
+                                                 const int            n_kv,
+                                                 const int            n_kv_padded,
+                                                 const int            n_head_kv,
+                                                 const int            n_batch,
+                                                 const ulong          src_nb1,
+                                                 const ulong          src_nb2,
+                                                 const ulong          src_nb3) {
+    const int x      = get_global_id(0);
+    const int flat_h = get_global_id(1);
+    const int d      = get_global_id(2);
+
+    const int kv_heads_total = n_head_kv * n_batch;
+    const int kpack          = d_head / 4;
+
+    if (x >= n_kv_padded || flat_h >= kv_heads_total || d >= kpack) {
+        return;
+    }
+
+    const int batch   = flat_h / n_head_kv;
+    const int head_kv = flat_h % n_head_kv;
+    const int c       = d * 4;
+
+    half4 out = (half4) (0.0h);
+    if (x < n_kv) {
+        const global char *  src_base = (const global char *) src_void + src_offset;
+        const global float * row_ptr =
+            (const global float *) (src_base + batch * src_nb3 + head_kv * src_nb2 + x * src_nb1);
+        out.x = convert_half(row_ptr[c + 0]);
+        if (c + 1 < d_head) {
+            out.y = convert_half(row_ptr[c + 1]);
+        }
+        if (c + 2 < d_head) {
+            out.z = convert_half(row_ptr[c + 2]);
+        }
+        if (c + 3 < d_head) {
+            out.w = convert_half(row_ptr[c + 3]);
+        }
+    }
+
+    write_imageh(dst_image2d, (int2) (x, flat_h * kpack + d), out);
+}
+
+__kernel void adreno_xmem_attn_kv_f16_to_img_gqa(const global void *  src_void,
+                                                 ulong                src_offset,
+                                                 write_only image2d_t dst_image2d,
+                                                 const int            d_head,
+                                                 const int            n_kv,
+                                                 const int            n_kv_padded,
+                                                 const int            n_head_kv,
+                                                 const int            n_batch,
+                                                 const ulong          src_nb1,
+                                                 const ulong          src_nb2,
+                                                 const ulong          src_nb3) {
+    const int x      = get_global_id(0);
+    const int flat_h = get_global_id(1);
+    const int d      = get_global_id(2);
+
+    const int kv_heads_total = n_head_kv * n_batch;
+    const int kpack          = d_head / 4;
+
+    if (x >= n_kv_padded || flat_h >= kv_heads_total || d >= kpack) {
+        return;
+    }
+
+    const int batch   = flat_h / n_head_kv;
+    const int head_kv = flat_h % n_head_kv;
+    const int c       = d * 4;
+
+    half4 out = (half4) (0.0h);
+    if (x < n_kv) {
+        const global char * src_base = (const global char *) src_void + src_offset;
+        const global half * row_ptr =
+            (const global half *) (src_base + batch * src_nb3 + head_kv * src_nb2 + x * src_nb1);
+        out.x = row_ptr[c + 0];
+        if (c + 1 < d_head) {
+            out.y = row_ptr[c + 1];
+        }
+        if (c + 2 < d_head) {
+            out.z = row_ptr[c + 2];
+        }
+        if (c + 3 < d_head) {
+            out.w = row_ptr[c + 3];
+        }
+    }
+
+    write_imageh(dst_image2d, (int2) (x, flat_h * kpack + d), out);
+}
+
+__kernel void adreno_xmem_attn_img_to_f32(global void *       dst_void,
+                                          ulong               dst_offset,
+                                          read_only image2d_t src_image2d,
+                                          const int           d_head,
+                                          const int           n_q,
+                                          const int           n_head,
+                                          const int           n_head_kv,
+                                          const int           n_batch,
+                                          const ulong         dst_nb1,
+                                          const ulong         dst_nb2,
+                                          const ulong         dst_nb3) {
+    const int x      = get_global_id(0);
+    const int flat_h = get_global_id(1);
+    const int d      = get_global_id(2);
+
+    const int heads_total = n_head * n_batch;
+    const int kpack       = d_head / 4;
+
+    if (x >= n_q || flat_h >= heads_total || d >= kpack) {
+        return;
+    }
+
+    const int batch = flat_h / n_head;
+    const int head  = flat_h % n_head;
+    const int gqa   = n_head / n_head_kv;
+    const int head_kv = head / gqa;
+    const int head_group = head - head_kv * gqa;
+    const int compact_h = batch * n_head_kv + head_kv;
+    const int compact_x = head_group * n_q + x;
+    const int c     = d * 4;
+
+    global char *  dst_base = (global char *) dst_void + dst_offset;
+    global float * row_ptr  = (global float *) (dst_base + batch * dst_nb3 + x * dst_nb2 + head * dst_nb1);
+
+    const half4 in_value = read_imageh(src_image2d, smp_zero, (int2) (compact_x, compact_h * kpack + d));
+    row_ptr[c + 0]       = convert_float(in_value.x);
+    if (c + 1 < d_head) {
+        row_ptr[c + 1] = convert_float(in_value.y);
+    }
+    if (c + 2 < d_head) {
+        row_ptr[c + 2] = convert_float(in_value.z);
+    }
+    if (c + 3 < d_head) {
+        row_ptr[c + 3] = convert_float(in_value.w);
+    }
+}
+
+__kernel void adreno_xmem_attn_k_gather(global half4 *      dst_tensor_buffer,
+                                        read_only image2d_t src_tensor_image2d,
+                                        const int4          shared_int4_0,
+                                        const int4          shared_int4_1) {
+    int X = get_global_id(0);
+    int Y = get_global_id(1);
+    int S = get_global_id(2);
+    if (X >= shared_int4_0.w || Y >= shared_int4_0.y || S >= shared_int4_0.z) {
+        return;
+    }
+    half temps[4];
+    temps[0] = (half) (0.f);
+    temps[1] = (half) (0.f);
+    temps[2] = (half) (0.f);
+    temps[3] = (half) (0.f);
+    for (int i = 0; i < 4; ++i) {
+        int dst_channel = S * 4 + i;
+        if (dst_channel < shared_int4_0.x) {
+            int s_y = Y;
+            int s_x = dst_channel;
+            int s_c = X;
+            {
+                int   slice_coord_TMP  = (s_c) / 4;
+                int   sub_ch_coord_TMP = (s_c) % 4;
+                half4 src_TMP          = read_imageh(src_tensor_image2d, smp_zero,
+                                                     (int2) ((s_x), ((s_y) *shared_int4_1.x + (slice_coord_TMP))));
+                temps[i]               = (half[4]){ src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w }[sub_ch_coord_TMP];
+            };
+        }
+    }
+    half4 result;
+    result.x                                                                  = temps[0];
+    result.y                                                                  = temps[1];
+    result.z                                                                  = temps[2];
+    result.w                                                                  = temps[3];
+    dst_tensor_buffer[(((S) *shared_int4_0.y + (Y)) * shared_int4_0.w + (X))] = result;
+}
+
+__kernel void adreno_xmem_attn_pack_k(global half4 *             dst_tensor_buffer,
+                                      read_only image1d_buffer_t src_image_buffer,
+                                      const int4                 shared_int4_0,
+                                      const int4                 shared_int4_1,
+                                      const int4                 shared_int4_2) {
+    int linear_index = get_global_id(0);
+    if (linear_index >= shared_int4_0.y) {
+        return;
+    }
+    if (get_global_id(1) != 0) {
+        return;
+    }
+    if (get_global_id(2) != 0) {
+        return;
+    }
+    int   dst_o_sp_i_ogroup = linear_index;
+    int   dst_ogroup        = dst_o_sp_i_ogroup % shared_int4_0.x;
+    int   dst_o_sp_i        = dst_o_sp_i_ogroup / shared_int4_0.x;
+    int   dst_i             = dst_o_sp_i % shared_int4_0.z;
+    int   dst_o_sp          = dst_o_sp_i / shared_int4_0.z;
+    int   dst_sp            = dst_o_sp % shared_int4_1.x;
+    int   dst_o             = dst_o_sp / shared_int4_1.x;
+    int   i_slice           = dst_i;
+    int   o_slice           = dst_o * shared_int4_0.x + dst_ogroup;
+    int   spatial_linear    = dst_sp;
+    int   W                 = spatial_linear % shared_int4_1.y;
+    int   H                 = spatial_linear / shared_int4_1.y;
+    half4 w0                = (half4) (0);
+    half4 w1                = (half4) (0);
+    half4 w2                = (half4) (0);
+    half4 w3                = (half4) (0);
+
+    if (i_slice * 4 < shared_int4_0.w && o_slice < shared_int4_1.w) {
+        w0 = read_imageh(src_image_buffer, (((o_slice) *shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4)));
+    }
+    if (i_slice * 4 + 1 < shared_int4_0.w && o_slice < shared_int4_1.w) {
+        w1 = read_imageh(src_image_buffer, (((o_slice) *shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4 + 1)));
+    }
+    if (i_slice * 4 + 2 < shared_int4_0.w && o_slice < shared_int4_1.w) {
+        w2 = read_imageh(src_image_buffer, (((o_slice) *shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4 + 2)));
+    }
+    if (i_slice * 4 + 3 < shared_int4_0.w && o_slice < shared_int4_1.w) {
+        w3 = read_imageh(src_image_buffer, (((o_slice) *shared_int4_1.z + (W)) * shared_int4_2.x + (i_slice * 4 + 3)));
+    }
+    half4 r0                                = w0;
+    half4 r1                                = w1;
+    half4 r2                                = w2;
+    half4 r3                                = w3;
+    dst_tensor_buffer[linear_index * 4 + 0] = r0;
+    dst_tensor_buffer[linear_index * 4 + 1] = r1;
+    dst_tensor_buffer[linear_index * 4 + 2] = r2;
+    dst_tensor_buffer[linear_index * 4 + 3] = r3;
+}
+
+__attribute__((qcom_max_concurrent_subgroups(12))) __kernel void adreno_xmem_attn_qk_gemm(
+    global half4 *      dst_tensor_buffer,
+    constant half8 *    weights_buffer __attribute__((sub_group_uniform)),
+    constant half8 *    xmem_buffer __attribute__((max_constant_size((6144)))),
+    read_only image2d_t src_tensor_image2d,
+    const int4          shared_int4_0,
+    const int4          shared_int4_1,
+    const int4          shared_int4_2) {
+    int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
+    int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
+    int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
+    if (X >= shared_int4_0.z || Y >= shared_int4_0.x) {
+        return;
+    }
+    if (Z * 8 >= shared_int4_0.y) {
+        return;
+    }
+
+    half4 r0      = (half4) (0.f);
+    half4 r1      = (half4) (0.f);
+    half4 r2      = (half4) (0.f);
+    half4 r3      = (half4) (0.f);
+    half4 r4      = (half4) (0.f);
+    half4 r5      = (half4) (0.f);
+    half4 r6      = (half4) (0.f);
+    half4 r7      = (half4) (0.f);
+    int   x_coord = mad24(X, shared_int4_2.y, shared_int4_1.y);
+    int   y_coord = mad24(Y, shared_int4_2.z, shared_int4_1.z);
+    int   coord_x, coord_y, coord_s;
+    int   f_offset = (Z * shared_int4_1.w + Y) * shared_int4_1.x * 32;
+
+    int subgroup_id                   = (int) ((0x1F & qcom_get_physical_sub_group_id()));
+    subgroup_id                       = subgroup_id % 12;
+    int                 c_offset      = mul24(subgroup_id, shared_int4_0.w);
+    __constant half16 * weights_cache = (__constant half16 *) &xmem_buffer[c_offset];
+    coord_y                           = Y;
+    coord_x                           = X;
+    coord_s                           = 0;
+    do {
+        half4 src0 =
+            read_imageh(src_tensor_image2d, smp_zero, (int2) ((coord_x), ((coord_y) *shared_int4_2.x + (coord_s))));
+        coord_s++;
+        half4 src1 =
+            read_imageh(src_tensor_image2d, smp_zero, (int2) ((coord_x), ((coord_y) *shared_int4_2.x + (coord_s))));
+        coord_s++;
+        qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
+        f_offset += 64;
+        qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
+        r0 += src0.x * weights_cache[0].s0123;
+        r0 += src0.y * weights_cache[0].s4567;
+        r0 += src0.z * weights_cache[0].s89ab;
+        r0 += src0.w * weights_cache[0].scdef;
+        r1 += src0.x * weights_cache[1].s0123;
+        r1 += src0.y * weights_cache[1].s4567;
+        r1 += src0.z * weights_cache[1].s89ab;
+        r1 += src0.w * weights_cache[1].scdef;
+        r2 += src0.x * weights_cache[2].s0123;
+        r2 += src0.y * weights_cache[2].s4567;
+        r2 += src0.z * weights_cache[2].s89ab;
+        r2 += src0.w * weights_cache[2].scdef;
+        r3 += src0.x * weights_cache[3].s0123;
+        r3 += src0.y * weights_cache[3].s4567;
+        r3 += src0.z * weights_cache[3].s89ab;
+        r3 += src0.w * weights_cache[3].scdef;
+        r4 += src0.x * weights_cache[4].s0123;
+        r4 += src0.y * weights_cache[4].s4567;
+        r4 += src0.z * weights_cache[4].s89ab;
+        r4 += src0.w * weights_cache[4].scdef;
+        r5 += src0.x * weights_cache[5].s0123;
+        r5 += src0.y * weights_cache[5].s4567;
+        r5 += src0.z * weights_cache[5].s89ab;
+        r5 += src0.w * weights_cache[5].scdef;
+        r6 += src0.x * weights_cache[6].s0123;
+        r6 += src0.y * weights_cache[6].s4567;
+        r6 += src0.z * weights_cache[6].s89ab;
+        r6 += src0.w * weights_cache[6].scdef;
+        r7 += src0.x * weights_cache[7].s0123;
+        r7 += src0.y * weights_cache[7].s4567;
+        r7 += src0.z * weights_cache[7].s89ab;
+        r7 += src0.w * weights_cache[7].scdef;
+        r0 += src1.x * weights_cache[8].s0123;
+        r0 += src1.y * weights_cache[8].s4567;
+        r0 += src1.z * weights_cache[8].s89ab;
+        r0 += src1.w * weights_cache[8].scdef;
+        r1 += src1.x * weights_cache[9].s0123;
+        r1 += src1.y * weights_cache[9].s4567;
+        r1 += src1.z * weights_cache[9].s89ab;
+        r1 += src1.w * weights_cache[9].scdef;
+        r2 += src1.x * weights_cache[10].s0123;
+        r2 += src1.y * weights_cache[10].s4567;
+        r2 += src1.z * weights_cache[10].s89ab;
+        r2 += src1.w * weights_cache[10].scdef;
+        r3 += src1.x * weights_cache[11].s0123;
+        r3 += src1.y * weights_cache[11].s4567;
+        r3 += src1.z * weights_cache[11].s89ab;
+        r3 += src1.w * weights_cache[11].scdef;
+        r4 += src1.x * weights_cache[12].s0123;
+        r4 += src1.y * weights_cache[12].s4567;
+        r4 += src1.z * weights_cache[12].s89ab;
+        r4 += src1.w * weights_cache[12].scdef;
+        r5 += src1.x * weights_cache[13].s0123;
+        r5 += src1.y * weights_cache[13].s4567;
+        r5 += src1.z * weights_cache[13].s89ab;
+        r5 += src1.w * weights_cache[13].scdef;
+        r6 += src1.x * weights_cache[14].s0123;
+        r6 += src1.y * weights_cache[14].s4567;
+        r6 += src1.z * weights_cache[14].s89ab;
+        r6 += src1.w * weights_cache[14].scdef;
+        r7 += src1.x * weights_cache[15].s0123;
+        r7 += src1.y * weights_cache[15].s4567;
+        r7 += src1.z * weights_cache[15].s89ab;
+        r7 += src1.w * weights_cache[15].scdef;
+    } while (coord_s < shared_int4_2.x);
+
+    coord_s = mul24(Z, 8);
+    coord_x = X;
+    coord_y = Y;
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r0);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r1);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r2);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r3);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r4);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r5);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r6);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r7);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image2d, smp_zero, (int2) ((0), ((0) * shared_int4_2.x + (0))));
+        }
+        dst_tensor_buffer[(((coord_s) *shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
+        coord_s++;
+    }
+}
+
+__kernel void adreno_xmem_attn_softmax_reduce_basic(read_only image1d_buffer_t src_tensor_image_buffer,
+                                                    write_only image2d_t       dst_tensor_image2d,
+                                                    const int4                 shared_int4_0,
+                                                    const int4                 shared_int4_1) {
+    int X = get_global_id(0);
+    int Y = get_global_id(1);
+    if (X >= shared_int4_0.z || Y >= shared_int4_0.x) {
+        return;
+    }
+    float sum                     = 0.0f;
+    int   end_channel             = shared_int4_0.w;
+    int   end_slice               = (end_channel + 3) / 4;
+    int   start_channel           = 0;
+    int   start_slice             = start_channel / 4;
+    bool  need_per_channels_check = start_channel % 4 != 0 || end_channel % 4 != 0;
+    float maximum;
+    {
+        int    slice_coord_TMP  = (start_channel) / 4;
+        int    sub_ch_coord_TMP = (start_channel) % 4;
+        float4 src_TMP          = convert_float4(
+            read_imageh(src_tensor_image_buffer, ((slice_coord_TMP) *shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
+        maximum = (float[4]){ src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w }[sub_ch_coord_TMP];
+    };
+    for (int d = start_slice; d < end_slice; d += 1) {
+        float4 mask_dot = (float4) (1.f);
+        float4 src =
+            convert_float4(read_imageh(src_tensor_image_buffer, ((d) *shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
+        if (need_per_channels_check && (d == start_slice || d == end_slice - 1)) {
+            if (d * 4 + 0 < start_channel || d * 4 + 0 >= end_channel) {
+                mask_dot.x = 0.f;
+                src.x      = maximum;
+            }
+            if (d * 4 + 1 < start_channel || d * 4 + 1 >= end_channel) {
+                mask_dot.y = 0.f;
+                src.y      = maximum;
+            }
+            if (d * 4 + 2 < start_channel || d * 4 + 2 >= end_channel) {
+                mask_dot.z = 0.f;
+                src.z      = maximum;
+            }
+            if (d * 4 + 3 < start_channel || d * 4 + 3 >= end_channel) {
+                mask_dot.w = 0.f;
+                src.w      = maximum;
+            }
+        }
+        float new_max = max(src.x, src.y);
+        new_max       = max(new_max, src.z);
+        new_max       = max(new_max, src.w);
+        new_max       = max(new_max, maximum);
+        float scale   = native_exp(maximum - new_max);
+        maximum       = new_max;
+        sum *= scale;
+        float4 exp_res = native_exp(src - maximum);
+        sum += dot(mask_dot, exp_res);
+    }
+    if (!isfinite(maximum) || sum == 0.0f) {
+        write_imageh(dst_tensor_image2d, (int2) (X, Y), (half4) (0.0h));
+        return;
+    }
+    write_imageh(dst_tensor_image2d, (int2) (X, Y),
+                 (half4) (convert_half(1.0f / sum), convert_half(maximum), 0.0h, 0.0h));
+}
+
+__kernel void adreno_xmem_attn_softmax_apply_basic(global half4 *             dst_tensor_buffer,
+                                                   read_only image1d_buffer_t src_tensor_image_buffer,
+                                                   read_only image2d_t        src_tensor_1_image2d,
+                                                   const int4                 shared_int4_0,
+                                                   const int4                 shared_int4_1) {
+    int X = get_global_id(0);
+    int Y = get_global_id(1);
+    int Z = get_global_id(2);
+    if (X >= shared_int4_0.z || Y >= shared_int4_0.x || Z >= shared_int4_0.y) {
+        return;
+    }
+    half4 src = read_imageh(src_tensor_image_buffer, ((Z) *shared_int4_1.x + (Y)) * shared_int4_1.y + (X));
+    {
+        half4 src_final;
+        {
+            {
+                half4 exp_val = read_imageh(src_tensor_1_image2d, smp_zero, (int2) (X, Y));
+                src_final = exp(src - exp_val.y) * exp_val.x;
+                const int k = Z * 4;
+                const int n_kv = shared_int4_1.z;
+                if (k + 0 >= n_kv) {
+                    src_final.x = 0.0h;
+                }
+                if (k + 1 >= n_kv) {
+                    src_final.y = 0.0h;
+                }
+                if (k + 2 >= n_kv) {
+                    src_final.z = 0.0h;
+                }
+                if (k + 3 >= n_kv) {
+                    src_final.w = 0.0h;
+                }
+            }
+        }
+        dst_tensor_buffer[(((Z) *shared_int4_0.x + (Y)) * shared_int4_0.z + (X))] = src_final;
+    };
+}
+
+__kernel void adreno_xmem_attn_mask_scores(global half4 *             dst_score_tensor_buffer,
+                                           read_only image1d_buffer_t src_score_image_buffer,
+                                           const global half *        mask,
+                                           const ulong                mask_offset,
+                                           const int                  q_width,
+                                           const int                  n_q,
+                                           const int                  n_kv,
+                                           const int                  n_kv_padded,
+                                           const int                  kv_heads_total,
+                                           const int                  n_head,
+                                           const int                  n_head_kv,
+                                           const ulong                mask_nb1,
+                                           const ulong                mask_nb2,
+                                           const ulong                mask_nb3,
+                                           const int                  mask_ne2,
+                                           const int                  mask_ne3) {
+    const int X     = get_global_id(0);
+    const int Y     = get_global_id(1);
+    const int Z     = get_global_id(2);
+    const int npack = n_kv_padded / 4;
+    if (X >= q_width || Y >= kv_heads_total || Z >= npack) {
+        return;
+    }
+
+    const int           gqa            = n_head / n_head_kv;
+    const int           head_kv        = Y % n_head_kv;
+    const int           batch          = Y / n_head_kv;
+    const int           head_group     = X / n_q;
+    const int           q              = X - head_group * n_q;
+    const int           head           = head_kv * gqa + head_group;
+    const int           mask_head_idx  = head % mask_ne2;
+    const int           mask_batch_idx = batch % mask_ne3;
+    const global char *  mask_base      = (const global char *) mask + mask_offset;
+    const global half *  mask_row       = (const global half *) (mask_base + mask_batch_idx * mask_nb3 +
+                                                                 mask_head_idx * mask_nb2 + q * mask_nb1);
+
+    const half4 score   = read_imageh(src_score_image_buffer, ((Z * kv_heads_total + Y) * q_width + X));
+    float       vals[4] = {
+        convert_float(score.x),
+        convert_float(score.y),
+        convert_float(score.z),
+        convert_float(score.w),
+    };
+
+    for (int lane = 0; lane < 4; ++lane) {
+        const int k_idx = Z * 4 + lane;
+        if (k_idx >= n_kv) {
+            vals[lane] = -INFINITY;
+        } else {
+            vals[lane] += convert_float(mask_row[k_idx]);
+        }
+    }
+
+    dst_score_tensor_buffer[((Z * kv_heads_total + Y) * q_width + X)] =
+        (half4) (convert_half(vals[0]), convert_half(vals[1]), convert_half(vals[2]), convert_half(vals[3]));
+}
+
+__kernel void adreno_xmem_attn_pack_v(global half4 *      dst_tensor_buffer,
+                                      read_only image2d_t src_image2d,
+                                      const int4          shared_int4_0,
+                                      const int4          shared_int4_1) {
+    int linear_index = get_global_id(0);
+    if (linear_index >= shared_int4_0.y) {
+        return;
+    }
+    if (get_global_id(1) != 0) {
+        return;
+    }
+    if (get_global_id(2) != 0) {
+        return;
+    }
+    int   dst_o_sp_i_ogroup = linear_index;
+    int   dst_ogroup        = dst_o_sp_i_ogroup % shared_int4_0.x;
+    int   dst_o_sp_i        = dst_o_sp_i_ogroup / shared_int4_0.x;
+    int   dst_i             = dst_o_sp_i % shared_int4_0.z;
+    int   dst_o_sp          = dst_o_sp_i / shared_int4_0.z;
+    int   dst_sp            = dst_o_sp % shared_int4_1.x;
+    int   dst_o             = dst_o_sp / shared_int4_1.x;
+    int   i_slice           = dst_i;
+    int   o_slice           = dst_o * shared_int4_0.x + dst_ogroup;
+    int   spatial_linear    = dst_sp;
+    int   W                 = spatial_linear % shared_int4_1.y;
+    int   H                 = spatial_linear / shared_int4_1.y;
+    half4 w0                = (half4) (0);
+    half4 w1                = (half4) (0);
+    half4 w2                = (half4) (0);
+    half4 w3                = (half4) (0);
+
+    if (i_slice * 4 < shared_int4_0.w && o_slice < shared_int4_1.z) {
+        w0 = read_imageh(src_image2d, smp_zero, (int2) ((i_slice * 4), ((W) *shared_int4_1.z + (o_slice))));
+    }
+    if (i_slice * 4 + 1 < shared_int4_0.w && o_slice < shared_int4_1.z) {
+        w1 = read_imageh(src_image2d, smp_zero, (int2) ((i_slice * 4 + 1), ((W) *shared_int4_1.z + (o_slice))));
+    }
+    if (i_slice * 4 + 2 < shared_int4_0.w && o_slice < shared_int4_1.z) {
+        w2 = read_imageh(src_image2d, smp_zero, (int2) ((i_slice * 4 + 2), ((W) *shared_int4_1.z + (o_slice))));
+    }
+    if (i_slice * 4 + 3 < shared_int4_0.w && o_slice < shared_int4_1.z) {
+        w3 = read_imageh(src_image2d, smp_zero, (int2) ((i_slice * 4 + 3), ((W) *shared_int4_1.z + (o_slice))));
+    }
+    half4 r0                                = w0;
+    half4 r1                                = w1;
+    half4 r2                                = w2;
+    half4 r3                                = w3;
+    dst_tensor_buffer[linear_index * 4 + 0] = r0;
+    dst_tensor_buffer[linear_index * 4 + 1] = r1;
+    dst_tensor_buffer[linear_index * 4 + 2] = r2;
+    dst_tensor_buffer[linear_index * 4 + 3] = r3;
+}
+
+__attribute__((qcom_max_concurrent_subgroups(12))) __kernel void adreno_xmem_attn_pv_gemm(
+    constant half8 *           weights_buffer __attribute__((sub_group_uniform)),
+    constant half8 *           xmem_buffer __attribute__((max_constant_size((6144)))),
+    read_only image1d_buffer_t src_tensor_image_buffer,
+    write_only image2d_t       dst_tensor_image2d,
+    const int4                 shared_int4_0,
+    const int4                 shared_int4_1,
+    const int4                 shared_int4_2,
+    const int4                 shared_int4_3) {
+    int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
+    int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
+    int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
+    if (X >= shared_int4_0.z || Y >= shared_int4_0.x) {
+        return;
+    }
+    if (Z * 8 >= shared_int4_0.y) {
+        return;
+    }
+
+    half4 r0      = (half4) (0.f);
+    half4 r1      = (half4) (0.f);
+    half4 r2      = (half4) (0.f);
+    half4 r3      = (half4) (0.f);
+    half4 r4      = (half4) (0.f);
+    half4 r5      = (half4) (0.f);
+    half4 r6      = (half4) (0.f);
+    half4 r7      = (half4) (0.f);
+    int   x_coord = mad24(X, shared_int4_2.w, shared_int4_1.y);
+    int   y_coord = mad24(Y, shared_int4_3.x, shared_int4_1.z);
+    int   coord_x, coord_y, coord_s;
+    int   f_offset = (Z * shared_int4_1.w + Y) * shared_int4_1.x * 32;
+
+    int subgroup_id                   = (int) ((0x1F & qcom_get_physical_sub_group_id()));
+    subgroup_id                       = subgroup_id % 12;
+    int                 c_offset      = mul24(subgroup_id, shared_int4_0.w);
+    __constant half16 * weights_cache = (__constant half16 *) &xmem_buffer[c_offset];
+    coord_y                           = Y;
+    coord_x                           = X;
+    int addr                          = (((0) * shared_int4_1.w + (coord_y)) * shared_int4_2.z + (coord_x));
+    int dz                            = shared_int4_2.x;
+    coord_s                           = 0;
+    do {
+        half4 src0 = read_imageh(src_tensor_image_buffer, addr);
+        addr += dz;
+        coord_s++;
+        half4 src1 = read_imageh(src_tensor_image_buffer, addr);
+        addr += dz;
+        coord_s++;
+        qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
+        f_offset += 64;
+        qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
+        r0 += src0.x * weights_cache[0].s0123;
+        r0 += src0.y * weights_cache[0].s4567;
+        r0 += src0.z * weights_cache[0].s89ab;
+        r0 += src0.w * weights_cache[0].scdef;
+        r1 += src0.x * weights_cache[1].s0123;
+        r1 += src0.y * weights_cache[1].s4567;
+        r1 += src0.z * weights_cache[1].s89ab;
+        r1 += src0.w * weights_cache[1].scdef;
+        r2 += src0.x * weights_cache[2].s0123;
+        r2 += src0.y * weights_cache[2].s4567;
+        r2 += src0.z * weights_cache[2].s89ab;
+        r2 += src0.w * weights_cache[2].scdef;
+        r3 += src0.x * weights_cache[3].s0123;
+        r3 += src0.y * weights_cache[3].s4567;
+        r3 += src0.z * weights_cache[3].s89ab;
+        r3 += src0.w * weights_cache[3].scdef;
+        r4 += src0.x * weights_cache[4].s0123;
+        r4 += src0.y * weights_cache[4].s4567;
+        r4 += src0.z * weights_cache[4].s89ab;
+        r4 += src0.w * weights_cache[4].scdef;
+        r5 += src0.x * weights_cache[5].s0123;
+        r5 += src0.y * weights_cache[5].s4567;
+        r5 += src0.z * weights_cache[5].s89ab;
+        r5 += src0.w * weights_cache[5].scdef;
+        r6 += src0.x * weights_cache[6].s0123;
+        r6 += src0.y * weights_cache[6].s4567;
+        r6 += src0.z * weights_cache[6].s89ab;
+        r6 += src0.w * weights_cache[6].scdef;
+        r7 += src0.x * weights_cache[7].s0123;
+        r7 += src0.y * weights_cache[7].s4567;
+        r7 += src0.z * weights_cache[7].s89ab;
+        r7 += src0.w * weights_cache[7].scdef;
+        r0 += src1.x * weights_cache[8].s0123;
+        r0 += src1.y * weights_cache[8].s4567;
+        r0 += src1.z * weights_cache[8].s89ab;
+        r0 += src1.w * weights_cache[8].scdef;
+        r1 += src1.x * weights_cache[9].s0123;
+        r1 += src1.y * weights_cache[9].s4567;
+        r1 += src1.z * weights_cache[9].s89ab;
+        r1 += src1.w * weights_cache[9].scdef;
+        r2 += src1.x * weights_cache[10].s0123;
+        r2 += src1.y * weights_cache[10].s4567;
+        r2 += src1.z * weights_cache[10].s89ab;
+        r2 += src1.w * weights_cache[10].scdef;
+        r3 += src1.x * weights_cache[11].s0123;
+        r3 += src1.y * weights_cache[11].s4567;
+        r3 += src1.z * weights_cache[11].s89ab;
+        r3 += src1.w * weights_cache[11].scdef;
+        r4 += src1.x * weights_cache[12].s0123;
+        r4 += src1.y * weights_cache[12].s4567;
+        r4 += src1.z * weights_cache[12].s89ab;
+        r4 += src1.w * weights_cache[12].scdef;
+        r5 += src1.x * weights_cache[13].s0123;
+        r5 += src1.y * weights_cache[13].s4567;
+        r5 += src1.z * weights_cache[13].s89ab;
+        r5 += src1.w * weights_cache[13].scdef;
+        r6 += src1.x * weights_cache[14].s0123;
+        r6 += src1.y * weights_cache[14].s4567;
+        r6 += src1.z * weights_cache[14].s89ab;
+        r6 += src1.w * weights_cache[14].scdef;
+        r7 += src1.x * weights_cache[15].s0123;
+        r7 += src1.y * weights_cache[15].s4567;
+        r7 += src1.z * weights_cache[15].s89ab;
+        r7 += src1.w * weights_cache[15].scdef;
+    } while (coord_s < shared_int4_2.y);
+
+    coord_s = mul24(Z, 8);
+    coord_x = X;
+    coord_y = Y;
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r0);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r1);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r2);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r3);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r4);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r5);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r6);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+    if (coord_s < shared_int4_0.y) {
+        half4 res = convert_half4(r7);
+        if (coord_s < 0) {
+            res += read_imageh(src_tensor_image_buffer, ((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0));
+        }
+        write_imageh(dst_tensor_image2d, (int2) ((coord_x), ((coord_y) *shared_int4_0.y + (coord_s))), res);
+        coord_s++;
+    }
+}
diff --git src/ggml-opencl/kernels/unary_ext.cl src/ggml-opencl/kernels/unary_ext.cl
new file mode 100644
index 00000000..e86eadfa
--- /dev/null
+++ src/ggml-opencl/kernels/unary_ext.cl
@@ -0,0 +1,85 @@
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+
+//------------------------------------------------------------------------------
+// Extended elementwise unary ops, same variant shape as abs.cl:
+//   f32, f32_4 (vec4), f16, f16_4 (vec4), f32_nc, f16_nc (stride-addressed).
+//
+//   sgn, step, elu, hardswish, hardsigmoid, floor, ceil, round, trunc.
+//
+// Semantics match the ggml CPU reference (ggml.c). Values are computed in float
+// (the f16 variants read/write half and convert), so the conditional ops match
+// the CPU bit-for-bit within tolerance. SEXPR is the scalar form, VEXPR the
+// float4 form (vector ternaries need select()).
+//------------------------------------------------------------------------------
+
+#define UNARY_EXT(NAME, SEXPR, VEXPR)                                           \
+kernel void kernel_##NAME##_f32(                                               \
+        global const float * src0, ulong offset0,                             \
+        global       float * dst,  ulong offsetd) {                           \
+    src0 = (global float*)((global char*)src0 + offset0);                     \
+    dst  = (global float*)((global char*)dst  + offsetd);                     \
+    float x = src0[get_global_id(0)];                                         \
+    dst[get_global_id(0)] = (SEXPR);                                          \
+}                                                                             \
+kernel void kernel_##NAME##_f32_4(                                            \
+        global const float4 * src0, ulong offset0,                           \
+        global       float4 * dst,  ulong offsetd) {                         \
+    src0 = (global float4*)((global char*)src0 + offset0);                   \
+    dst  = (global float4*)((global char*)dst  + offsetd);                   \
+    float4 x = src0[get_global_id(0)];                                       \
+    dst[get_global_id(0)] = (VEXPR);                                         \
+}                                                                             \
+kernel void kernel_##NAME##_f16(                                             \
+        global const half * src0, ulong offset0,                            \
+        global       half * dst,  ulong offsetd) {                          \
+    src0 = (global half*)((global char*)src0 + offset0);                    \
+    dst  = (global half*)((global char*)dst  + offsetd);                    \
+    float x = src0[get_global_id(0)];                                       \
+    dst[get_global_id(0)] = (SEXPR);                                        \
+}                                                                            \
+kernel void kernel_##NAME##_f16_4(                                          \
+        global const half4 * src0, ulong offset0,                          \
+        global       half4 * dst,  ulong offsetd) {                        \
+    src0 = (global half4*)((global char*)src0 + offset0);                  \
+    dst  = (global half4*)((global char*)dst  + offsetd);                  \
+    float4 x = convert_float4(src0[get_global_id(0)]);                     \
+    dst[get_global_id(0)] = convert_half4(VEXPR);                          \
+}                                                                           \
+kernel void kernel_##NAME##_f32_nc(                                         \
+        global const char * src0, ulong offset0,                          \
+        global       char * dst,  ulong offsetd,                          \
+        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,         \
+        ulong nb0, ulong nb1, ulong nb2, ulong nb3) {                     \
+    src0 = src0 + offset0; dst = dst + offsetd;                            \
+    const int i3 = get_group_id(2);                                       \
+    const int i2 = get_group_id(1);                                       \
+    const int i1 = get_group_id(0);                                       \
+    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {  \
+        float x = *(global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00); \
+        *(global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = (SEXPR);            \
+    }                                                                     \
+}                                                                         \
+kernel void kernel_##NAME##_f16_nc(                                       \
+        global const char * src0, ulong offset0,                        \
+        global       char * dst,  ulong offsetd,                        \
+        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,       \
+        ulong nb0, ulong nb1, ulong nb2, ulong nb3) {                   \
+    src0 = src0 + offset0; dst = dst + offsetd;                          \
+    const int i3 = get_group_id(2);                                     \
+    const int i2 = get_group_id(1);                                     \
+    const int i1 = get_group_id(0);                                     \
+    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {\
+        float x = *(global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00); \
+        *(global half *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = (SEXPR);            \
+    }                                                                   \
+}
+
+UNARY_EXT(sgn,          sign(x),                                           sign(x))
+UNARY_EXT(step,         x > 0.0f ? 1.0f : 0.0f,                            select((float4)0.0f, (float4)1.0f, x > 0.0f))
+UNARY_EXT(elu,          x > 0.0f ? x : expm1(x),                           select(expm1(x), x, x > 0.0f))
+UNARY_EXT(hardswish,    x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f)),     x * fmin((float4)1.0f, fmax((float4)0.0f, (x + 3.0f) / 6.0f)))
+UNARY_EXT(hardsigmoid,  fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f)),         fmin((float4)1.0f, fmax((float4)0.0f, (x + 3.0f) / 6.0f)))
+UNARY_EXT(floor,        floor(x),                                          floor(x))
+UNARY_EXT(ceil,         ceil(x),                                           ceil(x))
+UNARY_EXT(round,        round(x),                                          round(x))
+UNARY_EXT(trunc,        trunc(x),                                          trunc(x))
