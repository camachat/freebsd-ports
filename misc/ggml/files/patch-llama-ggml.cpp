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
