--- src/gallium/drivers/llvmpipe/lp_context.c.orig	2023-09-22 14:07:54 UTC
+++ src/gallium/drivers/llvmpipe/lp_context.c
@@ -263,7 +263,7 @@ llvmpipe_create_context(struct pipe_screen *screen, vo
    if (!llvmpipe->context)
       goto fail;
 
-#if LLVM_VERSION_MAJOR >= 15
+#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR < 17
    LLVMContextSetOpaquePointers(llvmpipe->context, false);
 #endif
 
