--- src/gallium/drivers/llvmpipe/lp_test_printf.c.orig	2023-09-22 14:07:54 UTC
+++ src/gallium/drivers/llvmpipe/lp_test_printf.c
@@ -96,7 +96,7 @@ test_printf(unsigned verbose, FILE *fp,
    boolean success = TRUE;
 
    context = LLVMContextCreate();
-#if LLVM_VERSION_MAJOR >= 15
+#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR < 17
    LLVMContextSetOpaquePointers(context, false);
 #endif
    gallivm = gallivm_create("test_module", context, NULL);
