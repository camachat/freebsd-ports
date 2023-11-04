--- src/gallium/drivers/llvmpipe/lp_test_format.c.orig	2023-09-22 14:07:54 UTC
+++ src/gallium/drivers/llvmpipe/lp_test_format.c
@@ -150,7 +150,7 @@ test_format_float(unsigned verbose, FILE *fp,
    unsigned i, j, k, l;
 
    context = LLVMContextCreate();
-#if LLVM_VERSION_MAJOR >= 15
+#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR < 17
    LLVMContextSetOpaquePointers(context, false);
 #endif
    gallivm = gallivm_create("test_module_float", context, NULL);
