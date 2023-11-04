--- src/gallium/drivers/llvmpipe/lp_test_conv.c.orig	2023-09-22 14:07:54 UTC
+++ src/gallium/drivers/llvmpipe/lp_test_conv.c
@@ -223,7 +223,7 @@ test_one(unsigned verbose,
    }
 
    context = LLVMContextCreate();
-#if LLVM_VERSION_MAJOR >= 15
+#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR < 17
    LLVMContextSetOpaquePointers(context, false);
 #endif
    gallivm = gallivm_create("test_module", context, NULL);
