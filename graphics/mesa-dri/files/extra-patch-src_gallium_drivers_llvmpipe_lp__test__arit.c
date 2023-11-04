--- src/gallium/drivers/llvmpipe/lp_test_arit.c.orig	2023-09-22 14:07:54 UTC
+++ src/gallium/drivers/llvmpipe/lp_test_arit.c
@@ -434,7 +434,7 @@ test_unary(unsigned verbose, FILE *fp, const struct un
    }
 
    context = LLVMContextCreate();
-#if LLVM_VERSION_MAJOR >= 15
+#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR < 17
    LLVMContextSetOpaquePointers(context, false);
 #endif
    gallivm = gallivm_create("test_module", context, NULL);
