--- src/gallium/auxiliary/gallivm/lp_bld_init.c.orig	2023-03-08 18:37:09 UTC
+++ src/gallium/auxiliary/gallivm/lp_bld_init.c
@@ -42,9 +42,11 @@
 
 #include <llvm/Config/llvm-config.h>
 #include <llvm-c/Analysis.h>
+#if LLVM_VERSION_MAJOR < 17
 #include <llvm-c/Transforms/Scalar.h>
 #if LLVM_VERSION_MAJOR >= 7
 #include <llvm-c/Transforms/Utils.h>
+#endif
 #endif
 #include <llvm-c/BitWriter.h>
 #if GALLIVM_USE_NEW_PASS == 1
