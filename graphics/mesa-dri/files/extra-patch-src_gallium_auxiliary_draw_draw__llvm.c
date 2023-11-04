--- src/gallium/auxiliary/draw/draw_llvm.c.orig	2023-09-22 14:07:54 UTC
+++ src/gallium/auxiliary/draw/draw_llvm.c
@@ -784,7 +784,7 @@ draw_llvm_create(struct draw_context *draw, LLVMContex
    if (!llvm->context) {
       llvm->context = LLVMContextCreate();
 
-#if LLVM_VERSION_MAJOR >= 15
+#if LLVM_VERSION_MAJOR >= 15 && LLVM_VERSION_MAJOR < 17
       LLVMContextSetOpaquePointers(llvm->context, false);
 #endif
 
