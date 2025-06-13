--- src/compiler/clc/clc_helpers.cpp.orig	2025-06-13 02:13:49 UTC
+++ src/compiler/clc/clc_helpers.cpp
@@ -841,9 +841,15 @@ clc_compile_to_llvm_module(LLVMContext &llvm_ctx,
    // http://www.llvm.org/bugs/show_bug.cgi?id=19735
    c->getDiagnosticOpts().ShowCarets = false;
 
+#if LLVM_VERSION_MAJOR >= 20
+   c->createDiagnostics(*llvm::vfs::getRealFileSystem(), new clang::TextDiagnosticPrinter(
+                           diag_log_stream,
+                           &c->getDiagnosticOpts()));
+#else
    c->createDiagnostics(new clang::TextDiagnosticPrinter(
                            diag_log_stream,
                            &c->getDiagnosticOpts()));
+#endif
 
    c->setTarget(clang::TargetInfo::CreateTargetInfo(
                    c->getDiagnostics(), c->getInvocation().TargetOpts));
@@ -889,7 +895,11 @@ clc_compile_to_llvm_module(LLVMContext &llvm_ctx,
    // GetResourcePath is a way to retrive the actual libclang resource dir based on a given binary
    // or library.
    auto clang_res_path =
+#if LLVM_VERSION_MAJOR >= 20
+      fs::path(Driver::GetResourcesPath(std::string(clang_path))) / "include";
+#else
       fs::path(Driver::GetResourcesPath(std::string(clang_path), CLANG_RESOURCE_DIR)) / "include";
+#endif
    free(clang_path);
 
    c->getHeaderSearchOpts().UseBuiltinIncludes = true;
