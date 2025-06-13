--- src/gallium/frontends/clover/llvm/invocation.cpp.orig	2025-06-13 02:18:44 UTC
+++ src/gallium/frontends/clover/llvm/invocation.cpp
@@ -312,9 +312,15 @@ namespace {
                                 ::llvm::Triple(target.triple),
                                 get_language_version(opts, device_clc_version));
 
+#if LLVM_VERSION_MAJOR >= 20
+      c->createDiagnostics(*llvm::vfs::getRealFileSystem(), new clang::TextDiagnosticPrinter(
+                              *new raw_string_ostream(r_log),
+                              &c->getDiagnosticOpts(), true));
+#else
       c->createDiagnostics(new clang::TextDiagnosticPrinter(
                               *new raw_string_ostream(r_log),
                               &c->getDiagnosticOpts(), true));
+#endif
 
       c->setTarget(clang::TargetInfo::CreateTargetInfo(
                            c->getDiagnostics(), c->getInvocation().TargetOpts));
