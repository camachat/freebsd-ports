--- src/gallium/auxiliary/gallivm/lp_bld_misc.cpp.orig	2023-03-08 18:37:09 UTC
+++ src/gallium/auxiliary/gallivm/lp_bld_misc.cpp
@@ -56,7 +56,11 @@
 #include <llvm-c/ExecutionEngine.h>
 #include <llvm/Target/TargetOptions.h>
 #include <llvm/ExecutionEngine/ExecutionEngine.h>
+#if LLVM_VERSION_MAJOR < 17
 #include <llvm/ADT/Triple.h>
+#else
+#include <llvm/TargetParser/Triple.h>
+#endif
 #include <llvm/Analysis/TargetLibraryInfo.h>
 #include <llvm/ExecutionEngine/SectionMemoryManager.h>
 #include <llvm/Support/CommandLine.h>
