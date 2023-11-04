--- src/amd/llvm/ac_llvm_util.c.orig	2023-03-08 18:37:09 UTC
+++ src/amd/llvm/ac_llvm_util.c
@@ -31,9 +31,11 @@
 #include "util/u_math.h"
 #include <llvm-c/Core.h>
 #include <llvm-c/Support.h>
+#if LLVM_VERSION_MAJOR < 17
 #include <llvm-c/Transforms/IPO.h>
 #include <llvm-c/Transforms/Scalar.h>
 #include <llvm-c/Transforms/Utils.h>
+#endif
 
 #include <assert.h>
 #include <stdio.h>
@@ -220,6 +222,7 @@ static LLVMTargetMachineRef ac_create_target_machine(e
 static LLVMPassManagerRef ac_create_passmgr(LLVMTargetLibraryInfoRef target_library_info,
                                             bool check_ir)
 {
+#if LLVM_VERSION_MAJOR < 17
    LLVMPassManagerRef passmgr = LLVMCreatePassManager();
    if (!passmgr)
       return NULL;
@@ -247,6 +250,9 @@ static LLVMPassManagerRef ac_create_passmgr(LLVMTarget
    LLVMAddEarlyCSEMemSSAPass(passmgr);
    LLVMAddInstructionCombiningPass(passmgr);
    return passmgr;
+#else
+   return NULL;
+#endif
 }
 
 static const char *attr_to_str(enum ac_func_attr attr)
