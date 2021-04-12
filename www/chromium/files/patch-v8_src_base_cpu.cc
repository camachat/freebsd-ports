<<<<<<< HEAD
--- v8/src/base/cpu.cc.orig	2020-11-13 06:42:28 UTC
+++ v8/src/base/cpu.cc
@@ -534,6 +534,7 @@ CPU::CPU()
=======
--- v8/src/base/cpu.cc.orig	2021-03-13 00:03:47 UTC
+++ v8/src/base/cpu.cc
@@ -545,6 +545,7 @@ CPU::CPU()
>>>>>>> upstream/main
 
 #if V8_OS_LINUX
 
+#if V8_OS_LINUX
   CPUInfo cpu_info;
 
   // Extract implementor from the "CPU implementer" field.
<<<<<<< HEAD
@@ -567,6 +568,7 @@ CPU::CPU()
=======
@@ -578,6 +579,7 @@ CPU::CPU()
>>>>>>> upstream/main
     }
     delete[] part;
   }
+#endif
 
   // Extract architecture from the "CPU Architecture" field.
   // The list is well-known, unlike the the output of
