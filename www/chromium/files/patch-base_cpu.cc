<<<<<<< HEAD
--- base/cpu.cc.orig	2021-01-18 21:28:44 UTC
=======
--- base/cpu.cc.orig	2021-03-12 23:57:15 UTC
>>>>>>> upstream/main
+++ base/cpu.cc
@@ -16,7 +16,7 @@
 
 #include "base/stl_util.h"
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD) || \
     defined(OS_AIX)
 #include "base/containers/flat_set.h"
 #include "base/files/file_util.h"
<<<<<<< HEAD
@@ -182,6 +182,14 @@ std::string* CpuInfoBrand() {
 
   return brand;
=======
@@ -212,6 +212,14 @@ const ProcCpuInfo& ParseProcCpu() {
 
   return *info;
>>>>>>> upstream/main
 }
+#elif defined(OS_BSD)
+std::string* CpuInfoBrand() {
+  static std::string* brand = []() {
+    return new std::string(SysInfo::CPUModelName());
+  }();
+
+  return brand;
+}
 #endif  // defined(ARCH_CPU_ARM_FAMILY) && (defined(OS_ANDROID) ||
         // defined(OS_LINUX) || defined(OS_CHROMEOS))
 
<<<<<<< HEAD
@@ -305,7 +313,7 @@ void CPU::Initialize() {
     }
   }
 #elif defined(ARCH_CPU_ARM_FAMILY)
-#if defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   cpu_brand_ = *CpuInfoBrand();
 #elif defined(OS_WIN)
   // Windows makes high-resolution thread timing information available in
=======
@@ -348,6 +356,8 @@ void CPU::Initialize() {
   has_bti_ = hwcap2 & HWCAP2_BTI;
 #endif
 
+#elif defined(OS_BSD)
+  cpu_brand_ = *CpuInfoBrand();
 #elif defined(OS_WIN)
   // Windows makes high-resolution thread timing information available in
   // user-space.
>>>>>>> upstream/main
