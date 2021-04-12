<<<<<<< HEAD
--- base/system/sys_info_posix.cc.orig	2020-11-13 06:36:34 UTC
+++ base/system/sys_info_posix.cc
@@ -223,6 +223,8 @@ std::string SysInfo::OperatingSystemArchitecture() {
=======
--- base/system/sys_info_posix.cc.orig	2021-03-12 23:57:15 UTC
+++ base/system/sys_info_posix.cc
@@ -25,6 +25,11 @@
 #if defined(OS_ANDROID)
 #include <sys/vfs.h>
 #define statvfs statfs  // Android uses a statvfs-like statfs struct and call.
+#elif defined(OS_BSD)
+#include <sys/param.h>
+#include <sys/mount.h>
+#define statvfs statfs
+#define f_frsize f_bsize
 #else
 #include <sys/statvfs.h>
 #endif
@@ -224,6 +229,8 @@ std::string SysInfo::OperatingSystemArchitecture() {
>>>>>>> upstream/main
     arch = "x86";
   } else if (arch == "amd64") {
     arch = "x86_64";
+  } else if (arch == "arm64") {
+    arch = "aarch64";
   } else if (std::string(info.sysname) == "AIX") {
     arch = "ppc64";
   }
