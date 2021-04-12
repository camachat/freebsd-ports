<<<<<<< HEAD
--- tools/gn/src/util/sys_info.cc.orig	2020-11-13 06:49:31 UTC
+++ tools/gn/src/util/sys_info.cc
@@ -33,6 +33,8 @@ std::string OperatingSystemArchitecture() {
=======
--- tools/gn/src/util/sys_info.cc.orig	2021-03-13 00:10:18 UTC
+++ tools/gn/src/util/sys_info.cc
@@ -34,6 +34,8 @@ std::string OperatingSystemArchitecture() {
>>>>>>> upstream/main
     arch = "x86_64";
   } else if (arch == "amd64") {
     arch = "x86_64";
+  } else if (arch == "arm64") {
+    arch = "aarch64";
<<<<<<< HEAD
   } else if (std::string(info.sysname) == "AIX") {
=======
   } else if (os == "AIX" || os == "OS400") {
>>>>>>> upstream/main
     arch = "ppc64";
   }
