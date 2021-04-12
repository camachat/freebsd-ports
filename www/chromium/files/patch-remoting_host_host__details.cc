<<<<<<< HEAD
--- remoting/host/host_details.cc.orig	2020-11-13 06:36:46 UTC
+++ remoting/host/host_details.cc
@@ -22,7 +22,7 @@ std::string GetHostOperatingSystemName() {
   return "Mac";
 #elif defined(OS_CHROMEOS)
   return "ChromeOS";
-#elif defined(OS_LINUX)
+#elif defined(OS_LINUX) || defined(OS_BSD)
   return "Linux";
 #elif defined(OS_ANDROID)
   return "Android";
=======
--- remoting/host/host_details.cc.orig	2021-03-12 23:57:28 UTC
+++ remoting/host/host_details.cc
@@ -25,6 +25,8 @@ std::string GetHostOperatingSystemName() {
   return "ChromeOS";
 #elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
   return "Linux";
+#elif defined(OS_FREEBSD)
+  return "FreeBSD";
 #elif defined(OS_ANDROID)
   return "Android";
 #else
>>>>>>> upstream/main
