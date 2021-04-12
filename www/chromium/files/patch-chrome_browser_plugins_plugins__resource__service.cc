<<<<<<< HEAD
--- chrome/browser/plugins/plugins_resource_service.cc.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/plugins/plugins_resource_service.cc
@@ -62,7 +62,7 @@ GURL GetPluginsServerURL() {
   filename = "plugins_win.json";
 #elif defined(OS_CHROMEOS)
   filename = "plugins_chromeos.json";
-#elif defined(OS_LINUX)
+#elif defined(OS_LINUX) || defined(OS_BSD)
=======
--- chrome/browser/plugins/plugins_resource_service.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/plugins/plugins_resource_service.cc
@@ -62,7 +62,7 @@ GURL GetPluginsServerURL() {
   filename = "plugins_win.json";
 #elif BUILDFLAG(IS_CHROMEOS_ASH)
   filename = "plugins_chromeos.json";
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   filename = "plugins_linux.json";
 #elif defined(OS_MAC)
   filename = "plugins_mac.json";
