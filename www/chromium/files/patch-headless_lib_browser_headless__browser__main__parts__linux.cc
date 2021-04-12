<<<<<<< HEAD
--- headless/lib/browser/headless_browser_main_parts_linux.cc.orig	2020-11-13 06:36:44 UTC
+++ headless/lib/browser/headless_browser_main_parts_linux.cc
@@ -10,7 +10,7 @@
 namespace headless {
 
 void HeadlessBrowserMainParts::PostMainMessageLoopStart() {
-#if defined(USE_DBUS) && !defined(OS_CHROMEOS)
+#if defined(USE_DBUS) && !defined(OS_CHROMEOS) && !defined(OS_BSD)
=======
--- headless/lib/browser/headless_browser_main_parts_linux.cc.orig	2021-03-12 23:57:25 UTC
+++ headless/lib/browser/headless_browser_main_parts_linux.cc
@@ -11,7 +11,7 @@
 namespace headless {
 
 void HeadlessBrowserMainParts::PostMainMessageLoopStart() {
-#if defined(USE_DBUS) && !BUILDFLAG(IS_CHROMEOS_ASH)
+#if defined(USE_DBUS) && !BUILDFLAG(IS_CHROMEOS_ASH) && !defined(OS_BSD)
>>>>>>> upstream/main
   bluez::BluezDBusManager::Initialize(/*system_bus=*/nullptr);
 #endif
 }
