<<<<<<< HEAD
--- chrome/browser/chrome_browser_main_linux.cc.orig	2020-11-13 06:36:36 UTC
+++ chrome/browser/chrome_browser_main_linux.cc
@@ -81,6 +81,7 @@ void ChromeBrowserMainPartsLinux::PreProfileInit() {
=======
--- chrome/browser/chrome_browser_main_linux.cc.orig	2021-03-12 23:57:17 UTC
+++ chrome/browser/chrome_browser_main_linux.cc
@@ -82,6 +82,7 @@ void ChromeBrowserMainPartsLinux::PreProfileInit() {
>>>>>>> upstream/main
 void ChromeBrowserMainPartsLinux::PostProfileInit() {
   ChromeBrowserMainPartsPosix::PostProfileInit();
 
+#if !defined(OS_BSD)
   bool breakpad_registered;
   if (crash_reporter::IsCrashpadEnabled()) {
     // If we're using crashpad, there's no breakpad and crashpad is always
<<<<<<< HEAD
@@ -98,10 +99,11 @@ void ChromeBrowserMainPartsLinux::PostProfileInit() {
=======
@@ -99,10 +100,11 @@ void ChromeBrowserMainPartsLinux::PostProfileInit() {
>>>>>>> upstream/main
   }
   g_browser_process->metrics_service()->RecordBreakpadRegistration(
       breakpad_registered);
+#endif
 }
 
 void ChromeBrowserMainPartsLinux::PostMainMessageLoopStart() {
<<<<<<< HEAD
-#if !defined(OS_CHROMEOS)
+#if !defined(OS_CHROMEOS) && !defined(OS_BSD)
   bluez::BluezDBusManager::Initialize(nullptr /* system_bus */);
 #endif
 
@@ -109,7 +111,7 @@ void ChromeBrowserMainPartsLinux::PostMainMessageLoopS
 }
 
 void ChromeBrowserMainPartsLinux::PostDestroyThreads() {
-#if !defined(OS_CHROMEOS)
+#if !defined(OS_CHROMEOS) && !defined(OS_BSD)
=======
-#if !BUILDFLAG(IS_CHROMEOS_ASH)
+#if !BUILDFLAG(IS_CHROMEOS_ASH) && !defined(OS_BSD)
   bluez::BluezDBusManager::Initialize(nullptr /* system_bus */);
 #endif
 
@@ -110,7 +112,7 @@ void ChromeBrowserMainPartsLinux::PostMainMessageLoopS
 }
 
 void ChromeBrowserMainPartsLinux::PostDestroyThreads() {
-#if !BUILDFLAG(IS_CHROMEOS_ASH)
+#if !BUILDFLAG(IS_CHROMEOS_ASH) && !defined(OS_BSD)
>>>>>>> upstream/main
   bluez::BluezDBusManager::Shutdown();
   bluez::BluezDBusThreadManager::Shutdown();
 #endif
