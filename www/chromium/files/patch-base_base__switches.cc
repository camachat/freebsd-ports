<<<<<<< HEAD
--- base/base_switches.cc.orig	2020-11-13 06:36:34 UTC
+++ base/base_switches.cc
@@ -117,7 +117,7 @@ const char kDisableHighResTimer[] = "disable-highres-t
 const char kDisableUsbKeyboardDetect[]      = "disable-usb-keyboard-detect";
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS) && !BUILDFLAG(IS_LACROS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !BUILDFLAG(IS_LACROS)) || defined(OS_BSD)
 // The /dev/shm partition is too small in certain VM environments, causing
 // Chrome to fail or crash (see http://crbug.com/715363). Use this flag to
 // work-around this issue (a temporary directory will always be used to create
@@ -152,7 +152,7 @@ const char kForceFieldTrialParams[] = "force-fieldtria
=======
--- base/base_switches.cc.orig	2021-03-12 23:57:15 UTC
+++ base/base_switches.cc
@@ -120,8 +120,8 @@ const char kDisableUsbKeyboardDetect[]      = "disable
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_ASH) && \
-    !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_BSD) || (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_ASH) && \
+    !BUILDFLAG(IS_CHROMEOS_LACROS))
 // The /dev/shm partition is too small in certain VM environments, causing
 // Chrome to fail or crash (see http://crbug.com/715363). Use this flag to
 // work-around this issue (a temporary directory will always be used to create
@@ -156,7 +156,7 @@ const char kForceFieldTrialParams[] = "force-fieldtria
>>>>>>> upstream/main
 
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 // Controls whether or not retired instruction counts are surfaced for threads
 // in trace events on Linux.
 //
