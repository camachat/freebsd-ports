<<<<<<< HEAD
--- base/base_switches.h.orig	2020-11-13 06:36:34 UTC
+++ base/base_switches.h
@@ -39,7 +39,7 @@ extern const char kDisableHighResTimer[];
 extern const char kDisableUsbKeyboardDetect[];
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS) && !BUILDFLAG(IS_LACROS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !BUILDFLAG(IS_LACROS)) || defined(OS_BSD)
 extern const char kDisableDevShmUsage[];
 #endif
 
@@ -55,7 +55,7 @@ extern const char kEnableIdleTracing[];
=======
--- base/base_switches.h.orig	2021-03-12 23:57:15 UTC
+++ base/base_switches.h
@@ -41,8 +41,8 @@ extern const char kDisableUsbKeyboardDetect[];
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_ASH) && \
-    !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_BSD) || (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_ASH) && \
+    !BUILDFLAG(IS_CHROMEOS_LACROS))
 extern const char kDisableDevShmUsage[];
 #endif
 
@@ -58,7 +58,7 @@ extern const char kEnableIdleTracing[];
>>>>>>> upstream/main
 extern const char kForceFieldTrialParams[];
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 extern const char kEnableThreadInstructionCount[];
 #endif
 
