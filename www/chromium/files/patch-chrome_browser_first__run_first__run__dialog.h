<<<<<<< HEAD
--- chrome/browser/first_run/first_run_dialog.h.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/first_run/first_run_dialog.h
@@ -9,7 +9,7 @@
 #include "build/build_config.h"
 
 // Hide this function on platforms where the dialog does not exist.
-#if defined(OS_MAC) || (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+#if defined(OS_MAC) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/first_run/first_run_dialog.h.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/first_run/first_run_dialog.h
@@ -12,7 +12,7 @@
 // Hide this function on platforms where the dialog does not exist.
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_MAC) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if defined(OS_MAC) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
 
 class Profile;
 
