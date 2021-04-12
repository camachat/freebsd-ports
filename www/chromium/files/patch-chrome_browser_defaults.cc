<<<<<<< HEAD
--- chrome/browser/defaults.cc.orig	2020-11-13 06:36:36 UTC
+++ chrome/browser/defaults.cc
@@ -43,7 +43,7 @@ const bool kSyncAutoStarts = true;
 const bool kSyncAutoStarts = false;
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
=======
--- chrome/browser/defaults.cc.orig	2021-03-12 23:57:17 UTC
+++ chrome/browser/defaults.cc
@@ -46,7 +46,7 @@ const bool kSyncAutoStarts = false;
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 const bool kScrollEventChangesTab = true;
 #else
 const bool kScrollEventChangesTab = false;
