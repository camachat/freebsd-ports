<<<<<<< HEAD
--- chrome/browser/web_applications/extensions/bookmark_app_finalizer_utils.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/web_applications/extensions/bookmark_app_finalizer_utils.cc
@@ -15,7 +15,7 @@ namespace {
 
 #if !defined(OS_CHROMEOS)
 bool CanOsAddDesktopShortcuts() {
-#if defined(OS_LINUX) || defined(OS_WIN)
+#if defined(OS_LINUX) || defined(OS_WIN) || defined(OS_BSD)
=======
--- chrome/browser/web_applications/extensions/bookmark_app_finalizer_utils.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/web_applications/extensions/bookmark_app_finalizer_utils.cc
@@ -18,7 +18,7 @@ namespace {
 bool CanOsAddDesktopShortcuts() {
 // TODO(crbug.com/1052397): Revisit once build flag switch of lacros-chrome is
 // complete.
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_WIN)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_WIN) || defined(OS_BSD)
>>>>>>> upstream/main
   return true;
 #else
   return false;
