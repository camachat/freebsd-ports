<<<<<<< HEAD
--- chrome/browser/ui/views/download/download_item_view.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/views/download/download_item_view.cc
@@ -190,7 +190,7 @@ bool UseNewWarnings() {
=======
--- chrome/browser/ui/views/download/download_item_view.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/views/download/download_item_view.cc
@@ -189,7 +189,7 @@ bool UseNewWarnings() {
 }
 
 int GetFilenameStyle(const views::Label& label) {
-#if !defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if !defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_BSD)
   if (UseNewWarnings())
     return STYLE_EMPHASIZED;
 #endif
@@ -197,7 +197,7 @@ int GetFilenameStyle(const views::Label& label) {
>>>>>>> upstream/main
 }
 
 int GetFilenameStyle(const views::StyledLabel& label) {
-#if !defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if !defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_BSD)
   if (UseNewWarnings())
     return STYLE_EMPHASIZED;
 #endif
