<<<<<<< HEAD
--- chrome/browser/ui/views/frame/browser_frame.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/views/frame/browser_frame.cc
@@ -38,14 +38,14 @@
 #include "components/user_manager/user_manager.h"
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 #include "ui/display/screen.h"
 #endif
 
 namespace {
 
 bool IsUsingGtkTheme(Profile* profile) {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
   return ThemeServiceFactory::GetForProfile(profile)->UsingSystemTheme();
 #else
   return false;
@@ -204,7 +204,7 @@ const ui::NativeTheme* BrowserFrame::GetNativeTheme() 
 
 void BrowserFrame::OnNativeWidgetWorkspaceChanged() {
   chrome::SaveWindowWorkspace(browser_view_->browser(), GetWorkspace());
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/ui/views/frame/browser_frame.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/views/frame/browser_frame.cc
@@ -44,7 +44,7 @@
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 #include "ui/display/screen.h"
 #endif
 
@@ -53,7 +53,7 @@ namespace {
 bool IsUsingGtkTheme(Profile* profile) {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   return ThemeServiceFactory::GetForProfile(profile)->UsingSystemTheme();
 #else
   return false;
@@ -226,7 +226,7 @@ void BrowserFrame::OnNativeWidgetWorkspaceChanged() {
                                            IsVisibleOnAllWorkspaces());
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   // If the window was sent to a different workspace, prioritize it if
   // it was sent to the current workspace and deprioritize it
   // otherwise.  This is done by MoveBrowsersInWorkspaceToFront()
