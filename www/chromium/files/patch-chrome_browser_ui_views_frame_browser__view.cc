<<<<<<< HEAD
--- chrome/browser/ui/views/frame/browser_view.cc.orig	2021-01-18 21:28:51 UTC
+++ chrome/browser/ui/views/frame/browser_view.cc
@@ -1506,7 +1506,7 @@ void BrowserView::ToolbarSizeChanged(bool is_animating
 void BrowserView::TabDraggingStatusChanged(bool is_dragging) {
   // TODO(crbug.com/1110266): Remove explicit OS_CHROMEOS check once OS_LINUX
   // CrOS cleanup is done.
-#if !defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if !defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   contents_web_view_->SetFastResize(is_dragging);
   if (!is_dragging) {
     // When tab dragging is ended, we need to make sure the web contents get
@@ -1887,7 +1887,7 @@ void BrowserView::UserChangedTheme(BrowserThemeChangeT
   const bool should_use_native_frame = frame_->ShouldUseNativeFrame();
 
   bool must_regenerate_frame;
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/ui/views/frame/browser_view.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/views/frame/browser_view.cc
@@ -1504,7 +1504,7 @@ void BrowserView::TabDraggingStatusChanged(bool is_dra
   // CrOS cleanup is done.
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if !(defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if !(defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
   contents_web_view_->SetFastResize(is_dragging);
   if (!is_dragging) {
     // When tab dragging is ended, we need to make sure the web contents get
@@ -1886,7 +1886,7 @@ void BrowserView::UserChangedTheme(BrowserThemeChangeT
   bool must_regenerate_frame;
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   // GTK and user theme changes can both change frame buttons, so the frame
   // always needs to be regenerated on Linux.
   must_regenerate_frame = true;
