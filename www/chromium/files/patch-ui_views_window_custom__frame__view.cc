<<<<<<< HEAD
--- ui/views/window/custom_frame_view.cc.orig	2021-01-18 21:29:49 UTC
+++ ui/views/window/custom_frame_view.cc
@@ -257,7 +257,7 @@ int CustomFrameView::NonClientTopBorderHeight() const 
 int CustomFrameView::CaptionButtonY() const {
   // Maximized buttons start at window top so that even if their images aren't
   // drawn flush with the screen edge, they still obey Fitts' Law.
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if defined(OS_LINUX) && !defined(OS_CHROMEOS) || defined(OS_BSD)
=======
--- ui/views/window/custom_frame_view.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/views/window/custom_frame_view.cc
@@ -259,7 +259,7 @@ int CustomFrameView::CaptionButtonY() const {
   // drawn flush with the screen edge, they still obey Fitts' Law.
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   return FrameBorderThickness();
 #else
   return frame_->IsMaximized() ? FrameBorderThickness() : kFrameShadowThickness;
