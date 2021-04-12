<<<<<<< HEAD
--- ui/aura/test/ui_controls_ozone.cc.orig	2020-11-16 14:58:19 UTC
+++ ui/aura/test/ui_controls_ozone.cc
@@ -348,7 +348,7 @@ bool UIControlsOzone::ScreenDIPToHostPixels(gfx::Point
=======
--- ui/aura/test/ui_controls_ozone.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/aura/test/ui_controls_ozone.cc
@@ -349,7 +349,7 @@ bool UIControlsOzone::ScreenDIPToHostPixels(gfx::Point
>>>>>>> upstream/main
 // To avoid multiple definitions when use_x11 && use_ozone is true, disable this
 // factory method for OS_LINUX as Linux has a factory method that decides what
 // UIControls to use based on IsUsingOzonePlatform feature flag.
-#if !defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if !defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_BSD)
 ui_controls::UIControlsAura* CreateUIControlsAura(WindowTreeHost* host) {
   return new UIControlsOzone(host);
 }
