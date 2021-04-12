<<<<<<< HEAD
--- components/viz/test/fake_output_surface.cc.orig	2020-11-13 06:36:42 UTC
+++ components/viz/test/fake_output_surface.cc
@@ -115,7 +115,7 @@ gfx::OverlayTransform FakeOutputSurface::GetDisplayTra
                                          : gfx::OVERLAY_TRANSFORM_NONE;
 }
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- components/viz/test/fake_output_surface.cc.orig	2021-03-12 23:57:23 UTC
+++ components/viz/test/fake_output_surface.cc
@@ -118,7 +118,7 @@ gfx::OverlayTransform FakeOutputSurface::GetDisplayTra
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 void FakeOutputSurface::SetNeedsSwapSizeNotifications(
     bool needs_swap_size_notifications) {}
 #endif
