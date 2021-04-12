<<<<<<< HEAD
--- components/viz/test/fake_output_surface.h.orig	2020-11-13 06:36:42 UTC
+++ components/viz/test/fake_output_surface.h
@@ -86,7 +86,7 @@ class FakeOutputSurface : public OutputSurface {
       UpdateVSyncParametersCallback callback) override;
   void SetDisplayTransformHint(gfx::OverlayTransform transform) override;
   gfx::OverlayTransform GetDisplayTransform() override;
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- components/viz/test/fake_output_surface.h.orig	2021-03-12 23:57:23 UTC
+++ components/viz/test/fake_output_surface.h
@@ -89,7 +89,7 @@ class FakeOutputSurface : public OutputSurface {
   gfx::OverlayTransform GetDisplayTransform() override;
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   void SetNeedsSwapSizeNotifications(
       bool needs_swap_size_notifications) override;
 #endif
