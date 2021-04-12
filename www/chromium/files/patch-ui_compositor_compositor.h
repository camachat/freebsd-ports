<<<<<<< HEAD
--- ui/compositor/compositor.h.orig	2020-11-13 06:37:05 UTC
+++ ui/compositor/compositor.h
@@ -366,7 +366,7 @@ class COMPOSITOR_EXPORT Compositor : public cc::LayerT
   void StopThroughtputTracker(TrackerId tracker_id) override;
   void CancelThroughtputTracker(TrackerId tracker_id) override;
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- ui/compositor/compositor.h.orig	2021-03-12 23:57:48 UTC
+++ ui/compositor/compositor.h
@@ -370,7 +370,7 @@ class COMPOSITOR_EXPORT Compositor : public cc::LayerT
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   void OnCompleteSwapWithNewSize(const gfx::Size& size);
 #endif
 
