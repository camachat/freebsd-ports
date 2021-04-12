<<<<<<< HEAD
--- ui/compositor/compositor.cc.orig	2021-01-18 21:29:48 UTC
+++ ui/compositor/compositor.cc
@@ -732,7 +732,7 @@ void Compositor::CancelThroughtputTracker(TrackerId tr
   throughput_tracker_map_.erase(tracker_id);
 }
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- ui/compositor/compositor.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/compositor/compositor.cc
@@ -750,7 +750,7 @@ void Compositor::CancelThroughtputTracker(TrackerId tr
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 void Compositor::OnCompleteSwapWithNewSize(const gfx::Size& size) {
   for (auto& observer : observer_list_)
     observer.OnCompositingCompleteSwapWithNewSize(this, size);
