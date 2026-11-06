--- src/hotspot/share/runtime/notificationThread.cpp.orig	2022-02-04 21:44:09.000000000 -0600
+++ src/hotspot/share/runtime/notificationThread.cpp	2022-04-25 13:44:10.965643000 -0500
@@ -104,10 +104,10 @@ void NotificationThread::notification_thread_entry(Jav
       // only the first recognized bit of work, to avoid frequently true early
       // tests from potentially starving later work.  Hence the use of
       // arithmetic-or to combine results; we don't want short-circuiting.
-      while (((sensors_changed = LowMemoryDetector::has_pending_requests()) |
-              (has_dcmd_notification_event = DCmdFactory::has_pending_jmx_notification()) |
+      while (((sensors_changed = LowMemoryDetector::has_pending_requests()) ||
+              (has_dcmd_notification_event = DCmdFactory::has_pending_jmx_notification()) ||
               (has_gc_notification_event = GCNotifier::has_event()))
-             == 0) {
+             == false) {
         // Wait until notified that there is some work to do.
         ml.wait(0);
       }
