--- src/hotspot/share/runtime/serviceThread.cpp.orig	2022-02-04 21:44:09.000000000 -0600
+++ src/hotspot/share/runtime/serviceThread.cpp	2022-04-25 16:24:07.086767000 -0500
@@ -165,20 +165,20 @@ void ServiceThread::service_thread_entry(JavaThread* j
       // only the first recognized bit of work, to avoid frequently true early
       // tests from potentially starving later work.  Hence the use of
       // arithmetic-or to combine results; we don't want short-circuiting.
-      while (((sensors_changed = (!UseNotificationThread && LowMemoryDetector::has_pending_requests())) |
-              (has_jvmti_events = _jvmti_service_queue.has_events()) |
-              (has_gc_notification_event = (!UseNotificationThread && GCNotifier::has_event())) |
-              (has_dcmd_notification_event = (!UseNotificationThread && DCmdFactory::has_pending_jmx_notification())) |
-              (stringtable_work = StringTable::has_work()) |
-              (symboltable_work = SymbolTable::has_work()) |
-              (resolved_method_table_work = ResolvedMethodTable::has_work()) |
-              (thread_id_table_work = ThreadIdTable::has_work()) |
-              (protection_domain_table_work = SystemDictionary::pd_cache_table()->has_work()) |
-              (oopstorage_work = OopStorage::has_cleanup_work_and_reset()) |
-              (oop_handles_to_release = (_oop_handle_list != NULL)) |
-              (cldg_cleanup_work = ClassLoaderDataGraph::should_clean_metaspaces_and_reset()) |
+      while (((sensors_changed = (!UseNotificationThread && LowMemoryDetector::has_pending_requests())) ||
+              (has_jvmti_events = _jvmti_service_queue.has_events()) ||
+              (has_gc_notification_event = (!UseNotificationThread && GCNotifier::has_event())) ||
+              (has_dcmd_notification_event = (!UseNotificationThread && DCmdFactory::has_pending_jmx_notification())) ||
+              (stringtable_work = StringTable::has_work()) ||
+              (symboltable_work = SymbolTable::has_work()) ||
+              (resolved_method_table_work = ResolvedMethodTable::has_work()) ||
+              (thread_id_table_work = ThreadIdTable::has_work()) ||
+              (protection_domain_table_work = SystemDictionary::pd_cache_table()->has_work()) ||
+              (oopstorage_work = OopStorage::has_cleanup_work_and_reset()) ||
+              (oop_handles_to_release = (_oop_handle_list != NULL)) ||
+              (cldg_cleanup_work = ClassLoaderDataGraph::should_clean_metaspaces_and_reset()) ||
               (jvmti_tagmap_work = JvmtiTagMap::has_object_free_events_and_reset())
-             ) == 0) {
+             ) == false) {
         // Wait until notified that there is some work to do.
         ml.wait();
       }
