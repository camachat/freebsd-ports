<<<<<<< HEAD
--- chrome/browser/memory_details_linux.cc.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/memory_details_linux.cc
@@ -69,8 +69,10 @@ ProcessData GetProcessDataMemoryInformation(
=======
--- chrome/browser/memory_details_linux.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/memory_details_linux.cc
@@ -70,8 +70,10 @@ ProcessData GetProcessDataMemoryInformation(
>>>>>>> upstream/main
 
     std::unique_ptr<base::ProcessMetrics> metrics(
         base::ProcessMetrics::CreateProcessMetrics(pid));
+#if !defined(OS_BSD)
     pmi.num_open_fds = metrics->GetOpenFdCount();
     pmi.open_fds_soft_limit = metrics->GetOpenFdSoftLimit();
+#endif
 
     process_data.processes.push_back(pmi);
   }
