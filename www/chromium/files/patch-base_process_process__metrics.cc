<<<<<<< HEAD
--- base/process/process_metrics.cc.orig	2020-11-13 06:36:34 UTC
+++ base/process/process_metrics.cc
@@ -49,7 +49,7 @@ SystemMetrics SystemMetrics::Sample() {
=======
--- base/process/process_metrics.cc.orig	2021-03-12 23:57:15 UTC
+++ base/process/process_metrics.cc
@@ -50,7 +50,7 @@ SystemMetrics SystemMetrics::Sample() {
>>>>>>> upstream/main
   SystemMetrics system_metrics;
 
   system_metrics.committed_memory_ = GetSystemCommitCharge();
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
   GetSystemMemoryInfo(&system_metrics.memory_info_);
   GetVmStatInfo(&system_metrics.vmstat_info_);
   GetSystemDiskInfo(&system_metrics.disk_info_);
<<<<<<< HEAD
@@ -68,7 +68,7 @@ std::unique_ptr<Value> SystemMetrics::ToValue() const 
=======
@@ -69,7 +69,7 @@ std::unique_ptr<Value> SystemMetrics::ToValue() const 
>>>>>>> upstream/main
   std::unique_ptr<DictionaryValue> res(new DictionaryValue());
 
   res->SetIntKey("committed_memory", static_cast<int>(committed_memory_));
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
   std::unique_ptr<DictionaryValue> meminfo = memory_info_.ToValue();
   std::unique_ptr<DictionaryValue> vmstat = vmstat_info_.ToValue();
   meminfo->MergeDictionary(vmstat.get());
<<<<<<< HEAD
@@ -119,7 +119,7 @@ double ProcessMetrics::GetPlatformIndependentCPUUsage(
=======
@@ -120,7 +120,7 @@ double ProcessMetrics::GetPlatformIndependentCPUUsage(
>>>>>>> upstream/main
 }
 #endif
 
-#if defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_AIX)
 int ProcessMetrics::CalculateIdleWakeupsPerSecond(
     uint64_t absolute_idle_wakeups) {
<<<<<<< HEAD
@@ -132,7 +132,7 @@ int ProcessMetrics::GetIdleWakeupsPerSecond() {
=======
@@ -133,7 +133,7 @@ int ProcessMetrics::GetIdleWakeupsPerSecond() {
>>>>>>> upstream/main
   NOTIMPLEMENTED();  // http://crbug.com/120488
   return 0;
 }
-#endif  // defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_CHROMEOS) ||
+#endif  // defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) ||
         // defined(OS_AIX)
 
 #if defined(OS_APPLE)
