<<<<<<< HEAD
--- chrome/browser/task_manager/sampling/task_group.h.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/task_manager/sampling/task_group.h
@@ -39,7 +39,7 @@ constexpr int kUnsupportedVMRefreshFlags =
=======
--- chrome/browser/task_manager/sampling/task_group.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/task_manager/sampling/task_group.h
@@ -40,7 +40,7 @@ constexpr int kUnsupportedVMRefreshFlags =
>>>>>>> upstream/main
     REFRESH_TYPE_WEBCACHE_STATS | REFRESH_TYPE_NETWORK_USAGE |
     REFRESH_TYPE_NACL | REFRESH_TYPE_IDLE_WAKEUPS | REFRESH_TYPE_HANDLES |
     REFRESH_TYPE_START_TIME | REFRESH_TYPE_CPU_TIME | REFRESH_TYPE_PRIORITY |
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
     REFRESH_TYPE_FD_COUNT |
 #endif
     REFRESH_TYPE_HARD_FAULTS;
<<<<<<< HEAD
@@ -122,9 +122,9 @@ class TaskGroup {
=======
@@ -123,9 +123,9 @@ class TaskGroup {
>>>>>>> upstream/main
   int nacl_debug_stub_port() const { return nacl_debug_stub_port_; }
 #endif  // BUILDFLAG(ENABLE_NACL)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
   int open_fd_count() const { return open_fd_count_; }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
 
   int idle_wakeups_per_second() const { return idle_wakeups_per_second_; }
 
<<<<<<< HEAD
@@ -138,9 +138,9 @@ class TaskGroup {
=======
@@ -139,9 +139,9 @@ class TaskGroup {
>>>>>>> upstream/main
   void RefreshNaClDebugStubPort(int child_process_unique_id);
   void OnRefreshNaClDebugStubPortDone(int port);
 #endif
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
   void OnOpenFdCountRefreshDone(int open_fd_count);
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
 
   void OnCpuRefreshDone(double cpu_usage);
   void OnSwappedMemRefreshDone(int64_t swapped_mem_bytes);
<<<<<<< HEAD
@@ -209,10 +209,10 @@ class TaskGroup {
=======
@@ -210,10 +210,10 @@ class TaskGroup {
>>>>>>> upstream/main
 #if BUILDFLAG(ENABLE_NACL)
   int nacl_debug_stub_port_;
 #endif  // BUILDFLAG(ENABLE_NACL)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
   // The number of file descriptors currently open by the process.
   int open_fd_count_;
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
   int idle_wakeups_per_second_;
   bool gpu_memory_has_duplicates_;
   bool is_backgrounded_;
