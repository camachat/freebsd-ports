<<<<<<< HEAD
--- chrome/browser/task_manager/sampling/task_group.cc.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/task_manager/sampling/task_group.cc
@@ -32,9 +32,9 @@ const int kBackgroundRefreshTypesMask =
=======
--- chrome/browser/task_manager/sampling/task_group.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/task_manager/sampling/task_group.cc
@@ -33,9 +33,9 @@ const int kBackgroundRefreshTypesMask =
>>>>>>> upstream/main
 #if defined(OS_WIN)
     REFRESH_TYPE_START_TIME | REFRESH_TYPE_CPU_TIME |
 #endif  // defined(OS_WIN)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
     REFRESH_TYPE_FD_COUNT |
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
 #if BUILDFLAG(ENABLE_NACL)
     REFRESH_TYPE_NACL |
 #endif  // BUILDFLAG(ENABLE_NACL)
<<<<<<< HEAD
@@ -113,9 +113,9 @@ TaskGroup::TaskGroup(
=======
@@ -114,9 +114,9 @@ TaskGroup::TaskGroup(
>>>>>>> upstream/main
 #if BUILDFLAG(ENABLE_NACL)
       nacl_debug_stub_port_(nacl::kGdbDebugStubPortUnknown),
 #endif  // BUILDFLAG(ENABLE_NACL)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
       open_fd_count_(-1),
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
       idle_wakeups_per_second_(-1),
       gpu_memory_has_duplicates_(false),
       is_backgrounded_(false) {
<<<<<<< HEAD
@@ -128,10 +128,10 @@ TaskGroup::TaskGroup(
=======
@@ -129,10 +129,10 @@ TaskGroup::TaskGroup(
>>>>>>> upstream/main
                             weak_ptr_factory_.GetWeakPtr()),
         base::BindRepeating(&TaskGroup::OnIdleWakeupsRefreshDone,
                             weak_ptr_factory_.GetWeakPtr()),
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
         base::BindRepeating(&TaskGroup::OnOpenFdCountRefreshDone,
                             weak_ptr_factory_.GetWeakPtr()),
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
         base::BindRepeating(&TaskGroup::OnProcessPriorityDone,
                             weak_ptr_factory_.GetWeakPtr()));
 
<<<<<<< HEAD
@@ -299,14 +299,14 @@ void TaskGroup::OnRefreshNaClDebugStubPortDone(int nac
=======
@@ -300,14 +300,14 @@ void TaskGroup::OnRefreshNaClDebugStubPortDone(int nac
>>>>>>> upstream/main
 }
 #endif  // BUILDFLAG(ENABLE_NACL)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
 void TaskGroup::OnOpenFdCountRefreshDone(int open_fd_count) {
   DCHECK_CURRENTLY_ON(content::BrowserThread::UI);
 
   open_fd_count_ = open_fd_count;
   OnBackgroundRefreshTypeFinished(REFRESH_TYPE_FD_COUNT);
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
 
 void TaskGroup::OnCpuRefreshDone(double cpu_usage) {
   DCHECK_CURRENTLY_ON(content::BrowserThread::UI);
