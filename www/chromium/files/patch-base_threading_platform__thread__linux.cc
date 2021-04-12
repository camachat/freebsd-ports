<<<<<<< HEAD
--- base/threading/platform_thread_linux.cc.orig	2020-11-13 06:36:34 UTC
+++ base/threading/platform_thread_linux.cc
@@ -24,7 +24,9 @@
=======
--- base/threading/platform_thread_linux.cc.orig	2021-03-12 23:57:15 UTC
+++ base/threading/platform_thread_linux.cc
@@ -27,7 +27,9 @@
>>>>>>> upstream/main
 
 #if !defined(OS_NACL) && !defined(OS_AIX)
 #include <pthread.h>
+#if !defined(OS_BSD)
 #include <sys/prctl.h>
+#endif
 #include <sys/resource.h>
 #include <sys/time.h>
 #include <sys/types.h>
<<<<<<< HEAD
@@ -264,7 +266,7 @@ const ThreadPriorityToNiceValuePair kThreadPriorityToN
=======
@@ -298,7 +300,7 @@ const ThreadPriorityToNiceValuePair kThreadPriorityToN
>>>>>>> upstream/main
 
 Optional<bool> CanIncreaseCurrentThreadPriorityForPlatform(
     ThreadPriority priority) {
-#if !defined(OS_NACL)
+#if !defined(OS_NACL) && !defined(OS_BSD)
   // A non-zero soft-limit on RLIMIT_RTPRIO is required to be allowed to invoke
   // pthread_setschedparam in SetCurrentThreadPriorityForPlatform().
   struct rlimit rlim;
<<<<<<< HEAD
@@ -314,7 +316,7 @@ Optional<ThreadPriority> GetCurrentThreadPriorityForPl
=======
@@ -348,7 +350,7 @@ Optional<ThreadPriority> GetCurrentThreadPriorityForPl
>>>>>>> upstream/main
 void PlatformThread::SetName(const std::string& name) {
   ThreadIdNameManager::GetInstance()->SetName(name);
 
-#if !defined(OS_NACL) && !defined(OS_AIX)
+#if !defined(OS_NACL) && !defined(OS_AIX) && !defined(OS_BSD)
   // On linux we can get the thread names to show up in the debugger by setting
   // the process name for the LWP.  We don't want to do this for the main
   // thread because that would rename the process, causing tools like killall
