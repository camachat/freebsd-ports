<<<<<<< HEAD
--- base/threading/thread_task_runner_handle.cc.orig	2020-11-13 06:36:34 UTC
=======
--- base/threading/thread_task_runner_handle.cc.orig	2021-03-12 23:57:15 UTC
>>>>>>> upstream/main
+++ base/threading/thread_task_runner_handle.cc
@@ -8,6 +8,7 @@
 #include <utility>
 
 #include "base/bind.h"
+#include "base/callback_helpers.h"
 #include "base/check_op.h"
 #include "base/lazy_instance.h"
 #include "base/run_loop.h"
<<<<<<< HEAD
@@ -38,6 +39,7 @@ bool ThreadTaskRunnerHandle::IsSet() {
   return !!thread_task_runner_tls.Pointer()->Get();
=======
@@ -33,6 +34,7 @@ const scoped_refptr<SingleThreadTaskRunner>& ThreadTas
   return current->task_runner_;
>>>>>>> upstream/main
 }
 
+#if defined(OS_BSD)
 // static
<<<<<<< HEAD
 ScopedClosureRunner ThreadTaskRunnerHandle::OverrideForTesting(
     scoped_refptr<SingleThreadTaskRunner> overriding_task_runner) {
@@ -82,6 +84,7 @@ ScopedClosureRunner ThreadTaskRunnerHandle::OverrideFo
       base::Unretained(ttrh->task_runner_.get()),
       std::move(no_running_during_override)));
 }
+#endif
 
 ThreadTaskRunnerHandle::ThreadTaskRunnerHandle(
     scoped_refptr<SingleThreadTaskRunner> task_runner)
=======
 bool ThreadTaskRunnerHandle::IsSet() {
   return !!thread_task_runner_tls.Pointer()->Get();
@@ -80,6 +82,7 @@ ThreadTaskRunnerHandleOverride::ThreadTaskRunnerHandle
   if (!allow_nested_runloop)
     no_running_during_override_.emplace();
 }
+#endif
 
 ThreadTaskRunnerHandleOverride::~ThreadTaskRunnerHandleOverride() {
   if (task_runner_to_restore_) {
>>>>>>> upstream/main
