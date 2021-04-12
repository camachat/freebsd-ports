<<<<<<< HEAD
--- base/test/launcher/test_launcher.cc.orig	2021-01-18 21:28:45 UTC
+++ base/test/launcher/test_launcher.cc
@@ -57,6 +57,7 @@
=======
--- base/test/launcher/test_launcher.cc.orig	2021-03-12 23:57:15 UTC
+++ base/test/launcher/test_launcher.cc
@@ -58,6 +58,7 @@
>>>>>>> upstream/main
 #include "testing/gtest/include/gtest/gtest.h"
 
 #if defined(OS_POSIX)
+#include <signal.h>
 #include <fcntl.h>
 
 #include "base/files/file_descriptor_watcher_posix.h"
<<<<<<< HEAD
@@ -598,7 +599,7 @@ ChildProcessResults DoLaunchChildTestProcess(
=======
@@ -599,7 +600,7 @@ ChildProcessResults DoLaunchChildTestProcess(
>>>>>>> upstream/main
 #if !defined(OS_FUCHSIA)
   options.new_process_group = true;
 #endif
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   options.kill_on_parent_death = true;
 #endif
 
<<<<<<< HEAD
@@ -1515,7 +1516,7 @@ bool TestLauncher::Init(CommandLine* command_line) {
=======
@@ -1516,7 +1517,7 @@ bool TestLauncher::Init(CommandLine* command_line) {
>>>>>>> upstream/main
   results_tracker_.AddGlobalTag("OS_IOS");
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   results_tracker_.AddGlobalTag("OS_LINUX");
 #endif
 
