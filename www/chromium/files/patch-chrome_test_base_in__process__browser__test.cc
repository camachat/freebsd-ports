<<<<<<< HEAD
--- chrome/test/base/in_process_browser_test.cc.orig	2021-01-18 21:28:52 UTC
+++ chrome/test/base/in_process_browser_test.cc
@@ -79,6 +79,10 @@
=======
--- chrome/test/base/in_process_browser_test.cc.orig	2021-03-12 23:57:20 UTC
+++ chrome/test/base/in_process_browser_test.cc
@@ -80,6 +80,10 @@
>>>>>>> upstream/main
 #include "chrome/test/base/scoped_bundle_swizzler_mac.h"
 #endif
 
+#if defined(OS_FREEBSD)
+#include <signal.h>
+#endif
+
 #if defined(OS_WIN)
 #include "base/win/scoped_com_initializer.h"
 #include "base/win/windows_version.h"
<<<<<<< HEAD
@@ -304,7 +308,7 @@ void InProcessBrowserTest::SetUp() {
=======
@@ -305,7 +309,7 @@ void InProcessBrowserTest::SetUp() {
>>>>>>> upstream/main
   // Cookies). Without this on Mac and Linux, many tests will hang waiting for a
   // user to approve KeyChain/kwallet access. On Windows this is not needed as
   // OS APIs never block.
-#if defined(OS_MAC) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_MAC) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   OSCryptMocker::SetUp();
 #endif
 
<<<<<<< HEAD
@@ -364,7 +368,7 @@ void InProcessBrowserTest::TearDown() {
=======
@@ -368,7 +372,7 @@ void InProcessBrowserTest::TearDown() {
>>>>>>> upstream/main
   com_initializer_.reset();
 #endif
   BrowserTestBase::TearDown();
-#if defined(OS_MAC) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_MAC) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   OSCryptMocker::TearDown();
 #endif
 
