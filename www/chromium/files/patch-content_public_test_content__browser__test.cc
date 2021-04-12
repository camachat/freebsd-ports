<<<<<<< HEAD
--- content/public/test/content_browser_test.cc.orig	2020-11-13 06:36:43 UTC
+++ content/public/test/content_browser_test.cc
@@ -26,7 +26,7 @@
 #include "base/mac/foundation_util.h"
 #endif
 
-#if !defined(OS_CHROMEOS) && defined(OS_LINUX)
+#if (!defined(OS_CHROMEOS) && defined(OS_LINUX)) || defined(OS_BSD)
 #include "ui/base/ime/init/input_method_initializer.h"
 #endif
 
@@ -87,7 +87,7 @@ void ContentBrowserTest::SetUp() {
 #endif
 
   // LinuxInputMethodContextFactory has to be initialized.
-#if !defined(OS_CHROMEOS) && defined(OS_LINUX)
+#if (!defined(OS_CHROMEOS) && defined(OS_LINUX)) || defined(OS_BSD)
   ui::InitializeInputMethodForTesting();
 #endif
 
@@ -100,7 +100,7 @@ void ContentBrowserTest::TearDown() {
   BrowserTestBase::TearDown();
 
   // LinuxInputMethodContextFactory has to be shutdown.
-#if !defined(OS_CHROMEOS) && defined(OS_LINUX)
+#if (!defined(OS_CHROMEOS) && defined(OS_LINUX)) || defined(OS_BSD)
=======
--- content/public/test/content_browser_test.cc.orig	2021-03-12 23:57:24 UTC
+++ content/public/test/content_browser_test.cc
@@ -29,7 +29,7 @@
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX)
+#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX) || defined(OS_BSD)
 #include "ui/base/ime/init/input_method_initializer.h"
 #endif
 
@@ -92,7 +92,7 @@ void ContentBrowserTest::SetUp() {
   // LinuxInputMethodContextFactory has to be initialized.
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX)
+#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX) || defined(OS_BSD)
   ui::InitializeInputMethodForTesting();
 #endif
 
@@ -107,7 +107,7 @@ void ContentBrowserTest::TearDown() {
   // LinuxInputMethodContextFactory has to be shutdown.
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX)
+#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX) || defined(OS_BSD)
>>>>>>> upstream/main
   ui::ShutdownInputMethodForTesting();
 #endif
 }
