<<<<<<< HEAD
--- components/os_crypt/os_crypt_unittest.cc.orig	2020-11-13 06:36:41 UTC
+++ components/os_crypt/os_crypt_unittest.cc
@@ -18,7 +18,7 @@
 #include "components/os_crypt/os_crypt_mocker.h"
 #include "testing/gtest/include/gtest/gtest.h"
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
=======
--- components/os_crypt/os_crypt_unittest.cc.orig	2021-03-12 23:57:22 UTC
+++ components/os_crypt/os_crypt_unittest.cc
@@ -21,7 +21,7 @@
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 #include "components/os_crypt/os_crypt_mocker_linux.h"
 #endif
 
