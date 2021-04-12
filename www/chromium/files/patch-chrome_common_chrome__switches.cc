<<<<<<< HEAD
--- chrome/common/chrome_switches.cc.orig	2021-01-18 21:28:52 UTC
+++ chrome/common/chrome_switches.cc
@@ -796,7 +796,7 @@ const char kAllowNaClFileHandleAPI[]        = "allow-n
=======
--- chrome/common/chrome_switches.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/chrome_switches.cc
@@ -810,7 +810,7 @@ const char kAllowNaClFileHandleAPI[]        = "allow-n
>>>>>>> upstream/main
 const char kAllowNaClSocketAPI[]            = "allow-nacl-socket-api";
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) || \
     defined(OS_WIN)
 const char kEnableNewAppMenuIcon[] = "enable-new-app-menu-icon";
 
