<<<<<<< HEAD
--- chrome/browser/intranet_redirect_detector.h.orig	2020-11-16 14:19:24 UTC
=======
--- chrome/browser/intranet_redirect_detector.h.orig	2021-03-12 23:57:18 UTC
>>>>>>> upstream/main
+++ chrome/browser/intranet_redirect_detector.h
@@ -27,7 +27,7 @@ class SimpleURLLoader;
 
 class PrefRegistrySimple;
 
-#if !(defined(OS_MAC) || defined(OS_WIN) || defined(OS_LINUX) || \
+#if !(defined(OS_MAC) || defined(OS_WIN) || defined(OS_LINUX) || defined(OS_BSD) ||\
       defined(OS_CHROMEOS))
 #error "IntranetRedirectDetector should only be built on Desktop platforms."
 #endif
