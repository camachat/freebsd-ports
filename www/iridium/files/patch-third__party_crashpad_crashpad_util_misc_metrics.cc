--- third_party/crashpad/crashpad/util/misc/metrics.cc.orig	2025-05-07 06:48:23 UTC
+++ third_party/crashpad/crashpad/util/misc/metrics.cc
@@ -25,7 +25,7 @@
 #define METRICS_OS_NAME "Win"
 #elif BUILDFLAG(IS_ANDROID)
 #define METRICS_OS_NAME "Android"
-#elif BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
+#elif BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_BSD)
 #define METRICS_OS_NAME "Linux"
 #elif BUILDFLAG(IS_FUCHSIA)
 #define METRICS_OS_NAME "Fuchsia"
