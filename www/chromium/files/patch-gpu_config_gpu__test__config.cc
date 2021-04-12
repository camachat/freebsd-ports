<<<<<<< HEAD
--- gpu/config/gpu_test_config.cc.orig	2021-01-18 21:28:59 UTC
+++ gpu/config/gpu_test_config.cc
@@ -28,7 +28,7 @@ namespace {
 GPUTestConfig::OS GetCurrentOS() {
 #if BUILDFLAG(IS_ASH)
=======
--- gpu/config/gpu_test_config.cc.orig	2021-03-12 23:57:25 UTC
+++ gpu/config/gpu_test_config.cc
@@ -28,7 +28,7 @@ namespace {
 GPUTestConfig::OS GetCurrentOS() {
 #if BUILDFLAG(IS_CHROMEOS_ASH)
>>>>>>> upstream/main
   return GPUTestConfig::kOsChromeOS;
-#elif defined(OS_LINUX) || defined(OS_OPENBSD)
+#elif defined(OS_LINUX) || defined(OS_BSD)
   return GPUTestConfig::kOsLinux;
 #elif defined(OS_WIN)
   int32_t major_version = 0;
