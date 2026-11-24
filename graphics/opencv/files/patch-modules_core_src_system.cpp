- workaround for https://github.com/opencv/opencv/issues/25527

--- modules/core/src/system.cpp.orig	2026-06-05 18:50:05 UTC
+++ modules/core/src/system.cpp
@@ -636,6 +636,7 @@ struct HWFeatures
     #ifdef __aarch64__
         have[CV_CPU_NEON] = true;
         have[CV_CPU_FP16] = true;
+#if 0 // disable until https://github.com/opencv/opencv/issues/25527 is fixed
         int cpufile = open("/proc/self/auxv", O_RDONLY);
 
         if (cpufile >= 0)
@@ -662,6 +663,7 @@ struct HWFeatures
 
             close(cpufile);
         }
+#endif
     #elif defined __arm__ && defined __ANDROID__
       #if defined HAVE_CPUFEATURES
         CV_LOG_INFO(NULL, "calling android_getCpuFeatures() ...");
