<<<<<<< HEAD
--- content/browser/gpu/compositor_util.cc.orig	2020-11-16 14:31:58 UTC
+++ content/browser/gpu/compositor_util.cc
@@ -131,11 +131,11 @@ const GpuFeatureData GetGpuFeatureData(
     {"video_decode",
      SafeGetFeatureStatus(gpu_feature_info,
                           gpu::GPU_FEATURE_TYPE_ACCELERATED_VIDEO_DECODE),
-#if defined(OS_LINUX) && !defined(OS_ANDROID) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_ANDROID) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
      !command_line.HasSwitch(switches::kEnableAcceleratedVideoDecode),
 #else
      command_line.HasSwitch(switches::kDisableAcceleratedVideoDecode),
-#endif  // defined(OS_LINUX) && !defined(OS_ANDROID) && !defined(OS_CHROMEOS)
+#endif  // (defined(OS_LINUX) && !defined(OS_ANDROID) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
      DisableInfo::Problem(
          "Accelerated video decode has been disabled, either via blocklist, "
          "about:flags or the command line."),
=======
--- content/browser/gpu/compositor_util.cc.orig	2021-03-12 23:57:24 UTC
+++ content/browser/gpu/compositor_util.cc
@@ -127,11 +127,11 @@ const GpuFeatureData GetGpuFeatureData(
     {"video_decode",
      SafeGetFeatureStatus(gpu_feature_info,
                           gpu::GPU_FEATURE_TYPE_ACCELERATED_VIDEO_DECODE),
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && !defined(OS_ANDROID)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && !defined(OS_ANDROID)
      !base::FeatureList::IsEnabled(media::kVaapiVideoDecodeLinux),
 #else
      command_line.HasSwitch(switches::kDisableAcceleratedVideoDecode),
-#endif  // ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) &&
+#endif  // ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) &&
         // !defined(OS_ANDROID)
      DisableInfo::Problem(
          "Accelerated video decode has been disabled, either via blocklist, "
>>>>>>> upstream/main
