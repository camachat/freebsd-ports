<<<<<<< HEAD
--- ui/gl/gl_features.cc.orig	2020-11-16 15:04:56 UTC
+++ ui/gl/gl_features.cc
@@ -14,7 +14,7 @@ namespace features {
 const base::Feature kDefaultPassthroughCommandDecoder{
   "DefaultPassthroughCommandDecoder",
 #if defined(OS_WIN) || \
-    (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(CHROMECAST_BUILD))
+    (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(CHROMECAST_BUILD)) || defined(OS_BSD)
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
       base::FEATURE_DISABLED_BY_DEFAULT
=======
--- ui/gl/gl_features.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/gl/gl_features.cc
@@ -23,8 +23,8 @@ const base::Feature kGpuVsync{"GpuVsync", base::FEATUR
 // Launched on Windows, still experimental on other platforms.
 const base::Feature kDefaultPassthroughCommandDecoder{
   "DefaultPassthroughCommandDecoder",
-#if defined(OS_WIN) ||                                       \
-    ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && \
+#if defined(OS_WIN) ||                                                          \
+    ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && \
      !defined(CHROMECAST_BUILD))
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
>>>>>>> upstream/main
