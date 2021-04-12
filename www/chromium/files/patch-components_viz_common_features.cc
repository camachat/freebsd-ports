<<<<<<< HEAD
--- components/viz/common/features.cc.orig	2021-01-18 21:28:57 UTC
+++ components/viz/common/features.cc
@@ -25,7 +25,7 @@ const base::Feature kForcePreferredIntervalForVideo{
 const base::Feature kUseSkiaRenderer {
   "UseSkiaRenderer",
 #if defined(OS_WIN) || \
-    (defined(OS_LINUX) && !(defined(OS_CHROMEOS) || BUILDFLAG(IS_CHROMECAST)))
+    (defined(OS_LINUX) && !(defined(OS_CHROMEOS) || BUILDFLAG(IS_CHROMECAST))) || defined(OS_BSD)
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
       base::FEATURE_DISABLED_BY_DEFAULT
=======
--- components/viz/common/features.cc.orig	2021-03-12 23:57:23 UTC
+++ components/viz/common/features.cc
@@ -32,7 +32,7 @@ const base::Feature kEnableOverlayPrioritization {
 // Use the SkiaRenderer.
 const base::Feature kUseSkiaRenderer {
   "UseSkiaRenderer",
-#if defined(OS_WIN) || (defined(OS_LINUX) && !(BUILDFLAG(IS_CHROMEOS_ASH) || \
+#if defined(OS_WIN) || defined(OS_BSD) || (defined(OS_LINUX) && !(BUILDFLAG(IS_CHROMEOS_ASH) || \
                                                BUILDFLAG(IS_CHROMECAST)))
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
>>>>>>> upstream/main
