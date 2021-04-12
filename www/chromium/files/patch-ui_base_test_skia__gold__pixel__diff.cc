<<<<<<< HEAD
--- ui/base/test/skia_gold_pixel_diff.cc.orig	2021-01-18 21:29:48 UTC
+++ ui/base/test/skia_gold_pixel_diff.cc
@@ -165,7 +165,7 @@ std::string SkiaGoldPixelDiff::GetPlatform() {
   return "windows";
 #elif defined(OS_APPLE)
   return "macOS";
-#elif defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#elif (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- ui/base/test/skia_gold_pixel_diff.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/base/test/skia_gold_pixel_diff.cc
@@ -168,7 +168,7 @@ std::string SkiaGoldPixelDiff::GetPlatform() {
   return "macOS";
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   return "linux";
 #endif
 }
