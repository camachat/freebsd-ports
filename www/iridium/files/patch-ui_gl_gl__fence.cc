--- ui/gl/gl_fence.cc.orig	2025-05-07 06:48:23 UTC
+++ ui/gl/gl_fence.cc
@@ -17,7 +17,7 @@
 #include "ui/gl/gl_surface_egl.h"
 #include "ui/gl/gl_version_info.h"
 
-#if BUILDFLAG(IS_ANDROID) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_LINUX)
+#if BUILDFLAG(IS_ANDROID) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_BSD)
 #define USE_GL_FENCE_ANDROID_NATIVE_FENCE_SYNC
 #include "ui/gl/gl_fence_android_native_fence_sync.h"
 #endif
