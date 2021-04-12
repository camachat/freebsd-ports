<<<<<<< HEAD
--- third_party/angle/src/libANGLE/renderer/vulkan/DisplayVk_api.h.orig	2021-01-18 21:31:44 UTC
+++ third_party/angle/src/libANGLE/renderer/vulkan/DisplayVk_api.h
@@ -19,13 +19,13 @@ bool IsVulkanWin32DisplayAvailable();
=======
--- third_party/angle/src/libANGLE/renderer/vulkan/DisplayVk_api.h.orig	2021-03-13 00:03:31 UTC
+++ third_party/angle/src/libANGLE/renderer/vulkan/DisplayVk_api.h
@@ -19,7 +19,7 @@ bool IsVulkanWin32DisplayAvailable();
>>>>>>> upstream/main
 DisplayImpl *CreateVulkanWin32Display(const egl::DisplayState &state);
 #endif  // defined(ANGLE_PLATFORM_WINDOWS)
 
-#if defined(ANGLE_PLATFORM_LINUX)
+#if defined(ANGLE_PLATFORM_POSIX)
 bool IsVulkanXcbDisplayAvailable();
 DisplayImpl *CreateVulkanXcbDisplay(const egl::DisplayState &state);
 
<<<<<<< HEAD
 bool IsVulkanSimpleDisplayAvailable();
 DisplayImpl *CreateVulkanSimpleDisplay(const egl::DisplayState &state);
=======
@@ -28,7 +28,7 @@ DisplayImpl *CreateVulkanSimpleDisplay(const egl::Disp
 
 bool IsVulkanHeadlessDisplayAvailable();
 DisplayImpl *CreateVulkanHeadlessDisplay(const egl::DisplayState &state);
>>>>>>> upstream/main
-#endif  // defined(ANGLE_PLATFORM_LINUX)
+#endif  // defined(ANGLE_PLATFORM_POSIX)
 
 #if defined(ANGLE_PLATFORM_ANDROID)
 bool IsVulkanAndroidDisplayAvailable();
