<<<<<<< HEAD
--- gpu/vulkan/vulkan_function_pointers.h.orig	2021-01-18 21:28:59 UTC
+++ gpu/vulkan/vulkan_function_pointers.h
@@ -235,10 +235,10 @@ struct COMPONENT_EXPORT(VULKAN) VulkanFunctionPointers
=======
--- gpu/vulkan/vulkan_function_pointers.h.orig	2021-03-12 23:57:25 UTC
+++ gpu/vulkan/vulkan_function_pointers.h
@@ -240,10 +240,10 @@ struct COMPONENT_EXPORT(VULKAN) VulkanFunctionPointers
>>>>>>> upstream/main
       vkGetAndroidHardwareBufferPropertiesANDROID;
 #endif  // defined(OS_ANDROID)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
   VulkanFunction<PFN_vkGetSemaphoreFdKHR> vkGetSemaphoreFdKHR;
   VulkanFunction<PFN_vkImportSemaphoreFdKHR> vkImportSemaphoreFdKHR;
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 
 #if defined(OS_WIN)
   VulkanFunction<PFN_vkGetSemaphoreWin32HandleKHR> vkGetSemaphoreWin32HandleKHR;
<<<<<<< HEAD
@@ -246,10 +246,10 @@ struct COMPONENT_EXPORT(VULKAN) VulkanFunctionPointers
=======
@@ -251,10 +251,10 @@ struct COMPONENT_EXPORT(VULKAN) VulkanFunctionPointers
>>>>>>> upstream/main
       vkImportSemaphoreWin32HandleKHR;
 #endif  // defined(OS_WIN)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
   VulkanFunction<PFN_vkGetMemoryFdKHR> vkGetMemoryFdKHR;
   VulkanFunction<PFN_vkGetMemoryFdPropertiesKHR> vkGetMemoryFdPropertiesKHR;
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 
 #if defined(OS_WIN)
   VulkanFunction<PFN_vkGetMemoryWin32HandleKHR> vkGetMemoryWin32HandleKHR;
<<<<<<< HEAD
@@ -954,7 +954,7 @@ ALWAYS_INLINE VkResult vkGetAndroidHardwareBufferPrope
=======
@@ -291,10 +291,10 @@ struct COMPONENT_EXPORT(VULKAN) VulkanFunctionPointers
   VulkanFunction<PFN_vkGetSwapchainImagesKHR> vkGetSwapchainImagesKHR;
   VulkanFunction<PFN_vkQueuePresentKHR> vkQueuePresentKHR;
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   VulkanFunction<PFN_vkGetImageDrmFormatModifierPropertiesEXT>
       vkGetImageDrmFormatModifierPropertiesEXT;
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 };
 
 }  // namespace gpu
@@ -985,7 +985,7 @@ ALWAYS_INLINE VkResult vkGetAndroidHardwareBufferPrope
>>>>>>> upstream/main
 }
 #endif  // defined(OS_ANDROID)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 ALWAYS_INLINE VkResult
 vkGetSemaphoreFdKHR(VkDevice device,
                     const VkSemaphoreGetFdInfoKHR* pGetFdInfo,
<<<<<<< HEAD
@@ -968,7 +968,7 @@ ALWAYS_INLINE VkResult vkImportSemaphoreFdKHR(
=======
@@ -999,7 +999,7 @@ ALWAYS_INLINE VkResult vkImportSemaphoreFdKHR(
>>>>>>> upstream/main
   return gpu::GetVulkanFunctionPointers()->vkImportSemaphoreFdKHR(
       device, pImportSemaphoreFdInfo);
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 
 #if defined(OS_WIN)
 ALWAYS_INLINE VkResult vkGetSemaphoreWin32HandleKHR(
<<<<<<< HEAD
@@ -987,7 +987,7 @@ vkImportSemaphoreWin32HandleKHR(VkDevice device,
=======
@@ -1018,7 +1018,7 @@ vkImportSemaphoreWin32HandleKHR(VkDevice device,
>>>>>>> upstream/main
 }
 #endif  // defined(OS_WIN)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 ALWAYS_INLINE VkResult vkGetMemoryFdKHR(VkDevice device,
                                         const VkMemoryGetFdInfoKHR* pGetFdInfo,
                                         int* pFd) {
<<<<<<< HEAD
@@ -1002,7 +1002,7 @@ vkGetMemoryFdPropertiesKHR(VkDevice device,
=======
@@ -1033,7 +1033,7 @@ vkGetMemoryFdPropertiesKHR(VkDevice device,
>>>>>>> upstream/main
   return gpu::GetVulkanFunctionPointers()->vkGetMemoryFdPropertiesKHR(
       device, handleType, fd, pMemoryFdProperties);
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 
 #if defined(OS_WIN)
 ALWAYS_INLINE VkResult vkGetMemoryWin32HandleKHR(
<<<<<<< HEAD
@@ -1082,4 +1082,4 @@ ALWAYS_INLINE VkResult vkQueuePresentKHR(VkQueue queue
                                                              pPresentInfo);
 }
 
=======
@@ -1113,7 +1113,7 @@ ALWAYS_INLINE VkResult vkQueuePresentKHR(VkQueue queue
                                                              pPresentInfo);
 }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 ALWAYS_INLINE VkResult vkGetImageDrmFormatModifierPropertiesEXT(
     VkDevice device,
     VkImage image,
@@ -1121,6 +1121,6 @@ ALWAYS_INLINE VkResult vkGetImageDrmFormatModifierProp
   return gpu::GetVulkanFunctionPointers()
       ->vkGetImageDrmFormatModifierPropertiesEXT(device, image, pProperties);
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
>>>>>>> upstream/main
-#endif  // GPU_VULKAN_VULKAN_FUNCTION_POINTERS_H_
\ No newline at end of file
+#endif  // GPU_VULKAN_VULKAN_FUNCTION_POINTERS_H_
