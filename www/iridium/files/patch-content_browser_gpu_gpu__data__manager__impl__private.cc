--- content/browser/gpu/gpu_data_manager_impl_private.cc.orig	2025-06-19 07:37:57 UTC
+++ content/browser/gpu/gpu_data_manager_impl_private.cc
@@ -1697,7 +1697,7 @@ void GpuDataManagerImplPrivate::RecordCompositingMode(
   UMA_HISTOGRAM_ENUMERATION("GPU.CompositingMode", compositing_mode);
 }
 
-#if BUILDFLAG(IS_LINUX)
+#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_BSD)
 bool GpuDataManagerImplPrivate::IsGpuMemoryBufferNV12Supported() {
   return is_gpu_memory_buffer_NV12_supported_;
 }
