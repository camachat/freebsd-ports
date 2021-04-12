<<<<<<< HEAD
--- media/video/video_encode_accelerator_adapter.cc.orig	2021-01-22 12:10:35 UTC
+++ media/video/video_encode_accelerator_adapter.cc
@@ -40,7 +40,7 @@ VideoEncodeAccelerator::Config SetUpVeaConfig(
       opts.bitrate.value_or(opts.frame_size.width() * opts.frame_size.height() *
                             kVEADefaultBitratePerPixel));
=======
--- media/video/video_encode_accelerator_adapter.cc.orig	2021-03-12 23:57:27 UTC
+++ media/video/video_encode_accelerator_adapter.cc
@@ -50,7 +50,7 @@ VideoEncodeAccelerator::Config SetUpVeaConfig(
   if (is_rgb)
     config.input_format = PIXEL_FORMAT_I420;
>>>>>>> upstream/main
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   if (storage_type == VideoFrame::STORAGE_DMABUFS ||
       storage_type == VideoFrame::STORAGE_GPU_MEMORY_BUFFER) {
<<<<<<< HEAD
     config.storage_type = VideoEncodeAccelerator::Config::StorageType::kDmabuf;
@@ -269,7 +269,7 @@ void VideoEncodeAcceleratorAdapter::EncodeOnAccelerato
     return;
   }
=======
     if (is_rgb)
@@ -253,7 +253,7 @@ void VideoEncodeAcceleratorAdapter::InitializeInternal
   auto vea_config =
       SetUpVeaConfig(profile_, options_, format, first_frame->storage_type());
>>>>>>> upstream/main
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // Linux/ChromeOS require a special configuration to use dmabuf storage.
<<<<<<< HEAD
   const bool is_same_storage_type = storage_type_ == frame->storage_type();
 #else
=======
   // We need to keep sending frames the same way the first frame was sent.
   // Other platforms will happily mix GpuMemoryBuffer storage with regular
>>>>>>> upstream/main
