<<<<<<< HEAD
--- chrome/browser/media_galleries/fileapi/mtp_device_map_service.cc.orig	2020-11-13 06:36:37 UTC
=======
--- chrome/browser/media_galleries/fileapi/mtp_device_map_service.cc.orig	2021-03-12 23:57:18 UTC
>>>>>>> upstream/main
+++ chrome/browser/media_galleries/fileapi/mtp_device_map_service.cc
@@ -39,10 +39,12 @@ void MTPDeviceMapService::RegisterMTPFileSystem(
     // Note that this initializes the delegate asynchronously, but since
     // the delegate will only be used from the IO thread, it is guaranteed
     // to be created before use of it expects it to be there.
+#if !defined(OS_FREEBSD)
     CreateMTPDeviceAsyncDelegate(
         device_location, read_only,
<<<<<<< HEAD
         base::Bind(&MTPDeviceMapService::AddAsyncDelegate,
                    base::Unretained(this), device_location, read_only));
=======
         base::BindOnce(&MTPDeviceMapService::AddAsyncDelegate,
                        base::Unretained(this), device_location, read_only));
>>>>>>> upstream/main
+#endif
     mtp_device_usage_map_[key] = 0;
   }
 
