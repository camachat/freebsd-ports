<<<<<<< HEAD
--- chrome/browser/sync/device_info_sync_service_factory.cc.orig	2021-01-19 11:26:56 UTC
+++ chrome/browser/sync/device_info_sync_service_factory.cc
@@ -42,12 +42,12 @@ class DeviceInfoSyncClient : public syncer::DeviceInfo
   std::string GetSigninScopedDeviceId() const override {
 // Since the local sync backend is currently only supported on Windows, Mac and
 // Linux don't even check the pref on other os-es.
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX)
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD)
=======
--- chrome/browser/sync/device_info_sync_service_factory.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/sync/device_info_sync_service_factory.cc
@@ -47,12 +47,12 @@ class DeviceInfoSyncClient : public syncer::DeviceInfo
 // in lacros-chrome once build flag switch of lacros-chrome is
 // complete.
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
     syncer::SyncPrefs prefs(profile_->GetPrefs());
     if (prefs.IsLocalSyncEnabled()) {
       return "local_device";
     }
<<<<<<< HEAD
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX)
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD)
 
     return GetSigninScopedDeviceIdForProfile(profile_);
   }
=======
-#endif  // defined(OS_WIN) || defined(OS_MAC) || (defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || (defined(OS_LINUX) || defined(OS_BSD) ||
         // BUILDFLAG(IS_CHROMEOS_LACROS))
 
     return GetSigninScopedDeviceIdForProfile(profile_);
>>>>>>> upstream/main
