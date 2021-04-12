<<<<<<< HEAD
--- components/sync_device_info/local_device_info_util_linux.cc.orig	2020-11-13 06:36:42 UTC
+++ components/sync_device_info/local_device_info_util_linux.cc
@@ -37,8 +37,9 @@ std::string GetPersonalizableDeviceNameInternal() {
 #if defined(OS_CHROMEOS)
=======
--- components/sync_device_info/local_device_info_util_linux.cc.orig	2021-03-12 23:57:23 UTC
+++ components/sync_device_info/local_device_info_util_linux.cc
@@ -38,8 +38,9 @@ std::string GetPersonalizableDeviceNameInternal() {
 #if BUILDFLAG(IS_CHROMEOS_ASH)
>>>>>>> upstream/main
   return GetChromeOSDeviceNameFromType();
 #else
-  char hostname[HOST_NAME_MAX];
-  if (gethostname(hostname, HOST_NAME_MAX) == 0)  // Success.
+  int len = sysconf(_SC_HOST_NAME_MAX);
+  char hostname[len];
+  if (gethostname(hostname, _SC_HOST_NAME_MAX) == 0)  // Success.
     return hostname;
   return base::GetLinuxDistro();
 #endif
