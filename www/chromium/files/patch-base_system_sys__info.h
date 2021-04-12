<<<<<<< HEAD
--- base/system/sys_info.h.orig	2021-01-18 21:28:45 UTC
+++ base/system/sys_info.h
@@ -208,6 +208,8 @@ class BASE_EXPORT SysInfo {
=======
--- base/system/sys_info.h.orig	2021-03-12 23:57:15 UTC
+++ base/system/sys_info.h
@@ -202,6 +202,8 @@ class BASE_EXPORT SysInfo {
>>>>>>> upstream/main
   // On Desktop this returns true when memory <= 512MB.
   static bool IsLowEndDevice();
 
+  static uint64_t MaxSharedMemorySize();
+
  private:
   FRIEND_TEST_ALL_PREFIXES(SysInfoTest, AmountOfAvailablePhysicalMemory);
   FRIEND_TEST_ALL_PREFIXES(debug::SystemMetricsTest, ParseMeminfo);
<<<<<<< HEAD
@@ -217,7 +219,7 @@ class BASE_EXPORT SysInfo {
=======
@@ -211,7 +213,7 @@ class BASE_EXPORT SysInfo {
>>>>>>> upstream/main
   static bool IsLowEndDeviceImpl();
   static HardwareInfo GetHardwareInfoSync();
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD) || \
     defined(OS_AIX)
   static int64_t AmountOfAvailablePhysicalMemory(
       const SystemMemoryInfoKB& meminfo);
