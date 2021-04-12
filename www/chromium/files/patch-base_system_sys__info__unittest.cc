<<<<<<< HEAD
--- base/system/sys_info_unittest.cc.orig	2021-01-18 21:28:45 UTC
+++ base/system/sys_info_unittest.cc
@@ -50,13 +50,13 @@ TEST_F(SysInfoTest, AmountOfMem) {
=======
--- base/system/sys_info_unittest.cc.orig	2021-03-12 23:57:15 UTC
+++ base/system/sys_info_unittest.cc
@@ -62,13 +62,13 @@ TEST_F(SysInfoTest, AmountOfMem) {
>>>>>>> upstream/main
   EXPECT_GE(SysInfo::AmountOfVirtualMemory(), 0);
 }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #define MAYBE_AmountOfAvailablePhysicalMemory \
   DISABLED_AmountOfAvailablePhysicalMemory
 #else
 #define MAYBE_AmountOfAvailablePhysicalMemory AmountOfAvailablePhysicalMemory
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 TEST_F(SysInfoTest, MAYBE_AmountOfAvailablePhysicalMemory) {
   // Note: info is in _K_bytes.
   SystemMemoryInfoKB info;
<<<<<<< HEAD
@@ -87,7 +87,7 @@ TEST_F(SysInfoTest, MAYBE_AmountOfAvailablePhysicalMem
=======
@@ -99,7 +99,7 @@ TEST_F(SysInfoTest, MAYBE_AmountOfAvailablePhysicalMem
>>>>>>> upstream/main
   EXPECT_GT(amount, static_cast<int64_t>(info.free) * 1024);
   EXPECT_LT(amount / 1024, info.total);
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID)
<<<<<<< HEAD
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSDD)
 
 TEST_F(SysInfoTest, AmountOfFreeDiskSpace) {
   // We aren't actually testing that it's correct, just that it's sane.
@@ -137,7 +137,7 @@ TEST_F(SysInfoTest, NestedVolumesAmountOfTotalDiskSpac
=======
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 
 TEST_F(SysInfoTest, AmountOfFreeDiskSpace) {
   // We aren't actually testing that it's correct, just that it's sane.
@@ -149,7 +149,7 @@ TEST_F(SysInfoTest, NestedVolumesAmountOfTotalDiskSpac
>>>>>>> upstream/main
 }
 #endif  // defined(OS_FUCHSIA)
 
-#if defined(OS_WIN) || defined(OS_APPLE) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS) || defined(OS_FUCHSIA)
 TEST_F(SysInfoTest, OperatingSystemVersionNumbers) {
   int32_t os_major_version = -1;
<<<<<<< HEAD
@@ -198,7 +198,7 @@ TEST_F(SysInfoTest, GetHardwareInfo) {
=======
@@ -210,7 +210,7 @@ TEST_F(SysInfoTest, GetHardwareInfo) {
>>>>>>> upstream/main
   EXPECT_TRUE(IsStringUTF8(hardware_info->model));
   bool empty_result_expected =
 #if defined(OS_ANDROID) || defined(OS_APPLE) || defined(OS_WIN) || \
-    defined(OS_LINUX) || defined(OS_CHROMEOS)
+    defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
       false;
 #else
       true;
