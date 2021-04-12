<<<<<<< HEAD
--- services/device/hid/hid_service.cc.orig	2020-11-13 06:36:46 UTC
+++ services/device/hid/hid_service.cc
@@ -16,6 +16,8 @@
=======
--- services/device/hid/hid_service.cc.orig	2021-03-12 23:57:28 UTC
+++ services/device/hid/hid_service.cc
@@ -18,6 +18,8 @@
>>>>>>> upstream/main
 
 #if (defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(USE_UDEV)
 #include "services/device/hid/hid_service_linux.h"
+#elif defined(OS_FREEBSD)
+#include "services/device/hid/hid_service_freebsd.h"
 #elif defined(OS_MAC)
 #include "services/device/hid/hid_service_mac.h"
 #elif defined(OS_WIN)
<<<<<<< HEAD
@@ -36,6 +38,8 @@ constexpr base::TaskTraits HidService::kBlockingTaskTr
=======
@@ -58,6 +60,8 @@ constexpr base::TaskTraits HidService::kBlockingTaskTr
>>>>>>> upstream/main
 std::unique_ptr<HidService> HidService::Create() {
 #if (defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(USE_UDEV)
   return base::WrapUnique(new HidServiceLinux());
+#elif defined(OS_FREEBSD)
+  return base::WrapUnique(new HidServiceFreeBSD());
 #elif defined(OS_MAC)
   return base::WrapUnique(new HidServiceMac());
 #elif defined(OS_WIN)
