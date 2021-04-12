<<<<<<< HEAD
--- net/base/network_change_notifier.cc.orig	2021-01-18 21:29:01 UTC
+++ net/base/network_change_notifier.cc
@@ -35,7 +35,7 @@
 #include "net/base/network_change_notifier_linux.h"
 #elif defined(OS_APPLE)
 #include "net/base/network_change_notifier_mac.h"
-#elif defined(OS_CHROMEOS) || defined(OS_ANDROID)
+#elif defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD)
 #include "net/base/network_change_notifier_posix.h"
 #elif defined(OS_FUCHSIA)
 #include "net/base/network_change_notifier_fuchsia.h"
@@ -249,8 +249,11 @@ std::unique_ptr<NetworkChangeNotifier> NetworkChangeNo
 #elif defined(OS_FUCHSIA)
   return std::make_unique<NetworkChangeNotifierFuchsia>(
       fuchsia::hardware::ethernet::Features());
+#elif defined(OS_BSD)
+  return std::make_unique<MockNetworkChangeNotifier>(
+      std::make_unique<SystemDnsConfigChangeNotifier>(
+          nullptr /* task_runner */, nullptr /* dns_config_service */));
 #else
-  NOTIMPLEMENTED();
   return NULL;
 #endif
 }
=======
--- net/base/network_change_notifier.cc.orig	2021-03-12 23:57:27 UTC
+++ net/base/network_change_notifier.cc
@@ -38,7 +38,7 @@
 #include "net/base/network_change_notifier_linux.h"
 #elif defined(OS_APPLE)
 #include "net/base/network_change_notifier_mac.h"
-#elif BUILDFLAG(IS_CHROMEOS_ASH) || defined(OS_ANDROID)
+#elif BUILDFLAG(IS_CHROMEOS_ASH) || defined(OS_ANDROID) || defined(OS_BSD)
 #include "net/base/network_change_notifier_posix.h"
 #elif defined(OS_FUCHSIA)
 #include "net/base/network_change_notifier_fuchsia.h"
@@ -241,7 +241,7 @@ std::unique_ptr<NetworkChangeNotifier> NetworkChangeNo
   // service in a separate process.
   return std::make_unique<NetworkChangeNotifierPosix>(initial_type,
                                                       initial_subtype);
-#elif BUILDFLAG(IS_CHROMEOS_ASH)
+#elif BUILDFLAG(IS_CHROMEOS_ASH) || defined(OS_BSD)
   return std::make_unique<NetworkChangeNotifierPosix>(initial_type,
                                                       initial_subtype);
 #elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
>>>>>>> upstream/main
