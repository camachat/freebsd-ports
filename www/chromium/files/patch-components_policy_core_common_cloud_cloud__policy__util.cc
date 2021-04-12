<<<<<<< HEAD
--- components/policy/core/common/cloud/cloud_policy_util.cc.orig	2021-01-18 21:28:55 UTC
+++ components/policy/core/common/cloud/cloud_policy_util.cc
@@ -17,7 +17,7 @@
 #include <wincred.h>
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS) || defined(OS_APPLE)
+#if defined(OS_LINUX) && !defined(OS_CHROMEOS) || defined(OS_APPLE) || defined(OS_BSD)
 #include <pwd.h>
 #include <sys/types.h>
 #include <unistd.h>
@@ -32,7 +32,7 @@
 #import <SystemConfiguration/SCDynamicStoreCopySpecific.h>
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 #include <limits.h>  // For HOST_NAME_MAX
 #endif
 
@@ -66,7 +66,7 @@
 #include "base/system/sys_info.h"
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 #include "base/system/sys_info.h"
 #endif
 
@@ -75,9 +75,14 @@ namespace policy {
 namespace em = enterprise_management;
 
 std::string GetMachineName() {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
+#if defined(OS_BSD)
+  char hostname[MAXHOSTNAMELEN];
+  if (gethostname(hostname, MAXHOSTNAMELEN) == 0)
+#else
   char hostname[HOST_NAME_MAX];
   if (gethostname(hostname, HOST_NAME_MAX) == 0)  // Success.
+#endif
     return hostname;
   return std::string();
 #elif defined(OS_APPLE)
@@ -136,7 +141,7 @@ std::string GetMachineName() {
=======
--- components/policy/core/common/cloud/cloud_policy_util.cc.orig	2021-03-12 23:57:22 UTC
+++ components/policy/core/common/cloud/cloud_policy_util.cc
@@ -18,7 +18,7 @@
 #include <wincred.h>
 #endif
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_APPLE)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_APPLE) || defined(OS_BSD)
 #include <pwd.h>
 #include <sys/types.h>
 #include <unistd.h>
@@ -35,7 +35,7 @@
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 #include <limits.h>  // For HOST_NAME_MAX
 #endif
 
@@ -71,7 +71,7 @@
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 #include "base/system/sys_info.h"
 #endif
 
@@ -100,6 +100,10 @@ std::string GetMachineName() {
   if (gethostname(hostname, HOST_NAME_MAX) == 0)  // Success.
     return hostname;
   return std::string();
+#elif defined(OS_BSD)
+  char hostname[MAXHOSTNAMELEN];
+  if (gethostname(hostname, MAXHOSTNAMELEN) == 0)
+    return hostname;
 #elif defined(OS_IOS)
   // Use the Vendor ID as the machine name.
   return ios::device_util::GetVendorId();
@@ -148,7 +152,7 @@ std::string GetMachineName() {
>>>>>>> upstream/main
 }
 
 std::string GetOSVersion() {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_APPLE)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_APPLE) || defined(OS_BSD)
   return base::SysInfo::OperatingSystemVersion();
 #elif defined(OS_WIN)
   base::win::OSInfo::VersionNumber version_number =
<<<<<<< HEAD
@@ -159,7 +164,7 @@ std::string GetOSArchitecture() {
 }
 
 std::string GetOSUsername() {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS) || defined(OS_APPLE)
+#if defined(OS_LINUX) && !defined(OS_CHROMEOS) || defined(OS_APPLE) || defined(OS_BSD)
=======
@@ -171,7 +175,7 @@ std::string GetOSArchitecture() {
 }
 
 std::string GetOSUsername() {
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_APPLE)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_APPLE) || defined(OS_BSD)
>>>>>>> upstream/main
   struct passwd* creds = getpwuid(getuid());
   if (!creds || !creds->pw_name)
     return std::string();
