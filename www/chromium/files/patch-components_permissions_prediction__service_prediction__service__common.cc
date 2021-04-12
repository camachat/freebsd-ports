<<<<<<< HEAD
--- components/permissions/prediction_service/prediction_service_common.cc.orig	2021-01-19 11:31:48 UTC
+++ components/permissions/prediction_service/prediction_service_common.cc
@@ -10,7 +10,7 @@ namespace permissions {
 ClientFeatures_Platform GetCurrentPlatformProto() {
 #if defined(OS_WIN)
   return permissions::ClientFeatures_Platform_PLATFORM_WINDOWS;
-#elif defined(OS_LINUX)
+#elif defined(OS_LINUX) || defined(OS_BSD)
=======
--- components/permissions/prediction_service/prediction_service_common.cc.orig	2021-03-12 23:57:22 UTC
+++ components/permissions/prediction_service/prediction_service_common.cc
@@ -11,7 +11,7 @@ namespace permissions {
 ClientFeatures_Platform GetCurrentPlatformProto() {
 #if defined(OS_WIN)
   return permissions::ClientFeatures_Platform_PLATFORM_WINDOWS;
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   return permissions::ClientFeatures_Platform_PLATFORM_LINUX;
 #elif defined(OS_ANDROID)
   return permissions::ClientFeatures_Platform_PLATFORM_ANDROID;
