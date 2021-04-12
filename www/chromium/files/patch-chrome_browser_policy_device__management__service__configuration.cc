<<<<<<< HEAD
--- chrome/browser/policy/device_management_service_configuration.cc.orig	2021-01-18 21:28:50 UTC
+++ chrome/browser/policy/device_management_service_configuration.cc
@@ -18,7 +18,7 @@
=======
--- chrome/browser/policy/device_management_service_configuration.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/policy/device_management_service_configuration.cc
@@ -20,7 +20,7 @@
>>>>>>> upstream/main
 #endif
 
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && !defined(OS_ANDROID))
+    ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && !defined(OS_ANDROID)) || defined(OS_BSD)
 #include "chrome/browser/enterprise/connectors/common.h"
<<<<<<< HEAD
 #include "chrome/browser/enterprise/connectors/connectors_manager.h"
 #endif
@@ -95,7 +95,7 @@ DeviceManagementServiceConfiguration::GetEncryptedRepo
 std::string
 DeviceManagementServiceConfiguration::GetReportingConnectorServerUrl() {
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && !defined(OS_ANDROID))
+    ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && !defined(OS_ANDROID)) || defined(OS_BSD)
   auto settings =
       enterprise_connectors::ConnectorsManager::GetInstance()
           ->GetReportingSettings(
=======
 #include "chrome/browser/enterprise/connectors/connectors_service.h"
 #endif
@@ -98,7 +98,7 @@ std::string
 DeviceManagementServiceConfiguration::GetReportingConnectorServerUrl(
     content::BrowserContext* context) {
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && !defined(OS_ANDROID))
+    ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && !defined(OS_ANDROID)) || defined(OS_BSD)
   auto* service =
       enterprise_connectors::ConnectorsServiceFactory::GetForBrowserContext(
           context);
>>>>>>> upstream/main
