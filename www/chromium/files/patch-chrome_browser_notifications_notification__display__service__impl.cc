<<<<<<< HEAD
--- chrome/browser/notifications/notification_display_service_impl.cc.orig	2021-01-18 21:28:50 UTC
+++ chrome/browser/notifications/notification_display_service_impl.cc
@@ -29,7 +29,7 @@
=======
--- chrome/browser/notifications/notification_display_service_impl.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/notifications/notification_display_service_impl.cc
@@ -30,7 +30,7 @@
>>>>>>> upstream/main
 #include "chrome/browser/extensions/api/notifications/extension_notification_handler.h"
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) || \
     defined(OS_WIN)
 #include "chrome/browser/send_tab_to_self/desktop_notification_handler.h"
 #include "chrome/browser/sharing/sharing_notification_handler.h"
<<<<<<< HEAD
@@ -63,7 +63,7 @@ NotificationDisplayServiceImpl* NotificationDisplaySer
 // static
 void NotificationDisplayServiceImpl::RegisterProfilePrefs(
     user_prefs::PrefRegistrySyncable* registry) {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
   registry->RegisterBooleanPref(prefs::kAllowNativeNotifications, true);
 #endif
 }
@@ -78,7 +78,7 @@ NotificationDisplayServiceImpl::NotificationDisplaySer
=======
@@ -66,7 +66,7 @@ void NotificationDisplayServiceImpl::RegisterProfilePr
     user_prefs::PrefRegistrySyncable* registry) {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   registry->RegisterBooleanPref(prefs::kAllowNativeNotifications, true);
 #endif
 }
@@ -81,7 +81,7 @@ NotificationDisplayServiceImpl::NotificationDisplaySer
>>>>>>> upstream/main
     AddNotificationHandler(NotificationHandler::Type::WEB_PERSISTENT,
                            std::make_unique<PersistentNotificationHandler>());
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) || \
     defined(OS_WIN)
     AddNotificationHandler(
         NotificationHandler::Type::SEND_TAB_TO_SELF,
