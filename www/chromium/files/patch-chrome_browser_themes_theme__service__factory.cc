<<<<<<< HEAD
--- chrome/browser/themes/theme_service_factory.cc.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/themes/theme_service_factory.cc
@@ -22,7 +22,7 @@
 #include "chrome/browser/themes/theme_helper_win.h"
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 #include "chrome/browser/themes/theme_service_aura_linux.h"
 #include "ui/views/linux_ui/linux_ui.h"
 #endif
@@ -77,7 +77,7 @@ ThemeServiceFactory::~ThemeServiceFactory() {}
 
 KeyedService* ThemeServiceFactory::BuildServiceInstanceFor(
     content::BrowserContext* profile) const {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
   using ThemeService = ThemeServiceAuraLinux;
 #endif
 
@@ -89,7 +89,7 @@ KeyedService* ThemeServiceFactory::BuildServiceInstanc
 
 void ThemeServiceFactory::RegisterProfilePrefs(
     user_prefs::PrefRegistrySyncable* registry) {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/themes/theme_service_factory.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/themes/theme_service_factory.cc
@@ -25,7 +25,7 @@
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 #include "chrome/browser/themes/theme_service_aura_linux.h"
 #include "ui/views/linux_ui/linux_ui.h"
 #endif
@@ -82,7 +82,7 @@ KeyedService* ThemeServiceFactory::BuildServiceInstanc
     content::BrowserContext* profile) const {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   using ThemeService = ThemeServiceAuraLinux;
 #endif
 
@@ -96,7 +96,7 @@ void ThemeServiceFactory::RegisterProfilePrefs(
     user_prefs::PrefRegistrySyncable* registry) {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   bool default_uses_system_theme = false;
 
   const views::LinuxUI* linux_ui = views::LinuxUI::instance();
