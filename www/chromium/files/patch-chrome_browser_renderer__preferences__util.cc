<<<<<<< HEAD
--- chrome/browser/renderer_preferences_util.cc.orig	2021-01-18 21:28:50 UTC
+++ chrome/browser/renderer_preferences_util.cc
@@ -34,7 +34,7 @@
 #include "ui/base/cocoa/defaults_utils.h"
 #endif
 
-#if defined(USE_AURA) && defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if defined(USE_AURA) && (defined(OS_BSD) || defined(OS_LINUX)) && !defined(OS_CHROMEOS)
 #include "chrome/browser/themes/theme_service.h"
 #include "chrome/browser/themes/theme_service_factory.h"
 #include "ui/views/linux_ui/linux_ui.h"
@@ -153,7 +153,7 @@ void UpdateFromSystemSettings(blink::RendererPreferenc
     prefs->caret_blink_interval = interval;
 #endif
 
-#if defined(USE_AURA) && defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if defined(USE_AURA) && (defined(OS_BSD) || defined(OS_LINUX)) && !defined(OS_CHROMEOS)
   views::LinuxUI* linux_ui = views::LinuxUI::instance();
   if (linux_ui) {
     if (ThemeServiceFactory::GetForProfile(profile)->UsingSystemTheme()) {
@@ -172,7 +172,7 @@ void UpdateFromSystemSettings(blink::RendererPreferenc
=======
--- chrome/browser/renderer_preferences_util.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/renderer_preferences_util.cc
@@ -35,7 +35,7 @@
 #include "ui/base/cocoa/defaults_utils.h"
 #endif
 
-#if defined(USE_AURA) && (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if defined(USE_AURA) && (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
 #include "chrome/browser/themes/theme_service.h"
 #include "chrome/browser/themes/theme_service_factory.h"
 #include "ui/views/linux_ui/linux_ui.h"
@@ -154,7 +154,7 @@ void UpdateFromSystemSettings(blink::RendererPreferenc
     prefs->caret_blink_interval = interval;
 #endif
 
-#if defined(USE_AURA) && (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if defined(USE_AURA) && (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
   views::LinuxUI* linux_ui = views::LinuxUI::instance();
   if (linux_ui) {
     if (ThemeServiceFactory::GetForProfile(profile)->UsingSystemTheme()) {
@@ -173,7 +173,7 @@ void UpdateFromSystemSettings(blink::RendererPreferenc
>>>>>>> upstream/main
   }
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_ANDROID) || defined(OS_BSD) || \
     defined(OS_WIN)
   content::UpdateFontRendererPreferencesFromSystemSettings(prefs);
 #endif
