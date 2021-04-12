<<<<<<< HEAD
--- chrome/browser/ui/webui/settings/settings_localized_strings_provider.cc.orig	2021-01-18 21:28:52 UTC
+++ chrome/browser/ui/webui/settings/settings_localized_strings_provider.cc
@@ -341,7 +341,7 @@ void AddAppearanceStrings(content::WebUIDataSource* ht
     {"minimumFont", IDS_SETTINGS_MINIMUM_FONT_SIZE_LABEL},
     {"tiny", IDS_SETTINGS_TINY_FONT_SIZE},
     {"huge", IDS_SETTINGS_HUGE_FONT_SIZE},
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
     {"systemTheme", IDS_SETTINGS_SYSTEM_THEME},
     {"useSystemTheme", IDS_SETTINGS_USE_SYSTEM_THEME},
     {"classicTheme", IDS_SETTINGS_CLASSIC_THEME},
=======
--- chrome/browser/ui/webui/settings/settings_localized_strings_provider.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/webui/settings/settings_localized_strings_provider.cc
@@ -343,7 +343,7 @@ void AddAppearanceStrings(content::WebUIDataSource* ht
     {"huge", IDS_SETTINGS_HUGE_FONT_SIZE},
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
     {"systemTheme", IDS_SETTINGS_SYSTEM_THEME},
     {"useSystemTheme", IDS_SETTINGS_USE_SYSTEM_THEME},
     {"classicTheme", IDS_SETTINGS_CLASSIC_THEME},
@@ -351,7 +351,7 @@ void AddAppearanceStrings(content::WebUIDataSource* ht
 #else
     {"resetToDefaultTheme", IDS_SETTINGS_RESET_TO_DEFAULT_THEME},
 #endif
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
     {"showWindowDecorations", IDS_SHOW_WINDOW_DECORATIONS},
 #endif
 #if defined(OS_MAC)
>>>>>>> upstream/main
