<<<<<<< HEAD
--- chrome/browser/ui/webui/settings/appearance_handler.h.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/webui/settings/appearance_handler.h
@@ -36,7 +36,7 @@ class AppearanceHandler : public SettingsPageUIHandler
   // Changes the UI theme of the browser to the default theme.
   void HandleUseDefaultTheme(const base::ListValue* args);
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_BSD) || defined(OS_LINUX)) && !defined(OS_CHROMEOS)
=======
--- chrome/browser/ui/webui/settings/appearance_handler.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/webui/settings/appearance_handler.h
@@ -39,7 +39,7 @@ class AppearanceHandler : public SettingsPageUIHandler
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   // Changes the UI theme of the browser to the system (GTK+) theme.
   void HandleUseSystemTheme(const base::ListValue* args);
 #endif
