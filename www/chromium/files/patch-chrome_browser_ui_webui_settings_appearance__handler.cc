<<<<<<< HEAD
--- chrome/browser/ui/webui/settings/appearance_handler.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/webui/settings/appearance_handler.cc
@@ -27,7 +27,7 @@ void AppearanceHandler::RegisterMessages() {
       "useDefaultTheme",
       base::BindRepeating(&AppearanceHandler::HandleUseDefaultTheme,
                           base::Unretained(this)));
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_BSD) && !defined(OS_CHROMEOS)
   web_ui()->RegisterMessageCallback(
       "useSystemTheme",
       base::BindRepeating(&AppearanceHandler::HandleUseSystemTheme,
@@ -39,7 +39,7 @@ void AppearanceHandler::HandleUseDefaultTheme(const ba
   ThemeServiceFactory::GetForProfile(profile_)->UseDefaultTheme();
 }
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_BSD) && !defined(OS_CHROMEOS)
=======
--- chrome/browser/ui/webui/settings/appearance_handler.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/webui/settings/appearance_handler.cc
@@ -31,7 +31,7 @@ void AppearanceHandler::RegisterMessages() {
                           base::Unretained(this)));
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
   web_ui()->RegisterMessageCallback(
       "useSystemTheme",
       base::BindRepeating(&AppearanceHandler::HandleUseSystemTheme,
@@ -45,7 +45,7 @@ void AppearanceHandler::HandleUseDefaultTheme(const ba
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
 void AppearanceHandler::HandleUseSystemTheme(const base::ListValue* args) {
   if (profile_->IsSupervised())
     NOTREACHED();
