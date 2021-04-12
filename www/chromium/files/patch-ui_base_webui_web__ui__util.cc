<<<<<<< HEAD
--- ui/base/webui/web_ui_util.cc.orig	2021-01-18 21:29:48 UTC
+++ ui/base/webui/web_ui_util.cc
@@ -221,7 +221,7 @@ void AppendWebUiCssTextDefaults(std::string* html) {
 std::string GetFontFamily() {
   std::string font_family = l10n_util::GetStringUTF8(IDS_WEB_FONT_FAMILY);
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- ui/base/webui/web_ui_util.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/base/webui/web_ui_util.cc
@@ -224,7 +224,7 @@ std::string GetFontFamily() {
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   std::string font_name = ui::ResourceBundle::GetSharedInstance()
                               .GetFont(ui::ResourceBundle::BaseFont)
                               .GetFontName();
