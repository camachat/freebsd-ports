<<<<<<< HEAD
--- chrome/browser/ui/webui/chrome_web_ui_controller_factory.cc.orig	2021-01-18 21:28:51 UTC
+++ chrome/browser/ui/webui/chrome_web_ui_controller_factory.cc
@@ -241,7 +241,7 @@
=======
--- chrome/browser/ui/webui/chrome_web_ui_controller_factory.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/webui/chrome_web_ui_controller_factory.cc
@@ -242,7 +242,7 @@
>>>>>>> upstream/main
 #include "chrome/browser/ui/webui/app_launcher_page_ui.h"
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
<<<<<<< HEAD
 #include "chrome/browser/ui/webui/webui_js_exception/webui_js_exception_ui.h"
 #endif
 
@@ -262,12 +262,12 @@
=======
 #include "chrome/browser/ui/webui/webui_js_error/webui_js_error_ui.h"
 #endif
 
@@ -263,12 +263,12 @@
>>>>>>> upstream/main
 #include "chrome/browser/ui/webui/conflicts/conflicts_ui.h"
 #endif
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 #include "chrome/browser/ui/webui/discards/discards_ui.h"
 #endif
 
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_ANDROID)
 #include "chrome/browser/ui/webui/sandbox/sandbox_internals_ui.h"
 #endif
<<<<<<< HEAD
@@ -473,7 +473,7 @@ bool IsAboutUI(const GURL& url) {
=======
@@ -465,7 +465,7 @@ bool IsAboutUI(const GURL& url) {
>>>>>>> upstream/main
 #if !defined(OS_ANDROID)
           || url.host_piece() == chrome::kChromeUITermsHost
 #endif
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_OPENBSD)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
           || url.host_piece() == chrome::kChromeUILinuxProxyConfigHost
 #endif
<<<<<<< HEAD
 #if defined(OS_CHROMEOS)
@@ -808,7 +808,7 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
   }
 #endif  // !defined(OFFICIAL_BUILD)
 #endif  // defined(OS_CHROMEOS)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   if (url.host_piece() == chrome::kChromeUIWebUIJsExceptionHost)
     return &NewWebUI<WebUIJsExceptionUI>;
 #endif
@@ -876,7 +876,7 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
=======
 #if BUILDFLAG(IS_CHROMEOS_ASH)
@@ -811,7 +811,7 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
   }
 #endif  // !defined(OFFICIAL_BUILD)
 #endif  // BUILDFLAG(IS_CHROMEOS_ASH)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   if (url.host_piece() == chrome::kChromeUIWebUIJsErrorHost)
     return &NewWebUI<WebUIJsErrorUI>;
 #endif
@@ -873,7 +873,7 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
>>>>>>> upstream/main
   if (url.host_piece() == chrome::kChromeUINaClHost)
     return &NewWebUI<NaClUI>;
 #endif
-#if ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(TOOLKIT_VIEWS)) || \
+#if ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(TOOLKIT_VIEWS)) || defined(OS_BSD) || \
     defined(USE_AURA)
   if (url.host_piece() == chrome::kChromeUITabModalConfirmDialogHost)
     return &NewWebUI<ConstrainedWebDialogUI>;
<<<<<<< HEAD
@@ -920,19 +920,19 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
=======
@@ -917,13 +917,13 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
>>>>>>> upstream/main
     return &NewWebUI<media_router::MediaRouterInternalsUI>;
   }
 #endif
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_ANDROID)
   if (url.host_piece() == chrome::kChromeUISandboxHost) {
     return &NewWebUI<SandboxInternalsUI>;
   }
 #endif
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
   if (url.host_piece() == chrome::kChromeUIDiscardsHost)
     return &NewWebUI<DiscardsUI>;
<<<<<<< HEAD
 #endif
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+    (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
@@ -931,7 +931,7 @@ WebUIFactoryFunction GetWebUIFactoryFunction(WebUI* we
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
>>>>>>> upstream/main
   if (url.host_piece() == chrome::kChromeUIBrowserSwitchHost)
     return &NewWebUI<BrowserSwitchUI>;
 #endif
