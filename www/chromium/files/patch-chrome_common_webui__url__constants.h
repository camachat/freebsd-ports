<<<<<<< HEAD
--- chrome/common/webui_url_constants.h.orig	2021-01-18 21:28:52 UTC
+++ chrome/common/webui_url_constants.h
@@ -292,12 +292,12 @@ bool IsSystemWebUIHost(base::StringPiece host);
 
 #endif  // defined(OS_CHROMEOS)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 extern const char kChromeUIWebUIJsExceptionHost[];
 extern const char kChromeUIWebUIJsExceptionURL[];
=======
--- chrome/common/webui_url_constants.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/webui_url_constants.h
@@ -295,12 +295,12 @@ bool IsSystemWebUIHost(base::StringPiece host);
 
 #endif  // BUILDFLAG(IS_CHROMEOS_ASH)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 extern const char kChromeUIWebUIJsErrorHost[];
 extern const char kChromeUIWebUIJsErrorURL[];
>>>>>>> upstream/main
 #endif
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 extern const char kChromeUIDiscardsHost[];
 extern const char kChromeUIDiscardsURL[];
<<<<<<< HEAD
@@ -314,13 +314,13 @@ extern const char kChromeUINearbyShareURL[];
=======
@@ -317,7 +317,7 @@ extern const char kChromeUINearbyShareURL[];
>>>>>>> upstream/main
 extern const char kChromeUILinuxProxyConfigHost[];
 #endif
 
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_ANDROID)
 extern const char kChromeUISandboxHost[];
 #endif
<<<<<<< HEAD
 
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+    (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 extern const char kChromeUIBrowserSwitchHost[];
 extern const char kChromeUIBrowserSwitchURL[];
 extern const char kChromeUIProfileCustomizationHost[];
@@ -330,7 +330,7 @@ extern const char kChromeUIProfilePickerUrl[];
=======
@@ -325,7 +325,7 @@ extern const char kChromeUISandboxHost[];
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
 extern const char kChromeUIBrowserSwitchHost[];
 extern const char kChromeUIBrowserSwitchURL[];
 extern const char kChromeUIProfileCustomizationHost[];
@@ -335,7 +335,7 @@ extern const char kChromeUIProfilePickerUrl[];
>>>>>>> upstream/main
 extern const char kChromeUIProfilePickerStartupQuery[];
 #endif
 
-#if ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(TOOLKIT_VIEWS)) || \
+#if ((defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(TOOLKIT_VIEWS)) || defined(OS_BSD) || \
     defined(USE_AURA)
 extern const char kChromeUITabModalConfirmDialogHost[];
 #endif
