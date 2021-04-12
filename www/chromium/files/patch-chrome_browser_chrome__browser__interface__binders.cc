<<<<<<< HEAD
--- chrome/browser/chrome_browser_interface_binders.cc.orig	2021-01-18 21:28:49 UTC
+++ chrome/browser/chrome_browser_interface_binders.cc
@@ -135,7 +135,7 @@
 #include "media/mojo/mojom/speech_recognition_service.mojom.h"
 #endif
=======
--- chrome/browser/chrome_browser_interface_binders.cc.orig	2021-03-12 23:57:17 UTC
+++ chrome/browser/chrome_browser_interface_binders.cc
@@ -138,7 +138,7 @@
 #include "mojo/public/cpp/bindings/self_owned_receiver.h"
 #endif  // defined(OS_ANDROID)
>>>>>>> upstream/main
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 #include "chrome/browser/ui/webui/discards/discards.mojom.h"
 #include "chrome/browser/ui/webui/discards/discards_ui.h"
<<<<<<< HEAD
@@ -775,7 +775,7 @@ void PopulateChromeWebUIFrameBinders(
=======
@@ -835,7 +835,7 @@ void PopulateChromeWebUIFrameBinders(
>>>>>>> upstream/main
   }
 #endif
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
   RegisterWebUIControllerInterfaceBinder<discards::mojom::DetailsProvider,
                                          DiscardsUI>(map);
