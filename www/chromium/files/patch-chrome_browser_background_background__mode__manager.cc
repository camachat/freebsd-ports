<<<<<<< HEAD
--- chrome/browser/background/background_mode_manager.cc.orig	2021-01-18 21:28:49 UTC
+++ chrome/browser/background/background_mode_manager.cc
@@ -850,7 +850,7 @@ gfx::ImageSkia GetStatusTrayIcon() {
=======
--- chrome/browser/background/background_mode_manager.cc.orig	2021-03-12 23:57:17 UTC
+++ chrome/browser/background/background_mode_manager.cc
@@ -873,7 +873,7 @@ gfx::ImageSkia GetStatusTrayIcon() {
>>>>>>> upstream/main
     return gfx::ImageSkia();
 
   return family->CreateExact(size).AsImageSkia();
-#elif defined(OS_LINUX) || defined(OS_CHROMEOS)
+#elif defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   return *ui::ResourceBundle::GetSharedInstance().GetImageSkiaNamed(
       IDR_PRODUCT_LOGO_128);
 #elif defined(OS_MAC)
