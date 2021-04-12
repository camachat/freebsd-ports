<<<<<<< HEAD
--- chrome/browser/ui/test/test_browser_dialog.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/test/test_browser_dialog.cc
@@ -114,7 +114,7 @@ bool TestBrowserDialog::VerifyUi() {
 
   views::Widget* dialog_widget = *(added.begin());
 // TODO(https://crbug.com/958242) support Mac for pixel tests.
-#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/ui/test/test_browser_dialog.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/test/test_browser_dialog.cc
@@ -117,7 +117,7 @@ bool TestBrowserDialog::VerifyUi() {
 // TODO(https://crbug.com/958242) support Mac for pixel tests.
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
   dialog_widget->SetBlockCloseForTesting(true);
   // Deactivate before taking screenshot. Deactivated dialog pixel outputs
   // is more predictable than activated dialog.
