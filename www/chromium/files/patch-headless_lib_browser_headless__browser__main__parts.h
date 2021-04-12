<<<<<<< HEAD
--- headless/lib/browser/headless_browser_main_parts.h.orig	2020-11-13 06:36:44 UTC
+++ headless/lib/browser/headless_browser_main_parts.h
@@ -32,7 +32,7 @@ class HeadlessBrowserMainParts : public content::Brows
=======
--- headless/lib/browser/headless_browser_main_parts.h.orig	2021-03-12 23:57:25 UTC
+++ headless/lib/browser/headless_browser_main_parts.h
@@ -37,7 +37,7 @@ class HeadlessBrowserMainParts : public content::Brows
>>>>>>> upstream/main
 #if defined(OS_MAC)
   void PreMainMessageLoopStart() override;
 #endif
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   void PostMainMessageLoopStart() override;
 #endif
   void QuitMainMessageLoop();
