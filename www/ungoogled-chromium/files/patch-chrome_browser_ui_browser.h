--- chrome/browser/ui/browser.h.orig	2024-11-16 12:20:41 UTC
+++ chrome/browser/ui/browser.h
@@ -303,7 +303,7 @@ class Browser : public TabStripModelObserver,
     std::optional<int64_t> display_id;
 #endif
 
-#if BUILDFLAG(IS_LINUX)
+#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_BSD)
     // When the browser window is shown, the desktop environment is notified
     // using this ID.  In response, the desktop will stop playing the "waiting
     // for startup" animation (if any).
