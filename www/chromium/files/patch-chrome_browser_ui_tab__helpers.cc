<<<<<<< HEAD
--- chrome/browser/ui/tab_helpers.cc.orig	2021-01-18 21:28:51 UTC
+++ chrome/browser/ui/tab_helpers.cc
@@ -149,7 +149,7 @@
=======
--- chrome/browser/ui/tab_helpers.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/tab_helpers.cc
@@ -150,7 +150,7 @@
>>>>>>> upstream/main
 #include "chrome/browser/ui/app_list/search/cros_action_history/cros_action_recorder_tab_tracker.h"
 #endif
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 #include "chrome/browser/ui/blocked_content/framebust_block_tab_helper.h"
 #include "chrome/browser/ui/hats/hats_helper.h"
<<<<<<< HEAD
@@ -409,11 +409,11 @@ void TabHelpers::AttachTabHelpers(WebContents* web_con
 #endif
 
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+    (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
@@ -422,11 +422,11 @@ void TabHelpers::AttachTabHelpers(WebContents* web_con
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
   metrics::DesktopSessionDurationObserver::CreateForWebContents(web_contents);
 #endif
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
   if (base::FeatureList::IsEnabled(
           features::kHappinessTrackingSurveysForDesktop) ||
