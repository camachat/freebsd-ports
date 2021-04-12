<<<<<<< HEAD
--- chrome/browser/ui/startup/bad_flags_prompt.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/startup/bad_flags_prompt.cc
@@ -88,7 +88,7 @@ static const char* kBadFlags[] = {
     extensions::switches::kExtensionsOnChromeURLs,
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
=======
--- chrome/browser/ui/startup/bad_flags_prompt.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/startup/bad_flags_prompt.cc
@@ -96,7 +96,7 @@ static const char* kBadFlags[] = {
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
     // Speech dispatcher is buggy, it can crash and it can make Chrome freeze.
     // http://crbug.com/327295
     switches::kEnableSpeechDispatcher,
