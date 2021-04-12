<<<<<<< HEAD
--- chrome/common/channel_info.h.orig	2020-11-13 06:36:38 UTC
+++ chrome/common/channel_info.h
@@ -61,7 +61,7 @@ bool IsSideBySideCapable();
 std::string GetChannelSuffixForDataDir();
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/common/channel_info.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/channel_info.h
@@ -70,7 +70,7 @@ std::string GetChannelSuffixForDataDir();
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 // Returns the channel-specific filename of the desktop shortcut used to launch
 // the browser.
 std::string GetDesktopName(base::Environment* env);
