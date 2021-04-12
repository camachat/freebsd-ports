<<<<<<< HEAD
--- chrome/common/pref_names.h.orig	2021-01-18 21:28:52 UTC
+++ chrome/common/pref_names.h
@@ -357,7 +357,7 @@ extern const char kAllowDeletingBrowserHistory[];
 extern const char kForceGoogleSafeSearch[];
 extern const char kForceYouTubeRestrict[];
 extern const char kAllowedDomainsForApps[];
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
 extern const char kUsesSystemTheme[];
 #endif
 extern const char kCurrentThemePackFilename[];
@@ -384,7 +384,7 @@ extern const char kDefaultBrowserSettingEnabled[];
 #if defined(OS_MAC)
 extern const char kShowUpdatePromotionInfoBar[];
 #endif
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
 extern const char kUseCustomChromeFrame[];
 #endif
 #if BUILDFLAG(ENABLE_PLUGINS)
@@ -554,7 +554,7 @@ extern const char kDownloadExtensionsToOpen[];
=======
--- chrome/common/pref_names.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/pref_names.h
@@ -364,7 +364,7 @@ extern const char kForceYouTubeRestrict[];
 extern const char kAllowedDomainsForApps[];
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 extern const char kUsesSystemTheme[];
 #endif
 extern const char kCurrentThemePackFilename[];
@@ -393,7 +393,7 @@ extern const char kShowUpdatePromotionInfoBar[];
 #endif
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 extern const char kUseCustomChromeFrame[];
 #endif
 #if BUILDFLAG(ENABLE_PLUGINS)
@@ -564,7 +564,7 @@ extern const char kDownloadExtensionsToOpen[];
>>>>>>> upstream/main
 extern const char kDownloadExtensionsToOpenByPolicy[];
 extern const char kDownloadAllowedURLsForOpenByPolicy[];
 extern const char kDownloadDirUpgraded[];
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
 extern const char kOpenPdfDownloadInSystemReader[];
 #endif
<<<<<<< HEAD
@@ -783,9 +783,9 @@ extern const char kGloballyScopeHTTPAuthCacheEnabled[]
=======
@@ -795,9 +795,9 @@ extern const char kGloballyScopeHTTPAuthCacheEnabled[]
>>>>>>> upstream/main
 extern const char kAmbientAuthenticationInPrivateModesEnabled[];
 extern const char kBasicAuthOverHttpEnabled[];
 
-#if defined(OS_LINUX) || defined(OS_MAC) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_MAC) || defined(OS_CHROMEOS) || defined(OS_BSD)
 extern const char kAuthNegotiateDelegateByKdcPolicy[];
-#endif  // defined(OS_LINUX) || defined(OS_MAC) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_MAC) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 #if defined(OS_POSIX)
 extern const char kNtlmV2Enabled[];
<<<<<<< HEAD
@@ -1014,7 +1014,7 @@ extern const char kAutoplayWhitelist[];
 extern const char kBlockAutoplayEnabled[];
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
@@ -1028,7 +1028,7 @@ extern const char kBlockAutoplayEnabled[];
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 extern const char kAllowNativeNotifications[];
 #endif
 
