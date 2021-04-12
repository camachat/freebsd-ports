<<<<<<< HEAD
--- chrome/common/chrome_features.h.orig	2021-01-18 21:28:52 UTC
+++ chrome/common/chrome_features.h
@@ -82,10 +82,10 @@ extern const base::Feature kAppShimNewCloseBehavior;
=======
--- chrome/common/chrome_features.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/chrome_features.h
@@ -78,10 +78,10 @@ extern const base::Feature kAppShimNewCloseBehavior;
>>>>>>> upstream/main
 
 COMPONENT_EXPORT(CHROME_FEATURES) extern const base::Feature kAsyncDns;
 
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 COMPONENT_EXPORT(CHROME_FEATURES)
 extern const base::Feature kBackgroundModeAllowRestart;
-#endif  // defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
<<<<<<< HEAD
 #if defined(OS_CHROMEOS)
 COMPONENT_EXPORT(CHROME_FEATURES) extern const base::Feature kBorealis;
@@ -227,11 +227,11 @@ extern const base::Feature kEnableAmbientAuthenticatio
 COMPONENT_EXPORT(CHROME_FEATURES)
 extern const base::Feature kEnableAmbientAuthenticationInIncognito;
 
-#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) || \
     defined(OS_MAC)
 COMPONENT_EXPORT(CHROME_FEATURES)
 extern const base::Feature kEnableEphemeralGuestProfilesOnDesktop;
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) &6 !defined(OS_CHROMEOS)) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) ||
         // defined(OS_MAC)
=======
 #if BUILDFLAG(IS_CHROMEOS_ASH)
 COMPONENT_EXPORT(CHROME_FEATURES) extern const base::Feature kBorealis;
@@ -230,11 +230,11 @@ extern const base::Feature kEnableAmbientAuthenticatio
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD) || \
     defined(OS_MAC)
 COMPONENT_EXPORT(CHROME_FEATURES)
 extern const base::Feature kEnableEphemeralGuestProfilesOnDesktop;
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) || defined(OS_BSD) ||
         // BUILDFLAG(IS_CHROMEOS_LACROS)) ||  defined(OS_MAC)
>>>>>>> upstream/main
 
 #if defined(OS_WIN)
