<<<<<<< HEAD
--- chrome/browser/about_flags.cc.orig	2021-01-18 21:28:49 UTC
+++ chrome/browser/about_flags.cc
@@ -188,7 +188,7 @@
=======
--- chrome/browser/about_flags.cc.orig	2021-03-12 23:57:17 UTC
+++ chrome/browser/about_flags.cc
@@ -191,7 +191,7 @@
>>>>>>> upstream/main
 #include "ui/gl/gl_switches.h"
 #include "ui/native_theme/native_theme_features.h"
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "base/allocator/buildflags.h"
 #endif
 
<<<<<<< HEAD
@@ -886,7 +886,7 @@ const FeatureEntry::Choice kMemlogSamplingRateChoices[
=======
@@ -918,7 +918,7 @@ const FeatureEntry::Choice kMemlogSamplingRateChoices[
>>>>>>> upstream/main
      heap_profiling::kMemlogSamplingRate5MB},
 };
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) || \
     defined(OS_WIN)
 const FeatureEntry::FeatureParam kOmniboxDocumentProviderServerScoring[] = {
     {"DocumentUseServerScore", "true"},
<<<<<<< HEAD
@@ -1236,7 +1236,7 @@ const FeatureEntry::FeatureVariation kOmniboxBookmarkP
=======
@@ -1268,7 +1268,7 @@ const FeatureEntry::FeatureVariation kOmniboxBookmarkP
>>>>>>> upstream/main
     },
 };
 
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) ||
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) ||
         // defined(OS_WIN)
 
 const FeatureEntry::FeatureVariation
<<<<<<< HEAD
@@ -2869,7 +2869,7 @@ const FeatureEntry kFeatureEntries[] = {
      FEATURE_VALUE_TYPE(ash::features::kSystemTrayMicGainSetting)},
 #endif  // OS_CHROMEOS
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_ANDROID)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_ANDROID)) || defined(OS_BSD)
     {
         "enable-accelerated-video-decode",
         flag_descriptions::kAcceleratedVideoDecodeName,
@@ -2885,7 +2885,7 @@ const FeatureEntry kFeatureEntries[] = {
         kOsMac | kOsWin | kOsCrOS | kOsAndroid,
         SINGLE_DISABLE_VALUE_TYPE(switches::kDisableAcceleratedVideoDecode),
     },
-#endif  // defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_ANDROID)
+#endif  // (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_ANDROID)) || defined(OS_BSD)
     {
         "disable-accelerated-video-encode",
         flag_descriptions::kAcceleratedVideoEncodeName,
@@ -3208,7 +3208,7 @@ const FeatureEntry kFeatureEntries[] = {
      flag_descriptions::kEnableOfflinePreviewsDescription, kOsAndroid,
      FEATURE_VALUE_TYPE(previews::features::kOfflinePreviews)},
 #endif  // OS_ANDROID
=======
@@ -3009,7 +3009,7 @@ const FeatureEntry kFeatureEntries[] = {
      FEATURE_VALUE_TYPE(ash::features::kSystemTrayMicGainSetting)},
 #endif  // BUILDFLAG(IS_CHROMEOS_ASH)
 
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && !defined(OS_ANDROID)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && !defined(OS_ANDROID)
     {
         "enable-accelerated-video-decode",
         flag_descriptions::kAcceleratedVideoDecodeName,
@@ -3025,7 +3025,7 @@ const FeatureEntry kFeatureEntries[] = {
         kOsMac | kOsWin | kOsCrOS | kOsAndroid,
         SINGLE_DISABLE_VALUE_TYPE(switches::kDisableAcceleratedVideoDecode),
     },
-#endif  // (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) &&
+#endif  // (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) &&
         // !defined(OS_ANDROID)
     {
         "disable-accelerated-video-encode",
@@ -3367,7 +3367,7 @@ const FeatureEntry kFeatureEntries[] = {
     {"enable-login-detection", flag_descriptions::kEnableLoginDetectionName,
      flag_descriptions::kEnableLoginDetectionDescription, kOsAll,
      FEATURE_VALUE_TYPE(login_detection::kLoginDetection)},
>>>>>>> upstream/main
-#if defined(OS_CHROMEOS) || defined(OS_LINUX)
+#if defined(OS_CHROMEOS) || defined(OS_LINUX) || defined(OS_BSD)
     {"enable-save-data", flag_descriptions::kEnableSaveDataName,
      flag_descriptions::kEnableSaveDataDescription, kOsCrOS | kOsLinux,
      SINGLE_VALUE_TYPE(
<<<<<<< HEAD
@@ -3222,7 +3222,7 @@ const FeatureEntry kFeatureEntries[] = {
      flag_descriptions::kEnableNavigationPredictorRendererWarmupName,
      flag_descriptions::kEnableNavigationPredictorRendererWarmupDescription,
      kOsAll, FEATURE_VALUE_TYPE(features::kNavigationPredictorRendererWarmup)},
-#endif  // OS_CHROMEOS || OS_LINUX
+#endif  // OS_CHROMEOS || OS_LINUX || OS_BSD
     {"enable-preconnect-to-search",
      flag_descriptions::kEnablePreconnectToSearchName,
      flag_descriptions::kEnablePreconnectToSearchDescription, kOsAll,
@@ -3977,7 +3977,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -3381,7 +3381,7 @@ const FeatureEntry kFeatureEntries[] = {
      flag_descriptions::kEnableNavigationPredictorRendererWarmupName,
      flag_descriptions::kEnableNavigationPredictorRendererWarmupDescription,
      kOsAll, FEATURE_VALUE_TYPE(features::kNavigationPredictorRendererWarmup)},
-#endif  // BUILDFLAG(IS_CHROMEOS_ASH) || OS_LINUX
+#endif  // BUILDFLAG(IS_CHROMEOS_ASH) || OS_LINUX || defined(OS_BSD)
     {"enable-preconnect-to-search",
      flag_descriptions::kEnablePreconnectToSearchName,
      flag_descriptions::kEnablePreconnectToSearchDescription, kOsAll,
@@ -4173,7 +4173,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      kOsAll,
      FEATURE_VALUE_TYPE(omnibox::kOmniboxTrendingZeroPrefixSuggestionsOnNTP)},
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) || \
     defined(OS_WIN)
     {"omnibox-experimental-keyword-mode",
      flag_descriptions::kOmniboxExperimentalKeywordModeName,
<<<<<<< HEAD
@@ -4054,7 +4054,7 @@ const FeatureEntry kFeatureEntries[] = {
      FEATURE_WITH_PARAMS_VALUE_TYPE(omnibox::kBookmarkPaths,
                                     kOmniboxBookmarkPathsVariations,
                                     "OmniboxBundledExperimentV1")},
=======
@@ -4254,7 +4254,7 @@ const FeatureEntry kFeatureEntries[] = {
      flag_descriptions::kOmniboxDisableCGIParamMatchingName,
      flag_descriptions::kOmniboxDisableCGIParamMatchingDescription, kOsDesktop,
      FEATURE_VALUE_TYPE(omnibox::kDisableCGIParamMatching)},
>>>>>>> upstream/main
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) ||
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) ||
         // defined(OS_WIN)
 
     {"enable-speculative-service-worker-start-on-query-input",
<<<<<<< HEAD
@@ -4344,14 +4344,14 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -4563,14 +4563,14 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      flag_descriptions::kClickToOpenPDFDescription, kOsAll,
      FEATURE_VALUE_TYPE(features::kClickToOpenPDFPlaceholder)},
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
     {"direct-manipulation-stylus",
      flag_descriptions::kDirectManipulationStylusName,
      flag_descriptions::kDirectManipulationStylusDescription,
      kOsWin | kOsMac | kOsLinux,
      FEATURE_VALUE_TYPE(features::kDirectManipulationStylus)},
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
 #if !defined(OS_ANDROID)
<<<<<<< HEAD
@@ -5043,7 +5043,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -5258,7 +5258,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      FEATURE_VALUE_TYPE(kClickToCallUI)},
 #endif  // BUILDFLAG(ENABLE_CLICK_TO_CALL)
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
     {"remote-copy-receiver", flag_descriptions::kRemoteCopyReceiverName,
      flag_descriptions::kRemoteCopyReceiverDescription, kOsDesktop,
<<<<<<< HEAD
@@ -5060,7 +5060,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -5275,7 +5275,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      flag_descriptions::kRemoteCopyProgressNotificationName,
      flag_descriptions::kRemoteCopyProgressNotificationDescription, kOsDesktop,
      FEATURE_VALUE_TYPE(kRemoteCopyProgressNotification)},
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
     {"restrict-gamepad-access", flag_descriptions::kRestrictGamepadAccessName,
<<<<<<< HEAD
@@ -5643,7 +5643,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -5852,7 +5852,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      flag_descriptions::kMouseSubframeNoImplicitCaptureDescription, kOsAll,
      FEATURE_VALUE_TYPE(features::kMouseSubframeNoImplicitCapture)},
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
     {"global-media-controls", flag_descriptions::kGlobalMediaControlsName,
      flag_descriptions::kGlobalMediaControlsDescription,
<<<<<<< HEAD
@@ -5684,7 +5684,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -5893,7 +5893,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      flag_descriptions::kGlobalMediaControlsOverlayControlsDescription,
      kOsWin | kOsMac | kOsLinux,
      FEATURE_VALUE_TYPE(media::kGlobalMediaControlsOverlayControls)},
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
 #if BUILDFLAG(ENABLE_SPELLCHECK) && defined(OS_WIN)
<<<<<<< HEAD
@@ -5861,7 +5861,7 @@ const FeatureEntry kFeatureEntries[] = {
      FEATURE_VALUE_TYPE(
          password_manager::features::kEnablePasswordsAccountStorage)},
=======
@@ -6072,7 +6072,7 @@ const FeatureEntry kFeatureEntries[] = {
          kPasswordsAccountStorageVariations,
          "ButterForPasswords")},
>>>>>>> upstream/main
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
     {"passwords-account-storage-iph",
      flag_descriptions::kEnablePasswordsAccountStorageIPHName,
<<<<<<< HEAD
@@ -5869,7 +5869,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -6080,7 +6080,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      kOsWin | kOsMac | kOsLinux,
      FEATURE_VALUE_TYPE(
          feature_engagement::kIPHPasswordsAccountStorageFeature)},
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
     {"autofill-always-return-cloud-tokenized-card",
<<<<<<< HEAD
@@ -6602,7 +6602,7 @@ const FeatureEntry kFeatureEntries[] = {
=======
@@ -6794,7 +6794,7 @@ const FeatureEntry kFeatureEntries[] = {
>>>>>>> upstream/main
      FEATURE_VALUE_TYPE(ash::features::kEnhancedDeskAnimations)},
 #endif
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
     {"enable-oop-print-drivers", flag_descriptions::kEnableOopPrintDriversName,
      flag_descriptions::kEnableOopPrintDriversDescription, kOsDesktop,
<<<<<<< HEAD
@@ -6636,14 +6636,14 @@ const FeatureEntry kFeatureEntries[] = {
      FEATURE_VALUE_TYPE(features::kMuteNotificationsDuringScreenShare)},
 #endif  // !defined(OS_ANDROID)
 
-#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) || \
=======
@@ -6833,14 +6833,14 @@ const FeatureEntry kFeatureEntries[] = {
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD) || \
>>>>>>> upstream/main
     defined(OS_MAC)
     {"enable-ephemeral-guest-profiles-on-desktop",
      flag_descriptions::kEnableEphemeralGuestProfilesOnDesktopName,
      flag_descriptions::kEnableEphemeralGuestProfilesOnDesktopDescription,
      kOsWin | kOsLinux | kOsMac,
      FEATURE_VALUE_TYPE(features::kEnableEphemeralGuestProfilesOnDesktop)},
<<<<<<< HEAD
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) ||
         // defined(OS_MAC)
=======
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) || defined(OS_BSD) ||
         // BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_MAC)
>>>>>>> upstream/main
 
 #if defined(OS_ANDROID)
