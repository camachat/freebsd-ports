<<<<<<< HEAD
--- chrome/browser/flag_descriptions.h.orig	2021-01-18 21:28:50 UTC
+++ chrome/browser/flag_descriptions.h
@@ -19,9 +19,9 @@
=======
--- chrome/browser/flag_descriptions.h.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/flag_descriptions.h
@@ -20,9 +20,9 @@
>>>>>>> upstream/main
 #include "ppapi/buildflags/buildflags.h"
 #include "printing/buildflags/buildflags.h"
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "base/allocator/buildflags.h"
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 // This file declares strings used in chrome://flags. These messages are not
 // translated, because instead of end-users they target Chromium developers and
<<<<<<< HEAD
@@ -2699,7 +2699,7 @@ extern const char kEnableNewBadgeOnMenuItemsDescriptio
=======
@@ -2790,7 +2790,7 @@ extern const char kEnableNewBadgeOnMenuItemsDescriptio
>>>>>>> upstream/main
 
 // Random platform combinations -----------------------------------------------
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 
 extern const char kEnableMediaFeedsName[];
<<<<<<< HEAD
@@ -2726,7 +2726,7 @@ extern const char kRemoteCopyProgressNotificationDescr
=======
@@ -2817,15 +2817,15 @@ extern const char kRemoteCopyProgressNotificationDescr
>>>>>>> upstream/main
 extern const char kDirectManipulationStylusName[];
 extern const char kDirectManipulationStylusDescription[];
 
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
<<<<<<< HEAD
 #if defined(OS_WIN) || defined(OS_MAC) || defined(OS_CHROMEOS)
@@ -2736,12 +2736,12 @@ extern const char kWebContentsOcclusionDescription[];
 
 #endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_CHROMEOS)
 
=======
>>>>>>> upstream/main
-#if defined(OS_CHROMEOS) || defined(OS_LINUX)
+#if defined(OS_CHROMEOS) || defined(OS_LINUX) || defined(OS_BSD)
 #if BUILDFLAG(USE_TCMALLOC)
 extern const char kDynamicTcmallocName[];
 extern const char kDynamicTcmallocDescription[];
 #endif  // BUILDFLAG(USE_TCMALLOC)
-#endif  // #if defined(OS_CHROMEOS) || defined(OS_LINUX)
+#endif  // #if defined(OS_CHROMEOS) || defined(OS_LINUX) || defined(OS_BSD)
 
<<<<<<< HEAD
 #if !defined(OS_ANDROID) && !defined(OS_CHROMEOS)
 extern const char kUserDataSnapshotName[];
@@ -2753,11 +2753,11 @@ extern const char kWebShareName[];
 extern const char kWebShareDescription[];
 #endif  // defined(OS_WIN) || defined(OS_CHROMEOS)
 
-#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) || \
     defined(OS_MAC)
 extern const char kEnableEphemeralGuestProfilesOnDesktopName[];
 extern const char kEnableEphemeralGuestProfilesOnDesktopDescription[];
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS) || defined(OS_BSD) ||
         // defined(OS_MAC)
 
 // Feature flags --------------------------------------------------------------
=======
 #if !defined(OS_ANDROID) && !BUILDFLAG(IS_CHROMEOS_ASH)
 extern const char kUserDataSnapshotName[];
@@ -2839,11 +2839,11 @@ extern const char kWebShareDescription[];
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD) || \
     defined(OS_MAC)
 extern const char kEnableEphemeralGuestProfilesOnDesktopName[];
 extern const char kEnableEphemeralGuestProfilesOnDesktopDescription[];
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) || defined(OS_BSD) ||
         // BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_MAC)
 
 // Feature flags --------------------------------------------------------------
@@ -2906,7 +2906,7 @@ extern const char kAutofillCreditCardUploadDescription
 
 #endif  // defined(TOOLKIT_VIEWS) || defined(OS_ANDROID)
 
-#if !defined(OS_WIN) && !defined(OS_FUCHSIA)
+#if !defined(OS_WIN) && !defined(OS_FUCHSIA) && !defined(OS_BSD)
 extern const char kSendWebUIJavaScriptErrorReportsName[];
 extern const char kSendWebUIJavaScriptErrorReportsDescription[];
 #endif
>>>>>>> upstream/main
