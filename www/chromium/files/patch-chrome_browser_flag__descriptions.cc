<<<<<<< HEAD
--- chrome/browser/flag_descriptions.cc.orig	2021-01-18 21:28:50 UTC
+++ chrome/browser/flag_descriptions.cc
@@ -4609,7 +4609,7 @@ const char kEnableNewBadgeOnMenuItemsDescription[] =
=======
--- chrome/browser/flag_descriptions.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/flag_descriptions.cc
@@ -4772,7 +4772,7 @@ const char kEnableNewBadgeOnMenuItemsDescription[] =
>>>>>>> upstream/main
 
 // Random platform combinations -----------------------------------------------
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 
 const char kEnableMediaFeedsName[] = "Enables Media Feeds";
<<<<<<< HEAD
@@ -4653,17 +4653,17 @@ const char kRemoteCopyProgressNotificationDescription[
=======
@@ -4816,27 +4816,27 @@ const char kRemoteCopyProgressNotificationDescription[
>>>>>>> upstream/main
     "Enables progress notifications to be shown for the remote copy feature "
     "when receiving a message.";
 
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
 
 const char kDirectManipulationStylusName[] = "Direct Manipulation Stylus";
 const char kDirectManipulationStylusDescription[] =
     "If enabled, Chrome will scroll web pages on stylus drag.";
 
-#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 
<<<<<<< HEAD
 #if defined(OS_WIN) || defined(OS_MAC) || defined(OS_CHROMEOS)
@@ -4675,14 +4675,14 @@ const char kWebContentsOcclusionDescription[] =
 
 #endif  // defined(OS_WIN) || defined(OS_MAC) || defined(OS_CHROMEOS)
 
=======
>>>>>>> upstream/main
-#if defined(OS_CHROMEOS) || defined(OS_LINUX)
+#if defined(OS_CHROMEOS) || defined(OS_LINUX) || defined(OS_BSD)
 #if BUILDFLAG(USE_TCMALLOC)
 const char kDynamicTcmallocName[] = "Dynamic Tcmalloc Tuning";
 const char kDynamicTcmallocDescription[] =
     "Allows tcmalloc to dynamically adjust tunables based on system resource "
     "utilization.";
 #endif  // BUILDFLAG(USE_TCMALLOC)
-#endif  // #if defined(OS_CHROMEOS) || defined(OS_LINUX)
+#endif  // #if defined(OS_CHROMEOS) || defined(OS_LINUX) || defined(OS_BSD)
 
<<<<<<< HEAD
 #if !defined(OS_ANDROID) && !defined(OS_CHROMEOS)
 const char kUserDataSnapshotName[] = "Enable user data snapshots";
@@ -4698,13 +4698,13 @@ const char kWebShareDescription[] =
     "platforms.";
 #endif  // defined(OS_WIN) || defined(OS_CHROMEOS)
 
-#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) || \
=======
 #if !defined(OS_ANDROID) && !BUILDFLAG(IS_CHROMEOS_ASH)
 const char kUserDataSnapshotName[] = "Enable user data snapshots";
@@ -4854,13 +4854,13 @@ const char kWebShareDescription[] =
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD) || \
>>>>>>> upstream/main
     defined(OS_MAC)
 const char kEnableEphemeralGuestProfilesOnDesktopName[] =
     "Enable ephemeral Guest profiles on Desktop";
 const char kEnableEphemeralGuestProfilesOnDesktopDescription[] =
     "Enables ephemeral Guest profiles on Windows, Linux, and Mac.";
<<<<<<< HEAD
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) ||
         // defined(OS_MAC)
 
 // Feature flags --------------------------------------------------------------
=======
-#endif  // defined(OS_WIN) || (defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || (defined(OS_LINUX) || defined(OS_BSD) ||
         // BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_MAC)
 
 // Feature flags --------------------------------------------------------------
@@ -4953,7 +4953,7 @@ const char kAutofillCreditCardUploadDescription[] =
 
 #endif  // defined(TOOLKIT_VIEWS) || defined(OS_ANDROID)
 
-#if !defined(OS_WIN) && !defined(OS_FUCHSIA)
+#if !defined(OS_WIN) && !defined(OS_FUCHSIA) && !defined(OS_BSD)
 const char kSendWebUIJavaScriptErrorReportsName[] =
     "Send WebUI JavaScript Error Reports";
 const char kSendWebUIJavaScriptErrorReportsDescription[] =
>>>>>>> upstream/main
