<<<<<<< HEAD
--- chrome/browser/first_run/first_run_internal_posix.cc.orig	2020-11-13 06:36:37 UTC
+++ chrome/browser/first_run/first_run_internal_posix.cc
@@ -45,7 +45,7 @@ enum class ForcedShowDialogState {
 ForcedShowDialogState g_forced_show_dialog_state =
     ForcedShowDialogState::kNotForced;
 
-#if !defined(OS_CHROMEOS)
+#if !defined(OS_CHROMEOS) && !defined(OS_BSD)
 // Returns whether the first run dialog should be shown. This is only true for
 // certain builds, and only if the user has not already set preferences. In a
 // real, official-build first run, initializes the default metrics reporting if
@@ -96,7 +96,7 @@ void ForceFirstRunDialogShownForTesting(bool shown) {
 }
 
 void DoPostImportPlatformSpecificTasks(Profile* profile) {
-#if !defined(OS_CHROMEOS)
+#if !defined(OS_CHROMEOS) && !defined(OS_BSD)
=======
--- chrome/browser/first_run/first_run_internal_posix.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/first_run/first_run_internal_posix.cc
@@ -46,7 +46,7 @@ enum class ForcedShowDialogState {
 ForcedShowDialogState g_forced_show_dialog_state =
     ForcedShowDialogState::kNotForced;
 
-#if !BUILDFLAG(IS_CHROMEOS_ASH)
+#if !BUILDFLAG(IS_CHROMEOS_ASH) && !defined(OS_BSD)
 // Returns whether the first run dialog should be shown. This is only true for
 // certain builds, and only if the user has not already set preferences. In a
 // real, official-build first run, initializes the default metrics reporting if
@@ -97,7 +97,7 @@ void ForceFirstRunDialogShownForTesting(bool shown) {
 }
 
 void DoPostImportPlatformSpecificTasks(Profile* profile) {
-#if !BUILDFLAG(IS_CHROMEOS_ASH)
+#if !BUILDFLAG(IS_CHROMEOS_ASH) && !defined(OS_BSD)
>>>>>>> upstream/main
   if (!ShouldShowFirstRunDialog())
     return;
 
