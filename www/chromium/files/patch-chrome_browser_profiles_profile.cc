<<<<<<< HEAD
--- chrome/browser/profiles/profile.cc.orig	2021-01-19 11:26:11 UTC
+++ chrome/browser/profiles/profile.cc
@@ -387,7 +387,7 @@ bool Profile::IsIncognitoProfile() const {
 
 // static
 bool Profile::IsEphemeralGuestProfileEnabled() {
-#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD) || \
=======
--- chrome/browser/profiles/profile.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/profiles/profile.cc
@@ -360,7 +360,7 @@ bool Profile::IsIncognitoProfile() const {
 bool Profile::IsEphemeralGuestProfileEnabled() {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || \
+#if defined(OS_WIN) || (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD) || \
>>>>>>> upstream/main
     defined(OS_MAC)
   return base::FeatureList::IsEnabled(
       features::kEnableEphemeralGuestProfilesOnDesktop);
