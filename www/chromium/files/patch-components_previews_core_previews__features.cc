<<<<<<< HEAD
--- components/previews/core/previews_features.cc.orig	2020-11-13 06:36:41 UTC
=======
--- components/previews/core/previews_features.cc.orig	2021-03-12 23:57:23 UTC
>>>>>>> upstream/main
+++ components/previews/core/previews_features.cc
@@ -14,12 +14,12 @@ namespace features {
 // are enabled are controlled by other features.
 const base::Feature kPreviews {
   "Previews",
-#if defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
       // Previews allowed for Android (but also allow on Linux for dev/debug).
       base::FEATURE_ENABLED_BY_DEFAULT
-#else   // !defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#else   // !defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
       base::FEATURE_DISABLED_BY_DEFAULT
-#endif  // defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 };
 
<<<<<<< HEAD
 // Enables the Offline previews on android slow connections.
=======
 // Provides slow page triggering parameters.
>>>>>>> upstream/main
