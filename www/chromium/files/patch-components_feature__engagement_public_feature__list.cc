<<<<<<< HEAD
--- components/feature_engagement/public/feature_list.cc.orig	2021-01-18 21:28:55 UTC
+++ components/feature_engagement/public/feature_list.cc
@@ -75,7 +75,7 @@ const base::Feature* const kAllFeatures[] = {
=======
--- components/feature_engagement/public/feature_list.cc.orig	2021-03-12 23:57:22 UTC
+++ components/feature_engagement/public/feature_list.cc
@@ -78,7 +78,7 @@ const base::Feature* const kAllFeatures[] = {
>>>>>>> upstream/main
     &kIPHBadgedTranslateManualTriggerFeature,
     &kIPHDiscoverFeedHeaderFeature,
 #endif  // defined(OS_IOS)
-#if defined(OS_WIN) || defined(OS_APPLE) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
     &kIPHDesktopTabGroupsNewGroupFeature,
     &kIPHFocusModeFeature,
<<<<<<< HEAD
@@ -85,7 +85,7 @@ const base::Feature* const kAllFeatures[] = {
=======
@@ -88,7 +88,7 @@ const base::Feature* const kAllFeatures[] = {
>>>>>>> upstream/main
     &kIPHReopenTabFeature,
     &kIPHWebUITabStripFeature,
     &kIPHDesktopPwaInstallFeature,
-#endif  // defined(OS_WIN) || defined(OS_APPLE) || defined(OS_LINUX) ||
+#endif  // defined(OS_WIN) || defined(OS_APPLE) || defined(OS_LINUX) || defined(OS_BSD) ||
         // defined(OS_CHROMEOS)
 };
 }  // namespace
