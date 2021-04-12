<<<<<<< HEAD
--- net/traffic_annotation/network_traffic_annotation.h.orig	2020-11-13 06:36:46 UTC
+++ net/traffic_annotation/network_traffic_annotation.h
@@ -359,7 +359,7 @@ struct MutablePartialNetworkTrafficAnnotationTag {
=======
--- net/traffic_annotation/network_traffic_annotation.h.orig	2021-03-12 23:57:27 UTC
+++ net/traffic_annotation/network_traffic_annotation.h
@@ -360,7 +360,7 @@ struct MutablePartialNetworkTrafficAnnotationTag {
>>>>>>> upstream/main
 }  // namespace net
 
 // Placeholder for unannotated usages.
-#if !defined(OS_WIN) && !defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if !defined(OS_WIN) && !defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_BSD)
 #define TRAFFIC_ANNOTATION_WITHOUT_PROTO(ANNOTATION_ID) \
   net::DefineNetworkTrafficAnnotation(ANNOTATION_ID, "No proto yet.")
 #endif
<<<<<<< HEAD
@@ -370,7 +370,7 @@ struct MutablePartialNetworkTrafficAnnotationTag {
 //
 // On Linux and Windows, use MISSING_TRAFFIC_ANNOTATION or
 // TRAFFIC_ANNOTATION_FOR_TESTS.
-#if (!defined(OS_WIN) && !defined(OS_LINUX)) || defined(OS_CHROMEOS)
+#if (!defined(OS_WIN) && !defined(OS_LINUX) && !defined(OS_BSD)) || defined(OS_CHROMEOS)
 #define NO_TRAFFIC_ANNOTATION_YET \
   net::DefineNetworkTrafficAnnotation("undefined", "Nothing here yet.")
 
=======
@@ -373,7 +373,7 @@ struct MutablePartialNetworkTrafficAnnotationTag {
 // TRAFFIC_ANNOTATION_FOR_TESTS.
 // TODO(crbug.com/1052397): Revisit once build flag switch of lacros-chrome is
 // complete.
-#if !defined(OS_WIN) && !(defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if !defined(OS_WIN) && !(defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
 
 #define NO_TRAFFIC_ANNOTATION_YET \
   net::DefineNetworkTrafficAnnotation("undefined", "Nothing here yet.")
>>>>>>> upstream/main
