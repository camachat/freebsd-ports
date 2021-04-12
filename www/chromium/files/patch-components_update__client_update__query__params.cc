<<<<<<< HEAD
--- components/update_client/update_query_params.cc.orig	2020-11-13 06:36:42 UTC
+++ components/update_client/update_query_params.cc
@@ -39,6 +39,8 @@ const char kOs[] =
=======
--- components/update_client/update_query_params.cc.orig	2021-03-12 23:57:23 UTC
+++ components/update_client/update_query_params.cc
@@ -40,6 +40,8 @@ const char kOs[] =
>>>>>>> upstream/main
     "fuchsia";
 #elif defined(OS_OPENBSD)
     "openbsd";
+#elif defined(OS_FREEBSD)
+    "freebsd";
 #else
 #error "unknown os"
 #endif
