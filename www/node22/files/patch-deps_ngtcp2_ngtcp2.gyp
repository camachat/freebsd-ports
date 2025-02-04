--- deps/ngtcp2/ngtcp2.gyp.orig	2025-01-20 19:04:55.000000000 -0600
+++ deps/ngtcp2/ngtcp2.gyp	2025-02-03 18:50:43.855971000 -0600
@@ -81,8 +81,7 @@
       'nghttp3/lib/nghttp3_unreachable.c',
       'nghttp3/lib/nghttp3_vec.c',
       'nghttp3/lib/nghttp3_version.c',
-      # sfparse is also used by nghttp2 and is included by nghttp2.gyp
-      # 'nghttp3/lib/sfparse.c'
+      'nghttp3/lib/sfparse/sfparse.c'
     ]
   },
   'targets': [
