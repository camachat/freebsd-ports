--- modules/videoio/src/cap_v4l.cpp.orig	2026-06-05 18:50:05 UTC
+++ modules/videoio/src/cap_v4l.cpp
@@ -227,7 +227,9 @@ make & enjoy!
 #include <poll.h>
 
 #ifdef HAVE_CAMV4L2
+#ifdef __linux__
 #include <asm/types.h>          /* for videodev2.h */
+#endif
 #include <linux/videodev2.h>
 #endif
 
