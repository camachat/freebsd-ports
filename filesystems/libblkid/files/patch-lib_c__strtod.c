--- lib/c_strtod.c.orig	2026-02-17 02:54:10 UTC
+++ lib/c_strtod.c
@@ -8,15 +8,15 @@
  */
 #include "c.h"
 
+#if defined(__APPLE__) || defined(__FreeBSD__)
+# include <xlocale.h>
+#endif
+
 #include <locale.h>
 #include <stdlib.h>
 #include <string.h>
 
 #include "c_strtod.h"
-
-#ifdef __APPLE__
-# include <xlocale.h>
-#endif
 
 #if defined(HAVE_NEWLOCALE) && (defined(HAVE_STRTOD_L) || defined(HAVE_USELOCALE))
 # define USE_CLOCALE
