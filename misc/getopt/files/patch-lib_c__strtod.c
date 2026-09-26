--- lib/c_strtod.c.orig	2026-02-13 17:09:41 UTC
+++ lib/c_strtod.c
@@ -8,8 +8,9 @@
  */
 #include "c.h"
 
-#include <locale.h>
 #include <stdlib.h>
+#include <xlocale.h>
+#include <locale.h>
 #include <string.h>
 
 #include "c_strtod.h"
