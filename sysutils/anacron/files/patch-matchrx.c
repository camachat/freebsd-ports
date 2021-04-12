--- matchrx.c.orig	2000-06-20 23:12:18 UTC
+++ matchrx.c
<<<<<<< HEAD
@@ -23,6 +23,7 @@
=======
@@ -23,9 +23,11 @@
>>>>>>> upstream/main
 
 
 #include <stdio.h>
+#include <sys/types.h>
 #include <regex.h>
 #include <stdarg.h>
 #include <stdlib.h>
<<<<<<< HEAD
=======
+#include <string.h>
 #include "matchrx.h"
 
 int
>>>>>>> upstream/main
