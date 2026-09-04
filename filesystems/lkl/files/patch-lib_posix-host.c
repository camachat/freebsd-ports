--- lib/posix-host.c.orig	2025-09-04 17:12:33 UTC
+++ lib/posix-host.c
@@ -2,6 +2,7 @@
 #define _GNU_SOURCE
 #define _DEFAULT_SOURCE /* glibc preadv/pwritev */
 #include <pthread.h>
+#include <pthread_np.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <sys/time.h>
