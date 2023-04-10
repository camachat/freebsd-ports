--- src/nvidia/nvlink_freebsd.c.orig	2022-10-26 09:02:40 UTC
+++ src/nvidia/nvlink_freebsd.c
@@ -119,7 +119,7 @@ void nvlink_assert(int cond)
 {
 }
 
-void * nvlink_allocLock()
+void * nvlink_allocLock(void)
 {
     return NULL;
 }
