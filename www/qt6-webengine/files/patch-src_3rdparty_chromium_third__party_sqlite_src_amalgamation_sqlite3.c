--- src/3rdparty/chromium/third_party/sqlite/src/amalgamation/sqlite3.c.orig	2024-05-15 13:57:21 UTC
+++ src/3rdparty/chromium/third_party/sqlite/src/amalgamation/sqlite3.c
@@ -20107,8 +20107,8 @@ SQLITE_PRIVATE int sqlite3HeapNearlyFull(void);
 ** that deal with sqlite3StackAlloc() failures to be unreachable.
 */
 #ifdef SQLITE_USE_ALLOCA
-# define sqlite3StackAllocRaw(D,N)   alloca(N)
-# define sqlite3StackAllocRawNN(D,N) alloca(N)
+# define sqlite3StackAllocRaw(D,N)   __builtin_alloca(N)
+# define sqlite3StackAllocRawNN(D,N) __builtin_alloca(N)
 # define sqlite3StackFree(D,P)
 # define sqlite3StackFreeNN(D,P)
 #else
@@ -43844,7 +43844,12 @@ static int unixRandomness(sqlite3_vfs *NotUsed, int nB
   memset(zBuf, 0, nBuf);
   randomnessPid = osGetpid(0);
 #if !defined(SQLITE_TEST) && !defined(SQLITE_OMIT_RANDOMNESS)
+#if defined(__OpenBSD__)
   {
+    arc4random_buf(zBuf, nBuf);
+  }
+#else
+  {
     int fd, got;
     fd = robust_open("/dev/urandom", O_RDONLY, 0);
     if( fd<0 ){
@@ -43859,6 +43864,7 @@ static int unixRandomness(sqlite3_vfs *NotUsed, int nB
       robust_close(0, fd, __LINE__);
     }
   }
+#endif
 #endif
   return nBuf;
 }
