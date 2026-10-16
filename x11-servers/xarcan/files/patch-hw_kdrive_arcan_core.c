--- hw/kdrive/arcan/core.c.orig	2023-12-18 20:29:59 UTC
+++ hw/kdrive/arcan/core.c
@@ -82,6 +82,18 @@ static void enqueueKeyboard(uint16_t scancode, int act
 }
 #endif
 
+static int64_t arcan_timemillis(void) {
+    struct timespec ts;
+    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
+        // handle error (very rare in practice)
+        perror("clock_gettime");
+        return -1;
+    }
+
+    // seconds → ms + nanoseconds → ms
+    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
+}
+
 static uint8_t code_tbl[512];
 static struct arcan_shmif_initial* arcan_init;
 static DevPrivateKeyRec pixmapPriv;
