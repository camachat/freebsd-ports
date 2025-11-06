--- toxcore/network.c.orig	2025-10-06 05:37:58.000000000 -0500
+++ toxcore/network.c	2025-11-06 09:47:33.019821000 -0600
@@ -1318,6 +1318,7 @@
      */
     int n = 1024 * 1024 * 2;
 
+#if !defined(__FreeBSD__)
     if (net_setsockopt(ns, temp->sock, SOL_SOCKET, SO_RCVBUF, &n, sizeof(n)) != 0) {
         LOGGER_WARNING(log, "failed to set socket option %d", SO_RCVBUF);
     }
@@ -1325,6 +1326,7 @@
     if (net_setsockopt(ns, temp->sock, SOL_SOCKET, SO_SNDBUF, &n, sizeof(n)) != 0) {
         LOGGER_WARNING(log, "failed to set socket option %d", SO_SNDBUF);
     }
+#endif
 
     /* Enable broadcast on socket */
     int broadcast = 1;
