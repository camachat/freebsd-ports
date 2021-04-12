<<<<<<< HEAD
--- net/socket/socks5_client_socket.cc.orig	2020-11-13 06:36:46 UTC
=======
--- net/socket/socks5_client_socket.cc.orig	2021-03-12 23:57:27 UTC
>>>>>>> upstream/main
+++ net/socket/socks5_client_socket.cc
@@ -4,6 +4,10 @@
 
 #include "net/socket/socks5_client_socket.h"
 
+#if defined(OS_BSD)
+#include <netinet/in.h>
+#endif
+
 #include <utility>
 
 #include "base/bind.h"
