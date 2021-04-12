<<<<<<< HEAD
--- third_party/perfetto/src/base/unix_socket.cc.orig	2021-01-18 21:31:50 UTC
+++ third_party/perfetto/src/base/unix_socket.cc
@@ -38,7 +38,7 @@
 #include "perfetto/ext/base/string_utils.h"
 #include "perfetto/ext/base/utils.h"
=======
--- third_party/perfetto/src/base/unix_socket.cc.orig	2021-03-13 00:03:38 UTC
+++ third_party/perfetto/src/base/unix_socket.cc
@@ -41,7 +41,7 @@
 #include <unistd.h>
 #endif
>>>>>>> upstream/main
 
-#if PERFETTO_BUILDFLAG(PERFETTO_OS_APPLE)
+#if PERFETTO_BUILDFLAG(PERFETTO_OS_APPLE) || PERFETTO_BUILDFLAG(PERFETTO_OS_FREEBSD)
 #include <sys/ucred.h>
 #endif
 
<<<<<<< HEAD
@@ -630,7 +630,7 @@ void UnixSocket::ReadPeerCredentials() {
   if (sock_raw_.family() != SockFamily::kUnix)
     return;
 
-#if PERFETTO_BUILDFLAG(PERFETTO_OS_LINUX) || \
+#if (PERFETTO_BUILDFLAG(PERFETTO_OS_LINUX) && !PERFETTO_BUILDFLAG(PERFETTO_OS_FREEBSD)) || \
     PERFETTO_BUILDFLAG(PERFETTO_OS_ANDROID)
   struct ucred user_cred;
   socklen_t len = sizeof(user_cred);
=======
@@ -758,7 +758,8 @@ void UnixSocket::ReadPeerCredentialsPosix() {
   PERFETTO_CHECK(res == 0);
   peer_uid_ = user_cred.uid;
   peer_pid_ = user_cred.pid;
-#elif PERFETTO_BUILDFLAG(PERFETTO_OS_APPLE)
+#elif PERFETTO_BUILDFLAG(PERFETTO_OS_APPLE) || \
+      PERFETTO_BUILDFLAG(PERFETTO_OS_FREEBSD)
   struct xucred user_cred;
   socklen_t len = sizeof(user_cred);
   int res = getsockopt(sock_raw_.fd(), 0, LOCAL_PEERCRED, &user_cred, &len);
>>>>>>> upstream/main
