<<<<<<< HEAD
--- remoting/host/heartbeat_sender.cc.orig	2021-01-19 11:48:10 UTC
+++ remoting/host/heartbeat_sender.cc
@@ -109,7 +109,7 @@ const net::BackoffEntry::Policy kBackoffPolicy = {
 };
 
 std::string GetHostname() {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- remoting/host/heartbeat_sender.cc.orig	2021-03-12 23:57:28 UTC
+++ remoting/host/heartbeat_sender.cc
@@ -112,7 +112,7 @@ const net::BackoffEntry::Policy kBackoffPolicy = {
 std::string GetHostname() {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag
 // switch of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   return net::GetHostName();
 #elif defined(OS_WIN)
   wchar_t buffer[MAX_PATH] = {0};
