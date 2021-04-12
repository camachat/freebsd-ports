<<<<<<< HEAD
--- remoting/host/it2me/it2me_native_messaging_host_main.cc.orig	2021-01-18 21:29:02 UTC
+++ remoting/host/it2me/it2me_native_messaging_host_main.cc
@@ -29,12 +29,12 @@
=======
--- remoting/host/it2me/it2me_native_messaging_host_main.cc.orig	2021-03-12 23:57:28 UTC
+++ remoting/host/it2me/it2me_native_messaging_host_main.cc
@@ -30,12 +30,12 @@
>>>>>>> upstream/main
 #include "remoting/host/switches.h"
 #include "remoting/host/usage_stats_consent.h"
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include <gtk/gtk.h>
 
 #include "base/linux_util.h"
 #include "ui/events/platform/x11/x11_event_source.h"
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 #if defined(OS_APPLE)
 #include "base/mac/mac_util.h"
<<<<<<< HEAD
@@ -112,7 +112,7 @@ int It2MeNativeMessagingHostMain(int argc, char** argv
=======
@@ -114,7 +114,7 @@ int It2MeNativeMessagingHostMain(int argc, char** argv
>>>>>>> upstream/main
 
   remoting::LoadResources("");
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // Create an X11EventSource so the global X11 connection
   // (x11::Connection::Get()) can dispatch X events.
   auto event_source =
<<<<<<< HEAD
@@ -130,7 +130,7 @@ int It2MeNativeMessagingHostMain(int argc, char** argv
=======
@@ -132,7 +132,7 @@ int It2MeNativeMessagingHostMain(int argc, char** argv
>>>>>>> upstream/main
   // Need to prime the host OS version value for linux to prevent IO on the
   // network thread. base::GetLinuxDistro() caches the result.
   base::GetLinuxDistro();
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
   base::File read_file;
   base::File write_file;
