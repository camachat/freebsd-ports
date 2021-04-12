<<<<<<< HEAD
--- chromecast/external_mojo/public/cpp/external_mojo_broker.cc.orig	2020-11-13 06:36:40 UTC
+++ chromecast/external_mojo/public/cpp/external_mojo_broker.cc
@@ -7,7 +7,7 @@
 #include <map>
=======
--- chromecast/external_mojo/public/cpp/external_mojo_broker.cc.orig	2021-03-12 23:57:21 UTC
+++ chromecast/external_mojo/public/cpp/external_mojo_broker.cc
@@ -8,7 +8,7 @@
 #include <set>
>>>>>>> upstream/main
 #include <utility>
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include <sys/stat.h>
 #endif
 
<<<<<<< HEAD
@@ -412,7 +412,7 @@ ExternalMojoBroker::ExternalMojoBroker(const std::stri
=======
@@ -419,7 +419,7 @@ ExternalMojoBroker::ExternalMojoBroker(const std::stri
>>>>>>> upstream/main
       named_channel.TakeServerEndpoint();
   DCHECK(server_endpoint.is_valid());
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   chmod(broker_path.c_str(), 0770);
 #endif
 
