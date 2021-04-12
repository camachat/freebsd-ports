<<<<<<< HEAD
--- services/device/geolocation/location_arbitrator.cc.orig	2020-11-13 06:36:46 UTC
=======
--- services/device/geolocation/location_arbitrator.cc.orig	2021-03-12 23:57:28 UTC
>>>>>>> upstream/main
+++ services/device/geolocation/location_arbitrator.cc
@@ -156,7 +156,7 @@ LocationArbitrator::NewNetworkLocationProvider(
 
 std::unique_ptr<LocationProvider>
 LocationArbitrator::NewSystemLocationProvider() {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_FUCHSIA)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_FUCHSIA) || defined(OS_BSD)
   return nullptr;
 #else
   return device::NewSystemLocationProvider();
