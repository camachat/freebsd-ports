<<<<<<< HEAD
--- net/nqe/network_quality_estimator.cc.orig	2020-11-13 06:36:46 UTC
+++ net/nqe/network_quality_estimator.cc
@@ -108,7 +108,7 @@ nqe::internal::NetworkID DoGetCurrentNetworkID(
=======
--- net/nqe/network_quality_estimator.cc.orig	2021-03-12 23:57:27 UTC
+++ net/nqe/network_quality_estimator.cc
@@ -109,7 +109,7 @@ nqe::internal::NetworkID DoGetCurrentNetworkID(
>>>>>>> upstream/main
       case NetworkChangeNotifier::ConnectionType::CONNECTION_ETHERNET:
         break;
       case NetworkChangeNotifier::ConnectionType::CONNECTION_WIFI:
-#if defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_WIN)
         network_id.id = GetWifiSSID();
 #endif
