<<<<<<< HEAD
--- weblayer/browser/system_network_context_manager.cc.orig	2020-11-13 06:37:06 UTC
=======
--- weblayer/browser/system_network_context_manager.cc.orig	2021-03-12 23:57:49 UTC
>>>>>>> upstream/main
+++ weblayer/browser/system_network_context_manager.cc
@@ -64,10 +64,10 @@ void SystemNetworkContextManager::ConfigureDefaultNetw
     network::mojom::NetworkContextParams* network_context_params,
     const std::string& user_agent) {
   network_context_params->user_agent = user_agent;
<<<<<<< HEAD
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_WIN)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_WIN) || defined(OS_BSD)
   // We're not configuring the cookie encryption on these platforms yet.
   network_context_params->enable_encrypted_cookies = false;
-#endif // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_WIN)
+#endif // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_WIN) || defined(OS_BSD)
=======
-#if defined(OS_LINUX) || defined(OS_WIN)
+#if defined(OS_LINUX) || defined(OS_WIN) || defined(OS_BSD)
   // We're not configuring the cookie encryption on these platforms yet.
   network_context_params->enable_encrypted_cookies = false;
-#endif  // defined(OS_LINUX) || defined(OS_WIN)
+#endif  // defined(OS_LINUX) || defined(OS_WIN) || defined(OS_BSD)
>>>>>>> upstream/main
 }
 
 SystemNetworkContextManager::SystemNetworkContextManager(
