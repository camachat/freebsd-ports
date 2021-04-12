<<<<<<< HEAD
--- headless/lib/browser/headless_request_context_manager.cc.orig	2021-01-18 21:28:59 UTC
+++ headless/lib/browser/headless_request_context_manager.cc
@@ -26,7 +26,7 @@ namespace headless {
 
 namespace {
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 constexpr char kProductName[] = "HeadlessChrome";
 #endif
 
@@ -56,7 +56,7 @@ net::NetworkTrafficAnnotationTag GetProxyConfigTraffic
   return traffic_annotation;
 }
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 ::network::mojom::CryptConfigPtr BuildCryptConfigOnce(
     const base::FilePath& user_data_path) {
   static bool done_once = false;
@@ -218,7 +218,7 @@ HeadlessRequestContextManager::HeadlessRequestContextM
           base::ThreadTaskRunnerHandle::Get());
     }
   }
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
   auto crypt_config = BuildCryptConfigOnce(user_data_path_);
   if (crypt_config)
     content::GetNetworkService()->SetCryptConfig(std::move(crypt_config));
=======
--- headless/lib/browser/headless_request_context_manager.cc.orig	2021-03-12 23:57:25 UTC
+++ headless/lib/browser/headless_request_context_manager.cc
@@ -34,7 +34,7 @@ namespace {
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 constexpr char kProductName[] = "HeadlessChrome";
 #endif
 
@@ -72,7 +72,7 @@ void SetCryptConfigOnce(const base::FilePath& user_dat
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   ::network::mojom::CryptConfigPtr config =
       ::network::mojom::CryptConfig::New();
   config->store = base::CommandLine::ForCurrentProcess()->GetSwitchValueASCII(
>>>>>>> upstream/main
