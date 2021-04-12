<<<<<<< HEAD
--- services/network/network_service.cc.orig	2021-01-18 21:29:02 UTC
=======
--- services/network/network_service.cc.orig	2021-03-12 23:57:28 UTC
>>>>>>> upstream/main
+++ services/network/network_service.cc
@@ -73,7 +73,7 @@
 #include "third_party/boringssl/src/include/openssl/cpu.h"
 #endif
 
<<<<<<< HEAD
-#if (defined(OS_LINUX) || BUILDFLAG(IS_LACROS)) && !BUILDFLAG(IS_CHROMECAST)
+#if ((defined(OS_LINUX) || BUILDFLAG(IS_LACROS)) && !BUILDFLAG(IS_CHROMECAST)) || defined(OS_BSD)
 #include "components/os_crypt/key_storage_config_linux.h"
 #endif
 
@@ -678,7 +678,7 @@ void NetworkService::OnCertDBChanged() {
   net::CertDatabase::GetInstance()->NotifyObserversCertDBChanged();
 }
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_LACROS) || defined(OS_BSD)
=======
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && \
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && \
     !BUILDFLAG(IS_CHROMECAST)
 #include "components/os_crypt/key_storage_config_linux.h"
 #endif
@@ -680,7 +680,7 @@ void NetworkService::OnCertDBChanged() {
   net::CertDatabase::GetInstance()->NotifyObserversCertDBChanged();
 }
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 void NetworkService::SetCryptConfig(mojom::CryptConfigPtr crypt_config) {
 #if !BUILDFLAG(IS_CHROMECAST)
   DCHECK(!os_crypt_config_set_);
