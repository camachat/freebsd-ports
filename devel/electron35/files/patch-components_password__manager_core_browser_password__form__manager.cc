--- components/password_manager/core/browser/password_form_manager.cc.orig	2025-03-24 20:50:14 UTC
+++ components/password_manager/core/browser/password_form_manager.cc
@@ -63,7 +63,7 @@
 #include "components/webauthn/android/webauthn_cred_man_delegate.h"
 #endif  // BUILDFLAG(IS_ANDROID)
 
-#if BUILDFLAG(IS_MAC) || BUILDFLAG(IS_LINUX)
+#if BUILDFLAG(IS_MAC) || BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_BSD)
 #include "components/os_crypt/sync/os_crypt.h"
 #endif
 
@@ -239,7 +239,7 @@ bool ShouldUploadCrowdsourcingVotes(const FormOrDigest
   return false;
 }
 
-#if BUILDFLAG(IS_MAC) || BUILDFLAG(IS_LINUX)
+#if BUILDFLAG(IS_MAC) || BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_BSD)
 bool ShouldShowKeychainErrorBubble(
     std::optional<PasswordStoreBackendError> backend_error) {
   if (!backend_error.has_value()) {
@@ -897,7 +897,7 @@ void PasswordFormManager::OnFetchCompleted() {
         error.value().type);
   }
 
-#elif BUILDFLAG(IS_MAC) || BUILDFLAG(IS_LINUX)
+#elif BUILDFLAG(IS_MAC) || BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_BSD)
   if (ShouldShowKeychainErrorBubble(
           form_fetcher_->GetProfileStoreBackendError())) {
     client_->NotifyKeychainError();
