<<<<<<< HEAD
--- chrome/browser/password_manager/password_store_factory.cc.orig	2021-01-18 21:28:50 UTC
+++ chrome/browser/password_manager/password_store_factory.cc
@@ -171,7 +171,7 @@ PasswordStoreFactory::BuildServiceInstanceFor(
       network_context_getter);
 
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+    (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/password_manager/password_store_factory.cc.orig	2021-03-12 23:57:18 UTC
+++ chrome/browser/password_manager/password_store_factory.cc
@@ -163,7 +163,7 @@ PasswordStoreFactory::BuildServiceInstanceFor(
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
   std::unique_ptr<password_manager::PasswordStoreSigninNotifier> notifier =
       std::make_unique<password_manager::PasswordStoreSigninNotifierImpl>(
           IdentityManagerFactory::GetForProfile(profile));
