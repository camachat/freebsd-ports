<<<<<<< HEAD
--- weblayer/app/content_main_delegate_impl.cc.orig	2021-01-18 21:29:49 UTC
+++ weblayer/app/content_main_delegate_impl.cc
@@ -221,7 +221,7 @@ bool ContentMainDelegateImpl::ShouldCreateFeatureList(
 
 void ContentMainDelegateImpl::PreSandboxStartup() {
 #if defined(ARCH_CPU_ARM_FAMILY) && \
-    (defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS))
+    (defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD))
=======
--- weblayer/app/content_main_delegate_impl.cc.orig	2021-03-12 23:57:49 UTC
+++ weblayer/app/content_main_delegate_impl.cc
@@ -218,7 +218,7 @@ bool ContentMainDelegateImpl::ShouldCreateFeatureList(
 }
 
 void ContentMainDelegateImpl::PreSandboxStartup() {
-#if defined(ARCH_CPU_ARM_FAMILY) && (defined(OS_ANDROID) || defined(OS_LINUX))
+#if defined(ARCH_CPU_ARM_FAMILY) && (defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_BSD))
>>>>>>> upstream/main
   // Create an instance of the CPU class to parse /proc/cpuinfo and cache
   // cpu_brand info.
   base::CPU cpu_info;
