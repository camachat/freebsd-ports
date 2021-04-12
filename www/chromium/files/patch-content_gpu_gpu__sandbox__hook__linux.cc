<<<<<<< HEAD
--- content/gpu/gpu_sandbox_hook_linux.cc.orig	2020-11-19 08:18:33 UTC
+++ content/gpu/gpu_sandbox_hook_linux.cc
@@ -361,6 +361,7 @@ std::vector<BrokerFilePermission> FilePermissionsForGp
=======
--- content/gpu/gpu_sandbox_hook_linux.cc.orig	2021-03-12 23:57:24 UTC
+++ content/gpu/gpu_sandbox_hook_linux.cc
@@ -362,6 +362,7 @@ std::vector<BrokerFilePermission> FilePermissionsForGp
>>>>>>> upstream/main
 }
 
 void LoadArmGpuLibraries() {
+#if !defined(OS_BSD)
   // Preload the Mali library.
   if (UseChromecastSandboxAllowlist()) {
     for (const char* path : kAllowedChromecastPaths) {
<<<<<<< HEAD
@@ -375,6 +376,7 @@ void LoadArmGpuLibraries() {
=======
@@ -376,6 +377,7 @@ void LoadArmGpuLibraries() {
>>>>>>> upstream/main
     // Preload the Tegra V4L2 (video decode acceleration) library.
     dlopen(kLibTegraPath, dlopen_flag);
   }
+#endif
 }
 
 bool LoadAmdGpuLibraries() {
