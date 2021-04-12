<<<<<<< HEAD
--- content/public/common/content_features.cc.orig	2021-01-18 21:28:57 UTC
+++ content/public/common/content_features.cc
@@ -49,7 +49,7 @@ const base::Feature kAudioServiceLaunchOnStartup{
 const base::Feature kAudioServiceOutOfProcess {
   "AudioServiceOutOfProcess",
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) && !defined(OS_CHROMEOS))
+    (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
       base::FEATURE_DISABLED_BY_DEFAULT
@@ -758,8 +758,8 @@ const base::Feature kWebAssemblyThreads {
=======
--- content/public/common/content_features.cc.orig	2021-03-12 23:57:24 UTC
+++ content/public/common/content_features.cc
@@ -45,7 +45,7 @@ const base::Feature kAudioServiceOutOfProcess {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
 #if defined(OS_WIN) || defined(OS_MAC) || \
-    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+    (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
       base::FEATURE_DISABLED_BY_DEFAULT
@@ -773,8 +773,8 @@ const base::Feature kWebAssemblyThreads {
>>>>>>> upstream/main
 };
 
 // Enable WebAssembly trap handler.
-#if (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_WIN) || \
-     defined(OS_MAC)) &&                                             \
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_WIN) || defined(OS_BSD) || \
+     defined(OS_MAC)) &&                                                                \
     defined(ARCH_CPU_X86_64)
 const base::Feature kWebAssemblyTrapHandler{"WebAssemblyTrapHandler",
                                             base::FEATURE_ENABLED_BY_DEFAULT};
<<<<<<< HEAD
@@ -789,7 +789,7 @@ const base::Feature kWebAuth{"WebAuthentication",
 // https://w3c.github.io/webauthn
 const base::Feature kWebAuthCable {
   "WebAuthenticationCable",
-#if !defined(OS_CHROMEOS) && defined(OS_LINUX)
+#if (!defined(OS_CHROMEOS) && defined(OS_LINUX)) || defined(OS_BSD)
=======
@@ -795,7 +795,7 @@ const base::Feature kWebAuthCable {
   "WebAuthenticationCable",
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX)
+#if BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_LINUX) || defined(OS_BSD)
>>>>>>> upstream/main
       base::FEATURE_DISABLED_BY_DEFAULT
 #else
       base::FEATURE_ENABLED_BY_DEFAULT
