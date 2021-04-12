<<<<<<< HEAD
--- content/public/common/content_switches.h.orig	2021-01-18 21:28:57 UTC
=======
--- content/public/common/content_switches.h.orig	2021-03-12 23:57:24 UTC
>>>>>>> upstream/main
+++ content/public/common/content_switches.h
@@ -245,7 +245,7 @@ CONTENT_EXPORT extern const char kWebXrRuntimeOrientat
 CONTENT_EXPORT extern const char kWebXrRuntimeOpenXr[];
 CONTENT_EXPORT extern const char kWebXrRuntimeWMR[];
 
<<<<<<< HEAD
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_ANDROID)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_ANDROID)) || defined(OS_BSD)
 CONTENT_EXPORT extern const char kEnableAcceleratedVideoDecode[];
 #endif
 CONTENT_EXPORT extern const char kDisableAcceleratedVideoDecode[];
@@ -266,7 +266,7 @@ CONTENT_EXPORT extern const char kRemoteDebuggingSocke
 CONTENT_EXPORT extern const char kRendererWaitForJavaDebugger[];
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_FREEBSD)
=======
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && !defined(OS_ANDROID)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && !defined(OS_ANDROID)
 CONTENT_EXPORT extern const char kEnableAcceleratedVideoDecode[];
 #endif
 CONTENT_EXPORT extern const char kDisableAcceleratedVideoDecode[];
@@ -268,7 +268,7 @@ CONTENT_EXPORT extern const char kRendererWaitForJavaD
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 CONTENT_EXPORT extern const char kEnableSpeechDispatcher[];
 #endif
 
