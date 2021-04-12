<<<<<<< HEAD
--- extensions/shell/browser/api/runtime/shell_runtime_api_delegate.cc.orig	2021-01-18 21:28:59 UTC
+++ extensions/shell/browser/api/runtime/shell_runtime_api_delegate.cc
@@ -46,7 +46,7 @@ void ShellRuntimeAPIDelegate::OpenURL(const GURL& unin
=======
--- extensions/shell/browser/api/runtime/shell_runtime_api_delegate.cc.orig	2021-03-12 23:57:25 UTC
+++ extensions/shell/browser/api/runtime/shell_runtime_api_delegate.cc
@@ -45,7 +45,7 @@ void ShellRuntimeAPIDelegate::OpenURL(const GURL& unin
>>>>>>> upstream/main
 bool ShellRuntimeAPIDelegate::GetPlatformInfo(PlatformInfo* info) {
 #if BUILDFLAG(IS_CHROMEOS_ASH)
   info->os = api::runtime::PLATFORM_OS_CROS;
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   info->os = api::runtime::PLATFORM_OS_LINUX;
 #endif
   return true;
