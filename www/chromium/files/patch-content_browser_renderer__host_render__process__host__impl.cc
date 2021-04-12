<<<<<<< HEAD
--- content/browser/renderer_host/render_process_host_impl.cc.orig	2021-01-18 21:28:57 UTC
+++ content/browser/renderer_host/render_process_host_impl.cc
@@ -235,7 +235,7 @@
=======
--- content/browser/renderer_host/render_process_host_impl.cc.orig	2021-03-12 23:57:24 UTC
+++ content/browser/renderer_host/render_process_host_impl.cc
@@ -229,7 +229,7 @@
>>>>>>> upstream/main
 #include "third_party/blink/public/mojom/android_font_lookup/android_font_lookup.mojom.h"
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include <sys/resource.h>
 #include <sys/time.h>
 
<<<<<<< HEAD
@@ -1219,7 +1219,7 @@ static constexpr size_t kUnknownPlatformProcessLimit =
=======
@@ -1205,7 +1205,7 @@ static constexpr size_t kUnknownPlatformProcessLimit =
>>>>>>> upstream/main
 // to indicate failure and std::numeric_limits<size_t>::max() to indicate
 // unlimited.
 size_t GetPlatformProcessLimit() {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   struct rlimit limit;
   if (getrlimit(RLIMIT_NPROC, &limit) != 0)
     return kUnknownPlatformProcessLimit;
<<<<<<< HEAD
@@ -1230,7 +1230,7 @@ size_t GetPlatformProcessLimit() {
=======
@@ -1216,7 +1216,7 @@ size_t GetPlatformProcessLimit() {
>>>>>>> upstream/main
 #else
   // TODO(https://crbug.com/104689): Implement on other platforms.
   return kUnknownPlatformProcessLimit;
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 }
 #endif  // !defined(OS_ANDROID) && !BUILDFLAG(IS_CHROMEOS_ASH)
 
<<<<<<< HEAD
@@ -1315,7 +1315,7 @@ class RenderProcessHostImpl::IOThreadHostImpl : public
=======
@@ -1290,7 +1290,7 @@ class RenderProcessHostImpl::IOThreadHostImpl : public
>>>>>>> upstream/main
         return;
     }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
     if (auto font_receiver = receiver.As<font_service::mojom::FontService>()) {
       ConnectToFontService(std::move(font_receiver));
       return;
<<<<<<< HEAD
@@ -1747,7 +1747,7 @@ bool RenderProcessHostImpl::Init() {
=======
@@ -1720,7 +1720,7 @@ bool RenderProcessHostImpl::Init() {
>>>>>>> upstream/main
   renderer_prefix =
       browser_command_line.GetSwitchValueNative(switches::kRendererCmdPrefix);
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   int flags = renderer_prefix.empty() ? ChildProcessHost::CHILD_ALLOW_SELF
                                       : ChildProcessHost::CHILD_NORMAL;
 #elif defined(OS_MAC)
<<<<<<< HEAD
@@ -3254,11 +3254,11 @@ void RenderProcessHostImpl::PropagateBrowserCommandLin
=======
@@ -3161,8 +3161,8 @@ void RenderProcessHostImpl::PropagateBrowserCommandLin
>>>>>>> upstream/main
     switches::kDisableInProcessStackTraces,
     sandbox::policy::switches::kDisableSeccompFilterSandbox,
     sandbox::policy::switches::kNoSandbox,
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_ASH) && \
-    !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_BSD) || (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_ASH) && \
+    !BUILDFLAG(IS_CHROMEOS_LACROS))
     switches::kDisableDevShmUsage,
 #endif
<<<<<<< HEAD
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && !defined(OS_ANDROID)
+#if ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && !defined(OS_ANDROID)) || defined(OS_BSD)
     switches::kEnableAcceleratedVideoDecode,
 #endif
=======
>>>>>>> upstream/main
 #if defined(OS_MAC)
