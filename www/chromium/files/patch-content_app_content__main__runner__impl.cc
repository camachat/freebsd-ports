<<<<<<< HEAD
--- content/app/content_main_runner_impl.cc.orig	2021-01-18 21:28:57 UTC
+++ content/app/content_main_runner_impl.cc
@@ -134,7 +134,7 @@
=======
--- content/app/content_main_runner_impl.cc.orig	2021-03-12 23:57:24 UTC
+++ content/app/content_main_runner_impl.cc
@@ -135,7 +135,7 @@
>>>>>>> upstream/main
 
 #endif  // OS_POSIX || OS_FUCHSIA
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "base/native_library.h"
 #include "base/rand_util.h"
 #include "content/public/common/zygote/sandbox_support_linux.h"
<<<<<<< HEAD
@@ -154,7 +154,7 @@
=======
@@ -155,7 +155,7 @@
>>>>>>> upstream/main
 #include "content/public/common/content_client.h"
 #endif
 
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 #if BUILDFLAG(USE_ZYGOTE_HANDLE)
 #include "content/browser/sandbox_host_linux.h"
<<<<<<< HEAD
@@ -310,7 +310,7 @@ void InitializeZygoteSandboxForBrowserProcess(
=======
@@ -342,7 +342,7 @@ void InitializeZygoteSandboxForBrowserProcess(
>>>>>>> upstream/main
 }
 #endif  // BUILDFLAG(USE_ZYGOTE_HANDLE)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 #if BUILDFLAG(ENABLE_PLUGINS)
 // Loads the (native) libraries but does not initialize them (i.e., does not
<<<<<<< HEAD
@@ -401,7 +401,7 @@ void PreSandboxInit() {
=======
@@ -433,7 +433,7 @@ void PreSandboxInit() {
>>>>>>> upstream/main
 }
 #endif  // BUILDFLAG(USE_ZYGOTE_HANDLE)
 
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 }  // namespace
 
<<<<<<< HEAD
@@ -464,7 +464,7 @@ int RunZygote(ContentMainDelegate* delegate) {
=======
@@ -496,7 +496,7 @@ int RunZygote(ContentMainDelegate* delegate) {
>>>>>>> upstream/main
   delegate->ZygoteStarting(&zygote_fork_delegates);
   media::InitializeMediaLibrary();
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   PreSandboxInit();
 #endif
 
<<<<<<< HEAD
@@ -855,7 +855,7 @@ int ContentMainRunnerImpl::Run(bool start_service_mana
=======
@@ -900,7 +900,7 @@ int ContentMainRunnerImpl::Run(bool start_minimal_brow
>>>>>>> upstream/main
       mojo::core::InitFeatures();
     }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
     // If dynamic Mojo Core is being used, ensure that it's loaded very early in
     // the child/zygote process, before any sandbox is initialized. The library
     // is not fully initialized with IPC support until a ChildProcess is later
<<<<<<< HEAD
@@ -865,7 +865,7 @@ int ContentMainRunnerImpl::Run(bool start_service_mana
=======
@@ -910,7 +910,7 @@ int ContentMainRunnerImpl::Run(bool start_minimal_brow
>>>>>>> upstream/main
       CHECK_EQ(mojo::LoadCoreLibrary(GetMojoCoreSharedLibraryPath()),
                MOJO_RESULT_OK);
     }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   }
 
   MainFunctionParams main_params(command_line);
