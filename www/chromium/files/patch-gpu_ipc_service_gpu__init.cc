<<<<<<< HEAD
--- gpu/ipc/service/gpu_init.cc.orig	2021-01-18 21:28:59 UTC
+++ gpu/ipc/service/gpu_init.cc
@@ -107,7 +107,7 @@ void InitializePlatformOverlaySettings(GPUInfo* gpu_in
 #endif
 }
 
-#if BUILDFLAG(IS_LACROS) || (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMECAST))
+#if BUILDFLAG(IS_LACROS) || (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMECAST)) || defined(OS_BSD)
 bool CanAccessNvidiaDeviceFile() {
   bool res = true;
   base::ScopedBlockingCall scoped_blocking_call(FROM_HERE,
@@ -118,8 +118,7 @@ bool CanAccessNvidiaDeviceFile() {
   }
   return res;
 }
-#endif  // BUILDFLAG(IS_LACROS) || (defined(OS_LINUX)  &&
-        // !BUILDFLAG(IS_CHROMECAST))
+#endif  // BUILDFLAG(IS_LACROS) || (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMECAST)) || defined(OS_BSD)
 
 class GpuWatchdogInit {
  public:
@@ -205,7 +204,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
     device_perf_info_ = device_perf_info;
   }
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_LACROS) || defined(OS_BSD)
   if (gpu_info_.gpu.vendor_id == 0x10de &&  // NVIDIA
       gpu_info_.gpu.driver_vendor == "NVIDIA" && !CanAccessNvidiaDeviceFile())
     return false;
@@ -257,7 +256,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
--- gpu/ipc/service/gpu_init.cc.orig	2021-03-12 23:57:25 UTC
+++ gpu/ipc/service/gpu_init.cc
@@ -108,7 +108,7 @@ void InitializePlatformOverlaySettings(GPUInfo* gpu_in
 }
 
 #if BUILDFLAG(IS_CHROMEOS_LACROS) || \
-    (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMECAST))
+    (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMECAST)) || defined(OS_BSD)
 bool CanAccessNvidiaDeviceFile() {
   bool res = true;
   base::ScopedBlockingCall scoped_blocking_call(FROM_HERE,
@@ -119,7 +119,7 @@ bool CanAccessNvidiaDeviceFile() {
   }
   return res;
 }
-#endif  // BUILDFLAG(IS_CHROMEOS_LACROS) || (defined(OS_LINUX)  &&
+#endif  // BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD) || (defined(OS_LINUX)  &&
         // !BUILDFLAG(IS_CHROMECAST))
 
 class GpuWatchdogInit {
@@ -206,7 +206,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
     device_perf_info_ = device_perf_info;
   }
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   if (gpu_info_.gpu.vendor_id == 0x10de &&  // NVIDIA
       gpu_info_.gpu.driver_vendor == "NVIDIA" && !CanAccessNvidiaDeviceFile())
     return false;
@@ -258,7 +258,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
   delayed_watchdog_enable = true;
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // PreSandbox is mainly for resource handling and not related to the GPU
   // driver, it doesn't need the GPU watchdog. The loadLibrary may take long
   // time that killing and restarting the GPU process will not help.
<<<<<<< HEAD
@@ -297,7 +296,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -298,7 +298,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
   }
 
   bool attempted_startsandbox = false;
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // On Chrome OS ARM Mali, GPU driver userspace creates threads when
   // initializing a GL context, so start the sandbox early.
   // TODO(zmo): Need to collect OS version before this.
<<<<<<< HEAD
@@ -306,7 +305,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -307,7 +307,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
         watchdog_thread_.get(), &gpu_info_, gpu_preferences_);
     attempted_startsandbox = true;
   }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
   base::TimeTicks before_initialize_one_off = base::TimeTicks::Now();
 
<<<<<<< HEAD
@@ -334,14 +333,14 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -345,7 +345,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
   }
   if (gl_initialized && gl_use_swiftshader_ &&
       gl::GetGLImplementation() != gl::kGLImplementationSwiftShaderGL) {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
     VLOG(1) << "Quit GPU process launch to fallback to SwiftShader cleanly "
             << "on Linux";
     return false;
<<<<<<< HEAD
 #else
=======
@@ -353,7 +353,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
     SaveHardwareGpuInfoAndGpuFeatureInfo();
>>>>>>> upstream/main
     gl::init::ShutdownGL(true);
     gl_initialized = false;
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   }
 
   if (!gl_initialized) {
<<<<<<< HEAD
@@ -367,7 +366,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -379,7 +379,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
     }
   }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // The ContentSandboxHelper is currently the only one implementation of
   // GpuSandboxHelper and it has no dependency. Except on Linux where
   // VaapiWrapper checks the GL implementation to determine which display
<<<<<<< HEAD
@@ -421,7 +420,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -433,7 +433,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
           command_line, gpu_feature_info_,
           gpu_preferences_.disable_software_rasterizer, false);
       if (gl_use_swiftshader_) {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
         VLOG(1) << "Quit GPU process launch to fallback to SwiftShader cleanly "
                 << "on Linux";
         return false;
<<<<<<< HEAD
@@ -435,7 +434,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -448,7 +448,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
               << "failed";
           return false;
         }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
       }
     } else {  // gl_use_swiftshader_ == true
       switch (gpu_preferences_.use_vulkan) {
<<<<<<< HEAD
@@ -511,7 +510,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -524,7 +524,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
 
   InitializePlatformOverlaySettings(&gpu_info_, gpu_feature_info_);
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // Driver may create a compatibility profile context when collect graphics
   // information on Linux platform. Try to collect graphics information
   // based on core profile context after disabling platform extensions.
<<<<<<< HEAD
@@ -530,7 +529,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
=======
@@ -543,7 +543,7 @@ bool GpuInit::InitializeAndStartSandbox(base::CommandL
>>>>>>> upstream/main
       return false;
     }
   }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
   if (gl_use_swiftshader_) {
     AdjustInfoToSwiftShader();
<<<<<<< HEAD
@@ -700,7 +699,7 @@ void GpuInit::InitializeInProcess(base::CommandLine* c
=======
@@ -726,7 +726,7 @@ void GpuInit::InitializeInProcess(base::CommandLine* c
>>>>>>> upstream/main
 
   InitializePlatformOverlaySettings(&gpu_info_, gpu_feature_info_);
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // Driver may create a compatibility profile context when collect graphics
   // information on Linux platform. Try to collect graphics information
   // based on core profile context after disabling platform extensions.
<<<<<<< HEAD
@@ -720,7 +719,7 @@ void GpuInit::InitializeInProcess(base::CommandLine* c
=======
@@ -747,7 +747,7 @@ void GpuInit::InitializeInProcess(base::CommandLine* c
>>>>>>> upstream/main
       }
     }
   }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
   if (gl_use_swiftshader_) {
     AdjustInfoToSwiftShader();
