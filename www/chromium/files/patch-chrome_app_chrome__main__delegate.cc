<<<<<<< HEAD
--- chrome/app/chrome_main_delegate.cc.orig	2021-01-18 21:28:46 UTC
+++ chrome/app/chrome_main_delegate.cc
@@ -148,12 +148,12 @@
=======
--- chrome/app/chrome_main_delegate.cc.orig	2021-03-12 23:57:16 UTC
+++ chrome/app/chrome_main_delegate.cc
@@ -149,12 +149,12 @@
>>>>>>> upstream/main
 #include "v8/include/v8.h"
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "base/environment.h"
 #endif
 
 #if defined(OS_MAC) || defined(OS_WIN) || defined(OS_ANDROID) || \
-    defined(OS_LINUX) || defined(OS_CHROMEOS)
+    defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "chrome/browser/policy/policy_path_parser.h"
 #include "components/crash/core/app/crashpad.h"
 #endif
<<<<<<< HEAD
@@ -259,7 +259,7 @@ void SetUpExtendedCrashReporting(bool is_browser_proce
=======
@@ -260,7 +260,7 @@ void SetUpExtendedCrashReporting(bool is_browser_proce
>>>>>>> upstream/main
 
 #endif  // defined(OS_WIN)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS))
 void AdjustLinuxOOMScore(const std::string& process_type) {
   int score = -1;
 
<<<<<<< HEAD
@@ -294,13 +294,13 @@ void AdjustLinuxOOMScore(const std::string& process_ty
=======
@@ -295,7 +295,7 @@ void AdjustLinuxOOMScore(const std::string& process_ty
>>>>>>> upstream/main
   if (score > -1)
     base::AdjustOOMScore(base::GetCurrentProcId(), score);
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // (defined(OS_LINUX) || defined(OS_CHROMEOS))
 
 // Returns true if this subprocess type needs the ResourceBundle initialized
 // and resources loaded.
<<<<<<< HEAD
 bool SubprocessNeedsResourceBundle(const std::string& process_type) {
   return
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
       // The zygote process opens the resources for the renderers.
       process_type == switches::kZygoteProcess ||
 #endif
@@ -338,7 +338,7 @@ bool HandleVersionSwitches(const base::CommandLine& co
   return false;
 }
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 // Show the man page if --help or -h is on the command line.
 void HandleHelpSwitches(const base::CommandLine& command_line) {
   if (command_line.HasSwitch(switches::kHelp) ||
@@ -348,7 +348,7 @@ void HandleHelpSwitches(const base::CommandLine& comma
     PLOG(FATAL) << "execlp failed";
   }
 }
-#endif  // defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#endif  // (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 
 #if !defined(OS_MAC) && !defined(OS_ANDROID)
 void SIGTERMProfilingShutdown(int signal) {
@@ -402,7 +402,7 @@ void InitializeUserDataDir(base::CommandLine* command_
=======
@@ -341,7 +341,7 @@ bool HandleVersionSwitches(const base::CommandLine& co
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 // Show the man page if --help or -h is on the command line.
 void HandleHelpSwitches(const base::CommandLine& command_line) {
   if (command_line.HasSwitch(switches::kHelp) ||
@@ -351,7 +351,7 @@ void HandleHelpSwitches(const base::CommandLine& comma
     PLOG(FATAL) << "execlp failed";
   }
 }
-#endif  // defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#endif  // defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 
 #if !defined(OS_MAC) && !defined(OS_ANDROID)
 void SIGTERMProfilingShutdown(int signal) {
@@ -405,7 +405,7 @@ void InitializeUserDataDir(base::CommandLine* command_
>>>>>>> upstream/main
   std::string process_type =
       command_line->GetSwitchValueASCII(switches::kProcessType);
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // On Linux, Chrome does not support running multiple copies under different
   // DISPLAYs, so the profile directory can be specified in the environment to
   // support the virtual desktop use-case.
<<<<<<< HEAD
@@ -414,7 +414,7 @@ void InitializeUserDataDir(base::CommandLine* command_
=======
@@ -417,7 +417,7 @@ void InitializeUserDataDir(base::CommandLine* command_
>>>>>>> upstream/main
       user_data_dir = base::FilePath::FromUTF8Unsafe(user_data_dir_string);
     }
   }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #if defined(OS_MAC)
   policy::path_parser::CheckUserDataDirPolicy(&user_data_dir);
 #endif  // OS_MAC
<<<<<<< HEAD
@@ -486,7 +486,7 @@ void RecordMainStartupMetrics(base::TimeTicks applicat
 #endif
 
 #if defined(OS_MAC) || defined(OS_WIN) || defined(OS_LINUX) || \
-    defined(OS_CHROMEOS)
+    defined(OS_CHROMEOS) || defined(OS_BSD)
   // Record the startup process creation time on supported platforms. On Android
   // this is recorded in ChromeMainDelegateAndroid.
   startup_metric_utils::RecordStartupProcessCreationTime(
@@ -709,7 +709,7 @@ bool ChromeMainDelegate::BasicStartupComplete(int* exi
=======
@@ -488,7 +488,7 @@ void RecordMainStartupMetrics(base::TimeTicks applicat
   startup_metric_utils::RecordApplicationStartTime(now);
 #endif
 
-#if defined(OS_MAC) || defined(OS_WIN) || defined(OS_LINUX) || \
+#if defined(OS_MAC) || defined(OS_WIN) || defined(OS_LINUX) || defined(OS_BSD) || \
     defined(OS_CHROMEOS)
   // Record the startup process creation time on supported platforms. On Android
   // this is recorded in ChromeMainDelegateAndroid.
@@ -723,7 +723,7 @@ bool ChromeMainDelegate::BasicStartupComplete(int* exi
>>>>>>> upstream/main
   v8_crashpad_support::SetUp();
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS))
   if (!crash_reporter::IsCrashpadEnabled()) {
     breakpad::SetFirstChanceExceptionHandler(v8::TryHandleWebAssemblyTrapPosix);
   }
<<<<<<< HEAD
@@ -720,7 +720,7 @@ bool ChromeMainDelegate::BasicStartupComplete(int* exi
     *exit_code = 0;
     return true;  // Got a --version switch; exit with a success error code.
   }
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
   // This will directly exit if the user asked for help.
   HandleHelpSwitches(command_line);
 #endif
@@ -928,7 +928,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
=======
@@ -736,7 +736,7 @@ bool ChromeMainDelegate::BasicStartupComplete(int* exi
   }
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   // This will directly exit if the user asked for help.
   HandleHelpSwitches(command_line);
 #endif
@@ -945,7 +945,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
>>>>>>> upstream/main
 
   crash_reporter::InitializeCrashKeys();
 
-#if defined(OS_POSIX)
+#if defined(OS_POSIX) && !defined(OS_BSD)
   ChromeCrashReporterClient::Create();
 #endif
 
<<<<<<< HEAD
@@ -941,7 +941,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
=======
@@ -958,7 +958,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
>>>>>>> upstream/main
   child_process_logging::Init();
 #endif
 #if defined(ARCH_CPU_ARM_FAMILY) && \
-    (defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS))
+    (defined(OS_ANDROID) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD))
   // Create an instance of the CPU class to parse /proc/cpuinfo and cache
   // cpu_brand info.
   base::CPU cpu_info;
<<<<<<< HEAD
@@ -1058,7 +1058,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
=======
@@ -1075,7 +1075,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
>>>>>>> upstream/main
         locale;
   }
 
-#if defined(OS_POSIX) && !defined(OS_MAC)
+#if defined(OS_POSIX) && !defined(OS_MAC) && !defined(OS_BSD)
   // Zygote needs to call InitCrashReporter() in RunZygote().
   if (process_type != switches::kZygoteProcess) {
 #if defined(OS_ANDROID)
<<<<<<< HEAD
@@ -1079,7 +1079,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
=======
@@ -1096,7 +1096,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
>>>>>>> upstream/main
     }
 #endif  // defined(OS_ANDROID)
   }
-#endif  // defined(OS_POSIX) && !defined(OS_MAC)
+#endif  // defined(OS_POSIX) && !defined(OS_MAC) && !defined(OS_BSD)
 
 #if defined(OS_ANDROID)
   CHECK_EQ(base::android::GetLibraryProcessType(),
<<<<<<< HEAD
@@ -1099,7 +1099,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
=======
@@ -1116,7 +1116,7 @@ void ChromeMainDelegate::PreSandboxStartup() {
>>>>>>> upstream/main
 void ChromeMainDelegate::SandboxInitialized(const std::string& process_type) {
   // Note: If you are adding a new process type below, be sure to adjust the
   // AdjustLinuxOOMScore function too.
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS))
   AdjustLinuxOOMScore(process_type);
 #endif
 #if defined(OS_WIN)
<<<<<<< HEAD
@@ -1141,7 +1141,7 @@ int ChromeMainDelegate::RunProcess(
=======
@@ -1158,7 +1158,7 @@ int ChromeMainDelegate::RunProcess(
>>>>>>> upstream/main
 
     // This entry is not needed on Linux, where the NaCl loader
     // process is launched via nacl_helper instead.
-#if BUILDFLAG(ENABLE_NACL) && !defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if BUILDFLAG(ENABLE_NACL) && !defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_BSD)
     {switches::kNaClLoaderProcess, NaClMain},
 #else
     {"<invalid>", nullptr},  // To avoid constant array of size 0
<<<<<<< HEAD
@@ -1169,7 +1169,7 @@ void ChromeMainDelegate::ProcessExiting(const std::str
=======
@@ -1186,7 +1186,7 @@ void ChromeMainDelegate::ProcessExiting(const std::str
>>>>>>> upstream/main
 #endif  // !defined(OS_ANDROID)
 }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS))
 void ChromeMainDelegate::ZygoteStarting(
     std::vector<std::unique_ptr<content::ZygoteForkDelegate>>* delegates) {
<<<<<<< HEAD
 #if defined(OS_CHROMEOS)
@@ -1206,7 +1206,7 @@ void ChromeMainDelegate::ZygoteForked() {
=======
 #if BUILDFLAG(IS_CHROMEOS_ASH)
@@ -1223,7 +1223,7 @@ void ChromeMainDelegate::ZygoteForked() {
>>>>>>> upstream/main
   crash_keys::SetCrashKeysFromCommandLine(*command_line);
 }
 
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // (defined(OS_LINUX) || defined(OS_CHROMEOS))
 
 content::ContentClient* ChromeMainDelegate::CreateContentClient() {
   return &chrome_content_client_;
