<<<<<<< HEAD
--- chromecast/browser/cast_browser_main_parts.cc.orig	2021-01-18 21:28:54 UTC
+++ chromecast/browser/cast_browser_main_parts.cc
@@ -73,7 +73,7 @@
=======
--- chromecast/browser/cast_browser_main_parts.cc.orig	2021-03-12 23:57:21 UTC
+++ chromecast/browser/cast_browser_main_parts.cc
@@ -75,7 +75,7 @@
>>>>>>> upstream/main
 #include "ui/base/ui_base_switches.h"
 #include "ui/gl/gl_switches.h"
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include <fontconfig/fontconfig.h>
 #include <signal.h>
 #include <sys/prctl.h>
<<<<<<< HEAD
@@ -129,7 +129,7 @@
=======
@@ -131,7 +131,7 @@
>>>>>>> upstream/main
 #include "extensions/browser/extension_prefs.h"  // nogncheck
 #endif
 
-#if (defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(USE_OZONE)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)) && defined(USE_OZONE)
 #include "chromecast/browser/exo/wayland_server_controller.h"
 #endif
 
<<<<<<< HEAD
@@ -271,7 +271,7 @@ class CastViewsDelegate : public views::ViewsDelegate 
=======
@@ -273,7 +273,7 @@ class CastViewsDelegate : public views::ViewsDelegate 
>>>>>>> upstream/main
 
 #endif  // defined(USE_AURA)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 base::FilePath GetApplicationFontsDir() {
   std::unique_ptr<base::Environment> env(base::Environment::Create());
<<<<<<< HEAD
@@ -287,7 +287,7 @@ base::FilePath GetApplicationFontsDir() {
=======
@@ -289,7 +289,7 @@ base::FilePath GetApplicationFontsDir() {
>>>>>>> upstream/main
   }
 }
 
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 }  // namespace
 
<<<<<<< HEAD
@@ -316,7 +316,7 @@ const DefaultCommandLineSwitch kDefaultSwitches[] = {
=======
@@ -318,7 +318,7 @@ const DefaultCommandLineSwitch kDefaultSwitches[] = {
>>>>>>> upstream/main
     {cc::switches::kDisableThreadedAnimation, ""},
 #endif  // defined(OS_ANDROID)
 #endif  // BUILDFLAG(IS_CAST_AUDIO_ONLY)
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #if defined(ARCH_CPU_X86_FAMILY)
     // This is needed for now to enable the x11 Ozone platform to work with
     // current Linux/NVidia OpenGL drivers.
<<<<<<< HEAD
@@ -326,7 +326,7 @@ const DefaultCommandLineSwitch kDefaultSwitches[] = {
=======
@@ -328,7 +328,7 @@ const DefaultCommandLineSwitch kDefaultSwitches[] = {
>>>>>>> upstream/main
     {switches::kEnableHardwareOverlays, "cast"},
 #endif
 #endif
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
     // It's better to start GPU process on demand. For example, for TV platforms
     // cast starts in background and can't render until TV switches to cast
     // input.
<<<<<<< HEAD
@@ -476,7 +476,7 @@ void CastBrowserMainParts::ToolkitInitialized() {
=======
@@ -482,7 +482,7 @@ void CastBrowserMainParts::ToolkitInitialized() {
>>>>>>> upstream/main
     views_delegate_ = std::make_unique<CastViewsDelegate>();
 #endif  // defined(USE_AURA)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   base::FilePath dir_font = GetApplicationFontsDir();
   const FcChar8 *dir_font_char8 = reinterpret_cast<const FcChar8*>(dir_font.value().data());
   if (!FcConfigAppFontAddDir(gfx::GetGlobalFontConfig(), dir_font_char8)) {
<<<<<<< HEAD
@@ -660,7 +660,7 @@ void CastBrowserMainParts::PreMainMessageLoopRun() {
=======
@@ -666,7 +666,7 @@ void CastBrowserMainParts::PreMainMessageLoopRun() {
>>>>>>> upstream/main
       cast_browser_process_->browser_context());
 #endif
 
-#if (defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(USE_OZONE)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)) && defined(USE_OZONE)
   wayland_server_controller_ =
       std::make_unique<WaylandServerController>(window_manager_.get());
 #endif
<<<<<<< HEAD
@@ -742,7 +742,7 @@ bool CastBrowserMainParts::MainMessageLoopRun(int* res
=======
@@ -748,7 +748,7 @@ bool CastBrowserMainParts::MainMessageLoopRun(int* res
>>>>>>> upstream/main
 }
 
 void CastBrowserMainParts::PostMainMessageLoopRun() {
-#if (defined(OS_LINUX) || defined(OS_CHROMEOS)) && defined(USE_OZONE)
+#if (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)) && defined(USE_OZONE)
   wayland_server_controller_.reset();
 #endif
 #if BUILDFLAG(ENABLE_CHROMECAST_EXTENSIONS)
