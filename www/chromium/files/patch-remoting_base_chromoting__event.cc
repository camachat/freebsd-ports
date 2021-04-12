<<<<<<< HEAD
--- remoting/base/chromoting_event.cc.orig	2020-11-13 06:36:46 UTC
+++ remoting/base/chromoting_event.cc
@@ -188,7 +188,7 @@ void ChromotingEvent::AddSystemInfo() {
=======
--- remoting/base/chromoting_event.cc.orig	2021-03-12 23:57:28 UTC
+++ remoting/base/chromoting_event.cc
@@ -189,7 +189,7 @@ void ChromotingEvent::AddSystemInfo() {
>>>>>>> upstream/main
   SetString(kCpuKey, base::SysInfo::OperatingSystemArchitecture());
   SetString(kOsVersionKey, base::SysInfo::OperatingSystemVersion());
   SetString(kWebAppVersionKey, STRINGIZE(VERSION));
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   Os os = Os::CHROMOTING_LINUX;
<<<<<<< HEAD
 #elif defined(OS_CHROMEOS)
=======
 #elif BUILDFLAG(IS_CHROMEOS_ASH)
>>>>>>> upstream/main
   Os os = Os::CHROMOTING_CHROMEOS;
