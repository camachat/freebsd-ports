<<<<<<< HEAD
--- chrome/common/extensions/command.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/common/extensions/command.cc
@@ -293,7 +293,7 @@ std::string Command::CommandPlatform() {
   return values::kKeybindingPlatformMac;
 #elif defined(OS_CHROMEOS)
   return values::kKeybindingPlatformChromeOs;
-#elif defined(OS_LINUX)
+#elif defined(OS_LINUX) || defined(OS_BSD)
=======
--- chrome/common/extensions/command.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/extensions/command.cc
@@ -294,7 +294,7 @@ std::string Command::CommandPlatform() {
   return values::kKeybindingPlatformMac;
 #elif BUILDFLAG(IS_CHROMEOS_ASH)
   return values::kKeybindingPlatformChromeOs;
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   return values::kKeybindingPlatformLinux;
 #else
   return "";
