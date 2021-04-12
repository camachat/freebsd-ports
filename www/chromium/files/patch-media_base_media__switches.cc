<<<<<<< HEAD
--- media/base/media_switches.cc.orig	2021-01-18 21:29:00 UTC
+++ media/base/media_switches.cc
@@ -348,7 +348,7 @@ const base::Feature kGav1VideoDecoder{"Gav1VideoDecode
=======
--- media/base/media_switches.cc.orig	2021-03-12 23:57:26 UTC
+++ media/base/media_switches.cc
@@ -370,7 +370,7 @@ const base::Feature kGav1VideoDecoder{"Gav1VideoDecode
>>>>>>> upstream/main
 // Show toolbar button that opens dialog for controlling media sessions.
 const base::Feature kGlobalMediaControls {
   "GlobalMediaControls",
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
<<<<<<< HEAD
     BUILDFLAG(IS_LACROS)
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
@@ -390,7 +390,7 @@ const base::Feature kGlobalMediaControlsOverlayControl
=======
     BUILDFLAG(IS_CHROMEOS_LACROS)
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
@@ -412,7 +412,7 @@ const base::Feature kGlobalMediaControlsOverlayControl
>>>>>>> upstream/main
 // Show picture-in-picture button in Global Media Controls.
 const base::Feature kGlobalMediaControlsPictureInPicture {
   "GlobalMediaControlsPictureInPicture",
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
<<<<<<< HEAD
     BUILDFLAG(IS_LACROS)
=======
     BUILDFLAG(IS_CHROMEOS_LACROS)
>>>>>>> upstream/main
       base::FEATURE_ENABLED_BY_DEFAULT
 #else
