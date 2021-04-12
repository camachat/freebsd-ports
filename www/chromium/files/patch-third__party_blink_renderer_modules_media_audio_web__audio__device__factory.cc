<<<<<<< HEAD
--- third_party/blink/renderer/modules/media/audio/web_audio_device_factory.cc.orig	2021-01-18 21:29:05 UTC
=======
--- third_party/blink/renderer/modules/media/audio/web_audio_device_factory.cc.orig	2021-03-12 23:57:30 UTC
>>>>>>> upstream/main
+++ third_party/blink/renderer/modules/media/audio/web_audio_device_factory.cc
@@ -33,7 +33,7 @@ WebAudioDeviceFactory* WebAudioDeviceFactory::factory_
 
 namespace {
 
-#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || \
+#if defined(OS_WIN) || defined(OS_MAC) || defined(OS_LINUX) || defined(OS_BSD) || \
<<<<<<< HEAD
     BUILDFLAG(IS_LACROS)
=======
     BUILDFLAG(IS_CHROMEOS_LACROS)
>>>>>>> upstream/main
 // Due to driver deadlock issues on Windows (http://crbug/422522) there is a
 // chance device authorization response is never received from the browser side.
