<<<<<<< HEAD
--- third_party/blink/renderer/modules/peerconnection/webrtc_audio_renderer_test.cc.orig	2021-01-18 21:29:05 UTC
+++ third_party/blink/renderer/modules/peerconnection/webrtc_audio_renderer_test.cc
@@ -281,7 +281,7 @@ TEST_F(MAYBE_WebRtcAudioRendererTest, MultipleRenderer
 TEST_F(MAYBE_WebRtcAudioRendererTest, VerifySinkParameters) {
=======
--- third_party/blink/renderer/modules/peerconnection/webrtc_audio_renderer_test.cc.orig	2021-03-12 23:57:30 UTC
+++ third_party/blink/renderer/modules/peerconnection/webrtc_audio_renderer_test.cc
@@ -283,7 +283,7 @@ TEST_F(MAYBE_WebRtcAudioRendererTest, DISABLED_Multipl
 TEST_F(MAYBE_WebRtcAudioRendererTest, DISABLED_VerifySinkParameters) {
>>>>>>> upstream/main
   SetupRenderer(kDefaultOutputDeviceId);
   renderer_proxy_->Start();
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD) || \
     defined(OS_FUCHSIA)
   static const int kExpectedBufferSize = kHardwareSampleRate / 100;
 #elif defined(OS_ANDROID)
