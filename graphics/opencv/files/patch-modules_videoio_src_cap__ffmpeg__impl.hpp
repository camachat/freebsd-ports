--- modules/videoio/src/cap_ffmpeg_impl.hpp.orig	2026-06-05 18:50:05 UTC
+++ modules/videoio/src/cap_ffmpeg_impl.hpp
@@ -2629,8 +2629,14 @@ static AVCodecContext * icv_configure_video_stream_FFM
     c->time_base.den = frame_rate;
     c->time_base.num = frame_rate_base;
     /* adjust time base for supported framerates */
-    if(codec && codec->supported_framerates){
-        const AVRational *p= codec->supported_framerates;
+#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
+    const AVRational *supported_framerates = NULL;
+    avcodec_get_supported_config(NULL, codec, AV_CODEC_CONFIG_FRAME_RATE, 0, (const void **)&supported_framerates, NULL);
+#else
+    const AVRational *supported_framerates = codec ? codec->supported_framerates : NULL;
+#endif
+    if(codec && supported_framerates){
+        const AVRational *p= supported_framerates;
         AVRational req = {frame_rate, frame_rate_base};
         const AVRational *best=NULL;
         AVRational best_error= {INT_MAX, 1};
