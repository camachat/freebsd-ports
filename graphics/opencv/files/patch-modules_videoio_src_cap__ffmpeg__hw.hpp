--- modules/videoio/src/cap_ffmpeg_hw.hpp.orig	2026-06-05 18:50:05 UTC
+++ modules/videoio/src/cap_ffmpeg_hw.hpp
@@ -755,11 +755,18 @@ AVCodec *hw_find_codec(AVCodecID id, AVHWDeviceType hw
             if (hw_type == AV_HWDEVICE_TYPE_VAAPI)
                 hw_native_fmt = AV_PIX_FMT_VAAPI_VLD;
 #endif
+#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
+            const enum AVPixelFormat *pix_fmts = NULL;
+            if (avcodec_get_supported_config(NULL, c, AV_CODEC_CONFIG_PIX_FORMAT, 0, (const void **)&pix_fmts, NULL) < 0)
+                pix_fmts = NULL;
+#else
+            const enum AVPixelFormat *pix_fmts = c->pix_fmts;
+#endif
             if (hw_type == AV_HWDEVICE_TYPE_CUDA) // CUDA encoders don't support avcodec_get_hw_config()
                 hw_native_fmt = AV_PIX_FMT_CUDA;
-            if (av_codec_is_encoder(c) && hw_native_fmt != AV_PIX_FMT_NONE && c->pix_fmts) {
-                for (int i = 0; c->pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
-                    if (c->pix_fmts[i] == hw_native_fmt) {
+            if (av_codec_is_encoder(c) && hw_native_fmt != AV_PIX_FMT_NONE && pix_fmts) {
+                for (int i = 0; pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
+                    if (pix_fmts[i] == hw_native_fmt) {
                         *hw_pix_fmt = hw_native_fmt;
                         if (hw_check_codec(c, hw_type, disabled_codecs))
                             return c;
