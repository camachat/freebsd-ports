--- src/libjasper/jp2/jp2_dec.c.orig	2025-08-06 03:57:09 UTC
+++ src/libjasper/jp2/jp2_dec.c
@@ -309,7 +309,7 @@ jas_image_t *jp2_decode(jas_stream_t *in, const char *
 		jas_image_setclrspc(dec->image, jp2_getcs(&dec->colr->data.colr));
 		break;
 	case JP2_COLR_ICC:
-		iccprof = jas_iccprof_createfrombuf(dec->colr->data.colr.iccp,
+		iccprof = jas_iccprof_createfrombuf((const unsigned char*)dec->colr->data.colr.iccp,
 		  dec->colr->data.colr.iccplen);
 		if (!iccprof) {
 			jas_logerrorf("error: failed to parse ICC profile\n");
