--- lib/x86/adler32_template.h.orig	2025-11-01 02:19:27 UTC
+++ lib/x86/adler32_template.h
@@ -82,7 +82,7 @@
 #  define VADD8(a, b)		_mm256_add_epi8((a), (b))
 #  define VADD16(a, b)		_mm256_add_epi16((a), (b))
 #  define VADD32(a, b)		_mm256_add_epi32((a), (b))
-#  if USE_AVX512
+#  if USE_AVX512 && defined(__AVX512F__)
 #    define VDPBUSD(a, b, c)	_mm256_dpbusd_epi32((a), (b), (c))
 #  else
 #    define VDPBUSD(a, b, c)	_mm256_dpbusd_avx_epi32((a), (b), (c))
@@ -375,7 +375,7 @@ ADD_SUFFIX(adler32_x86)(u32 adler, const u8 *p, size_t
 			 * Process the last 0 < n <= VL bytes of the chunk.
 			 * Utilize a masked load if it's available.
 			 */
-		#if USE_AVX512
+		#if USE_AVX512 && defined(__AVX512F__)
 			data = VMASKZ_LOADU((mask_t)-1 >> (VL - n), p);
 		#else
 			data = zeroes;
