--- lib/x86/crc32_pclmul_template.h.orig	2025-11-01 02:19:27 UTC
+++ lib/x86/crc32_pclmul_template.h
@@ -116,7 +116,7 @@ ADD_SUFFIX(fold_vec256)(__m256i src, __m256i dst, __m2
 static forceinline ATTRIBUTES __m256i
 ADD_SUFFIX(fold_vec256)(__m256i src, __m256i dst, __m256i /* __v4du */ mults)
 {
-#if USE_AVX512
+#if USE_AVX512 && defined(__AVX512F__)
 	/* vpternlog with immediate 0x96 is a three-argument XOR. */
 	return _mm256_ternarylogic_epi32(
 			_mm256_clmulepi64_epi128(src, mults, 0x00),
@@ -206,7 +206,7 @@ ADD_SUFFIX(crc32_x86)(u32 crc, const u8 *p, size_t len
 		if (len < VL) {
 			STATIC_ASSERT(VL == 16 || VL == 32 || VL == 64);
 			if (len < 16) {
-			#if USE_AVX512
+			#if USE_AVX512 && defined(__AVX512F__)
 				if (len < 4)
 					return crc32_slice1(crc, p, len);
 				/*
@@ -387,7 +387,7 @@ less_than_16_remaining:
 	/* Handle any remainder of 1 to 15 bytes. */
 	if (len)
 		x0 = fold_lessthan16bytes(x0, p, len, mults_128b);
-#if USE_AVX512
+#if USE_AVX512 && defined(__AVX512F__)
 reduce_x0:
 #endif
 	/*
