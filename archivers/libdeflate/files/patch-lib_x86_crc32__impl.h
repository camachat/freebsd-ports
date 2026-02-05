--- lib/x86/crc32_impl.h.orig	2025-11-01 02:19:27 UTC
+++ lib/x86/crc32_impl.h
@@ -92,7 +92,7 @@ static const u8 MAYBE_UNUSED shift_tab[48] = {
 #endif
 
 #if (GCC_PREREQ(10, 1) || CLANG_PREREQ(6, 0, 10000000) || MSVC_PREREQ(1920)) && \
-	!defined(LIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_VPCLMULQDQ)
+	!defined(LIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_VPCLMULQDQ) && defined(__AVX512F__)
 /*
  * VPCLMULQDQ/AVX512 implementation using 256-bit vectors.  This is very similar
  * to the VPCLMULQDQ/AVX2 implementation but takes advantage of the vpternlog
