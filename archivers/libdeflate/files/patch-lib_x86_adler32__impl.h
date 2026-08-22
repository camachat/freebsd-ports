--- lib/x86/adler32_impl.h.orig	2025-11-01 02:19:27 UTC
+++ lib/x86/adler32_impl.h
@@ -72,7 +72,7 @@
 #endif
 
 #if (GCC_PREREQ(8, 1) || CLANG_PREREQ(6, 0, 10000000) || MSVC_PREREQ(1920)) && \
-	!defined(LIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_AVX512VNNI)
+	!defined(LIBDEFLATE_ASSEMBLER_DOES_NOT_SUPPORT_AVX512VNNI) && defined(__AVX512F__)
 /*
  * AVX512VNNI implementation using 256-bit vectors.  This is very similar to the
  * AVX-VNNI implementation but takes advantage of masking and more registers.
