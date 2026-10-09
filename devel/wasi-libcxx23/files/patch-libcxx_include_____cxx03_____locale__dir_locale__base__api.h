--- libcxx/include/__cxx03/__locale_dir/locale_base_api.h.orig	2026-08-12 04:58:33 UTC
+++ libcxx/include/__cxx03/__locale_dir/locale_base_api.h
@@ -24,7 +24,7 @@
 #elif defined(__Fuchsia__)
 #  include <__cxx03/__locale_dir/locale_base_api/fuchsia.h>
 #elif defined(__wasi__) || defined(_LIBCPP_HAS_MUSL_LIBC)
-#  include <__cxx03/__locale_dir/locale_base_api/musl.h>
+#  /* wasi-libc already provides locale_t and most *_l functions */
 #elif defined(__APPLE__) || defined(__FreeBSD__)
 #  include <__cxx03/xlocale.h>
 #endif
