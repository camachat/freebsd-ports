--- BaseTools/Source/C/Makefiles/header.makefile.orig	2022-06-25 17:00:04.055061000 +0000
+++ BaseTools/Source/C/Makefiles/header.makefile	2022-06-25 17:00:28.684235000 +0000
@@ -92,7 +92,7 @@ BUILD_CFLAGS = -MD -fshort-wchar -fno-strict-aliasing 
 -Wno-unused-result -nostdlib -g
 else
 BUILD_CFLAGS = -MD -fshort-wchar -fno-strict-aliasing -fwrapv \
--fno-delete-null-pointer-checks -Wall -Werror \
+-fno-delete-null-pointer-checks \
 -Wno-deprecated-declarations -Wno-stringop-truncation -Wno-restrict \
 -Wno-unused-result -nostdlib -g
 endif
