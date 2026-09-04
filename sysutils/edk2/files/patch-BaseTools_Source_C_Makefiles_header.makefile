--- BaseTools/Source/C/Makefiles/header.makefile.orig	2026-08-19 15:03:09 UTC
+++ BaseTools/Source/C/Makefiles/header.makefile
@@ -161,17 +161,17 @@ ifeq ($(DARWIN),Darwin)
 
 ifeq ($(DARWIN),Darwin)
 # assume clang or clang compatible flags on OS X
-CFLAGS = -MD -fshort-wchar -fno-strict-aliasing -Wall -Werror \
+CFLAGS = -MD -fshort-wchar -fno-strict-aliasing -Wall \
 -Wno-deprecated-declarations -Wno-self-assign -Wno-unused-result -nostdlib -g
 else
 ifneq ($(CLANG),)
 CFLAGS = -MD -fshort-wchar -fno-strict-aliasing -fwrapv \
--fno-delete-null-pointer-checks -Wall -Werror \
+-fno-delete-null-pointer-checks -Wall \
 -Wno-deprecated-declarations -Wno-self-assign \
 -Wno-unused-result -nostdlib -g
 else
 CFLAGS = -MD -fshort-wchar -fno-strict-aliasing -fwrapv \
--fno-delete-null-pointer-checks -Wall -Werror \
+-fno-delete-null-pointer-checks -Wall \
 -Wno-deprecated-declarations -Wno-stringop-truncation -Wno-restrict \
 -Wno-unused-result -nostdlib -g
 endif
