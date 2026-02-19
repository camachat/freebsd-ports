--- config/auto-aux/tclversion.c.orig	2026-02-19 22:32:07 UTC
+++ config/auto-aux/tclversion.c
@@ -18,7 +18,7 @@
 #include <tcl.h>
 #include <tk.h>
 
-main ()
+int main ()
 {
     puts(TCL_VERSION);
 }
