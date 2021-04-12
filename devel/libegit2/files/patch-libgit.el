<<<<<<< HEAD
--- libgit.el.orig	2020-05-15 17:59:08 UTC
+++ libgit.el
=======
--- libgit.el.orig	2020-05-15 17:59:08.000000000 +0000
+++ libgit.el	2021-04-01 04:02:35.637193000 +0000
>>>>>>> upstream/main
@@ -41,7 +41,7 @@
   "Directory where the libegit2 dynamic module file should be built.")
 
 (defvar libgit--module-file
-  (expand-file-name (concat "libegit2" module-file-suffix) libgit--build-dir)
<<<<<<< HEAD
+  (expand-file-name (concat "libegit2" module-file-suffix) "/usr/local/share/emacs/27.1/site-lisp")
=======
+  (expand-file-name (concat "libegit2" module-file-suffix) "%%LIBEGIT2_INSTALL_DIR%%")
>>>>>>> upstream/main
   "Path to the libegit2 dynamic module file.")
 
 (defun libgit--configure ()
