<<<<<<< HEAD
--- content/common/common_sandbox_support_linux.cc.orig	2020-11-13 06:36:42 UTC
=======
--- content/common/common_sandbox_support_linux.cc.orig	2021-03-12 23:57:24 UTC
>>>>>>> upstream/main
+++ content/common/common_sandbox_support_linux.cc
@@ -5,6 +5,7 @@
 #include "content/public/common/common_sandbox_support_linux.h"
 
 #include <sys/stat.h>
+#include <unistd.h>
 
 #include <limits>
 #include <memory>
