<<<<<<< HEAD
--- third_party/webrtc/modules/rtp_rtcp/source/forward_error_correction.cc.orig	2020-11-13 06:42:25 UTC
=======
--- third_party/webrtc/modules/rtp_rtcp/source/forward_error_correction.cc.orig	2021-03-13 00:03:43 UTC
>>>>>>> upstream/main
+++ third_party/webrtc/modules/rtp_rtcp/source/forward_error_correction.cc
@@ -13,6 +13,7 @@
 #include <string.h>
 
 #include <algorithm>
+#include <cstdlib>
 #include <utility>
 
 #include "absl/algorithm/container.h"
