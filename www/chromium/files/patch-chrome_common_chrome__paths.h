<<<<<<< HEAD
--- chrome/common/chrome_paths.h.orig	2021-01-18 21:28:52 UTC
+++ chrome/common/chrome_paths.h
@@ -51,7 +51,7 @@ enum {
                      // contains subdirectories.
 #endif
 #if defined(OS_CHROMEOS) || \
-    (defined(OS_LINUX) && BUILDFLAG(CHROMIUM_BRANDING)) || defined(OS_MAC)
+    (defined(OS_LINUX) && BUILDFLAG(CHROMIUM_BRANDING)) || defined(OS_MAC) || defined(OS_BSD)
   DIR_USER_EXTERNAL_EXTENSIONS,  // Directory for per-user external extensions
                                  // on Chrome Mac and Chromium Linux.
                                  // On Chrome OS, this path is used for OEM
@@ -59,7 +59,7 @@ enum {
=======
--- chrome/common/chrome_paths.h.orig	2021-03-12 23:57:19 UTC
+++ chrome/common/chrome_paths.h
@@ -53,9 +53,9 @@ enum {
 #endif
 // TODO(crbug.com/1052397): Revisit once build flag switch of lacros-chrome is
 // complete.
-#if defined(OS_CHROMEOS) ||                                  \
-    ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && \
-     BUILDFLAG(CHROMIUM_BRANDING)) ||                        \
+#if defined(OS_CHROMEOS) ||                                                     \
+    ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && \
+     BUILDFLAG(CHROMIUM_BRANDING)) ||                                           \
     defined(OS_MAC)
   DIR_USER_EXTERNAL_EXTENSIONS,  // Directory for per-user external extensions
                                  // on Chrome Mac and Chromium Linux.
@@ -64,7 +64,7 @@ enum {
>>>>>>> upstream/main
                                  // create it.
 #endif
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   DIR_STANDALONE_EXTERNAL_EXTENSIONS,  // Directory for 'per-extension'
                                        // definition manifest files that
                                        // describe extensions which are to be
<<<<<<< HEAD
@@ -116,7 +116,7 @@ enum {
   DIR_SUPERVISED_USER_INSTALLED_WHITELISTS,  // Directory where sanitized
                                              // supervised user whitelists are
                                              // installed.
=======
@@ -118,7 +118,7 @@ enum {
   DIR_CHROMEOS_CUSTOM_WALLPAPERS,     // Directory where custom wallpapers
                                       // reside.
 #endif
>>>>>>> upstream/main
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_MAC) || defined(OS_BSD)
   DIR_NATIVE_MESSAGING,       // System directory where native messaging host
                               // manifest files are stored.
   DIR_USER_NATIVE_MESSAGING,  // Directory with Native Messaging Hosts
