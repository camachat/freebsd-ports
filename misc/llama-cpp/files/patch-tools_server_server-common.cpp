--- tools/server/server-common.cpp.orig	2026-10-22 22:38:03 UTC
+++ tools/server/server-common.cpp
@@ -78,6 +78,13 @@ std::filesystem::path get_server_exec_path() {
         }
         throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
     }
+#elif defined(__FreeBSD__)
+    char path[FILENAME_MAX];
+    ssize_t count = readlink("/proc/curproc/exe", path, FILENAME_MAX);
+    if (count <= 0) {
+        throw std::runtime_error("failed to resolve /proc/curproc/exe");
+    }
+    return std::filesystem::path(std::string(path, count));
 #else
     char path[FILENAME_MAX];
     ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
