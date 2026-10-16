diff --git tools/server/server-models.cpp tools/server/server-models.cpp
index 93e940951..b61e60f29 100644
--- tools/server/server-models.cpp
+++ tools/server/server-models.cpp
@@ -289,7 +289,11 @@ static std::filesystem::path get_server_exec_path() {
     }
 #else
     char path[FILENAME_MAX];
-    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
+#    if defined(__linux__)
+        ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
+#    elif defined(__FreeBSD__)
+        ssize_t count = readlink("/proc/curproc/file", path, FILENAME_MAX);
+#    endif
     if (count <= 0) {
         throw std::runtime_error("failed to resolve /proc/self/exe");
     }
