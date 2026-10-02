--- src/fsutils.cpp.orig	2025-01-24 00:00:00 UTC
+++ src/fsutils.cpp
@@ -7,6 +7,10 @@
 #include <unistd.h>
 #include <cstring>
 #endif
+#if defined(__FreeBSD__)
+#include <sys/types.h>
+#include <sys/sysctl.h>
+#endif
 
 #include <spdlog/spdlog.h>
 
@@ -39,6 +43,42 @@ static std::filesystem::path get_executable_directory(
     // Create a std::filesystem::path from the filepath and return its parent path
     std::filesystem::path execpath(filepath.data());
     return execpath.parent_path();
+}
+#elif defined(__FreeBSD__)   // _WIN32
+static std::filesystem::path get_executable_directory() {
+    // Prepare sysctl MIB for KERN_PROC_PATHNAME (-1 = current process)
+    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
+
+    // First call: get required buffer size
+    std::size_t bufsize = 0;
+    if (sysctl(mib, 4, nullptr, &bufsize, nullptr, 0) == -1) {
+        // Should not happen unless kernel is very broken
+        logger()->error("sysctl KERN_PROC_PATHNAME (size query) failed: {}", std::strerror(errno));
+        return std::filesystem::path();
+    }
+
+    // Allocate buffer (bufsize includes null terminator)
+    std::string path(bufsize, '\0');
+
+    // Second call: actually retrieve the path
+    if (sysctl(mib, 4, path.data(), &bufsize, nullptr, 0) == -1) {
+        logger()->error("sysctl KERN_PROC_PATHNAME failed: {}", std::strerror(errno));
+        return std::filesystem::path();
+    }
+
+    // Trim trailing null characters
+    path.resize(std::strlen(path.c_str()));
+
+    // Convert to filesystem::path and get parent directory
+    std::filesystem::path filepath(std::move(path));
+
+    // Optional: check if it's absolute and non-empty
+    if (filepath.empty() || !filepath.is_absolute()) {
+        logger()->error("KERN_PROC_PATHNAME returned invalid path: {}", filepath.string());
+        return std::filesystem::path();
+    }
+
+    return filepath.parent_path();
 }
 #else   // _WIN32
 static std::filesystem::path get_executable_directory() {
