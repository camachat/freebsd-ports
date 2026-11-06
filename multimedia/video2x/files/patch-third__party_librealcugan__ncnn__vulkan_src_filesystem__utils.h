--- third_party/librealcugan_ncnn_vulkan/src/filesystem_utils.h.orig	2026-01-26 23:26:23 UTC
+++ third_party/librealcugan_ncnn_vulkan/src/filesystem_utils.h
@@ -20,6 +20,11 @@
 #include <mach-o/dyld.h>
 #endif
 
+#if __FreeBSD__
+#include <sys/types.h>
+#include <sys/sysctl.h>
+#endif
+
 #if _WIN32
 typedef std::wstring path_t;
 #define PATHSTR(X) L##X
@@ -131,6 +136,23 @@ static path_t get_executable_directory()
     char filepath[256];
     uint32_t size = sizeof(filepath);
     _NSGetExecutablePath(filepath, &size);
+
+    char* slash = strrchr(filepath, '/');
+    slash[1] = '\0';
+
+    return path_t(filepath);
+}
+#elif __FreeBSD__
+static path_t get_executable_directory()
+{
+    char filepath[256];
+    size_t size = sizeof(filepath);
+
+    // Prepare sysctl MIB for KERN_PROC_PATHNAME (-1 = current process)
+    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
+    if (sysctl(mib, 4, filepath, &size, nullptr, 0) == -1) {
+        strcpy(filepath, "./");
+    }
 
     char* slash = strrchr(filepath, '/');
     slash[1] = '\0';
