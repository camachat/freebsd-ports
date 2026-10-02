--- third_party/librealesrgan_ncnn_vulkan/src/filesystem_utils.h.orig	2026-01-26 23:27:02 UTC
+++ third_party/librealesrgan_ncnn_vulkan/src/filesystem_utils.h
@@ -16,6 +16,11 @@
 #include <dirent.h>
 #endif // _WIN32
 
+#if __FreeBSD__
+#include <sys/types.h>
+#include <sys/sysctl.h>
+#endif
+
 #if _WIN32
 typedef std::wstring path_t;
 #define PATHSTR(X) L##X
@@ -118,6 +123,21 @@ static path_t get_executable_directory()
 
     wchar_t* backslash = wcsrchr(filepath, L'\\');
     backslash[1] = L'\0';
+
+    return path_t(filepath);
+}
+#elif __FreeBSD__ // _WIN32
+static path_t get_executable_directory()
+{
+    char filepath[256];
+    size_t size = sizeof(filepath);
+
+    // Prepare sysctl MIB for KERN_PROC_PATHNAME (-1 = current process)
+    int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
+    sysctl(mib, 4, filepath, &size, nullptr, 0);
+
+    char* slash = strrchr(filepath, '/');
+    slash[1] = '\0';
 
     return path_t(filepath);
 }
