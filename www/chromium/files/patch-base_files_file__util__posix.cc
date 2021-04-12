<<<<<<< HEAD
--- base/files/file_util_posix.cc.orig	2021-01-18 21:28:44 UTC
+++ base/files/file_util_posix.cc
@@ -406,7 +406,7 @@ bool CreatePipe(ScopedFD* read_fd, ScopedFD* write_fd,
=======
--- base/files/file_util_posix.cc.orig	2021-03-12 23:57:15 UTC
+++ base/files/file_util_posix.cc
@@ -380,7 +380,7 @@ bool CreatePipe(ScopedFD* read_fd, ScopedFD* write_fd,
>>>>>>> upstream/main
 }
 
 bool CreateLocalNonBlockingPipe(int fds[2]) {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   return pipe2(fds, O_CLOEXEC | O_NONBLOCK) == 0;
 #else
   int raw_fds[2];
<<<<<<< HEAD
@@ -953,8 +953,12 @@ bool AllocateFileRegion(File* file, int64_t offset, si
=======
@@ -927,8 +927,12 @@ bool AllocateFileRegion(File* file, int64_t offset, si
>>>>>>> upstream/main
   // space. It can fail because the filesystem doesn't support it. In that case,
   // use the manual method below.
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
+#if defined(OS_BSD)
+  if (HANDLE_EINTR(posix_fallocate(file->GetPlatformFile(), offset, size)) != -1)
+#else
   if (HANDLE_EINTR(fallocate(file->GetPlatformFile(), 0, offset, size)) != -1)
+#endif
     return true;
   DPLOG(ERROR) << "fallocate";
 #elif defined(OS_APPLE)
<<<<<<< HEAD
@@ -1178,7 +1182,7 @@ PrefetchResult PreReadFile(const FilePath& file_path,
=======
@@ -1098,7 +1102,7 @@ int GetMaximumPathComponentLength(const FilePath& path
 #if !defined(OS_ANDROID)
 // This is implemented in file_util_android.cc for that platform.
 bool GetShmemTempDir(bool executable, FilePath* path) {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX) || defined(OS_BSD)
   bool disable_dev_shm = false;
 #if !BUILDFLAG(IS_CHROMEOS_ASH) && !BUILDFLAG(IS_CHROMEOS_LACROS)
   disable_dev_shm = CommandLine::ForCurrentProcess()->HasSwitch(
@@ -1114,7 +1118,7 @@ bool GetShmemTempDir(bool executable, FilePath* path) 
     *path = FilePath("/dev/shm");
     return true;
   }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX) || defined(OS_BSD)
   return GetTempDir(path);
 }
 #endif  // !defined(OS_ANDROID)
@@ -1152,7 +1156,7 @@ PrefetchResult PreReadFile(const FilePath& file_path,
>>>>>>> upstream/main
   // posix_fadvise() is only available in the Android NDK in API 21+. Older
   // versions may have the required kernel support, but don't have enough usage
   // to justify backporting.
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     (defined(OS_ANDROID) && __ANDROID_API__ >= 21)
   File file(file_path, File::FLAG_OPEN | File::FLAG_READ);
   if (!file.IsValid())
<<<<<<< HEAD
@@ -1214,7 +1218,7 @@ PrefetchResult PreReadFile(const FilePath& file_path,
=======
@@ -1188,7 +1192,7 @@ PrefetchResult PreReadFile(const FilePath& file_path,
>>>>>>> upstream/main
   return internal::PreReadFileSlow(file_path, max_bytes)
              ? PrefetchResult{PrefetchResultCode::kSlowSuccess}
              : PrefetchResult{PrefetchResultCode::kSlowFailed};
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || (defined(OS_ANDROID) &&
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || (defined(OS_ANDROID) &&
         // __ANDROID_API__ >= 21)
 }
 
<<<<<<< HEAD
@@ -1249,7 +1253,7 @@ bool MoveUnsafe(const FilePath& from_path, const FileP
=======
@@ -1223,7 +1227,7 @@ bool MoveUnsafe(const FilePath& from_path, const FileP
>>>>>>> upstream/main
 
 #endif  // !defined(OS_NACL_NONSFI)
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX) || defined(OS_BSD)
 BASE_EXPORT bool IsPathExecutable(const FilePath& path) {
   bool result = false;
   FilePath tmp_file_path;
<<<<<<< HEAD
@@ -1270,6 +1274,6 @@ BASE_EXPORT bool IsPathExecutable(const FilePath& path
=======
@@ -1244,6 +1248,6 @@ BASE_EXPORT bool IsPathExecutable(const FilePath& path
>>>>>>> upstream/main
   }
   return result;
 }
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_AIX) || defined(OS_BSD)
 
 }  // namespace base
