<<<<<<< HEAD
--- base/files/file_path_watcher_unittest.cc.orig	2021-01-18 21:28:44 UTC
+++ base/files/file_path_watcher_unittest.cc
@@ -438,12 +438,12 @@ TEST_F(FilePathWatcherTest, WatchDirectory) {
=======
--- base/files/file_path_watcher_unittest.cc.orig	2021-03-12 23:57:15 UTC
+++ base/files/file_path_watcher_unittest.cc
@@ -444,12 +444,12 @@ TEST_F(FilePathWatcherTest, WatchDirectory) {
>>>>>>> upstream/main
   VLOG(1) << "Waiting for file1 creation";
   ASSERT_TRUE(WaitForEvents());
 
-#if !defined(OS_APPLE)
+#if !defined(OS_APPLE) && !defined(OS_BSD)
   // Mac implementation does not detect files modified in a directory.
   ASSERT_TRUE(WriteFile(file1, "content v2"));
   VLOG(1) << "Waiting for file1 modification";
   ASSERT_TRUE(WaitForEvents());
-#endif  // !OS_APPLE
+#endif  // !OS_APPLE && !OS_BSD
 
   ASSERT_TRUE(base::DeleteFile(file1));
   VLOG(1) << "Waiting for file1 deletion";
<<<<<<< HEAD
@@ -650,7 +650,7 @@ TEST_F(FilePathWatcherTest, FileAttributesChanged) {
   ASSERT_TRUE(WaitForEvents());
 }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 // Verify that creating a symlink is caught.
 TEST_F(FilePathWatcherTest, CreateLink) {
@@ -816,7 +816,7 @@ TEST_F(FilePathWatcherTest, LinkedDirectoryPart3) {
=======
@@ -822,7 +822,7 @@ TEST_F(FilePathWatcherTest, LinkedDirectoryPart3) {
>>>>>>> upstream/main
   ASSERT_TRUE(WaitForEvents());
 }
 
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 enum Permission {
   Read,
<<<<<<< HEAD
@@ -824,7 +824,7 @@ enum Permission {
=======
@@ -830,7 +830,7 @@ enum Permission {
>>>>>>> upstream/main
   Execute
 };
 
-#if defined(OS_APPLE)
+#if defined(OS_APPLE) || defined(OS_BSD)
 bool ChangeFilePermissions(const FilePath& path, Permission perm, bool allow) {
   struct stat stat_buf;
 
<<<<<<< HEAD
@@ -853,9 +853,9 @@ bool ChangeFilePermissions(const FilePath& path, Permi
=======
@@ -859,9 +859,9 @@ bool ChangeFilePermissions(const FilePath& path, Permi
>>>>>>> upstream/main
   }
   return chmod(path.value().c_str(), stat_buf.st_mode) == 0;
 }
-#endif  // defined(OS_APPLE)
+#endif  // defined(OS_APPLE) || defined(OS_BSD)
 
-#if defined(OS_APPLE)
+#if defined(OS_APPLE) || defined(OS_BSD)
 // Linux implementation of FilePathWatcher doesn't catch attribute changes.
 // http://crbug.com/78043
 // Windows implementation of FilePathWatcher catches attribute changes that
<<<<<<< HEAD
@@ -891,7 +891,7 @@ TEST_F(FilePathWatcherTest, DirAttributesChanged) {
=======
@@ -897,7 +897,7 @@ TEST_F(FilePathWatcherTest, DirAttributesChanged) {
>>>>>>> upstream/main
   ASSERT_TRUE(ChangeFilePermissions(test_dir1, Execute, true));
 }
 
-#endif  // OS_APPLE
+#endif  // OS_APPLE || OS_BSD
 
 #if defined(OS_MAC)
 
