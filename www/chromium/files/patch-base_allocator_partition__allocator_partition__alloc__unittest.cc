<<<<<<< HEAD
--- base/allocator/partition_allocator/partition_alloc_unittest.cc.orig	2021-01-18 21:28:44 UTC
+++ base/allocator/partition_allocator/partition_alloc_unittest.cc
@@ -366,9 +366,13 @@ void FreeFullSlotSpan(PartitionRoot<base::internal::Th
   }
 }
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 bool CheckPageInCore(void* ptr, bool in_core) {
+#if defined(OS_BSD)
+  char ret = 0;
+#else
   unsigned char ret = 0;
+#endif
   EXPECT_EQ(0, mincore(ptr, SystemPageSize(), &ret));
   return in_core == (ret & 1);
 }
@@ -377,7 +381,7 @@ bool CheckPageInCore(void* ptr, bool in_core) {
   EXPECT_TRUE(CheckPageInCore(ptr, in_core))
 #else
 #define CHECK_PAGE_IN_CORE(ptr, in_core) (void)(0)
-#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS)
+#endif  // defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 
 class MockPartitionStatsDumper : public PartitionStatsDumper {
  public:
=======
--- base/allocator/partition_allocator/partition_alloc_unittest.cc.orig	2021-03-12 23:57:15 UTC
+++ base/allocator/partition_allocator/partition_alloc_unittest.cc
@@ -1588,7 +1588,7 @@ TEST_F(PartitionAllocTest, LostFreeSlotSpansBug) {
 // cause flake.
 #if !defined(OS_WIN) &&            \
     (!defined(ARCH_CPU_64_BITS) || \
-     (defined(OS_POSIX) && !(defined(OS_APPLE) || defined(OS_ANDROID))))
+     (defined(OS_POSIX) && !(defined(OS_APPLE) || defined(OS_ANDROID) || defined(OS_BSD))))
 
 // The following four tests wrap a called function in an expect death statement
 // to perform their test, because they are non-hermetic. Specifically they are
@@ -1634,7 +1634,7 @@ TEST_F(PartitionAllocDeathTest, RepeatedTryReallocRetu
 }
 
 #endif  // !defined(ARCH_CPU_64_BITS) || (defined(OS_POSIX) &&
-        // !(defined(OS_APPLE) || defined(OS_ANDROID)))
+        // !(defined(OS_APPLE) || defined(OS_ANDROID) || defined(OS_BSD)))
 
 // Make sure that malloc(-1) dies.
 // In the past, we had an integer overflow that would alias malloc(-1) to
>>>>>>> upstream/main
