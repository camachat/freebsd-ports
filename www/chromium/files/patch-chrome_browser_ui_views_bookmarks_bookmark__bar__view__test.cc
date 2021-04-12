<<<<<<< HEAD
--- chrome/browser/ui/views/bookmarks/bookmark_bar_view_test.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/views/bookmarks/bookmark_bar_view_test.cc
@@ -1848,7 +1848,7 @@ class BookmarkBarViewTest20 : public BookmarkBarViewEv
=======
--- chrome/browser/ui/views/bookmarks/bookmark_bar_view_test.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/views/bookmarks/bookmark_bar_view_test.cc
@@ -1856,7 +1856,7 @@ class BookmarkBarViewTest20 : public BookmarkBarViewEv
>>>>>>> upstream/main
   }
 
   void Step3() {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
     EXPECT_EQ(1, test_view_->press_count());
 #else
     EXPECT_EQ(2, test_view_->press_count());
