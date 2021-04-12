<<<<<<< HEAD
--- electron/shell/browser/native_window_views.cc.orig	2021-02-19 19:40:19 UTC
=======
--- electron/shell/browser/native_window_views.cc.orig	2021-04-02 17:02:59 UTC
>>>>>>> upstream/main
+++ electron/shell/browser/native_window_views.cc
@@ -306,7 +306,7 @@ NativeWindowViews::NativeWindowViews(const gin_helper:
     last_window_state_ = ui::SHOW_STATE_NORMAL;
 #endif
 
-#if defined(OS_LINUX)
+#if defined(OS_LINUX) || defined(OS_BSD)
   // Listen to move events.
   aura::Window* window = GetNativeWindow();
   if (window)
@@ -322,7 +322,7 @@ NativeWindowViews::~NativeWindowViews() {
   SetForwardMouseMessages(false);
 #endif
 
-#if defined(OS_LINUX)
+#if defined(OS_LINUX) || defined(OS_BSD)
   aura::Window* window = GetNativeWindow();
   if (window)
     window->RemovePreTargetHandler(this);
<<<<<<< HEAD
@@ -1375,7 +1375,7 @@ void NativeWindowViews::OnWidgetBoundsChanged(views::W
=======
@@ -1386,7 +1386,7 @@ void NativeWindowViews::OnWidgetBoundsChanged(views::W
>>>>>>> upstream/main
 }
 
 void NativeWindowViews::OnWidgetDestroying(views::Widget* widget) {
-#if defined(OS_LINUX)
+#if defined(OS_LINUX) || defined(OS_BSD)
   aura::Window* window = GetNativeWindow();
   if (window)
     window->RemovePreTargetHandler(this);
<<<<<<< HEAD
@@ -1485,7 +1485,7 @@ void NativeWindowViews::HandleKeyboardEvent(
=======
@@ -1496,7 +1496,7 @@ void NativeWindowViews::HandleKeyboardEvent(
>>>>>>> upstream/main
   if (widget_destroyed_)
     return;
 
-#if defined(OS_LINUX)
+#if defined(OS_LINUX) || defined(OS_BSD)
   if (event.windows_key_code == ui::VKEY_BROWSER_BACK)
     NotifyWindowExecuteAppCommand(kBrowserBackward);
   else if (event.windows_key_code == ui::VKEY_BROWSER_FORWARD)
<<<<<<< HEAD
@@ -1497,7 +1497,7 @@ void NativeWindowViews::HandleKeyboardEvent(
=======
@@ -1508,7 +1508,7 @@ void NativeWindowViews::HandleKeyboardEvent(
>>>>>>> upstream/main
   root_view_->HandleKeyEvent(event);
 }
 
-#if defined(OS_LINUX)
+#if defined(OS_LINUX) || defined(OS_BSD)
 void NativeWindowViews::OnMouseEvent(ui::MouseEvent* event) {
   if (event->type() != ui::ET_MOUSE_PRESSED)
     return;
