<<<<<<< HEAD
--- ui/views/widget/native_widget_aura.cc.orig	2021-01-18 21:29:49 UTC
+++ ui/views/widget/native_widget_aura.cc
@@ -65,7 +65,7 @@
=======
--- ui/views/widget/native_widget_aura.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/views/widget/native_widget_aura.cc
@@ -68,7 +68,7 @@
>>>>>>> upstream/main
 #endif
 
 #if BUILDFLAG(ENABLE_DESKTOP_AURA) && \
-    (defined(OS_LINUX) || defined(OS_CHROMEOS))
+    (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD))
 #include "ui/views/linux_ui/linux_ui.h"
 #include "ui/views/widget/desktop_aura/desktop_window_tree_host_linux.h"
 #endif
<<<<<<< HEAD
@@ -1090,7 +1090,7 @@ void NativeWidgetAura::SetInitialFocus(ui::WindowShowS
=======
@@ -1121,7 +1121,7 @@ void NativeWidgetAura::SetInitialFocus(ui::WindowShowS
>>>>>>> upstream/main
 
 namespace {
 #if BUILDFLAG(ENABLE_DESKTOP_AURA) && \
-    (defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS))
+    (defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD))
 void CloseWindow(aura::Window* window) {
   if (window) {
     Widget* widget = Widget::GetWidgetForNativeView(window);
<<<<<<< HEAD
@@ -1121,14 +1121,14 @@ void Widget::CloseAllSecondaryWidgets() {
=======
@@ -1152,14 +1152,14 @@ void Widget::CloseAllSecondaryWidgets() {
>>>>>>> upstream/main
 #endif
 
 #if BUILDFLAG(ENABLE_DESKTOP_AURA) && \
-    (defined(OS_LINUX) || defined(OS_CHROMEOS))
+    (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD))
   DesktopWindowTreeHostLinux::CleanUpWindowList(CloseWindow);
 #endif
 }
 
 const ui::NativeTheme* Widget::GetNativeTheme() const {
 #if BUILDFLAG(ENABLE_DESKTOP_AURA) && \
-    (defined(OS_LINUX) || defined(OS_CHROMEOS))
+    (defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD))
   const LinuxUI* linux_ui = LinuxUI::instance();
   if (linux_ui) {
     ui::NativeTheme* native_theme =
