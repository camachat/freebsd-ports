<<<<<<< HEAD
--- content/browser/renderer_host/render_widget_host_view_aura.cc.orig	2021-01-18 21:28:57 UTC
=======
--- content/browser/renderer_host/render_widget_host_view_aura.cc.orig	2021-03-12 23:57:24 UTC
>>>>>>> upstream/main
+++ content/browser/renderer_host/render_widget_host_view_aura.cc
@@ -111,7 +111,7 @@
 #include "ui/gfx/gdi_util.h"
 #endif
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 #include "content/browser/accessibility/browser_accessibility_auralinux.h"
 #include "ui/base/ime/linux/text_edit_command_auralinux.h"
 #include "ui/base/ime/linux/text_edit_key_bindings_delegate_auralinux.h"
<<<<<<< HEAD
@@ -492,7 +492,7 @@ gfx::NativeViewAccessible RenderWidgetHostViewAura::Ge
=======
@@ -474,7 +474,7 @@ gfx::NativeViewAccessible RenderWidgetHostViewAura::Ge
>>>>>>> upstream/main
   if (manager)
     return ToBrowserAccessibilityWin(manager->GetRoot())->GetCOM();
 
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   BrowserAccessibilityManager* manager =
       host()->GetOrCreateRootBrowserAccessibilityManager();
   if (manager && manager->GetRoot())
<<<<<<< HEAD
@@ -2234,7 +2234,7 @@ bool RenderWidgetHostViewAura::NeedsInputGrab() {
=======
@@ -2188,7 +2188,7 @@ bool RenderWidgetHostViewAura::NeedsInputGrab() {
>>>>>>> upstream/main
 }
 
 bool RenderWidgetHostViewAura::NeedsMouseCapture() {
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   return NeedsInputGrab();
 #else
   return false;
<<<<<<< HEAD
@@ -2398,7 +2398,7 @@ void RenderWidgetHostViewAura::ForwardKeyboardEventWit
=======
@@ -2354,7 +2354,7 @@ void RenderWidgetHostViewAura::ForwardKeyboardEventWit
>>>>>>> upstream/main
   if (!target_host)
     return;
 
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   ui::TextEditKeyBindingsDelegateAuraLinux* keybinding_delegate =
       ui::GetTextEditKeyBindingsDelegate();
   std::vector<ui::TextEditCommandAuraLinux> commands;
