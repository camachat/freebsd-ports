<<<<<<< HEAD
--- ui/views/style/platform_style.cc.orig	2021-01-18 21:29:49 UTC
+++ ui/views/style/platform_style.cc
@@ -54,7 +54,7 @@ const bool PlatformStyle::kInactiveWidgetControlsAppea
 // Linux clips bubble windows that extend outside their parent window
 // bounds.
 const bool PlatformStyle::kAdjustBubbleIfOffscreen =
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
     false;
 #else
     true;
@@ -89,7 +89,7 @@ View::FocusBehavior PlatformStyle::DefaultFocusBehavio
=======
--- ui/views/style/platform_style.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/views/style/platform_style.cc
@@ -58,7 +58,7 @@ const View::FocusBehavior PlatformStyle::kDefaultFocus
 const bool PlatformStyle::kAdjustBubbleIfOffscreen =
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
     false;
 #else
     true;
@@ -88,7 +88,7 @@ gfx::Range PlatformStyle::RangeToDeleteBackwards(const
>>>>>>> upstream/main
 #endif  // OS_APPLE
 
 #if !BUILDFLAG(ENABLE_DESKTOP_AURA) || \
-    (!defined(OS_LINUX) && !defined(OS_CHROMEOS))
+    (!defined(OS_LINUX) && !defined(OS_CHROMEOS) && !defined(OS_BSD))
 // static
 std::unique_ptr<Border> PlatformStyle::CreateThemedLabelButtonBorder(
     LabelButton* button) {
