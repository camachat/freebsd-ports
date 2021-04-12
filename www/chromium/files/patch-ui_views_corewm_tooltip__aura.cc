<<<<<<< HEAD
--- ui/views/corewm/tooltip_aura.cc.orig	2020-11-13 06:37:06 UTC
+++ ui/views/corewm/tooltip_aura.cc
@@ -46,7 +46,7 @@ constexpr int kVerticalPaddingBottom = 5;
 
 // TODO(varkha): Update if native widget can be transparent on Linux.
 bool CanUseTranslucentTooltipWidget() {
-#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_WIN)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_WIN) || defined(OS_BSD)
=======
--- ui/views/corewm/tooltip_aura.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/views/corewm/tooltip_aura.cc
@@ -49,7 +49,7 @@ constexpr int kVerticalPaddingBottom = 5;
 bool CanUseTranslucentTooltipWidget() {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_WIN)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_WIN) || defined(OS_BSD)
>>>>>>> upstream/main
   return false;
 #else
   return true;
