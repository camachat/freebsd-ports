<<<<<<< HEAD
--- ui/native_theme/native_theme_base.cc.orig	2020-11-13 06:37:06 UTC
+++ ui/native_theme/native_theme_base.cc
@@ -255,7 +255,7 @@ void NativeThemeBase::Paint(cc::PaintCanvas* canvas,
     case kCheckbox:
       PaintCheckbox(canvas, state, rect, extra.button, color_scheme);
       break;
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
=======
--- ui/native_theme/native_theme_base.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/native_theme/native_theme_base.cc
@@ -258,7 +258,7 @@ void NativeThemeBase::Paint(cc::PaintCanvas* canvas,
       break;
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
     case kFrameTopArea:
       PaintFrameTopArea(canvas, state, rect, extra.frame_top_area,
                         color_scheme);
