<<<<<<< HEAD
--- ui/gfx/native_widget_types.h.orig	2020-11-13 06:37:06 UTC
+++ ui/gfx/native_widget_types.h
@@ -103,7 +103,7 @@ class ViewAndroid;
 #endif
 class SkBitmap;
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 extern "C" {
 struct _AtkObject;
 typedef struct _AtkObject AtkObject;
@@ -204,7 +204,7 @@ typedef id NativeViewAccessible;
 #elif defined(OS_MAC)
 typedef NSFont* NativeFont;
 typedef id NativeViewAccessible;
-#elif defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#elif (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- ui/gfx/native_widget_types.h.orig	2021-03-12 23:57:48 UTC
+++ ui/gfx/native_widget_types.h
@@ -106,7 +106,7 @@ class SkBitmap;
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 extern "C" {
 struct _AtkObject;
 typedef struct _AtkObject AtkObject;
@@ -209,7 +209,7 @@ typedef NSFont* NativeFont;
 typedef id NativeViewAccessible;
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 // Linux doesn't have a native font type.
 typedef AtkObject* NativeViewAccessible;
 #else
