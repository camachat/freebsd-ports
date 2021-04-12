<<<<<<< HEAD
--- ui/gfx/gpu_memory_buffer.h.orig	2021-01-18 21:29:48 UTC
+++ ui/gfx/gpu_memory_buffer.h
@@ -15,7 +15,7 @@
 #include "ui/gfx/geometry/rect.h"
 #include "ui/gfx/gfx_export.h"
=======
--- ui/gfx/gpu_memory_buffer.h.orig	2021-03-12 23:57:48 UTC
+++ ui/gfx/gpu_memory_buffer.h
@@ -16,7 +16,7 @@
 #include "ui/gfx/gfx_export.h"
 #include "ui/gfx/hdr_metadata.h"
>>>>>>> upstream/main
 
-#if defined(USE_OZONE) || defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(USE_OZONE) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "ui/gfx/native_pixmap_handle.h"
 #elif defined(OS_MAC)
 #include "ui/gfx/mac/io_surface.h"
<<<<<<< HEAD
@@ -69,7 +69,7 @@ struct GFX_EXPORT GpuMemoryBufferHandle {
=======
@@ -70,7 +70,7 @@ struct GFX_EXPORT GpuMemoryBufferHandle {
>>>>>>> upstream/main
   base::UnsafeSharedMemoryRegion region;
   uint32_t offset = 0;
   int32_t stride = 0;
-#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_FUCHSIA)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_FUCHSIA) || defined(OS_BSD)
   NativePixmapHandle native_pixmap_handle;
 #elif defined(OS_MAC)
<<<<<<< HEAD
   gfx::ScopedIOSurface io_surface;
=======
   ScopedIOSurface io_surface;
>>>>>>> upstream/main
