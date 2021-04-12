<<<<<<< HEAD
--- components/viz/host/host_display_client.h.orig	2020-11-13 06:36:42 UTC
+++ components/viz/host/host_display_client.h
@@ -43,7 +43,7 @@ class VIZ_HOST_EXPORT HostDisplayClient : public mojom
       mojo::PendingReceiver<mojom::LayeredWindowUpdater> receiver) override;
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- components/viz/host/host_display_client.h.orig	2021-03-12 23:57:23 UTC
+++ components/viz/host/host_display_client.h
@@ -46,7 +46,7 @@ class VIZ_HOST_EXPORT HostDisplayClient : public mojom
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   void DidCompleteSwapWithNewSize(const gfx::Size& size) override;
 #endif
 
