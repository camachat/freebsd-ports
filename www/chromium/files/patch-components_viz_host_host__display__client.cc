<<<<<<< HEAD
--- components/viz/host/host_display_client.cc.orig	2020-11-13 06:36:42 UTC
+++ components/viz/host/host_display_client.cc
@@ -56,7 +56,7 @@ void HostDisplayClient::CreateLayeredWindowUpdater(
 }
 #endif
 
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- components/viz/host/host_display_client.cc.orig	2021-03-12 23:57:23 UTC
+++ components/viz/host/host_display_client.cc
@@ -60,7 +60,7 @@ void HostDisplayClient::CreateLayeredWindowUpdater(
 
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 void HostDisplayClient::DidCompleteSwapWithNewSize(const gfx::Size& size) {
   NOTIMPLEMENTED();
 }
