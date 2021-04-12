<<<<<<< HEAD
--- content/browser/compositor/viz_process_transport_factory.cc.orig	2020-11-13 06:36:42 UTC
+++ content/browser/compositor/viz_process_transport_factory.cc
@@ -108,7 +108,7 @@ class HostDisplayClient : public viz::HostDisplayClien
   HostDisplayClient& operator=(const HostDisplayClient&) = delete;
 
   // viz::HostDisplayClient:
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- content/browser/compositor/viz_process_transport_factory.cc.orig	2021-03-12 23:57:24 UTC
+++ content/browser/compositor/viz_process_transport_factory.cc
@@ -111,7 +111,7 @@ class HostDisplayClient : public viz::HostDisplayClien
   // viz::HostDisplayClient:
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   void DidCompleteSwapWithNewSize(const gfx::Size& size) override {
     compositor_->OnCompleteSwapWithNewSize(size);
   }
