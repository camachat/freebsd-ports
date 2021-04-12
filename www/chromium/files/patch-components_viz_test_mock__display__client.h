<<<<<<< HEAD
--- components/viz/test/mock_display_client.h.orig	2020-11-13 06:36:42 UTC
+++ components/viz/test/mock_display_client.h
@@ -36,7 +36,7 @@ class MockDisplayClient : public mojom::DisplayClient 
   MOCK_METHOD1(SetWideColorEnabled, void(bool enabled));
   MOCK_METHOD1(SetPreferredRefreshRate, void(float refresh_rate));
 #endif
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- components/viz/test/mock_display_client.h.orig	2021-03-12 23:57:23 UTC
+++ components/viz/test/mock_display_client.h
@@ -39,7 +39,7 @@ class MockDisplayClient : public mojom::DisplayClient 
 #endif
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   MOCK_METHOD1(DidCompleteSwapWithNewSize, void(const gfx::Size&));
 #endif
 
