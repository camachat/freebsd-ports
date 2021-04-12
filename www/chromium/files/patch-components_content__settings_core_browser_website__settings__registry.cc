<<<<<<< HEAD
--- components/content_settings/core/browser/website_settings_registry.cc.orig	2020-11-13 06:36:40 UTC
+++ components/content_settings/core/browser/website_settings_registry.cc
@@ -66,7 +66,7 @@ const WebsiteSettingsInfo* WebsiteSettingsRegistry::Re
 #if defined(OS_WIN)
   if (!(platform & PLATFORM_WINDOWS))
     return nullptr;
-#elif defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#elif (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
=======
--- components/content_settings/core/browser/website_settings_registry.cc.orig	2021-03-12 23:57:22 UTC
+++ components/content_settings/core/browser/website_settings_registry.cc
@@ -69,7 +69,7 @@ const WebsiteSettingsInfo* WebsiteSettingsRegistry::Re
     return nullptr;
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   if (!(platform & PLATFORM_LINUX))
     return nullptr;
 #elif defined(OS_MAC)
