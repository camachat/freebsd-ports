--- src/3rdparty/chromium/third_party/blink/renderer/platform/fonts/font_cache.cc.orig	2024-09-30 07:45:04 UTC
+++ src/3rdparty/chromium/third_party/blink/renderer/platform/fonts/font_cache.cc
@@ -81,7 +81,7 @@ extern const char kNotoColorEmojiCompat[] = "Noto Colo
 
 SkFontMgr* FontCache::static_font_manager_ = nullptr;
 
-#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
+#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_BSD)
 float FontCache::device_scale_factor_ = 1.0;
 #endif
 
@@ -132,7 +132,7 @@ const FontPlatformData* FontCache::SystemFontPlatformD
     const FontDescription& font_description) {
   const AtomicString& family = FontCache::SystemFontFamily();
 #if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_FUCHSIA) || \
-    BUILDFLAG(IS_IOS)
+    BUILDFLAG(IS_IOS) || BUILDFLAG(IS_BSD)
   if (family.empty() || family == font_family_names::kSystemUi)
     return nullptr;
 #else
