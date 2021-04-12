<<<<<<< HEAD
--- chrome/test/base/interactive_ui_tests_main.cc.orig	2021-01-18 21:28:52 UTC
+++ chrome/test/base/interactive_ui_tests_main.cc
@@ -16,7 +16,7 @@
 #if defined(USE_AURA)
 #include "ui/aura/test/ui_controls_factory_aura.h"
 #include "ui/base/test/ui_controls_aura.h"
-#if defined(USE_OZONE) && defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if defined(USE_OZONE) && ((defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD))
 #include "ui/base/ui_base_features.h"
 #include "ui/ozone/public/ozone_platform.h"
 #include "ui/views/test/ui_controls_factory_desktop_aura_ozone.h"
@@ -54,7 +54,7 @@ class InteractiveUITestSuite : public ChromeTestSuite 
     com_initializer_.reset(new base::win::ScopedCOMInitializer());
     ui_controls::InstallUIControlsAura(
         aura::test::CreateUIControlsAura(nullptr));
-#elif defined(OS_LINUX)
+#elif defined(OS_LINUX) || defined(OS_BSD)
=======
--- chrome/test/base/interactive_ui_tests_main.cc.orig	2021-03-12 23:57:20 UTC
+++ chrome/test/base/interactive_ui_tests_main.cc
@@ -17,7 +17,7 @@
 #if defined(USE_AURA)
 #include "ui/aura/test/ui_controls_factory_aura.h"
 #include "ui/base/test/ui_controls_aura.h"
-#if defined(USE_OZONE) && (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if defined(USE_OZONE) && (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
 #include "ui/base/ui_base_features.h"
 #include "ui/ozone/public/ozone_platform.h"
 #include "ui/views/test/ui_controls_factory_desktop_aura_ozone.h"
@@ -55,7 +55,7 @@ class InteractiveUITestSuite : public ChromeTestSuite 
     com_initializer_.reset(new base::win::ScopedCOMInitializer());
     ui_controls::InstallUIControlsAura(
         aura::test::CreateUIControlsAura(nullptr));
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 #if defined(USE_OZONE)
     if (features::IsUsingOzonePlatform()) {
       ui::OzonePlatform::InitParams params;
