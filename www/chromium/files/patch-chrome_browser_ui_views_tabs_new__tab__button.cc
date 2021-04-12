<<<<<<< HEAD
--- chrome/browser/ui/views/tabs/new_tab_button.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/views/tabs/new_tab_button.cc
@@ -57,7 +57,7 @@ class NewTabButton::HighlightPathGenerator
 NewTabButton::NewTabButton(TabStrip* tab_strip, PressedCallback callback)
     : views::ImageButton(std::move(callback)), tab_strip_(tab_strip) {
   SetAnimateOnStateChange(true);
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- chrome/browser/ui/views/tabs/new_tab_button.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/views/tabs/new_tab_button.cc
@@ -58,7 +58,7 @@ NewTabButton::NewTabButton(TabStrip* tab_strip, Presse
   SetAnimateOnStateChange(true);
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   SetTriggerableEventFlags(GetTriggerableEventFlags() |
                            ui::EF_MIDDLE_MOUSE_BUTTON);
 #endif
