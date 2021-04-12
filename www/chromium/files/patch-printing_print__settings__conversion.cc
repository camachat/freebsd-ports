<<<<<<< HEAD
--- printing/print_settings_conversion.cc.orig	2021-01-18 21:29:02 UTC
+++ printing/print_settings_conversion.cc
@@ -211,14 +211,14 @@ std::unique_ptr<PrintSettings> PrintSettingsFromJobSet
 #endif
   }
 
-#if defined(OS_CHROMEOS) || (defined(OS_LINUX) && defined(USE_CUPS))
+#if defined(OS_CHROMEOS) || ((defined(OS_LINUX) || defined(OS_BSD)) && defined(USE_CUPS))
   const base::Value* advanced_settings =
       job_settings.FindDictKey(kSettingAdvancedSettings);
   if (advanced_settings) {
     for (const auto& item : advanced_settings->DictItems())
       settings->advanced_settings().emplace(item.first, item.second.Clone());
   }
-#endif  // defined(OS_CHROMEOS) || (defined(OS_LINUX) && defined(USE_CUPS))
+#endif  // defined(OS_CHROMEOS) || ((defined(OS_LINUX) || defined(OS_BSD)) && defined(USE_CUPS))
 
 #if BUILDFLAG(IS_ASH)
   bool send_user_info =
=======
--- printing/print_settings_conversion.cc.orig	2021-03-12 23:57:28 UTC
+++ printing/print_settings_conversion.cc
@@ -213,8 +213,8 @@ std::unique_ptr<PrintSettings> PrintSettingsFromJobSet
 
 // TODO(crbug.com/1052397): Revisit once build flag switch of lacros-chrome is
 // complete.
-#if defined(OS_CHROMEOS) ||                                  \
-    ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) && \
+#if defined(OS_CHROMEOS) ||                                                     \
+    ((defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)) && \
      defined(USE_CUPS))
   const base::Value* advanced_settings =
       job_settings.FindDictKey(kSettingAdvancedSettings);
@@ -222,7 +222,7 @@ std::unique_ptr<PrintSettings> PrintSettingsFromJobSet
     for (const auto& item : advanced_settings->DictItems())
       settings->advanced_settings().emplace(item.first, item.second.Clone());
   }
-#endif  // defined(OS_CHROMEOS) || ((defined(OS_LINUX) ||
+#endif  // defined(OS_CHROMEOS) || ((defined(OS_LINUX) || defined(OS_BSD) ||
         // BUILDFLAG(IS_CHROMEOS_LACROS)) && defined(USE_CUPS))
 
 #if BUILDFLAG(IS_CHROMEOS_ASH)
>>>>>>> upstream/main
