<<<<<<< HEAD
--- chrome/browser/ui/views/frame/system_menu_model_builder.cc.orig	2020-11-13 06:36:38 UTC
+++ chrome/browser/ui/views/frame/system_menu_model_builder.cc
@@ -63,7 +63,7 @@ void SystemMenuModelBuilder::BuildMenu(ui::SimpleMenuM
 
 void SystemMenuModelBuilder::BuildSystemMenuForBrowserWindow(
     ui::SimpleMenuModel* model) {
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
   model->AddItemWithStringId(IDC_MINIMIZE_WINDOW, IDS_MINIMIZE_WINDOW_MENU);
   model->AddItemWithStringId(IDC_MAXIMIZE_WINDOW, IDS_MAXIMIZE_WINDOW_MENU);
   model->AddItemWithStringId(IDC_RESTORE_WINDOW, IDS_RESTORE_WINDOW_MENU);
@@ -78,7 +78,7 @@ void SystemMenuModelBuilder::BuildSystemMenuForBrowser
     model->AddSeparator(ui::NORMAL_SEPARATOR);
     model->AddItemWithStringId(IDC_TASK_MANAGER, IDS_TASK_MANAGER);
   }
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
   model->AddSeparator(ui::NORMAL_SEPARATOR);
   model->AddCheckItemWithStringId(IDC_USE_SYSTEM_TITLE_BAR,
                                   IDS_SHOW_WINDOW_DECORATIONS_MENU);
@@ -116,7 +116,7 @@ void SystemMenuModelBuilder::BuildSystemMenuForAppOrPo
     model->AddSeparator(ui::NORMAL_SEPARATOR);
     model->AddItemWithStringId(IDC_TASK_MANAGER, IDS_TASK_MANAGER);
   }
-#if defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#if (defined(OS_LINUX) || defined(OS_BSD)) && !defined(OS_CHROMEOS)
=======
--- chrome/browser/ui/views/frame/system_menu_model_builder.cc.orig	2021-03-12 23:57:19 UTC
+++ chrome/browser/ui/views/frame/system_menu_model_builder.cc
@@ -70,7 +70,7 @@ void SystemMenuModelBuilder::BuildSystemMenuForBrowser
     ui::SimpleMenuModel* model) {
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
   model->AddItemWithStringId(IDC_MINIMIZE_WINDOW, IDS_MINIMIZE_WINDOW_MENU);
   model->AddItemWithStringId(IDC_MAXIMIZE_WINDOW, IDS_MAXIMIZE_WINDOW_MENU);
   model->AddItemWithStringId(IDC_RESTORE_WINDOW, IDS_RESTORE_WINDOW_MENU);
@@ -87,7 +87,7 @@ void SystemMenuModelBuilder::BuildSystemMenuForBrowser
   }
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) && !BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
   model->AddSeparator(ui::NORMAL_SEPARATOR);
   model->AddCheckItemWithStringId(IDC_USE_SYSTEM_TITLE_BAR,
                                   IDS_SHOW_WINDOW_DECORATIONS_MENU);
@@ -130,7 +130,7 @@ void SystemMenuModelBuilder::BuildSystemMenuForAppOrPo
   }
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if (defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)) || defined(OS_BSD)
>>>>>>> upstream/main
   model->AddSeparator(ui::NORMAL_SEPARATOR);
   model->AddItemWithStringId(IDC_CLOSE_WINDOW, IDS_CLOSE);
 #endif
