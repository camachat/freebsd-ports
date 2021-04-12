<<<<<<< HEAD
--- base/util/memory_pressure/system_memory_pressure_evaluator.cc.orig	2020-11-16 14:03:42 UTC
+++ base/util/memory_pressure/system_memory_pressure_evaluator.cc
@@ -14,7 +14,7 @@
 #elif defined(OS_WIN)
 #include "base/util/memory_pressure/system_memory_pressure_evaluator_win.h"
 #include "base/win/windows_version.h"
-#elif defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#elif (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
 #include "base/util/memory_pressure/system_memory_pressure_evaluator_linux.h"
 #endif
 
@@ -45,7 +45,7 @@ SystemMemoryPressureEvaluator::CreateDefaultSystemEval
     evaluator->CreateOSSignalPressureEvaluator(monitor->CreateVoter());
   }
   return evaluator;
-#elif defined(OS_LINUX) && !defined(OS_CHROMEOS)
+#elif (defined(OS_LINUX) && !defined(OS_CHROMEOS)) || defined(OS_BSD)
=======
--- base/util/memory_pressure/system_memory_pressure_evaluator.cc.orig	2021-03-12 23:57:15 UTC
+++ base/util/memory_pressure/system_memory_pressure_evaluator.cc
@@ -17,7 +17,7 @@
 #include "base/win/windows_version.h"
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
 #include "base/util/memory_pressure/system_memory_pressure_evaluator_linux.h"
 #endif
 
@@ -50,7 +50,7 @@ SystemMemoryPressureEvaluator::CreateDefaultSystemEval
   return evaluator;
 // TODO(crbug.com/1052397): Revisit the macro expression once build flag switch
 // of lacros-chrome is complete.
-#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#elif defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
   return std::make_unique<util::os_linux::SystemMemoryPressureEvaluator>(
       monitor->CreateVoter());
 #endif
