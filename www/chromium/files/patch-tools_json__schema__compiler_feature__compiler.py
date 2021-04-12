<<<<<<< HEAD
--- tools/json_schema_compiler/feature_compiler.py.orig	2020-11-13 06:37:04 UTC
+++ tools/json_schema_compiler/feature_compiler.py
@@ -243,6 +243,7 @@ FEATURE_GRAMMAR = ({
=======
--- tools/json_schema_compiler/feature_compiler.py.orig	2021-03-12 23:57:47 UTC
+++ tools/json_schema_compiler/feature_compiler.py
@@ -252,6 +252,7 @@ FEATURE_GRAMMAR = ({
>>>>>>> upstream/main
                 'chromeos': 'Feature::CHROMEOS_PLATFORM',
                 'lacros': 'Feature::LACROS_PLATFORM',
                 'linux': 'Feature::LINUX_PLATFORM',
+                'bsd': 'Feature::LINUX_PLATFORM',
                 'mac': 'Feature::MACOSX_PLATFORM',
                 'win': 'Feature::WIN_PLATFORM',
             }
