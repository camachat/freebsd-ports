--- src/core/services/synccredentialsstore.cpp.orig	2026-06-29 07:53:08 UTC
+++ src/core/services/synccredentialsstore.cpp
@@ -25,8 +25,8 @@ void logKeychainUnavailableOnce() {
   static bool s_logged = false;
   if (!s_logged) {
     s_logged = true;
-    qWarning() << "QtKeychain unavailable; sync features will be disabled "
-                  "until keychain backend is installed";
+    qWarning("QtKeychain unavailable; sync features will be disabled "
+                  "until keychain backend is installed");
   }
 }
 #endif
