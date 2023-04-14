--- iocage_lib/ioc_common.py.orig	2023-04-14 21:05:11 UTC
+++ iocage_lib/ioc_common.py
@@ -77,7 +77,7 @@ def callback(_log, callback_exception):
         else:
             if not isinstance(message, str) and isinstance(
                 message,
-                collections.Iterable
+                collections.abc.Iterable
             ):
                 message = '\n'.join(message)
 
