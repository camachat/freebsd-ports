--- BaseTools/Source/Python/AutoGen/UniClassObject.py.orig	2019-11-25 08:14:18.000000000 -0600
+++ BaseTools/Source/Python/AutoGen/UniClassObject.py	2022-01-21 10:55:01.309389000 -0600
@@ -309,7 +309,7 @@ class UniFileClassObject(object):
 
     @staticmethod
     def VerifyUcs2Data(FileIn, FileName, Encoding):
-        Ucs2Info = codecs.lookup('ucs-2')
+        Ucs2Info = codecs.lookup('utf-16')
         #
         # Convert to unicode
         #
