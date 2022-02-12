--- BaseTools/Source/Python/Common/Misc.py.orig	2022-01-21 10:56:26.425526000 -0600
+++ BaseTools/Source/Python/Common/Misc.py	2022-01-21 11:00:47.051534000 -0600
@@ -1635,9 +1635,14 @@ class PeImageClass():
         ByteArray = array.array('B')
         ByteArray.fromfile(PeObject, 4)
         # PE signature should be 'PE\0\0'
-        if ByteArray.tostring() != b'PE\0\0':
-            self.ErrorInfo = self.FileName + ' has no valid PE signature PE00'
-            return
+        if sys.version_info < (3,9):
+            if ByteArray.tostring() != b'PE\0\0':
+                self.ErrorInfo = self.FileName + ' has no valid PE signature PE00'
+                return
+        else:
+            if ByteArray.tobytes() != b'PE\0\0':
+                self.ErrorInfo = self.FileName + ' has no valid PE signature PE00'
+                return
 
         # Read PE file header
         ByteArray = array.array('B')
