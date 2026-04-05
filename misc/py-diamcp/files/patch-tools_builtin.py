--- tools/builtin.py.orig	2026-04-01 12:53:39 UTC
+++ tools/builtin.py
@@ -144,7 +144,11 @@ def write_file(path: str, content: str) -> str:
     try:
         full_path.parent.mkdir(parents=True, exist_ok=True)
         full_path.write_text(content, encoding="utf-8")
-        return f"Successfully wrote to '{path}'"
+        result = f"Successfully wrote to '{path}'"
+        result += f"\n\n=== File Content ({full_path}) ===\n"
+        result += content
+        result += "\n=== End of File ===\n"
+        return result
     except Exception as e:
         return f"Error writing file: {e}"
 
