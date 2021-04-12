<<<<<<< HEAD
--- chrome/browser/diagnostics/diagnostics_writer.h.orig	2020-11-13 06:36:36 UTC
=======
--- chrome/browser/diagnostics/diagnostics_writer.h.orig	2021-03-12 23:57:17 UTC
>>>>>>> upstream/main
+++ chrome/browser/diagnostics/diagnostics_writer.h
@@ -15,6 +15,8 @@ namespace diagnostics {
 // Console base class used internally.
 class SimpleConsole;
 
+#undef MACHINE
+
 class DiagnosticsWriter : public DiagnosticsModel::Observer {
  public:
   // The type of formatting done by this writer.
