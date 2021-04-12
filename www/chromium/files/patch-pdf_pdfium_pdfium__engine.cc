<<<<<<< HEAD
--- pdf/pdfium/pdfium_engine.cc.orig	2021-01-18 21:29:02 UTC
+++ pdf/pdfium/pdfium_engine.cc
@@ -68,7 +68,7 @@
=======
--- pdf/pdfium/pdfium_engine.cc.orig	2021-03-12 23:57:28 UTC
+++ pdf/pdfium/pdfium_engine.cc
@@ -71,7 +71,7 @@
>>>>>>> upstream/main
 #include "ui/gfx/geometry/vector2d.h"
 #include "v8/include/v8.h"
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
 #include "pdf/pdfium/pdfium_font_linux.h"
 #endif
 
<<<<<<< HEAD
@@ -400,7 +400,7 @@ void InitializeSDK(bool enable_v8) {
=======
@@ -468,7 +468,7 @@ void InitializeSDK(bool enable_v8) {
>>>>>>> upstream/main
 
   FPDF_InitLibraryWithConfig(&config);
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   InitializeLinuxFontMapper();
 #endif
 
<<<<<<< HEAD
@@ -459,7 +459,7 @@ PDFiumEngine::PDFiumEngine(PDFEngine::Client* client,
=======
@@ -527,7 +527,7 @@ PDFiumEngine::PDFiumEngine(PDFEngine::Client* client,
>>>>>>> upstream/main
   IFSDK_PAUSE::user = nullptr;
   IFSDK_PAUSE::NeedToPauseNow = Pause_NeedToPauseNow;
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   // PreviewModeClient does not know its pp::Instance.
   SetLastInstance(client_->GetPluginInstance());
 #endif
<<<<<<< HEAD
@@ -924,7 +924,7 @@ pp::Buffer_Dev PDFiumEngine::PrintPagesAsRasterPdf(
=======
@@ -992,7 +992,7 @@ pp::Buffer_Dev PDFiumEngine::PrintPagesAsRasterPdf(
>>>>>>> upstream/main
 
   KillFormFocus();
 
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   SetLastInstance(client_->GetPluginInstance());
 #endif
 
<<<<<<< HEAD
@@ -3019,7 +3019,7 @@ bool PDFiumEngine::ContinuePaint(int progressive_index
=======
@@ -3123,7 +3123,7 @@ bool PDFiumEngine::ContinuePaint(int progressive_index
>>>>>>> upstream/main
   DCHECK_LT(static_cast<size_t>(progressive_index), progressive_paints_.size());
 
   last_progressive_start_time_ = base::Time::Now();
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   SetLastInstance(client_->GetPluginInstance());
 #endif
 
<<<<<<< HEAD
@@ -3506,7 +3506,7 @@ void PDFiumEngine::SetCurrentPage(int index) {
=======
@@ -3610,7 +3610,7 @@ void PDFiumEngine::SetCurrentPage(int index) {
>>>>>>> upstream/main
     FORM_DoPageAAction(old_page, form(), FPDFPAGE_AACTION_CLOSE);
   }
   most_visible_page_ = index;
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   SetLastInstance(client_->GetPluginInstance());
 #endif
   if (most_visible_page_ != -1 && called_do_document_action_) {
