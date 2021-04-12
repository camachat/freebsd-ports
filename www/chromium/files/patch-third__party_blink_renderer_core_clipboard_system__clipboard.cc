<<<<<<< HEAD
--- third_party/blink/renderer/core/clipboard/system_clipboard.cc.orig	2021-01-18 21:29:04 UTC
+++ third_party/blink/renderer/core/clipboard/system_clipboard.cc
@@ -42,10 +42,10 @@ SystemClipboard::SystemClipboard(LocalFrame* frame)
   frame->GetBrowserInterfaceBroker().GetInterface(
       clipboard_.BindNewPipeAndPassReceiver(
           frame->GetTaskRunner(TaskType::kUserInteraction)));
-#if defined(OS_LINUX) || BUILDFLAG(IS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_LACROS) || defined(OS_BSD)
   is_selection_buffer_available_ =
       frame->GetSettings()->GetSelectionClipboardBufferAvailable();
-#endif  // defined(OS_LINUX) || BUILDFLAG(IS_LACROS)
+#endif  // defined(OS_LINUX) || BUILDFLAG(IS_LACROS) || defined(OS_BSD)
=======
--- third_party/blink/renderer/core/clipboard/system_clipboard.cc.orig	2021-03-12 23:57:29 UTC
+++ third_party/blink/renderer/core/clipboard/system_clipboard.cc
@@ -43,10 +43,10 @@ SystemClipboard::SystemClipboard(LocalFrame* frame)
   frame->GetBrowserInterfaceBroker().GetInterface(
       clipboard_.BindNewPipeAndPassReceiver(
           frame->GetTaskRunner(TaskType::kUserInteraction)));
-#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#if defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
   is_selection_buffer_available_ =
       frame->GetSettings()->GetSelectionClipboardBufferAvailable();
-#endif  // defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS)
+#endif  // defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD)
>>>>>>> upstream/main
 }
 
 bool SystemClipboard::IsSelectionMode() const {
