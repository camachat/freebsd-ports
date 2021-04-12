<<<<<<< HEAD
--- ui/events/event.cc.orig	2020-11-16 15:03:40 UTC
+++ ui/events/event.cc
@@ -469,7 +469,7 @@ std::string LocatedEvent::ToString() const {
=======
--- ui/events/event.cc.orig	2021-03-12 23:57:48 UTC
+++ ui/events/event.cc
@@ -451,7 +451,7 @@ std::string LocatedEvent::ToString() const {
>>>>>>> upstream/main
 MouseEvent::MouseEvent(const PlatformEvent& native_event)
     : LocatedEvent(native_event),
       changed_button_flags_(GetChangedMouseButtonFlagsFromNative(native_event)),
-#if defined(OS_CHROMEOS) || defined(OS_LINUX)
+#if defined(OS_CHROMEOS) || defined(OS_LINUX) || defined(OS_BSD)
       movement_(GetMouseMovementFromNative(native_event)),
 #endif
       pointer_details_(GetMousePointerDetailsFromNative(native_event)) {
