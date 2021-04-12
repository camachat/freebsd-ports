<<<<<<< HEAD
--- third_party/perfetto/include/perfetto/base/time.h.orig	2020-11-13 06:42:20 UTC
+++ third_party/perfetto/include/perfetto/base/time.h
@@ -141,6 +141,9 @@ inline TimeNanos GetTimeInternalNs(clockid_t clk_id) {
=======
--- third_party/perfetto/include/perfetto/base/time.h.orig	2021-03-13 00:03:38 UTC
+++ third_party/perfetto/include/perfetto/base/time.h
@@ -142,6 +142,9 @@ inline TimeNanos GetTimeInternalNs(clockid_t clk_id) {
>>>>>>> upstream/main
 // Return ns from boot. Conversely to GetWallTimeNs, this clock counts also time
 // during suspend (when supported).
 inline TimeNanos GetBootTimeNs() {
+#if PERFETTO_BUILDFLAG(PERFETTO_OS_FREEBSD)
+  return GetTimeInternalNs(kWallTimeClockSource);
+#else
   // Determine if CLOCK_BOOTTIME is available on the first call.
   static const clockid_t kBootTimeClockSource = [] {
     struct timespec ts = {};
<<<<<<< HEAD
@@ -148,6 +151,7 @@ inline TimeNanos GetBootTimeNs() {
=======
@@ -149,6 +152,7 @@ inline TimeNanos GetBootTimeNs() {
>>>>>>> upstream/main
     return res == 0 ? CLOCK_BOOTTIME : kWallTimeClockSource;
   }();
   return GetTimeInternalNs(kBootTimeClockSource);
+#endif
 }
 
 inline TimeNanos GetWallTimeNs() {
