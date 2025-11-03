--- src/lib/fcitx-utils/log.cpp.orig	2025-01-23 12:40:37 UTC
+++ src/lib/fcitx-utils/log.cpp
@@ -215,15 +215,17 @@ LogMessageBuilder::LogMessageBuilder(std::ostream &out
         break;
     }
 
-#if FMT_VERSION >= 50300
+#ifdef __cpp_lib_format
     if (globalLogConfig.showTimeDate) {
         try {
-            auto now = std::chrono::system_clock::now();
-            auto floor = std::chrono::floor<std::chrono::seconds>(now);
-            auto micro = std::chrono::duration_cast<std::chrono::microseconds>(
-                now - floor);
-            auto t = fmt::localtime(std::chrono::system_clock::to_time_t(now));
-            auto timeString = fmt::format("{:%F %T}.{:06d}", t, micro.count());
+#if __cpp_lib_chrono >= 201907L
+            const auto *current_zone = std::chrono::current_zone();
+            std::chrono::zoned_time zoned_time{current_zone, now};
+
+            auto timeString = std::format("{:%F %T}", zoned_time);
+#else
+            auto timeString = std::format("{:%F %T}", now);
+#endif
             out_ << timeString << " ";
         } catch (...) {
         }
