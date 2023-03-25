--- Source/WTF/wtf/URLHelpers.cpp.orig	2021-10-21 08:52:08 UTC
+++ Source/WTF/wtf/URLHelpers.cpp
@@ -48,7 +48,7 @@ constexpr unsigned urlBytesBufferLength = 2048;
 //    WebKit was compiled.
 // This is only really important for platforms that load an external IDN allowed script list.
 // Not important for the compiled-in one.
-constexpr auto scriptCodeLimit = static_cast<UScriptCode>(256);
+constexpr auto scriptCodeLimit = static_cast<UScriptCode>(USCRIPT_INVALID_CODE);
 
 static uint32_t allowedIDNScriptBits[(scriptCodeLimit + 31) / 32];
 
