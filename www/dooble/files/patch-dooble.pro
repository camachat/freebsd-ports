--- dooble.pro.orig	2020-10-08 14:35:20 UTC
+++ dooble.pro
@@ -205,17 +205,13 @@ QMAKE_CLEAN     += Dooble
 
 freebsd-* {
 DEFINES += DOOBLE_FREEBSD_WEBENGINE_MISMATCH
-QMAKE_CXXFLAGS_RELEASE += -O3 \
+QMAKE_CXXFLAGS_RELEASE += \
                           -Wall \
                           -Wcast-align \
                           -Wcast-qual \
                           -Wdouble-promotion \
-                          -Werror \
                           -Wextra \
                           -Wformat=2 \
-                          -Wformat-overflow=2 \
-                          -Wformat-truncation=2 \
-                          -Wl,-z,relro \
                           -Woverloaded-virtual \
                           -Wpointer-arith \
                           -Wstack-protector \
@@ -225,10 +221,8 @@ QMAKE_CXXFLAGS_RELEASE += -O3 \
                           -fPIE \
                           -fstack-protector-all \
                           -fwrapv \
-                          -mtune=generic \
                           -pedantic \
                           -std=c++11
-QMAKE_CXXFLAGS_RELEASE -= -O2
 } else:macx {
 QMAKE_CXXFLAGS_RELEASE += -O3 \
                           -Wall \
