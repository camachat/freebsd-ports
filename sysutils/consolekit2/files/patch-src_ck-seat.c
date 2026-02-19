--- src/ck-seat.c.orig	2025-03-20 16:35:45 UTC
+++ src/ck-seat.c
@@ -1054,7 +1054,7 @@ dbus_get_sessions (ConsoleKitSeat        *ckseat,
         }
 
         g_ptr_array_add (sessions, NULL);
-        console_kit_seat_complete_get_sessions (ckseat, context, sessions->pdata);
+        console_kit_seat_complete_get_sessions (ckseat, context, (const gchar *const *)sessions->pdata);
         g_ptr_array_unref (sessions);
         return TRUE;
 }
