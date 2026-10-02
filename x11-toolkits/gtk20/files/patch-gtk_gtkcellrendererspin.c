--- gtk/gtkcellrendererspin.c.orig	2020-12-18 05:29:40 UTC
+++ gtk/gtkcellrendererspin.c
@@ -207,7 +207,7 @@ gtk_cell_renderer_spin_set_property (GObject      *obj
 	}
 
       if (obj)
-	priv->adjustment = g_object_ref_sink (obj);
+	priv->adjustment = (GtkAdjustment *)g_object_ref_sink (obj);
       break;
     case PROP_CLIMB_RATE:
       priv->climb_rate = g_value_get_double (value);
