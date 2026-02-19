--- gtk/gtkcomboboxtext.c.orig	2020-12-18 05:29:40 UTC
+++ gtk/gtkcomboboxtext.c
@@ -275,7 +275,7 @@ gtk_combo_box_text_buildable_custom_tag_start (GtkBuil
 
       parser_data = g_slice_new0 (ItemParserData);
       parser_data->builder = g_object_ref (builder);
-      parser_data->object = g_object_ref (buildable);
+      parser_data->object = (GObject *)g_object_ref (buildable);
       parser_data->domain = gtk_builder_get_translation_domain (builder);
       *parser = item_parser;
       *data = parser_data;
