--- src/java.desktop/share/native/libharfbuzz/hb-ot-layout-gpos-table.hh.orig	2022-07-20 00:18:35.000000000 -0500
+++ src/java.desktop/share/native/libharfbuzz/hb-ot-layout-gpos-table.hh	2022-08-18 21:18:04.319828000 -0500
@@ -1128,8 +1128,8 @@ struct PairSet
     if (record)
     {
       /* Note the intentional use of "|" instead of short-circuit "||". */
-      if (valueFormats[0].apply_value (c, this, &record->values[0], buffer->cur_pos()) |
-          valueFormats[1].apply_value (c, this, &record->values[len1], buffer->pos[pos]))
+      if ((0 != valueFormats[0].apply_value (c, this, &record->values[0], buffer->cur_pos())) ||
+          (0 != valueFormats[1].apply_value (c, this, &record->values[len1], buffer->pos[pos])))
         buffer->unsafe_to_break (buffer->idx, pos + 1);
       if (len2)
         pos++;
@@ -1414,8 +1414,8 @@ struct PairPosFormat2
 
     const Value *v = &values[record_len * (klass1 * class2Count + klass2)];
     /* Note the intentional use of "|" instead of short-circuit "||". */
-    if (valueFormat1.apply_value (c, this, v, buffer->cur_pos()) |
-        valueFormat2.apply_value (c, this, v + len1, buffer->pos[skippy_iter.idx]))
+    if ((0 != valueFormat1.apply_value (c, this, v, buffer->cur_pos())) ||
+        (0 != valueFormat2.apply_value (c, this, v + len1, buffer->pos[skippy_iter.idx])))
       buffer->unsafe_to_break (buffer->idx, skippy_iter.idx + 1);
 
     buffer->idx = skippy_iter.idx;
