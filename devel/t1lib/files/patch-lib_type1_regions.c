--- lib/type1/regions.c.orig	2026-02-17 02:50:38 UTC
+++ lib/type1/regions.c
@@ -604,6 +604,7 @@ void ChangeDirection(type, R, x, y, dy, x2, y2)
        register struct region *R;  /* region in which we are changing direction */
        fractpel x,y;         /* current beginning x,y                        */
        fractpel dy;          /* direction and magnitude of change in y       */
+       fractpel x2,y2;
 {
        register fractpel ymin,ymax;  /* minimum and maximum Y since last call */
        register fractpel x_at_ymin,x_at_ymax;  /* their respective X's       */
