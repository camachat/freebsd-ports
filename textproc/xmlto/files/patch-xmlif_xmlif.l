--- xmlif/xmlif.l.orig	2015-02-11 12:20:51 UTC
+++ xmlif/xmlif.l
@@ -43,7 +43,7 @@ static int nselections;		/* number of selections */
 
 static char **selections;	/* selection tokens */
 static int nselections;		/* number of selections */
-static ifsense;			/* sense of last `if' or unless seen */
+static int ifsense;		/* sense of last `if' or unless seen */
 static char *attribute;		/* last attribute scanned */
 
 struct stack_t {
@@ -239,7 +239,7 @@ int yywrap() {exit(0);};
 
 int yywrap() {exit(0);};
 
-main(int argc, char *argv[])
+int main(int argc, char *argv[])
 {
     int i;
 
