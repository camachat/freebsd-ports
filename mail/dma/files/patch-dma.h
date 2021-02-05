--- dma.h.orig	2021-02-05 18:21:50.690565000 +0000
+++ dma.h	2021-02-05 18:23:30.987015000 +0000
@@ -199,6 +199,7 @@
 void hmac_md5(unsigned char *, int, unsigned char *, int, unsigned char *);
 int smtp_auth_md5(int, char *, char *);
 int smtp_init_crypto(int, int, struct smtp_features*);
+int verify_server_fingerprint(const X509 *cert);
 
 /* dns.c */
 int dns_get_mx_list(const char *, int, struct mx_hostentry **, int);
