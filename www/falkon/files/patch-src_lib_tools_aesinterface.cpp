--- src/lib/tools/aesinterface.cpp.orig	2019-01-16 08:58:55.555405000 -0800
+++ src/lib/tools/aesinterface.cpp	2019-01-16 08:59:08.513498000 -0800
@@ -41,14 +41,14 @@ AesInterface::AesInterface(QObject* parent)
 {
     m_encodeCTX = EVP_CIPHER_CTX_new();
     m_decodeCTX = EVP_CIPHER_CTX_new();
-    EVP_CIPHER_CTX_init(m_encodeCTX);
-    EVP_CIPHER_CTX_init(m_decodeCTX);
+//-    EVP_CIPHER_CTX_init(m_encodeCTX);
+//-    EVP_CIPHER_CTX_init(m_decodeCTX);
 }
 
 AesInterface::~AesInterface()
 {
-    EVP_CIPHER_CTX_cleanup(m_encodeCTX);
-    EVP_CIPHER_CTX_cleanup(m_decodeCTX);
+//-    EVP_CIPHER_CTX_cleanup(m_encodeCTX);
+//-    EVP_CIPHER_CTX_cleanup(m_decodeCTX);
     EVP_CIPHER_CTX_free(m_encodeCTX);
     EVP_CIPHER_CTX_free(m_decodeCTX);
 }
