--- src/core/configmgr.cpp.orig	2026-06-30 16:15:53 UTC
+++ src/core/configmgr.cpp
@@ -343,7 +343,7 @@ void ConfigMgr::initAppPrefixPath() {
   // auto app_dir_path = VxCore::getInst().getExecutionFolderPath();
   auto app_dir_path = QCoreApplication::applicationDirPath();
   qInfo() << "app prefix path: " << app_dir_path;
-  potential_dirs << app_dir_path;
+  potential_dirs << app_dir_path << "%%DATADIR%%";
 
 #if defined(Q_OS_LINUX)
   QDir localBinDir(app_dir_path);
