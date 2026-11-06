--- src/core/configmgr2.cpp.orig	2026-07-14 18:03:19 UTC
+++ src/core/configmgr2.cpp
@@ -396,7 +396,7 @@ void ConfigMgr2::initAppPrefixPath() {
   QStringList potentialDirs;
   auto appDirPath = m_configService->getExecutionFolderPath();
   qInfo() << "App prefix path:" << appDirPath;
-  potentialDirs << appDirPath;
+  potentialDirs << appDirPath << "%%DATADIR%%";
 
 #if defined(Q_OS_LINUX)
   QDir localBinDir(appDirPath);
