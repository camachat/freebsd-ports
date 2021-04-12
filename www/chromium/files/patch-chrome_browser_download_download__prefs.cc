<<<<<<< HEAD
--- chrome/browser/download/download_prefs.cc.orig	2020-11-13 06:36:36 UTC
+++ chrome/browser/download/download_prefs.cc
@@ -67,7 +67,7 @@ namespace {
=======
--- chrome/browser/download/download_prefs.cc.orig	2021-03-12 23:57:17 UTC
+++ chrome/browser/download/download_prefs.cc
@@ -68,7 +68,7 @@ namespace {
>>>>>>> upstream/main
 // Consider downloads 'dangerous' if they go to the home directory on Linux and
 // to the desktop on any platform.
 bool DownloadPathIsDangerous(const base::FilePath& download_path) {
-#if defined(OS_LINUX) || defined(OS_CHROMEOS)
+#if defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD)
   base::FilePath home_dir = base::GetHomeDir();
   if (download_path == home_dir) {
     return true;
<<<<<<< HEAD
@@ -172,7 +172,7 @@ DownloadPrefs::DownloadPrefs(Profile* profile) : profi
                                 GetDefaultDownloadDirectoryForProfile()));
 #endif  // defined(OS_CHROMEOS)
=======
@@ -173,7 +173,7 @@ DownloadPrefs::DownloadPrefs(Profile* profile) : profi
                                 GetDefaultDownloadDirectoryForProfile()));
 #endif  // BUILDFLAG(IS_CHROMEOS_ASH)
>>>>>>> upstream/main
 
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
   should_open_pdf_in_system_reader_ =
       prefs->GetBoolean(prefs::kOpenPdfDownloadInSystemReader);
<<<<<<< HEAD
@@ -300,7 +300,7 @@ void DownloadPrefs::RegisterProfilePrefs(
=======
@@ -301,7 +301,7 @@ void DownloadPrefs::RegisterProfilePrefs(
>>>>>>> upstream/main
                                  default_download_path);
   registry->RegisterFilePathPref(prefs::kSaveFileDefaultDirectory,
                                  default_download_path);
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
   registry->RegisterBooleanPref(prefs::kOpenPdfDownloadInSystemReader, false);
 #endif
<<<<<<< HEAD
@@ -430,7 +430,7 @@ bool DownloadPrefs::IsDownloadPathManaged() const {
=======
@@ -431,7 +431,7 @@ bool DownloadPrefs::IsDownloadPathManaged() const {
>>>>>>> upstream/main
 }
 
 bool DownloadPrefs::IsAutoOpenByUserUsed() const {
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
   if (ShouldOpenPdfInSystemReader())
     return true;
<<<<<<< HEAD
@@ -445,7 +445,7 @@ bool DownloadPrefs::IsAutoOpenEnabled(const GURL& url,
=======
@@ -446,7 +446,7 @@ bool DownloadPrefs::IsAutoOpenEnabled(const GURL& url,
>>>>>>> upstream/main
     return false;
   DCHECK(extension[0] == base::FilePath::kExtensionSeparator);
   extension.erase(0, 1);
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
   if (base::FilePath::CompareEqualIgnoreCase(extension,
                                              FILE_PATH_LITERAL("pdf")) &&
<<<<<<< HEAD
@@ -496,7 +496,7 @@ void DownloadPrefs::DisableAutoOpenByUserBasedOnExtens
=======
@@ -497,7 +497,7 @@ void DownloadPrefs::DisableAutoOpenByUserBasedOnExtens
>>>>>>> upstream/main
   SaveAutoOpenState();
 }
 
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
 void DownloadPrefs::SetShouldOpenPdfInSystemReader(bool should_open) {
   if (should_open_pdf_in_system_reader_ == should_open)
<<<<<<< HEAD
@@ -518,7 +518,7 @@ bool DownloadPrefs::ShouldOpenPdfInSystemReader() cons
=======
@@ -519,7 +519,7 @@ bool DownloadPrefs::ShouldOpenPdfInSystemReader() cons
>>>>>>> upstream/main
 #endif
 
 void DownloadPrefs::ResetAutoOpenByUser() {
-#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || \
+#if defined(OS_WIN) || defined(OS_LINUX) || defined(OS_CHROMEOS) || defined(OS_BSD) || \
     defined(OS_MAC)
   SetShouldOpenPdfInSystemReader(false);
 #endif
