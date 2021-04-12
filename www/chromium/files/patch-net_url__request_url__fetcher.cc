<<<<<<< HEAD
--- net/url_request/url_fetcher.cc.orig	2020-11-13 06:36:46 UTC
+++ net/url_request/url_fetcher.cc
@@ -21,7 +21,7 @@ void URLFetcher::SetIgnoreCertificateRequests(bool ign
   URLFetcherImpl::SetIgnoreCertificateRequests(ignored);
 }
 
-#if (!defined(OS_WIN) && !defined(OS_LINUX)) || defined(OS_CHROMEOS)
+#if (!defined(OS_WIN) && !defined(OS_LINUX) && !defined(OS_BSD)) || defined(OS_CHROMEOS)
=======
--- net/url_request/url_fetcher.cc.orig	2021-03-12 23:57:27 UTC
+++ net/url_request/url_fetcher.cc
@@ -24,7 +24,7 @@ void URLFetcher::SetIgnoreCertificateRequests(bool ign
 
 // TODO(crbug.com/1052397): Revisit once build flag switch of lacros-chrome is
 // complete.
-#if !defined(OS_WIN) && !(defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS))
+#if !defined(OS_WIN) && !(defined(OS_LINUX) || BUILDFLAG(IS_CHROMEOS_LACROS) || defined(OS_BSD))
>>>>>>> upstream/main
 // static
 std::unique_ptr<URLFetcher> URLFetcher::Create(
     const GURL& url,
