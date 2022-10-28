--- src/freebsd/btop_collect.cpp.orig	2022-08-28 11:47:37 UTC
+++ src/freebsd/btop_collect.cpp
@@ -576,7 +576,7 @@ namespace Mem {
 		// this code is for ZFS mounts
 		for (string poolName : Mem::zpools) {
 			char sysCtl[1024];
-			snprintf(sysCtl, sizeof(sysCtl), "sysctl kstat.zfs.%s.dataset | egrep \'dataset_name|nread|nwritten\'", poolName.c_str());
+			snprintf(sysCtl, sizeof(sysCtl), "sysctl kstat.zfs.%s.dataset | grep -E \'dataset_name|nread|nwritten\'", poolName.c_str());
 			PipeWrapper f = PipeWrapper(sysCtl, "r");
 			if (f()) {
 				char buf[512];
