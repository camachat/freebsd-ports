--- src/share/poudriere/common.sh.orig	2022-02-20 13:16:22.796874000 -0600
+++ src/share/poudriere/common.sh	2022-02-20 13:16:45.338727000 -0600
@@ -4123,6 +4123,14 @@ build_pkg() {
 			;;
 		esac
 	done
+    if [ -f "${SHARED_LOCK_DIR}/allow_make_jobs.ctl" ]; then
+        sed -i '' '/DISABLE_MAKE_JOBS=poudriere/d' \
+            "${mnt}/etc/make.conf"
+        if [ `cat "${SHARED_LOCK_DIR}/allow_make_jobs.ctl"` -eq 0 ]; then
+            echo "DISABLE_MAKE_JOBS=poudriere" \
+                >> ${mnt}/etc/make.conf
+        fi
+    fi
 
 	buildlog_start "${ORIGINSPEC}"
 
