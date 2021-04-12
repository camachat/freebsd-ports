<<<<<<< HEAD
--- src/bootstrap-node.js.orig	2021-01-19 06:35:37 UTC
+++ src/bootstrap-node.js
@@ -94,7 +94,7 @@ exports.configurePortable = function (product) {
=======
--- src/bootstrap-node.js.orig	2021-03-30 12:04:46 UTC
+++ src/bootstrap-node.js
@@ -130,7 +130,7 @@ exports.configurePortable = function (product) {
>>>>>>> upstream/main
 			return process.env['VSCODE_PORTABLE'];
 		}
 
-		if (process.platform === 'win32' || process.platform === 'linux') {
+		if (process.platform === 'win32' || process.platform === 'linux' || process.platform === 'freebsd') {
 			return path.join(getApplicationPath(path), 'data');
 		}
 
