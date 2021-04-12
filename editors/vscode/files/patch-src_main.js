<<<<<<< HEAD
--- src/main.js.orig	2021-03-04 22:21:59 UTC
+++ src/main.js
@@ -150,7 +150,7 @@ function configureCommandlineSwitchesSync(cliArgs) {
=======
--- src/main.js.orig	2021-03-30 12:04:46 UTC
+++ src/main.js
@@ -147,7 +147,7 @@ function configureCommandlineSwitchesSync(cliArgs) {
>>>>>>> upstream/main
 		'force-color-profile'
 	];
 
-	if (process.platform === 'linux') {
+	if (process.platform === 'linux' || process.platform === 'freebsd') {
 
 		// Force enable screen readers on Linux via this flag
 		SUPPORTED_ELECTRON_SWITCHES.push('force-renderer-accessibility');
