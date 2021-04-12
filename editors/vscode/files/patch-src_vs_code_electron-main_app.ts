<<<<<<< HEAD
--- src/vs/code/electron-main/app.ts.orig	2021-03-04 22:21:59 UTC
+++ src/vs/code/electron-main/app.ts
@@ -525,7 +525,7 @@ export class CodeApplication extends Disposable {
=======
--- src/vs/code/electron-main/app.ts.orig	2021-03-30 12:04:46 UTC
+++ src/vs/code/electron-main/app.ts
@@ -527,7 +527,7 @@ export class CodeApplication extends Disposable {
>>>>>>> upstream/main
 				services.set(IUpdateService, new SyncDescriptor(Win32UpdateService));
 				break;
 
-			case 'linux':
+			case 'linux': case 'freebsd':
 				if (isLinuxSnap) {
 					services.set(IUpdateService, new SyncDescriptor(SnapUpdateService, [process.env['SNAP'], process.env['SNAP_REVISION']]));
 				} else {
