<<<<<<< HEAD
--- electron/spec-main/api-browser-window-spec.ts.orig	2021-02-02 17:38:53 UTC
=======
--- electron/spec-main/api-browser-window-spec.ts.orig	2021-03-23 15:10:33 UTC
>>>>>>> upstream/main
+++ electron/spec-main/api-browser-window-spec.ts
@@ -57,7 +57,7 @@ describe('BrowserWindow module', () => {
       }).not.to.throw();
     });
 
-    ifit(process.platform === 'linux')('does not crash when setting large window icons', async () => {
+    ifit(process.platform === 'linux' || process.platform === 'freebsd')('does not crash when setting large window icons', async () => {
       const appPath = path.join(__dirname, 'spec-main', 'fixtures', 'apps', 'xwindow-icon');
       const appProcess = childProcess.spawn(process.execPath, [appPath]);
       await new Promise((resolve) => { appProcess.once('exit', resolve); });
@@ -1036,7 +1036,7 @@ describe('BrowserWindow module', () => {
         });
       });
 
-      ifdescribe(process.platform !== 'linux')('Maximized state', () => {
+      ifdescribe(process.platform !== 'linux' && process.platform !== 'freebsd')('Maximized state', () => {
         it('checks normal bounds when maximized', async () => {
           const bounds = w.getBounds();
           const maximize = emittedOnce(w, 'maximize');
@@ -1097,7 +1097,7 @@ describe('BrowserWindow module', () => {
         });
       });
 
-      ifdescribe(process.platform !== 'linux')('Minimized state', () => {
+      ifdescribe(process.platform !== 'linux' && process.platform !== 'freebsd')('Minimized state', () => {
         it('checks normal bounds when minimized', async () => {
           const bounds = w.getBounds();
           const minimize = emittedOnce(w, 'minimize');
<<<<<<< HEAD
@@ -1669,7 +1669,7 @@ describe('BrowserWindow module', () => {
=======
@@ -1685,7 +1685,7 @@ describe('BrowserWindow module', () => {
>>>>>>> upstream/main
   describe('BrowserWindow.setOpacity(opacity)', () => {
     afterEach(closeAllWindows);
 
-    ifdescribe(process.platform !== 'linux')(('Windows and Mac'), () => {
+    ifdescribe(process.platform !== 'linux' && process.platform !== 'freebsd')(('Windows and Mac'), () => {
       it('make window with initial opacity', () => {
         const w = new BrowserWindow({ show: false, opacity: 0.5 });
         expect(w.getOpacity()).to.equal(0.5);
<<<<<<< HEAD
@@ -1695,7 +1695,7 @@ describe('BrowserWindow module', () => {
=======
@@ -1711,7 +1711,7 @@ describe('BrowserWindow module', () => {
>>>>>>> upstream/main
       });
     });
 
-    ifdescribe(process.platform === 'linux')(('Linux'), () => {
+    ifdescribe(process.platform === 'linux' || process.platform === 'freebsd')(('Linux'), () => {
       it('sets 1 regardless of parameter', () => {
         const w = new BrowserWindow({ show: false });
         w.setOpacity(0);
<<<<<<< HEAD
@@ -2531,7 +2531,7 @@ describe('BrowserWindow module', () => {
=======
@@ -2547,7 +2547,7 @@ describe('BrowserWindow module', () => {
>>>>>>> upstream/main
         expect(test.version).to.equal(process.version);
         expect(test.versions).to.deep.equal(process.versions);
 
-        if (process.platform === 'linux' && test.osSandbox) {
+        if ((process.platform === 'linux' || process.platform === 'freebsd') && test.osSandbox) {
           expect(test.creationTime).to.be.null('creation time');
           expect(test.systemMemoryInfo).to.be.null('system memory info');
         } else {
<<<<<<< HEAD
@@ -3057,7 +3057,7 @@ describe('BrowserWindow module', () => {
=======
@@ -3073,7 +3073,7 @@ describe('BrowserWindow module', () => {
>>>>>>> upstream/main
     });
   });
 
-  ifdescribe(process.platform !== 'linux')('max/minimize events', () => {
+  ifdescribe(process.platform !== 'linux' && process.platform !== 'freebsd')('max/minimize events', () => {
     afterEach(closeAllWindows);
     it('emits an event when window is maximized', async () => {
       const w = new BrowserWindow({ show: false });
<<<<<<< HEAD
@@ -3635,7 +3635,7 @@ describe('BrowserWindow module', () => {
=======
@@ -3651,7 +3651,7 @@ describe('BrowserWindow module', () => {
>>>>>>> upstream/main
     });
   });
 
-  ifdescribe(process.platform !== 'linux')('window states (excluding Linux)', () => {
+  ifdescribe(process.platform !== 'linux' && process.platform !== 'freebsd')('window states (excluding Linux)', () => {
     // Not implemented on Linux.
     afterEach(closeAllWindows);
 
