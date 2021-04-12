<<<<<<< HEAD
--- build/lib/layersChecker.ts.orig	2021-02-03 15:33:23 UTC
+++ build/lib/layersChecker.ts
@@ -299,7 +299,7 @@ function checkFile(program: ts.Program, sourceFile: ts
=======
--- build/lib/layersChecker.ts.orig	2021-03-30 12:04:46 UTC
+++ build/lib/layersChecker.ts
@@ -289,7 +289,7 @@ function checkFile(program: ts.Program, sourceFile: ts
>>>>>>> upstream/main
 function createProgram(tsconfigPath: string): ts.Program {
 	const tsConfig = ts.readConfigFile(tsconfigPath, ts.sys.readFile);
 
-	const configHostParser: ts.ParseConfigHost = { fileExists: existsSync, readDirectory: ts.sys.readDirectory, readFile: file => readFileSync(file, 'utf8'), useCaseSensitiveFileNames: process.platform === 'linux' };
+	const configHostParser: ts.ParseConfigHost = { fileExists: existsSync, readDirectory: ts.sys.readDirectory, readFile: file => readFileSync(file, 'utf8'), useCaseSensitiveFileNames: (process.platform === 'linux' || process.platform === 'freebsd') };
 	const tsConfigParsed = ts.parseJsonConfigFileContent(tsConfig.config, configHostParser, resolve(dirname(tsconfigPath)), { noEmit: true });
 
 	const compilerHost = ts.createCompilerHost(tsConfigParsed.options, true);
