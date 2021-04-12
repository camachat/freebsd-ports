<<<<<<< HEAD
--- ngs-sdk/setup/konfigure.perl.orig	2020-04-01 16:34:36 UTC
+++ ngs-sdk/setup/konfigure.perl
@@ -202,7 +202,7 @@ print "checking system type... " unless ($AUTORUN);
=======
--- ngs-sdk/setup/konfigure.perl.orig	2021-03-15 18:08:20 UTC
+++ ngs-sdk/setup/konfigure.perl
@@ -204,7 +204,7 @@ print "checking system type... " unless ($AUTORUN);
>>>>>>> upstream/main
 my ($OS, $ARCH, $OSTYPE, $MARCH, @ARCHITECTURES) = OsArch();
 println $OSTYPE unless ($AUTORUN);
 
-unless ($OSTYPE =~ /linux/i || $OSTYPE =~ /darwin/i || $OSTYPE eq 'win') {
<<<<<<< HEAD
+unless ($OSTYPE =~ /linux/i || $OSTYPE =~ /darwin/i || $OSTYPE eq 'win' || $OSTYPE eq 'FreeBSD') {
     println "configure: error: unsupported system '$OSTYPE'";
     exit 1;
 }
@@ -221,6 +221,10 @@ if ($OS eq 'linux') {
     println $OS_DISTRIBUTOR unless ($AUTORUN);
 }
 
+if ($MARCH eq 'amd64') {
+    $MARCH = 'x86_64';
+}
+
 print "checking machine architecture... " unless ($AUTORUN);
 println $MARCH unless ($AUTORUN);
 unless ($MARCH =~ /x86_64/i || $MARCH =~ /i?86/i) {
@@ -333,6 +337,16 @@ if ($OSTYPE =~ /linux/i) {
=======
+unless ($OSTYPE =~ /linux/i || $OSTYPE =~ /freebsd/i || $OSTYPE =~ /darwin/i || $OSTYPE eq 'win') {
     println "configure: error: unsupported system '$OSTYPE'";
     exit 1;
 }
@@ -225,7 +225,7 @@ if ($OS eq 'linux') {
 
 print "checking machine architecture... " unless ($AUTORUN);
 println $MARCH unless ($AUTORUN);
-unless ($MARCH =~ /x86_64/i || $MARCH =~ /i?86/i || $MARCH =~ /aarch64/) {
+unless ($MARCH =~ /x86_64/i || $MARCH =~ /amd64/i || $MARCH =~ /i?86/i || $MARCH =~ /aarch64/) {
     println "configure: error: unsupported architecture '$OSTYPE':'$MARCH'";
     exit 1;
 }
@@ -312,6 +312,8 @@ my $BITS;
 
 if ($MARCH =~ /x86_64/i) {
     $BITS = 64;
+} elsif ($MARCH =~ /amd64/i) {
+    $BITS = 64;
 } elsif ($MARCH eq 'fat86') {
     $BITS = '32_64';
 } elsif ($MARCH =~ /i?86/i) {
@@ -337,6 +339,25 @@ if ($OSTYPE =~ /linux/i) {
>>>>>>> upstream/main
     $OSINC = 'unix';
     $TOOLS = 'gcc' unless ($TOOLS);
     $PYTHON = 'python';
+} elsif ($OSTYPE =~ /freebsd/i) {
+    $LPFX = 'lib';
+    $OBJX = 'o';
+    $LOBX = 'pic.o';
+    $LIBX = 'a';
+    $SHLX = 'so';
+    $EXEX = '';
+    $OSINC = 'unix';
+    $TOOLS = 'clang' unless ($TOOLS);
<<<<<<< HEAD
+    $PYTHON = 'python';
 } elsif ($OSTYPE =~ /darwin/i) {
     $LPFX = 'lib';
     $OBJX = 'o';
@@ -379,11 +393,11 @@ if ($TOOLS =~ /gcc$/) {
 } elsif ($TOOLS eq 'clang') {
     $CPP  = 'clang++' unless ($CPP);
     $CC   = 'clang -c';
-    my $versionMin = '-mmacosx-version-min=10.10';
+    my $versionMin = '';
     $CP   = "$CPP -c $versionMin";
     if ($BITS ne '32_64') {
         $ARCH_FL = '-arch i386' if ($BITS == 32);
-        $OPT = '-O3';
+        $OPT = $ENV{'CXXFLAGS'};
         $AR      = 'ar rc';
         $LD      = "clang $ARCH_FL";
         $LP      = "$CPP $versionMin $ARCH_FL";
@@ -508,7 +522,7 @@ foreach my $href (DEPENDS()) {
             $I = $t if (-e $t);
         }
         push ( @L, File::Spec->catdir($OPT{$o}, 'lib') );
-        push ( @L, File::Spec->catdir($OPT{$o}, 'lib64') );
+        #push ( @L, File::Spec->catdir($OPT{$o}, 'lib64') );
     }
     my ($i, $l) = find_lib($_, $I, @L);
     if (defined $i || $l) {
@@ -929,8 +943,7 @@ EndText
     L($F, "PIC     = $PIC") if ($PIC);
     if ($PKG{LNG} eq 'C') {
         if ($TOOLS =~ /clang/i) {
-   L($F, 'SONAME  = -install_name ' .
-               '$(INST_LIBDIR)$(BITS)/$(subst $(VERSION),$(MAJVERS),$(@F)) \\');
+   L($F, 'SONAME = -Wl,-soname=$(subst $(VERSION),$(MAJVERS),$(@F)) \\');
    L($F, '    -compatibility_version $(MAJMIN) -current_version $(VERSION) \\');
    L($F, '    -flat_namespace -undefined suppress');
         } else {
=======
+    $PYTHON = $ENV{FREEBSD_PYTHON_CMD};
+} elsif ($OSTYPE =~ /darwin/i) {
+    $LPFX = 'lib';
+    $OBJX = 'o';
+    $LOBX = 'pic.o';
+    $LIBX = 'a';
+    $SHLX = 'dylib';
+    $EXEX = '';
+    $OSINC = 'unix';
+    $TOOLS = 'clang' unless ($TOOLS);
 } elsif ($OSTYPE =~ /darwin/i) {
     $LPFX = 'lib';
     $OBJX = 'o';
>>>>>>> upstream/main
