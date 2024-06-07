--- src/libXNVCtrl/utils.mk.orig	2021-11-10 19:10:56 UTC
+++ src/libXNVCtrl/utils.mk
@@ -136,6 +136,9 @@ endif
 ifeq ($(TARGET_ARCH),x86_64)
   CFLAGS += -DNV_X86_64 -DNV_ARCH_BITS=64
 endif
+ifeq ($(TARGET_ARCH),amd64)
+  CFLAGS += -DNV_X86_64 -DNV_ARCH_BITS=64
+endif
 
 ifeq ($(TARGET_ARCH),armv7l)
   CFLAGS += -DNV_ARMV7 -DNV_ARCH_BITS=32
@@ -540,7 +543,7 @@ define READ_ONLY_OBJECT_FROM_FILE_RULE
   $$(OUTPUTDIR)/$$(notdir $(1)).o: $(1)
 	$(at_if_quiet)$$(MKDIR) $$(OUTPUTDIR)
 	$(at_if_quiet)cd $$(dir $(1)); \
-	$$(call quiet_cmd_no_at,LD) -r -z noexecstack --format=binary \
+	$$(call quiet_cmd_no_at,LD) -m elf_$(if $(filter-out amd64,$(TARGET_ARCH)),$(TARGET_ARCH),x86_64) -r -z noexecstack --format=binary \
 	    $$(notdir $(1)) -o $$(OUTPUTDIR_ABSOLUTE)/$$(notdir $$@)
 	$$(call quiet_cmd,OBJCOPY) \
 	    --rename-section .data=.rodata,contents,alloc,load,data,readonly \
