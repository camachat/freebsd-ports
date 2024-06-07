--- utils.mk.orig	2023-04-15 02:17:00 UTC
+++ utils.mk
@@ -137,6 +137,10 @@ ifeq ($(TARGET_ARCH),x86_64)
   CFLAGS += -DNV_X86_64 -DNV_ARCH_BITS=64
 endif
 
+ifeq ($(TARGET_ARCH),amd64)
+  CFLAGS += -DNV_X86_64 -DNV_ARCH_BITS=64
+endif
+
 ifeq ($(TARGET_ARCH),armv7l)
   CFLAGS += -DNV_ARMV7 -DNV_ARCH_BITS=32
 endif
@@ -540,7 +544,7 @@ define READ_ONLY_OBJECT_FROM_FILE_RULE
   $$(OUTPUTDIR)/$$(notdir $(1)).o: $(1)
 	$(at_if_quiet)$$(MKDIR) $$(OUTPUTDIR)
 	$(at_if_quiet)cd $$(dir $(1)); \
-	$$(call quiet_cmd_no_at,LD) -r -z noexecstack --format=binary \
+	$$(call quiet_cmd_no_at,LD) -m elf_$(if $(filter-out amd64,$(TARGET_ARCH)),$(TARGET_ARCH),x86_64) -r -z noexecstack --format=binary \
 	    $$(notdir $(1)) -o $$(OUTPUTDIR_ABSOLUTE)/$$(notdir $$@)
 	$$(call quiet_cmd,OBJCOPY) \
 	    --rename-section .data=.rodata,contents,alloc,load,data,readonly \
