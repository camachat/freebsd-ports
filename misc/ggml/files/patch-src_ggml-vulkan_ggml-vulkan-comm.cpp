diff --git src/ggml-vulkan/ggml-vulkan-comm.cpp src/ggml-vulkan/ggml-vulkan-comm.cpp
index a2370b8c..f0f57638 100644
--- src/ggml-vulkan/ggml-vulkan-comm.cpp
+++ src/ggml-vulkan/ggml-vulkan-comm.cpp
@@ -367,7 +367,18 @@ static void ggml_vk_comm_sfd_exchange(ggml_backend_vk_comm_context * comm, uint6
             if (i == j) {
                 continue;
             }
-            const int imp_fd = fd < 0 || i == last ? fd : fcntl(fd, F_DUPFD_CLOEXEC, 0);
+            int imp_fd = fd;
+            if (fd >= 0 && i != last) {
+#ifdef F_DUPFD_CLOEXEC
+                imp_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
+#else
+                imp_fd = fcntl(fd, F_DUPFD, 0);
+                if (imp_fd >= 0 && fcntl(imp_fd, F_SETFD, FD_CLOEXEC) == -1) {
+                    close(imp_fd);
+                    imp_fd = -1;
+                }
+#endif
+            }
             GGML_ASSERT(imp_fd >= 0 || fd < 0);
             ggml_vk_comm_sfd_import(comm->devs[i], comm->devs[i].sfd.imp[j], imp_fd);
         }
