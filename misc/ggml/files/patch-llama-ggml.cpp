diff --git CMakeLists.txt CMakeLists.txt
index b7110fa1..9d807d5c 100644
--- CMakeLists.txt
+++ CMakeLists.txt
@@ -243,6 +243,7 @@ set   (GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                             "ggml: metal minimum macOS version")
 set   (GGML_METAL_STD "" CACHE STRING       "ggml: metal standard version (-std flag)")
 option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)
+option(GGML_OPENMP_FETCH                    "ggml: fetch LLVM OpenMP"                         OFF)
 option(GGML_RPC                             "ggml: use RPC"                                   OFF)
 option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
 option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
diff --git include/ggml-rpc.h include/ggml-rpc.h
index 276aea00..059e4496 100644
--- include/ggml-rpc.h
+++ include/ggml-rpc.h
@@ -7,7 +7,7 @@ extern "C" {
 #endif
 
 #define RPC_PROTO_MAJOR_VERSION    5
-#define RPC_PROTO_MINOR_VERSION    0
+#define RPC_PROTO_MINOR_VERSION    1
 #define RPC_PROTO_PATCH_VERSION    0
 
 #ifdef  __cplusplus
diff --git include/ggml.h include/ggml.h
index c2ccd972..32462d79 100644
--- include/ggml.h
+++ include/ggml.h
@@ -1981,6 +1981,14 @@ extern "C" {
             float                 beta_fast,
             float                 beta_slow);
 
+    // set the offset dims for RoPE
+    // a must be GGML_OP_ROPE or GGML_OP_ROPE_BACK
+    // vision RoPE is not supported
+    // example: (marking: x = rotated, 0 = unrotated)
+    //     n_embd = 10, n_dims = 4, offset = 2 --> [00xxxx0000]
+    GGML_API struct ggml_tensor * ggml_rope_set_offset(
+            struct ggml_tensor  * a,
+            int                   n_offs);
 
     // clamp
     // in-place, returns view(a)
diff --git src/CMakeLists.txt src/CMakeLists.txt
index 82e9480c..96535b49 100644
--- src/CMakeLists.txt
+++ src/CMakeLists.txt
@@ -222,9 +222,123 @@ if (GGML_SCHED_NO_REALLOC)
     target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
 endif()
 
-if (GGML_OPENMP)
+if (GGML_OPENMP_FETCH)
+    if (NOT GGML_OPENMP)
+        message(FATAL_ERROR "GGML_OPENMP_FETCH requires GGML_OPENMP")
+    elseif (NOT WIN32 OR NOT (CMAKE_C_COMPILER_ID MATCHES "Clang"))
+        message(FATAL_ERROR "GGML_OPENMP_FETCH currently requires Clang on Windows")
+    endif()
+
+    set(GGML_OPENMP_LLVM_VERSION "20.1.8")
+    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_LLVM_VERSION_MAJOR "${GGML_OPENMP_LLVM_VERSION}")
+    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_COMPILER_VERSION_MAJOR "${CMAKE_C_COMPILER_VERSION}")
+    if (NOT GGML_OPENMP_COMPILER_VERSION_MAJOR STREQUAL GGML_OPENMP_LLVM_VERSION_MAJOR)
+        message(FATAL_ERROR "LLVM OpenMP ${GGML_OPENMP_LLVM_VERSION} requires Clang ${GGML_OPENMP_LLVM_VERSION_MAJOR}.x")
+    endif()
+
+    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" GGML_OPENMP_SYSTEM_PROCESSOR)
+    if (GGML_OPENMP_SYSTEM_PROCESSOR MATCHES "^(amd64|x86_64)$")
+        set(GGML_OPENMP_ARCH "x64")
+        set(GGML_OPENMP_INSTALLER_SUFFIX "win64")
+        set(GGML_OPENMP_INSTALLER_SHA256 "3197846a2b19063687dd56e93e34cd941e3548d907f23a6131571321bdf9fe7b")
+    elseif (GGML_OPENMP_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
+        set(GGML_OPENMP_ARCH "arm64")
+        set(GGML_OPENMP_INSTALLER_SUFFIX "woa64")
+        set(GGML_OPENMP_INSTALLER_SHA256 "7c4ac97eb2ae6b960ca5f9caf3ff6124c8d2a18cc07a7840a4d2ea15537bad8e")
+    else()
+        message(FATAL_ERROR "GGML_OPENMP_FETCH does not support ${CMAKE_SYSTEM_PROCESSOR}")
+    endif()
+
+    set(GGML_OPENMP_CACHE_DIR "${CMAKE_BINARY_DIR}/_deps")
+    set(GGML_OPENMP_ROOT "${GGML_OPENMP_CACHE_DIR}/llvm-openmp-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_ARCH}")
+    set(GGML_OPENMP_LIBRARY "${GGML_OPENMP_ROOT}/lib/libomp.lib")
+    set(GGML_OPENMP_RUNTIME "${GGML_OPENMP_ROOT}/bin/libomp.dll")
+    set(GGML_OPENMP_HEADER "${GGML_OPENMP_ROOT}/include/omp.h")
+    set(GGML_OPENMP_LICENSE "${GGML_OPENMP_ROOT}/LICENSE.TXT")
+    set(GGML_OPENMP_LICENSE_SHA256 "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4")
+
+    if (NOT EXISTS "${GGML_OPENMP_LIBRARY}" OR NOT EXISTS "${GGML_OPENMP_RUNTIME}" OR NOT EXISTS "${GGML_OPENMP_HEADER}")
+        find_program(GGML_OPENMP_7Z NAMES 7z 7zz 7za)
+        if (NOT GGML_OPENMP_7Z)
+            message(FATAL_ERROR "GGML_OPENMP_FETCH requires 7-Zip to extract the LLVM installer")
+        endif()
+
+        set(GGML_OPENMP_INSTALLER "${GGML_OPENMP_ROOT}/LLVM-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_INSTALLER_SUFFIX}.exe")
+        set(GGML_OPENMP_EXTRACT_DIR "${GGML_OPENMP_ROOT}/extract")
+        set(GGML_OPENMP_INSTALLER_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${GGML_OPENMP_LLVM_VERSION}/LLVM-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_INSTALLER_SUFFIX}.exe")
+
+        file(MAKE_DIRECTORY "${GGML_OPENMP_EXTRACT_DIR}")
+        file(DOWNLOAD "${GGML_OPENMP_INSTALLER_URL}" "${GGML_OPENMP_INSTALLER}"
+            EXPECTED_HASH "SHA256=${GGML_OPENMP_INSTALLER_SHA256}"
+            SHOW_PROGRESS
+            STATUS GGML_OPENMP_DOWNLOAD_STATUS)
+        list(GET GGML_OPENMP_DOWNLOAD_STATUS 0 GGML_OPENMP_DOWNLOAD_RESULT)
+        if (NOT GGML_OPENMP_DOWNLOAD_RESULT EQUAL 0)
+            list(GET GGML_OPENMP_DOWNLOAD_STATUS 1 GGML_OPENMP_DOWNLOAD_ERROR)
+            message(FATAL_ERROR "Failed to download LLVM OpenMP: ${GGML_OPENMP_DOWNLOAD_ERROR}")
+        endif()
+
+        execute_process(
+            COMMAND "${GGML_OPENMP_7Z}" e -y "-o${GGML_OPENMP_EXTRACT_DIR}" "${GGML_OPENMP_INSTALLER}" -r libomp.lib libomp.dll omp.h
+            RESULT_VARIABLE GGML_OPENMP_EXTRACT_RESULT
+            OUTPUT_QUIET)
+        if (NOT GGML_OPENMP_EXTRACT_RESULT EQUAL 0 OR
+            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/libomp.lib" OR
+            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/libomp.dll" OR
+            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/omp.h")
+            message(FATAL_ERROR "Failed to extract libomp from ${GGML_OPENMP_INSTALLER}")
+        endif()
+
+        file(MAKE_DIRECTORY "${GGML_OPENMP_ROOT}/lib" "${GGML_OPENMP_ROOT}/bin" "${GGML_OPENMP_ROOT}/include")
+        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/libomp.lib" DESTINATION "${GGML_OPENMP_ROOT}/lib")
+        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/libomp.dll" DESTINATION "${GGML_OPENMP_ROOT}/bin")
+        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/omp.h" DESTINATION "${GGML_OPENMP_ROOT}/include")
+        file(REMOVE_RECURSE "${GGML_OPENMP_INSTALLER}" "${GGML_OPENMP_EXTRACT_DIR}")
+    endif()
+
+    # The NSIS installer embeds LLVM's general license in its UI but does not install it as a file; use OpenMP's license to include its additional notices.
+    if (EXISTS "${GGML_OPENMP_LICENSE}")
+        file(SHA256 "${GGML_OPENMP_LICENSE}" GGML_OPENMP_LICENSE_ACTUAL_SHA256)
+    endif()
+    if (NOT GGML_OPENMP_LICENSE_ACTUAL_SHA256 STREQUAL GGML_OPENMP_LICENSE_SHA256)
+        file(DOWNLOAD "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-${GGML_OPENMP_LLVM_VERSION}/openmp/LICENSE.TXT" "${GGML_OPENMP_LICENSE}"
+            EXPECTED_HASH "SHA256=${GGML_OPENMP_LICENSE_SHA256}")
+    endif()
+
+    if (COMMAND license_add_file)
+        license_add_file("LLVM OpenMP" "${GGML_OPENMP_LICENSE}")
+    endif()
+
+    add_library(ggml-openmp-c INTERFACE)
+    target_compile_options(ggml-openmp-c INTERFACE "$<$<COMPILE_LANGUAGE:C>:-fopenmp=libomp>")
+    target_include_directories(ggml-openmp-c SYSTEM INTERFACE "${GGML_OPENMP_ROOT}/include")
+    target_link_libraries(ggml-openmp-c INTERFACE "${GGML_OPENMP_LIBRARY}")
+
+    add_library(ggml-openmp-cxx INTERFACE)
+    target_compile_options(ggml-openmp-cxx INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-fopenmp=libomp>")
+    target_include_directories(ggml-openmp-cxx SYSTEM INTERFACE "${GGML_OPENMP_ROOT}/include")
+    target_link_libraries(ggml-openmp-cxx INTERFACE "${GGML_OPENMP_LIBRARY}")
+
+    set(GGML_OPENMP_RUNTIME_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
+    if (CMAKE_CONFIGURATION_TYPES)
+        string(APPEND GGML_OPENMP_RUNTIME_OUTPUT_DIR "/$<CONFIG>")
+    endif()
+    add_custom_target(ggml-openmp-runtime ALL
+        COMMAND ${CMAKE_COMMAND} -E make_directory "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}"
+        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_OPENMP_RUNTIME}" "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}/libomp.dll"
+        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_OPENMP_LICENSE}" "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}/LICENSE-LLVM-OpenMP")
+    add_dependencies(ggml-base ggml-openmp-runtime)
+    install(FILES "${GGML_OPENMP_RUNTIME}" DESTINATION ${CMAKE_INSTALL_BINDIR})
+    install(FILES "${GGML_OPENMP_LICENSE}" DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME LICENSE-LLVM-OpenMP)
+
+    set(GGML_OPENMP_TARGET_C ggml-openmp-c)
+    set(GGML_OPENMP_TARGET_CXX ggml-openmp-cxx)
+    set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
+elseif (GGML_OPENMP)
     find_package(OpenMP)
     if (OpenMP_FOUND)
+        set(GGML_OPENMP_TARGET_C OpenMP::OpenMP_C)
+        set(GGML_OPENMP_TARGET_CXX OpenMP::OpenMP_CXX)
         set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
     else()
         set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
@@ -236,7 +350,7 @@ endif()
 
 if (GGML_OPENMP_ENABLED)
     target_compile_definitions(ggml-base PRIVATE GGML_USE_OPENMP)
-    target_link_libraries(ggml-base PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
+    target_link_libraries(ggml-base PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
 endif()
 
 add_library(ggml
diff --git src/ggml-backend.cpp src/ggml-backend.cpp
index f6fb9179..3d6310f3 100644
--- src/ggml-backend.cpp
+++ src/ggml-backend.cpp
@@ -1599,11 +1599,23 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
     std::vector<int32_t> ids;
     std::vector<ggml_bitset_t> used_ids;
 
+    int prev_backend_id = -1;
+
     for (int split_id = 0; split_id < sched->n_splits; split_id++) {
         struct ggml_backend_sched_split * split = &splits[split_id];
         int split_backend_id = split->backend_id;
         ggml_backend_t split_backend = sched->backends[split_backend_id];
 
+        // ensure the previous split's async work has completed before we start
+        // this split, the allocator may have reused buffer regions across splits
+        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
+            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
+                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
+            } else {
+                ggml_backend_synchronize(sched->backends[prev_backend_id]);
+            }
+        }
+
         // copy the input tensors to the split backend
         for (int input_id = 0; input_id < split->n_inputs; input_id++) {
             ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
@@ -1766,12 +1778,12 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
             }
         }
 
-        // record the event of this copy
-        if (split->n_inputs > 0) {
-            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
-                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
-            }
+        // record the event of this split
+        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
+            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
         }
+
+        prev_backend_id = split_backend_id;
     }
 
     return GGML_STATUS_SUCCESS;
diff --git src/ggml-cann/ggml-cann.cpp src/ggml-cann/ggml-cann.cpp
index ffa361af..5e5541aa 100644
--- src/ggml-cann/ggml-cann.cpp
+++ src/ggml-cann/ggml-cann.cpp
@@ -2534,6 +2534,9 @@ static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, const ggml_ten
             }
         case GGML_OP_ROPE:
             {
+                if (((const int32_t *) op->op_params)[15] != 0) {
+                    return false; // FIXME: support ggml_rope_set_offset
+                }
                 if (op->src[0]->ne[0] > 896) {
                     return false;
                 }
diff --git src/ggml-cpu/CMakeLists.txt src/ggml-cpu/CMakeLists.txt
index 836bae4d..a6cc4958 100644
--- src/ggml-cpu/CMakeLists.txt
+++ src/ggml-cpu/CMakeLists.txt
@@ -74,7 +74,7 @@ function(ggml_add_cpu_backend_variant_impl tag_name)
 
     if (GGML_OPENMP_ENABLED)
         target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
-        target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
+        target_link_libraries(${GGML_CPU_NAME} PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
     endif()
 
     if (GGML_LLAMAFILE)
diff --git src/ggml-cpu/ops.cpp src/ggml-cpu/ops.cpp
index 001e1ae8..2b5f6844 100644
--- src/ggml-cpu/ops.cpp
+++ src/ggml-cpu/ops.cpp
@@ -5979,6 +5979,8 @@ static void ggml_compute_forward_rope_flt(
     memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
     memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
 
+    const int n_offs = ((int32_t *) dst->op_params)[15];
+
     GGML_TENSOR_UNARY_OP_LOCALS
 
     //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
@@ -5995,6 +5997,10 @@ static void ggml_compute_forward_rope_flt(
     GGML_ASSERT(n_dims <= ne0);
     GGML_ASSERT(n_dims % 2 == 0);
 
+    GGML_ASSERT(n_offs >= 0);
+    GGML_ASSERT(n_offs % 2 == 0);
+    GGML_ASSERT(n_offs + n_dims <= ne0);
+
     // rows per thread
     const int dr = (nr + nth - 1)/nth;
 
@@ -6020,6 +6026,7 @@ static void ggml_compute_forward_rope_flt(
 
     if (is_vision) {
         GGML_ASSERT(n_dims == ne0/2);
+        GGML_ASSERT(n_offs == 0);
     }
 
     const float * freq_factors = NULL;
@@ -6068,12 +6075,12 @@ static void ggml_compute_forward_rope_flt(
 
                 switch (mode) {
                     case GGML_ROPE_TYPE_NORMAL:
-                        rotate_pairs<T>(n_dims, 1, cache, src, dst_data, 1);
+                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                         break;
                     case GGML_ROPE_TYPE_NEOX:
                     case GGML_ROPE_TYPE_MROPE:
                     case GGML_ROPE_TYPE_IMROPE:
-                        rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst_data);
+                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                         break;
                     case GGML_ROPE_TYPE_VISION:
                         rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
@@ -6084,7 +6091,11 @@ static void ggml_compute_forward_rope_flt(
 
                 if (!is_vision) {
                     // fill the remain channels with data from src tensor
-                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
+                    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
+                        if (i0 == n_offs) {
+                            i0 += n_dims - 2; // skip the rotated channels
+                            continue;
+                        }
                         const T * const src = (T *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                         T * dst_data  = (T *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
 
diff --git src/ggml-cpu/simd-mappings.h src/ggml-cpu/simd-mappings.h
index fca5119e..10ce4bfc 100644
--- src/ggml-cpu/simd-mappings.h
+++ src/ggml-cpu/simd-mappings.h
@@ -29,13 +29,15 @@ extern "C" {
 // FP16 to FP32 conversion
 
 // 16-bit float
-// on Arm, we use __fp16
+// on Arm, we use __fp16, which requires the IEEE fp16 format: implied on
+// AArch64, selected by -mfp16-format=ieee on 32 bit Arm, where the compiler
+// may otherwise reject the type
 // on x86, we use uint16_t
 //
 // for old CUDA compilers (<= 11), we use uint16_t: ref https://github.com/ggml-org/llama.cpp/pull/10616
 // for     MUSA compilers        , we use uint16_t: ref https://github.com/ggml-org/llama.cpp/pull/11843
 //
-#if defined(__ARM_NEON) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
+#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
     #define GGML_CPU_COMPUTE_FP16_TO_FP32(x) neon_compute_fp16_to_fp32(x)
     #define GGML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)
 
@@ -326,7 +328,7 @@ inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
     #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
 #endif
 
-#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
+#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)
 
 #define GGML_SIMD
 
diff --git src/ggml-cuda/common.cuh src/ggml-cuda/common.cuh
index d27d8acb..14dd1098 100644
--- src/ggml-cuda/common.cuh
+++ src/ggml-cuda/common.cuh
@@ -1418,7 +1418,9 @@ struct ggml_backend_cuda_context {
     cudaEvent_t copy_event = nullptr;
 
     cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
-    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};
+    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
+    void * cublas_workspaces[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
+    size_t cublas_workspace_sizes[GGML_CUDA_MAX_DEVICES] = {0};
 
     int curr_stream_no = 0;
 
@@ -1495,17 +1497,22 @@ struct ggml_backend_cuda_context {
 
     ggml_cuda_stream_context & stream_context() { return concurrent_stream_context; }
 
-    cublasHandle_t cublas_handle(int device) {
-        if (cublas_handles[device] == nullptr) {
+    cublasHandle_t cublas_handle() {
+        if (cublas_handles[device][curr_stream_no] == nullptr) {
             ggml_cuda_set_device(device);
-            CUBLAS_CHECK(cublasCreate(&cublas_handles[device]));
-            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH));
+            CUBLAS_CHECK(cublasCreate(&cublas_handles[device][curr_stream_no]));
+            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device][curr_stream_no], CUBLAS_TF32_TENSOR_OP_MATH));
+            CUBLAS_CHECK(cublasSetStream(cublas_handles[device][curr_stream_no], stream()));
+#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUBLAS_VER_MAJOR > 11 || (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR >= 2))
+            if (cublas_workspace_sizes[device] == 0) {
+                const int cc = ggml_cuda_info().devices[device].cc;
+                cublas_workspace_sizes[device] = (cc >= GGML_CUDA_CC_HOPPER) ? 32 * 1024 * 1024 : 4 * 1024 * 1024;
+            }
+            CUDA_CHECK(cudaMalloc(&cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
+            CUBLAS_CHECK(cublasSetWorkspace(cublas_handles[device][curr_stream_no], cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
+#endif
         }
-        return cublas_handles[device];
-    }
-
-    cublasHandle_t cublas_handle() {
-        return cublas_handle(device);
+        return cublas_handles[device][curr_stream_no];
     }
 
     // pool
diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index f2e381ee..a8a1c09c 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -711,9 +711,12 @@ ggml_backend_cuda_context::~ggml_backend_cuda_context() {
             if (streams[i][j] != nullptr) {
                 CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
             }
-        }
-        if (cublas_handles[i] != nullptr) {
-            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
+            if (cublas_handles[i][j] != nullptr) {
+                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
+            }
+            if (cublas_workspaces[i][j] != nullptr) {
+                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
+            }
         }
     }
 }
@@ -1416,7 +1419,7 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
 
     const int64_t ne_dst = ggml_nelements(dst);
     cudaStream_t main_stream = ctx.stream();
-    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));
+    cublasHandle_t cublas_h = ctx.cublas_handle();
 
     const size_t src0_ts = ggml_type_size(src0->type);
     GGML_ASSERT(nb00 == src0_ts);
@@ -1539,14 +1542,14 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
     //     probably because the internal kernel selection logic is suboptimal.
     if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
         CUBLAS_CHECK(
-            cublasSgemm(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
+            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     ne01, ne11, ne10,
                     (const float *) alpha, (const float *) src0_ptr, s01,
                                            (const float *) src1_ptr, s11,
                     (const float *) beta,  (float       *)  dst_ptr, ne0));
     } else if (ne12 == 1 && ne13 == 1) {
         CUBLAS_CHECK(
-            cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
+            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     ne01, ne11, ne10,
                     alpha, src0_ptr, cu_data_type_a, s01,
                            src1_ptr, cu_data_type_b, s11,
@@ -1561,7 +1564,7 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
         // there is no broadcast and src0, src1 are contiguous across dims 2, 3
         // use cublasGemmStridedBatchedEx
         CUBLAS_CHECK(
-        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
+        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 ne01, ne11, ne10,
                 alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                        src1_ptr, cu_data_type_b, s11, smb,     // strideB
@@ -1599,7 +1602,7 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
         CUDA_CHECK(cudaGetLastError());
 
         CUBLAS_CHECK(
-        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
+        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 ne01, ne11, ne10,
                 alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                        (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
@@ -2723,6 +2726,12 @@ static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm
         return false;
     }
 
+    // ggml_rope_set_offset is not yet supported in the fused kernel
+    const int n_offs = ((const int32_t *) rope->op_params)[15];
+    if (n_offs != 0) {
+        return false;
+    }
+
     return true;
 }
 
diff --git src/ggml-cuda/mmvq.cu src/ggml-cuda/mmvq.cu
index c9992380..97053480 100644
--- src/ggml-cuda/mmvq.cu
+++ src/ggml-cuda/mmvq.cu
@@ -290,6 +290,42 @@ bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
     if (!ggml_is_quantized(type)) {
         return false;
     }
+    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
+    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
+    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
+        switch (type) { // tuned on RTX 4090
+            case GGML_TYPE_Q2_K:
+                return ne11 <= 4;
+            case GGML_TYPE_Q3_K:
+                return ne11 <= 6;
+            case GGML_TYPE_Q4_K:
+            case GGML_TYPE_Q5_K:
+                return ne11 <= 7;
+            default:
+                return ne11 <= MMVQ_MAX_BATCH_SIZE;
+        }
+    }
+    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
+        switch (type) { // tuned on RTX 5090
+            case GGML_TYPE_Q2_K:
+            case GGML_TYPE_Q3_K:
+            case GGML_TYPE_Q4_K:
+            case GGML_TYPE_Q5_K:
+                return ne11 <= 5;
+            case GGML_TYPE_Q6_K:
+                return ne11 <= 7;
+            default:
+                return ne11 <= MMVQ_MAX_BATCH_SIZE;
+        }
+    }
+    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
+        switch (type) { // tuned on DGX Spark GB10
+            case GGML_TYPE_Q2_K:
+                return ne11 <= 6;
+            default:
+                return ne11 <= MMVQ_MAX_BATCH_SIZE;
+        }
+    }
     if (GGML_CUDA_CC_IS_CDNA(cc)) {
         if (GGML_CUDA_CC_IS_CDNA1(cc)) {
             switch (type) {
diff --git src/ggml-cuda/out-prod.cu src/ggml-cuda/out-prod.cu
index 46b9f3a6..c46e0455 100644
--- src/ggml-cuda/out-prod.cu
+++ src/ggml-cuda/out-prod.cu
@@ -54,8 +54,6 @@ void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const float alpha = 1.0f;
     const float beta = 0.0f;
 
-    CUBLAS_CHECK(cublasSetStream(handle, stream));
-
     const int64_t lda = nb01 / sizeof(float);
     const int64_t ldc = nb1  / sizeof(float);
 
diff --git src/ggml-cuda/rope.cu src/ggml-cuda/rope.cu
index 504c6b81..e546fb65 100644
--- src/ggml-cuda/rope.cu
+++ src/ggml-cuda/rope.cu
@@ -53,6 +53,7 @@ static __global__ void rope_norm(const T *            x,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
+                                 const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
@@ -61,7 +62,8 @@ static __global__ void rope_norm(const T *            x,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const int64_t *      row_indices,
-                                 const int            set_rows_stride) {
+                                 const int            set_rows_stride,
+                                 const bool           inplace) {
     const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);
 
     if (i0 >= ne00) {
@@ -92,19 +94,24 @@ static __global__ void rope_norm(const T *            x,
             ggml_cuda_memcpy_1<4>(dst + idst, &v);
         }
     };
-    if (i0 >= n_dims) {
+    if (i0 < n_offs || i0 >= n_offs + n_dims) {
+        if (inplace) {
+            return;
+        }
         store_coaelsced(x[ix + 0], x[ix + 1]);
         return;
     }
 
-    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);
+    const int iw = i0 - n_offs; // relative idx
 
-    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;
+    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);
+
+    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;
 
     float cos_theta;
     float sin_theta;
 
-    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
+    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);
 
     const float x0 = x[ix + 0];
     const float x1 = x[ix + 1];
@@ -125,6 +132,7 @@ static __global__ void rope_neox(const T *            x,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
+                                 const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
@@ -133,7 +141,8 @@ static __global__ void rope_neox(const T *            x,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const int64_t *      row_indices,
-                                 const int            set_rows_stride) {
+                                 const int            set_rows_stride,
+                                 const bool           inplace) {
     ggml_cuda_pdl_lc();
     const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);
 
@@ -158,27 +167,33 @@ static __global__ void rope_neox(const T *            x,
         idst += row_indices[i2] * set_rows_stride;
     }
 
-    if (i0 >= n_dims) {
+    if (i0 < n_offs || i0 >= n_offs + n_dims) {
+        if (inplace) {
+            return;
+        }
         dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
         dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);
 
         return;
     }
 
-    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);
+    const int iw = i0 - n_offs; // relative idx
 
-    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;
+    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);
+
+    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;
 
     float cos_theta;
     float sin_theta;
 
-    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
+    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);
 
-    const float x0 = x[ix + 0];
-    const float x1 = x[ix + n_dims/2];
+    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
+    const float x0 = x[ix + n_offs/2 + 0];
+    const float x1 = x[ix + n_offs/2 + n_dims/2];
 
-    dst[idst + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
-    dst[idst + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
+    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
+    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
 }
 
 template <bool forward, bool has_ff, typename T>
@@ -194,6 +209,7 @@ static __global__ void rope_multi(const T *            x,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
+                                  const int            n_offs,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
@@ -202,7 +218,8 @@ static __global__ void rope_multi(const T *            x,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections,
-                                  const bool           is_imrope) {
+                                  const bool           is_imrope,
+                                  const bool           inplace) {
     const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);
 
     if (i0 >= ne00) {
@@ -219,52 +236,58 @@ static __global__ void rope_multi(const T *            x,
     const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
 
     ggml_cuda_pdl_sync();
-    if (i0 >= n_dims) {
+    if (i0 < n_offs || i0 >= n_offs + n_dims) {
+        if (inplace) {
+            return;
+        }
         dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
         dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];
 
         return;
     }
 
+    const int iw = i0 - n_offs; // relative idx
+
     const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
     const int sec_w = sections.v[1] + sections.v[0];
-    const int sector = (i0 / 2) % sect_dims;
+    const int sector = (iw / 2) % sect_dims;
 
     float theta_base = 0.0;
     if (is_imrope) {
         if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
-            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
         } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
-            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
         } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
-            theta_base = pos[i2] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
         } else {
-            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
         }
     } else {
         if (sector < sections.v[0]) {
-            theta_base = pos[i2] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
         } else if (sector >= sections.v[0] && sector < sec_w) {
-            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
         } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
-            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
         } else if (sector >= sec_w + sections.v[2]) {
-            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, i0 / 2.0f);
+            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
         }
     }
 
-    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;
+    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;
 
     float cos_theta;
     float sin_theta;
 
-    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
+    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);
 
-    const float x0 = x[ix + 0];
-    const float x1 = x[ix + n_dims/2];
+    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
+    const float x0 = x[ix + n_offs/2 + 0];
+    const float x1 = x[ix + n_offs/2 + n_dims/2];
 
-    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
-    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
+    dst[idst + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
+    dst[idst + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
 }
 
 template <bool forward, bool has_ff, typename T>
@@ -344,6 +367,7 @@ static void rope_norm_cuda(const T *            x,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
+                           const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
@@ -354,6 +378,7 @@ static void rope_norm_cuda(const T *            x,
                            const float *        freq_factors,
                            const int64_t *      row_indices,
                            const int            set_rows_stride,
+                           const bool           inplace,
                            cudaStream_t         stream) {
     GGML_ASSERT(ne00 % 2 == 0);
     const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
@@ -364,12 +389,12 @@ static void rope_norm_cuda(const T *            x,
 
     if (freq_factors == nullptr) {
         rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
-            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
-            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
+            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
+            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
     } else {
         rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
-            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
-            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
+            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
+            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
     }
 }
 
@@ -386,6 +411,7 @@ static void rope_neox_cuda(const T *            x,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
+                           const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
@@ -396,6 +422,7 @@ static void rope_neox_cuda(const T *            x,
                            const float *        freq_factors,
                            const int64_t *      row_indices,
                            const int            set_rows_stride,
+                           const bool           inplace,
                            cudaStream_t         stream) {
     GGML_ASSERT(ne00 % 2 == 0);
     const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
@@ -407,12 +434,12 @@ static void rope_neox_cuda(const T *            x,
 
     if (freq_factors == nullptr) {
         ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
-            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
-            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
+            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
+            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
     } else {
         ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
-            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
-            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
+            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
+            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
     }
 }
 
@@ -429,6 +456,7 @@ static void rope_multi_cuda(const T *            x,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
+                            const int            n_offs,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
@@ -439,6 +467,7 @@ static void rope_multi_cuda(const T *            x,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             const bool           is_imrope,
+                            const bool           inplace,
                             cudaStream_t         stream) {
     GGML_ASSERT(ne00 % 2 == 0);
     const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
@@ -450,13 +479,13 @@ static void rope_multi_cuda(const T *            x,
     if (freq_factors == nullptr) {
         const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
         ggml_cuda_kernel_launch(rope_multi<forward, false, T>, launch_params,
-            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
-            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
+            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
+            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
     } else {
         const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
         ggml_cuda_kernel_launch(rope_multi<forward, true, T>, launch_params,
-            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
-            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
+            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
+            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
     }
 }
 
@@ -552,8 +581,12 @@ void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
     const int mode       = ((int32_t *) dst->op_params)[2];
     //const int n_ctx      = ((int32_t *) dst->op_params)[3];
     const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
+    const int n_offs     = ((int32_t *) dst->op_params)[15];
     mrope_sections sections;
 
+    // when dst aliases src0, the channels outside the rotated window already hold the correct data
+    const bool inplace = dst_d == src0->data;
+
     // RoPE alteration for extended context
     float freq_base;
     float freq_scale;
@@ -581,6 +614,7 @@ void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
 
     if (is_vision) {
         GGML_ASSERT(n_dims == ne00/2);
+        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
     }
 
     const int32_t * pos = (const int32_t *) src1_d;
@@ -597,31 +631,31 @@ void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
     if (is_neox) {
         if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
             rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
-                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
+                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                   ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
-                                                  set_rows_stride, stream);
+                                                  set_rows_stride, inplace, stream);
         } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
             rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
-                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
+                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
-                                                 set_rows_stride, stream);
+                                                 set_rows_stride, inplace, stream);
         } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
             rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
-                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
+                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
-                                                set_rows_stride, stream);
+                                                set_rows_stride, inplace, stream);
         } else {
             GGML_ABORT("fatal error");
         }
     } else if (is_mrope && !is_vision) {
         if (src0->type == GGML_TYPE_F32) {
             rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
-                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
-                                     corr_dims, freq_factors, sections, is_imrope, stream);
+                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
+                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
         } else if (src0->type == GGML_TYPE_F16) {
             rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
-                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
-                                     corr_dims, freq_factors, sections, is_imrope, stream);
+                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
+                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
         } else {
             GGML_ABORT("fatal error");
         }
@@ -640,19 +674,19 @@ void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
     } else {
         if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
             rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
-                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
+                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                   ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
-                                                  set_rows_stride, stream);
+                                                  set_rows_stride, inplace, stream);
         } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
             rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
-                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
+                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
-                                                 set_rows_stride, stream);
+                                                 set_rows_stride, inplace, stream);
         } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
             rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
-                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
+                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
-                                                set_rows_stride, stream);
+                                                set_rows_stride, inplace, stream);
         } else {
             GGML_ABORT("fatal error");
         }
diff --git src/ggml-cuda/solve_tri.cu src/ggml-cuda/solve_tri.cu
index 07ca33f5..d9678342 100644
--- src/ggml-cuda/solve_tri.cu
+++ src/ggml-cuda/solve_tri.cu
@@ -65,15 +65,13 @@ static void solve_tri_f32_cublas(ggml_backend_cuda_context & ctx,
     get_batch_pointers<<<(total_batches + 255) / 256, 256, 0, stream>>>(A, X, A_ptrs_dev, X_ptrs_dev, ne02,
                                                                         total_batches, s02, s03, s2, s3);
 
-    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
-
     // Yes, this is necessary, without this we get RMSE errors
-    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_DEFAULT_MATH));
-    CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(id), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
+    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_DEFAULT_MATH));
+    CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, k, n, &alpha, A_ptrs_dev, n, X_ptrs_dev, k, total_batches));
 
     // revert to standard mode from common.cuh
-    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_TF32_TENSOR_OP_MATH));
+    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(), CUBLAS_TF32_TENSOR_OP_MATH));
 
     GGML_UNUSED_VARS(s12, s13);
 }
diff --git src/ggml-cuda/ssm-scan.cu src/ggml-cuda/ssm-scan.cu
index ef342f01..40cb38de 100644
--- src/ggml-cuda/ssm-scan.cu
+++ src/ggml-cuda/ssm-scan.cu
@@ -632,7 +632,6 @@ static void ssm_scan_ssd_f32_cuda(
     // Step 3: chunked SSD loop
     // Per chunk: pre_matmul (incl. M) + 4 cuBLAS (CB, Y, S@C, state update) + scale_state
     cublasHandle_t handle = ctx.cublas_handle();
-    CUBLAS_CHECK(cublasSetStream(handle, stream));
     const float alpha_one  = 1.0f;
     const float beta_zero  = 0.0f;
     const float beta_one   = 1.0f;
diff --git src/ggml-et/ggml-et.cpp src/ggml-et/ggml-et.cpp
index e8482f73..b87b189a 100644
--- src/ggml-et/ggml-et.cpp
+++ src/ggml-et/ggml-et.cpp
@@ -1061,9 +1061,11 @@ static bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, const ggm
                 const bool zero_view_offset = op->src[0]->view_src == nullptr || op->src[0]->view_offs == 0;
                 const bool has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                           ggml_get_op_params_i32(op, 13) > 0;
+                // FIXME: support ggml_rope_set_offset
+                const bool zero_rot_offset  = ggml_get_op_params_i32(op, 15) == 0;
 
                 supported =
-                    zero_view_offset && ndims <= 512 &&
+                    zero_view_offset && zero_rot_offset && ndims <= 512 &&
                     (is_normal || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
             } else {
                 supported = false;
diff --git src/ggml-hexagon/ggml-hexagon.cpp src/ggml-hexagon/ggml-hexagon.cpp
index f80c60a5..b262a73d 100644
--- src/ggml-hexagon/ggml-hexagon.cpp
+++ src/ggml-hexagon/ggml-hexagon.cpp
@@ -3180,6 +3180,10 @@ static bool ggml_hexagon_supported_argsort(const struct ggml_hexagon_session * s
 static bool ggml_hexagon_supported_rope(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
     const int32_t * op_params = &op->op_params[0];
 
+    if (op_params[15] != 0) {
+        return false; // FIXME: support ggml_rope_set_offset
+    }
+
     int mode = op_params[2];
 
     // n_dims == ne0/2, so the rotation spans the full row
diff --git src/ggml-hexagon/htp/flash-attn-ops.c src/ggml-hexagon/htp/flash-attn-ops.c
index fe78718c..81765629 100644
--- src/ggml-hexagon/htp/flash-attn-ops.c
+++ src/ggml-hexagon/htp/flash-attn-ops.c
@@ -132,8 +132,8 @@ struct hmx_fa_context {
     __fp16 *     vtcm_v_tiles[2];      // V tiles (column-major, double-buffered)
     __fp16 *     vtcm_s_tiles[2];      // S = QK^T [g_br, Bc] (double-buffered)
     __fp16 *     vtcm_p_tiles[2];      // P = softmax(S) [g_br, Bc]
-    __fp16 *     vtcm_d_tiles;         // Diagonal rescale [g_br, g_br]
-    __fp16 *     vtcm_d_inv_l;         // Diagonal rescale (1/l) [g_br, g_br]
+    __fp16 *     vtcm_d_tiles[2];      // Diagonal rescale, g_br/32 packed diagonal tiles (double-buffered)
+    __fp16 *     vtcm_d_inv_l;         // Diagonal rescale (1/l), same packed layout
     HVX_Vector * vtcm_m_vec;           // Row max [g_br]
     HVX_Vector * vtcm_l_vec;           // Row sum [g_br]
     HVX_Vector * vtcm_s_rowmax;        // Softmax intermediate [g_br]
@@ -782,13 +782,14 @@ static void fa_q_load_thread(unsigned int n, unsigned int i, void * data) {
             }
         }
 
-        // Initialize vtcm_d_tiles and vtcm_d_inv_l to 0
+        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
+        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
+        // them all.  The scatter only ever writes the diagonal, ignore the rest.
         const size_t d_bytes_per_t = hex_align_up(d_tile_bytes / n, 128);
         const size_t d_start       = i * d_bytes_per_t;
         const size_t d_end         = hex_smin(d_start + d_bytes_per_t, d_tile_bytes);
         if (d_start < d_tile_bytes) {
-            hvx_splat_u8_a((char *) factx->vtcm_d_tiles + d_start, 0, d_end - d_start);
-            hvx_splat_u8_a((char *) factx->vtcm_d_inv_l + d_start, 0, d_end - d_start);
+            hvx_splat_u8_a((char *) factx->vtcm_d_tiles[0] + d_start, 0, d_end - d_start);
         }
     }
 
@@ -1432,17 +1433,19 @@ static inline void fa_softmax_impl(
         const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
         HVX_Vector           v_exp_m_diff = exp_m_diff_f16;
 
+        __fp16 * const d_tiles_out = factx->vtcm_d_tiles[args->buf_idx];
+
         size_t t0 = r_vec_idx * 2;
         if (t0 < args->n_row_tiles) {
             const HVX_Vector v_content = v_exp_m_diff;
-            __fp16 *         out_base  = factx->vtcm_d_tiles + t0 * (args->n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
+            __fp16 *         out_base  = d_tiles_out + t0 * HMX_FP16_TILE_N_ELMS;
             Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
         }
 
         size_t t1 = r_vec_idx * 2 + 1;
         if (t1 < args->n_row_tiles) {
             const HVX_Vector v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
-            __fp16 *         out_base  = factx->vtcm_d_tiles + t1 * (args->n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
+            __fp16 *         out_base  = d_tiles_out + t1 * HMX_FP16_TILE_N_ELMS;
             Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
         }
     }
@@ -1506,7 +1509,7 @@ static __attribute__((noinline)) void fa_build_d_diag_inv_l(struct hmx_fa_contex
             v_content = Q6_V_vror_VR(v_content, 64);
         }
 
-        __fp16 * out_base = factx->vtcm_d_inv_l + i * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
+        __fp16 * out_base = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
         Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
     }
 }
@@ -1615,7 +1618,7 @@ static void hmx_fa_o_update_worker(void * data) {
     const size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
     const size_t v_stride = n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;
     for (size_t r = 0; r < n_row_tiles; ++r) {
-        const __fp16 * d_diag     = d_tiles + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
+        const __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
         const __fp16 * p_tile_in  = p_tiles + (r * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
         const __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
         const __fp16 * v_tile_in  = v_tiles;
@@ -1654,7 +1657,7 @@ static void hmx_fa_o_norm_worker(void * data) {
     asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
     const size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
     for (size_t r = 0; r < n_row_tiles; ++r) {
-        const __fp16 * d_diag = d_tiles + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
+        const __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
         const __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
         __fp16 *       o_out = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;
 
@@ -1882,7 +1885,8 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
     factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_s_tiles[1], pipeline);
     factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_p_tiles[0]);
     factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_p_tiles[1], pipeline);
-    factx.vtcm_d_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_tiles);
+    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_tiles[0]);
+    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_d_tiles[1], pipeline);
     factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_inv_l);
     factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_m_vec);
     factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_l_vec);
@@ -2039,7 +2043,30 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
                             }
                         }
 
-                        // ---- 3. Pop and run K-prep for next block & push next QK-dot ----
+                        // ---- 3. Start HMX O update for block kv_blk - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
+                        // O update relys on the previous block's P and V tiles.
+                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
+                        // oldest descriptor, so push order alone decides which pop waits for which job.
+                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
+                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
+                        if (kv_blk > 0) {
+                            const size_t prev_buf        = 1 - buf_idx;
+                            ou_job[prev_buf].o_curr      = o_tile_curr;
+                            ou_job[prev_buf].o_prev      = o_tile_prev;
+                            ou_job[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
+                            ou_job[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
+                            ou_job[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
+                            ou_job[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
+                            ou_job[prev_buf].n_row_tiles = n_row_tiles;
+                            ou_job[prev_buf].n_col_tiles =
+                                hmx_ceil_div(hex_smin(Bc, nek1 - (kv_blk - 1) * Bc), HMX_FP16_TILE_N_COLS);
+                            ou_job[prev_buf].n_row_tiles_g_br = n_row_tiles_g_br;
+                            ou_job[prev_buf].n_tiles_per_bc   = n_tiles_per_bc;
+                            ou_job[prev_buf].DV               = DV;
+                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[prev_buf]));
+                        }
+
+                        // ---- 4. Pop and run K-prep for next block & push next QK-dot ----
                         if (kv_blk + 1 < factx.n_kv_blocks) {
                             const uint32_t next_start = (kv_blk + 1) * Bc;
                             const uint32_t next_rows  = hex_smin(Bc, nek1 - next_start);
@@ -2059,10 +2086,10 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
                             hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[next_buf]));
                         }
 
-                        // ---- 4. Wait for current block's QK-dot to finish ----
+                        // ---- 5. Wait for current block's QK-dot to finish ----
                         hmx_queue_pop(hmx_q);
 
-                        // ---- 5. Phase 2: softmax + build_D ----
+                        // ---- 6. Phase 2: softmax + build_D ----
                         fa_softmax_args_t sargs;
                         memset(&sargs, 0, sizeof(sargs));
                         sargs.factx                = &factx;
@@ -2085,23 +2112,6 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
                         sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                         sargs.slopes               = factx.vtcm_slopes;
 
-                        // Start HMX O update for block kv_blk - 1 (reads P[1 - buf_idx], V[1 - buf_idx])
-                        if (kv_blk > 0) {
-                            const size_t prev_buf = 1 - buf_idx;
-                            ou_job[prev_buf].o_curr           = o_tile_curr;
-                            ou_job[prev_buf].o_prev           = o_tile_prev;
-                            ou_job[prev_buf].p_tiles          = factx.vtcm_p_tiles[prev_buf];
-                            ou_job[prev_buf].v_tiles          = factx.vtcm_v_tiles[prev_buf];
-                            ou_job[prev_buf].d_tiles          = factx.vtcm_d_tiles;
-                            ou_job[prev_buf].hmx_scales       = factx.vtcm_hmx_scales_id;
-                            ou_job[prev_buf].n_row_tiles      = n_row_tiles;
-                            ou_job[prev_buf].n_col_tiles      = hmx_ceil_div(hex_smin(Bc, nek1 - (kv_blk - 1) * Bc), HMX_FP16_TILE_N_COLS);
-                            ou_job[prev_buf].n_row_tiles_g_br = n_row_tiles_g_br;
-                            ou_job[prev_buf].n_tiles_per_bc   = n_tiles_per_bc;
-                            ou_job[prev_buf].DV               = DV;
-                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[prev_buf]));
-                        }
-
                         // Run Softmax on HVX (blocking call)
                         fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);
 
@@ -2128,7 +2138,7 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
                         ou_job[0].o_prev           = o_tile_prev;
                         ou_job[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                         ou_job[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
-                        ou_job[0].d_tiles          = factx.vtcm_d_tiles;
+                        ou_job[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                         ou_job[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                         ou_job[0].n_row_tiles      = n_row_tiles;
                         ou_job[0].n_col_tiles      = last_cols;
@@ -2232,7 +2242,7 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
                             ou_job.o_prev           = o_tile_prev;
                             ou_job.p_tiles          = factx.vtcm_p_tiles[0];
                             ou_job.v_tiles          = factx.vtcm_v_tiles[0];
-                            ou_job.d_tiles          = factx.vtcm_d_tiles;
+                            ou_job.d_tiles          = factx.vtcm_d_tiles[0];
                             ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                             ou_job.n_row_tiles      = n_row_tiles;
                             ou_job.n_col_tiles      = n_col_tiles;
diff --git src/ggml-hexagon/htp/flash-attn-ops.h src/ggml-hexagon/htp/flash-attn-ops.h
index efe5ce54..c4d19063 100644
--- src/ggml-hexagon/htp/flash-attn-ops.h
+++ src/ggml-hexagon/htp/flash-attn-ops.h
@@ -109,7 +109,7 @@ struct hmx_fa_vtcm_layout {
     size_t off_v_tiles[2];
     size_t off_s_tiles[2];
     size_t off_p_tiles[2];
-    size_t off_d_tiles;
+    size_t off_d_tiles[2];
     size_t off_d_inv_l;
     size_t off_m_vec;
     size_t off_l_vec;
@@ -125,7 +125,7 @@ struct hmx_fa_vtcm_layout {
     size_t q_tile_bytes;
     size_t o_tile_bytes;
     size_t s_tile_bytes;       // S and P tiles (same size)
-    size_t d_tile_bytes;
+    size_t d_tile_bytes;       // d_tiles[0..1] + d_inv_l, allocated back to back
     size_t m_line_bytes;       // one mask row
     size_t m_buf_slot_bytes;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
     size_t col_vec_bytes;
@@ -149,7 +149,12 @@ static inline void hmx_fa_vtcm_layout_build(struct hmx_fa_vtcm_layout * L,
     const size_t k_tile_size  = hex_align_up(Bc   * DK   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
     const size_t v_tile_size  = hex_align_up(Bc   * DV   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
     const size_t s_tile_size  = hex_align_up(g_br * Bc   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
-    const size_t d_tile_size  = hex_align_up(g_br * g_br * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
+
+    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
+    // tiles that sit on the diagonal, so store just those, packed back to back with
+    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
+    // times more than it used, which is also why a second D buffer was unaffordable.
+    const size_t d_tile_size = (g_br / HMX_FP16_TILE_N_ROWS) * HTP_FA_HMX_TILE_SIZE;
 
     const size_t q_dma_size   = hex_align_up(g_br * DK * (is_q_fp32 ? sizeof(float) : sizeof(__fp16)), 128);
     const size_t k_dma_size   = hex_align_up(Bc * hex_round_up(DK * sizeof(__fp16), 128), 128);
@@ -167,7 +172,8 @@ static inline void hmx_fa_vtcm_layout_build(struct hmx_fa_vtcm_layout * L,
     VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_size);
     VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_size);
     VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_size);
-    VTCM_LAYOUT_ALLOC(off, off_d_tiles,       d_tile_size);
+    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_size);
+    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_size, pipeline);
     VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_size);
 
     // Group B & C share start offset (Group B tiles must be 2KB aligned)
@@ -213,7 +219,10 @@ static inline void hmx_fa_vtcm_layout_build(struct hmx_fa_vtcm_layout * L,
     L->o_tile_bytes        = o_tile_size;
     L->col_vec_bytes       = col_vec_size;
     L->s_tile_bytes        = s_tile_size;
-    L->d_tile_bytes        = d_tile_size;
+    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
+    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
+    // cannot silently leave the tail of the run unzeroed.
+    L->d_tile_bytes        = (L->off_d_inv_l + d_tile_size) - L->off_d_tiles[0];
     L->m_line_bytes        = m_line_size;
     L->m_buf_slot_bytes    = m_buf_slot;
     L->row_buf_stride      = row_vec_size / 128;
diff --git src/ggml-metal/ggml-metal-device.cpp src/ggml-metal/ggml-metal-device.cpp
index 953c7575..52043696 100644
--- src/ggml-metal/ggml-metal-device.cpp
+++ src/ggml-metal/ggml-metal-device.cpp
@@ -1409,6 +1409,23 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_p
     return res;
 }
 
+ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
+        ggml_metal_library_t lib,
+        const ggml_tensor * op) {
+    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
+
+    char base[256];
+
+    snprintf(base, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));
+
+    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
+    if (!res.pipeline) {
+        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
+    }
+
+    return res;
+}
+
 ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
         ggml_metal_library_t lib,
         const struct ggml_tensor * op,
@@ -1460,7 +1477,10 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
         bool    has_bias,
         bool    has_scap,
         bool    has_kvpad,
-        int32_t nsg) {
+        int32_t nsg,
+        bool    use_kv_f16,
+        int32_t ns10,
+        int32_t ns20) {
     assert(op->op == GGML_OP_FLASH_ATTN_EXT);
 
     char base[256];
@@ -1469,15 +1489,14 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
     const int32_t dk = (int32_t) op->src[1]->ne[0];
     const int32_t dv = (int32_t) op->src[2]->ne[0];
 
-    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
-    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];
+    const char * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);
 
     // do bounds checks for the mask?
     const bool bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);
 
     snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
             "flash_attn_ext",
-            ggml_type_name(op->src[1]->type),
+            type,
             dk,
             dv);
 
@@ -1526,7 +1545,10 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_v
         bool    has_scap,
         bool    has_kvpad,
         int32_t nsg,
-        int32_t nwg) {
+        int32_t nwg,
+        bool    use_kv_f16,
+        int32_t ns10,
+        int32_t ns20) {
     assert(op->op == GGML_OP_FLASH_ATTN_EXT);
 
     char base[256];
@@ -1535,12 +1557,11 @@ ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_v
     const int32_t dk = (int32_t) op->src[1]->ne[0];
     const int32_t dv = (int32_t) op->src[2]->ne[0];
 
-    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
-    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];
+    const char * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);
 
     snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
             "flash_attn_ext_vec",
-            ggml_type_name(op->src[1]->type),
+            type,
             dk,
             dv);
 
diff --git src/ggml-metal/ggml-metal-device.h src/ggml-metal/ggml-metal-device.h
index 7e1deeaa..b7d46605 100644
--- src/ggml-metal/ggml-metal-device.h
+++ src/ggml-metal/ggml-metal-device.h
@@ -176,6 +176,10 @@ struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_att
         bool    has_mask,
         int32_t ncpsg);
 
+struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
+        ggml_metal_library_t lib,
+        const struct ggml_tensor * op);
+
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
         ggml_metal_library_t lib,
         const struct ggml_tensor * op,
@@ -190,7 +194,10 @@ struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_att
         bool    has_bias,
         bool    has_scap,
         bool    has_kvpad,
-        int32_t nsg);
+        int32_t nsg,
+        bool    use_kv_f16,
+        int32_t ns10,
+        int32_t ns20);
 
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
         ggml_metal_library_t lib,
@@ -201,7 +208,10 @@ struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_att
         bool    has_scap,
         bool    has_kvpad,
         int32_t nsg,
-        int32_t nwg);
+        int32_t nwg,
+        bool    use_kv_f16,
+        int32_t ns10,
+        int32_t ns20);
 
 struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
         ggml_metal_library_t lib,
diff --git src/ggml-metal/ggml-metal-impl.h src/ggml-metal/ggml-metal-impl.h
index 1f6e8c48..f0b77997 100644
--- src/ggml-metal/ggml-metal-impl.h
+++ src/ggml-metal/ggml-metal-impl.h
@@ -329,6 +329,7 @@ typedef struct {
     uint64_t nb3;
     int32_t  n_past;
     int32_t  n_dims;
+    int32_t  n_offs;
     int32_t  n_ctx_orig;
     float    freq_base;
     float    freq_scale;
@@ -341,8 +342,21 @@ typedef struct {
     int32_t  sect_2;
     int32_t  sect_3;
     bool     src2;
+    bool     inplace;
 } ggml_metal_kargs_rope;
 
+typedef struct {
+    int32_t  ne0;
+    int32_t  ne1;
+    int32_t  ne2;
+    int32_t  ne3;
+    uint64_t nb0;
+    uint64_t nb1;
+    uint64_t nb2;
+    uint64_t nb3;
+    int32_t  nblocks;
+} ggml_metal_kargs_flash_attn_ext_kv_f16;
+
 typedef struct {
     int32_t  ne11;
     int32_t  ne_12_2; // assume K and V are same shape
diff --git src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal-ops.cpp
index b7f9b2d0..8311544b 100644
--- src/ggml-metal/ggml-metal-ops.cpp
+++ src/ggml-metal/ggml-metal-ops.cpp
@@ -2801,6 +2801,51 @@ bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
     return (ne01 < 20) && (ne00 % 32 == 0);
 }
 
+// ref: https://github.com/ggml-org/llama.cpp/pull/27390
+// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
+static bool ggml_metal_op_flash_attn_ext_use_kv_f16(const ggml_tensor * op) {
+    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
+
+    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
+    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
+    // TODO: tune per device
+    if (op->src[0]->ne[1] < 32) {
+        return false;
+    }
+
+    switch (op->src[1]->type) {
+        case GGML_TYPE_Q4_0:
+        case GGML_TYPE_Q4_1:
+        case GGML_TYPE_Q5_0:
+        case GGML_TYPE_Q5_1:
+        case GGML_TYPE_Q8_0:
+            return true;
+        default:
+            return false;
+    }
+}
+
+// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
+// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
+// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
+static bool ggml_metal_op_flash_attn_ext_v_is_view_of_k(const ggml_tensor * op) {
+    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
+
+    const ggml_tensor * K = op->src[1];
+    const ggml_tensor * V = op->src[2];
+
+    return V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
+}
+
+// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
+static size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(const ggml_tensor * op) {
+    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
+
+    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
+
+    return GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
+}
+
 size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
     assert(op->op == GGML_OP_FLASH_ATTN_EXT);
 
@@ -2816,6 +2861,18 @@ size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
     size_t res = 0;
 
     const bool has_mask = op->src[3] != nullptr;
+    const bool use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);
+
+    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
+    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
+    const bool v_is_view_of_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
+    uint64_t nb11_pad = nb11;
+    uint64_t nb21_pad = nb21;
+
+    if (use_kv_f16) {
+        nb11_pad = sizeof(ggml_fp16_t)*ne10;
+        nb21_pad = sizeof(ggml_fp16_t)*(v_is_view_of_k ? ne10 : ne20);
+    }
 
     // note: the non-vec kernel requires more extra memory, so always reserve for it
     GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);
@@ -2828,8 +2885,8 @@ size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
 
         if (has_kvpad) {
             res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
-                nb11*ne12*ne13 +
-                nb21*ne22*ne23 +
+                nb11_pad*ne12*ne13 +
+                nb21_pad*ne22*ne23 +
                 (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
         }
     } else {
@@ -2838,8 +2895,8 @@ size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
 
         if (has_kvpad) {
             res += OP_FLASH_ATTN_EXT_NCPSG*(
-                nb11*ne12*ne13 +
-                nb21*ne22*ne23 +
+                nb11_pad*ne12*ne13 +
+                nb21_pad*ne22*ne23 +
                 (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
         }
     }
@@ -2915,6 +2972,29 @@ size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
     return res;
 }
 
+size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(const ggml_tensor * op) {
+    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
+
+    // note: always reserve the temp buffer to avoid graph reallocations
+    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
+    //    return 0;
+    //}
+
+    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
+
+    const size_t k_size = ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);
+
+    // when V is a view of K, the dequantized V is a view of the dequantized K
+    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
+    if (v_is_view_of_k) {
+        return k_size;
+    }
+
+    const size_t v_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);
+
+    return k_size + v_size;
+}
+
 int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
     ggml_tensor * op = ctx->node(idx);
 
@@ -2989,6 +3069,111 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
     ggml_metal_buffer_id bid_tmp = bid_blk;
     bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);
 
+    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
+    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);
+
+    const bool use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);
+
+    ggml_metal_buffer_id bid_k = bid_src1;
+    ggml_metal_buffer_id bid_v = bid_src2;
+
+    uint64_t nb10_attn = nb10;
+    uint64_t nb11_attn = nb11;
+    uint64_t nb12_attn = nb12;
+    uint64_t nb13_attn = nb13;
+    uint64_t nb20_attn = nb20;
+    uint64_t nb21_attn = nb21;
+    uint64_t nb22_attn = nb22;
+    uint64_t nb23_attn = nb23;
+
+    if (use_kv_f16) {
+        assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);
+
+        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
+
+        const int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
+        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
+        const int32_t nblocks1 = nblocks1_64;
+
+        ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
+        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);
+
+        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
+        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);
+
+        // K
+        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
+            /*.ne0    =*/ ne10,
+            /*.ne1    =*/ ne11,
+            /*.ne2    =*/ ne12,
+            /*.ne3    =*/ ne13,
+            /*.nb0    =*/ nb10,
+            /*.nb1    =*/ nb11,
+            /*.nb2    =*/ nb12,
+            /*.nb3    =*/ nb13,
+            /*.nblocks =*/ nblocks1,
+        };
+
+        ggml_metal_encoder_set_pipeline(enc, pipeline0);
+        ggml_metal_encoder_set_bytes   (enc, &args_k, sizeof(args_k), 0);
+        ggml_metal_encoder_set_buffer  (enc, bid_src1,        1);
+        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);
+
+        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);
+
+        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
+        if (!v_is_view_of_k) {
+            const int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
+            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
+            const int32_t nblocks2 = nblocks2_64;
+
+            ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
+                /*.ne0    =*/ ne20,
+                /*.ne1    =*/ ne21,
+                /*.ne2    =*/ ne22,
+                /*.ne3    =*/ ne23,
+                /*.nb0    =*/ nb20,
+                /*.nb1    =*/ nb21,
+                /*.nb2    =*/ nb22,
+                /*.nb3    =*/ nb23,
+                /*.nblocks =*/ nblocks2,
+            };
+
+            ggml_metal_encoder_set_pipeline(enc, pipeline0);
+            ggml_metal_encoder_set_bytes   (enc, &args_v, sizeof(args_v), 0);
+            ggml_metal_encoder_set_buffer  (enc, bid_src2,        1);
+            ggml_metal_encoder_set_buffer  (enc, bid_v_f16,       2);
+
+            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
+        }
+
+        // the pad and attention kernels read the dequantized KV
+        ggml_metal_op_concurrency_reset(ctx);
+
+        bid_k = bid_kv_f16;
+        bid_v = v_is_view_of_k ? bid_k : bid_v_f16;
+
+        // contiguous F16 layout of the dequantized K
+        nb10_attn = sizeof(ggml_fp16_t);
+        nb11_attn = nb10_attn*ne10;
+        nb12_attn = nb11_attn*ne11;
+        nb13_attn = nb12_attn*ne12;
+
+        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
+        if (v_is_view_of_k) {
+            nb20_attn = nb10_attn;
+            nb21_attn = nb11_attn;
+            nb22_attn = nb12_attn;
+            nb23_attn = nb13_attn;
+        } else {
+            // contiguous F16 layout of the dequantized V
+            nb20_attn = sizeof(ggml_fp16_t);
+            nb21_attn = nb20_attn*ne20;
+            nb22_attn = nb21_attn*ne21;
+            nb23_attn = nb22_attn*ne22;
+        }
+    }
+
     if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
         // half8x8 kernel
         const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
@@ -3009,12 +3194,12 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
                 /*.ne11    =*/ne11,
                 /*.ne_12_2 =*/ne12,
                 /*.ne_12_3 =*/ne13,
-                /*.nb11    =*/nb11,
-                /*.nb12    =*/nb12,
-                /*.nb13    =*/nb13,
-                /*.nb21    =*/nb21,
-                /*.nb22    =*/nb22,
-                /*.nb23    =*/nb23,
+                /*.nb11    =*/nb11_attn,
+                /*.nb12    =*/nb12_attn,
+                /*.nb13    =*/nb13_attn,
+                /*.nb21    =*/nb21_attn,
+                /*.nb22    =*/nb22_attn,
+                /*.nb23    =*/nb23_attn,
                 /*.ne31    =*/ne31,
                 /*.ne32    =*/ne32,
                 /*.ne33    =*/ne33,
@@ -3027,8 +3212,8 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
 
             ggml_metal_encoder_set_pipeline(enc, pipeline0);
             ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
-            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
-            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
+            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
+            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
             ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
             ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);
 
@@ -3073,7 +3258,7 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
             ggml_metal_op_concurrency_reset(ctx);
         }
 
-        const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;
+        const int is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;
 
         // 2*(2*ncpsg)
         // ncpsg soft_max values + ncpsg mask values
@@ -3104,6 +3289,9 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
 
         const size_t smem = FATTN_SMEM(nsg);
 
+        const int32_t ns10 = nb11_attn/nb10_attn;
+        const int32_t ns20 = nb21_attn/nb20_attn;
+
         ggml_metal_kargs_flash_attn_ext args = {
             /*.ne01          =*/ ne01,
             /*.ne02          =*/ ne02,
@@ -3114,14 +3302,14 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
             /*.ne11          =*/ ne11,
             /*.ne_12_2       =*/ ne12,
             /*.ne_12_3       =*/ ne13,
-            /*.ns10          =*/ int32_t(nb11/nb10),
-            /*.nb11          =*/ nb11,
-            /*.nb12          =*/ nb12,
-            /*.nb13          =*/ nb13,
-            /*.ns20          =*/ int32_t(nb21/nb20),
-            /*.nb21          =*/ nb21,
-            /*.nb22          =*/ nb22,
-            /*.nb23          =*/ nb23,
+            /*.ns10          =*/ ns10,
+            /*.nb11          =*/ nb11_attn,
+            /*.nb12          =*/ nb12_attn,
+            /*.nb13          =*/ nb13_attn,
+            /*.ns20          =*/ ns20,
+            /*.nb21          =*/ nb21_attn,
+            /*.nb22          =*/ nb22_attn,
+            /*.nb23          =*/ nb23_attn,
             /*.ne31          =*/ ne31,
             /*.ne32          =*/ ne32,
             /*.ne33          =*/ ne33,
@@ -3139,13 +3327,13 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
             /*.logit_softcap =*/ logit_softcap,
         };
 
-        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg);
+        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, use_kv_f16, ns10, ns20);
 
         ggml_metal_encoder_set_pipeline(enc, pipeline);
         ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
         ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
-        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
-        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
+        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
+        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
         ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
         ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
         ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
@@ -3177,12 +3365,12 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
                 /*.ne11    =*/ne11,
                 /*.ne_12_2 =*/ne12,
                 /*.ne_12_3 =*/ne13,
-                /*.nb11    =*/nb11,
-                /*.nb12    =*/nb12,
-                /*.nb13    =*/nb13,
-                /*.nb21    =*/nb21,
-                /*.nb22    =*/nb22,
-                /*.nb23    =*/nb23,
+                /*.nb11    =*/nb11_attn,
+                /*.nb12    =*/nb12_attn,
+                /*.nb13    =*/nb13_attn,
+                /*.nb21    =*/nb21_attn,
+                /*.nb22    =*/nb22_attn,
+                /*.nb23    =*/nb23_attn,
                 /*.ne31    =*/ne31,
                 /*.ne32    =*/ne32,
                 /*.ne33    =*/ne33,
@@ -3195,8 +3383,8 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
 
             ggml_metal_encoder_set_pipeline(enc, pipeline0);
             ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
-            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
-            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
+            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
+            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
             ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
             ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);
 
@@ -3242,6 +3430,9 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
             }
         }
 
+        const int32_t ns10 = nb11_attn/nb10_attn;
+        const int32_t ns20 = nb21_attn/nb20_attn;
+
         ggml_metal_kargs_flash_attn_ext_vec args = {
             /*.ne01          =*/ ne01,
             /*.ne02          =*/ ne02,
@@ -3252,14 +3443,14 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
             /*.ne11          =*/ ne11,
             /*.ne_12_2       =*/ ne12,
             /*.ne_12_3       =*/ ne13,
-            /*.ns10          =*/ int32_t(nb11/nb10),
-            /*.nb11          =*/ nb11,
-            /*.nb12          =*/ nb12,
-            /*.nb13          =*/ nb13,
-            /*.ns20          =*/ int32_t(nb21/nb20),
-            /*.nb21          =*/ nb21,
-            /*.nb22          =*/ nb22,
-            /*.nb23          =*/ nb23,
+            /*.ns10          =*/ ns10,
+            /*.nb11          =*/ nb11_attn,
+            /*.nb12          =*/ nb12_attn,
+            /*.nb13          =*/ nb13_attn,
+            /*.ns20          =*/ ns20,
+            /*.nb21          =*/ nb21_attn,
+            /*.nb22          =*/ nb22_attn,
+            /*.nb23          =*/ nb23_attn,
             /*.ne31          =*/ ne31,
             /*.ne32          =*/ ne32,
             /*.ne33          =*/ ne33,
@@ -3277,15 +3468,15 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
             /*.logit_softcap =*/ logit_softcap,
         };
 
-        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg);
+        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg, use_kv_f16, ns10, ns20);
 
         GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
 
         ggml_metal_encoder_set_pipeline(enc, pipeline);
         ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
         ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
-        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
-        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
+        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
+        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
         ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
         ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
 
@@ -3884,6 +4075,11 @@ int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
     const int sect_2 = ((const int32_t *) op->op_params)[13];
     const int sect_3 = ((const int32_t *) op->op_params)[14];
 
+    const int n_offs = ((const int32_t *) op->op_params)[15];
+
+    // when dst aliases src0, the channels outside the rotated window already hold the correct data
+    const bool inplace = op->data == op->src[0]->data;
+
     ggml_metal_kargs_rope args = {
         /*.ne00        =*/ ne00,
         /*.ne01        =*/ ne01,
@@ -3903,6 +4099,7 @@ int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
         /*.nb3         =*/ nb3,
         /*.n_past      =*/ n_past,
         /*.n_dims      =*/ n_dims,
+        /*.n_offs      =*/ n_offs,
         /*.n_ctx_orig  =*/ n_ctx_orig,
         /*.freq_base   =*/ freq_base,
         /*.freq_scale  =*/ freq_scale,
@@ -3915,6 +4112,7 @@ int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
         /* sect_2      =*/ sect_2,
         /* sect_3      =*/ sect_3,
         /* src2        =*/ op->src[2] != nullptr,
+        /* inplace     =*/ inplace,
     };
 
     auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);
diff --git src/ggml-metal/ggml-metal-ops.h src/ggml-metal/ggml-metal-ops.h
index b03b59e0..159a628d 100644
--- src/ggml-metal/ggml-metal-ops.h
+++ src/ggml-metal/ggml-metal-ops.h
@@ -42,6 +42,7 @@ bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor * op);
 size_t ggml_metal_op_flash_attn_ext_extra_pad(const struct ggml_tensor * op);
 size_t ggml_metal_op_flash_attn_ext_extra_blk(const struct ggml_tensor * op);
 size_t ggml_metal_op_flash_attn_ext_extra_tmp(const struct ggml_tensor * op);
+size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(const struct ggml_tensor * op);
 
 int ggml_metal_op_concat            (ggml_metal_op_t ctx, int idx);
 int ggml_metal_op_repeat            (ggml_metal_op_t ctx, int idx);
diff --git src/ggml-metal/ggml-metal.cpp src/ggml-metal/ggml-metal.cpp
index ef3c92f2..0e8d409e 100644
--- src/ggml-metal/ggml-metal.cpp
+++ src/ggml-metal/ggml-metal.cpp
@@ -225,6 +225,7 @@ static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_
                 res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                 res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                 res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
+                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
             } break;
         case GGML_OP_CUMSUM:
         case GGML_OP_ARGSORT:
diff --git src/ggml-metal/ggml-metal.metal src/ggml-metal/ggml-metal.metal
index 243c997f..949931c8 100644
--- src/ggml-metal/ggml-metal.metal
+++ src/ggml-metal/ggml-metal.metal
@@ -656,13 +656,13 @@ void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & r
 
 template <typename type4x4>
 void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
-    device const int8_t * qs = ((device const int8_t *)xb->qs);
+    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
     const float d = xb->d;
 
     float4x4 reg_f;
 
-    for (int i = 0; i < 16; i++) {
-        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
+    for (int i = 0; i < 4; ++i) {
+        reg_f[i] = float4(qs[4*il + i]) * d;
     }
 
     reg = (type4x4) reg_f;
@@ -670,12 +670,10 @@ void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg
 
 template <typename type4>
 void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
-    device const int8_t * qs = ((device const int8_t *)xb->qs);
+    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
     const float d = xb->d;
 
-    for (int i = 0; i < 4; i++) {
-        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
-    }
+    reg = (type4) (float4(qs[il]) * d);
 }
 
 template <typename type4x4>
@@ -4688,14 +4686,15 @@ kernel void kernel_rope_norm(
     float sin_theta;
 
     for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
-        if (i0 < args.n_dims) {
-            const int ic = i0/2;
+        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
+            const int iw = i0 - args.n_offs; // relative idx
+            const int ic = iw/2;
 
-            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);
+            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);
 
             const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;
 
-            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
+            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
 
             device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
             device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);
@@ -4706,6 +4705,10 @@ kernel void kernel_rope_norm(
             dst_data[0] = x0*cos_theta - x1*sin_theta;
             dst_data[1] = x0*sin_theta + x1*cos_theta;
         } else {
+            if (args.inplace) {
+                continue;
+            }
+
             device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
             device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);
 
@@ -4741,17 +4744,18 @@ kernel void kernel_rope_neox(
     float sin_theta;
 
     for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
-        if (i0 < args.n_dims) {
-            const int ic = i0/2;
+        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
+            const int iw = i0 - args.n_offs; // relative idx
+            const int ic = iw/2;
 
-            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);
+            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);
 
             const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;
 
-            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
+            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
 
-            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
-            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);
+            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + (args.n_offs + ic)*args.nb00);
+            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + (args.n_offs + ic)*args.nb0);
 
             const float x0 = src[0];
             const float x1 = src[args.n_dims/2];
@@ -4759,6 +4763,10 @@ kernel void kernel_rope_neox(
             dst_data[0]             = x0*cos_theta - x1*sin_theta;
             dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
         } else {
+            if (args.inplace) {
+                continue;
+            }
+
             device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
             device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);
 
@@ -4793,8 +4801,9 @@ kernel void kernel_rope_multi(
     float sin_theta;
 
     for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
-        if (i0 < args.n_dims) {
-            const int ic = i0/2;
+        if (i0 >= args.n_offs && i0 < args.n_offs + args.n_dims) {
+            const int iw = i0 - args.n_offs; // relative idx
+            const int ic = iw/2;
 
             // mrope theta calculations
             // note: the rest is the same as kernel_rope_neox
@@ -4827,14 +4836,14 @@ kernel void kernel_rope_multi(
             }
             // end of mrope
 
-            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);
+            const float theta = theta_base * pow(args.freq_base, inv_ndims*iw);
 
             const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;
 
-            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
+            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, iw, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
 
-            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
-            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);
+            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + (args.n_offs + ic)*args.nb00);
+            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + (args.n_offs + ic)*args.nb0);
 
             const float x0 = src[0];
             const float x1 = src[args.n_dims/2];
@@ -4842,6 +4851,10 @@ kernel void kernel_rope_multi(
             dst_data[0]             = x0*cos_theta - x1*sin_theta;
             dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
         } else {
+            if (args.inplace) {
+                continue;
+            }
+
             device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
             device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);
 
@@ -6305,6 +6318,53 @@ template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_t kernel_fwht_f
 template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_t kernel_fwht_f32<256>;
 template [[host_name("kernel_fwht_f32_512")]] kernel kernel_fwht_t kernel_fwht_f32<512>;
 
+// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
+// - one thread per block; dispatched separately for K and V
+// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
+template <
+    typename block_t,
+    short QK,
+    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
+kernel void kernel_flash_attn_ext_kv_f16(
+        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
+        device const char * x,
+        device       half * x_dst,
+        uint gid [[thread_position_in_grid]]) {
+    if (gid >= (uint) args.nblocks) {
+        return;
+    }
+
+    const uint nb = args.ne0/QK;
+    const uint i0 = gid%nb;
+    uint ib       = gid/nb;
+    const uint i1 = ib%args.ne1;
+    ib /= args.ne1;
+    const uint i2 = ib%args.ne2;
+    const uint i3 = ib/args.ne2;
+
+    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;
+
+    device const block_t * src = (device const block_t *) (x + offs);
+    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;
+
+    for (short i = 0; i < QK/16; ++i) {
+        float4x4 reg;
+        deq_t4x4(src, i, reg);
+        dst[4*i + 0] = (half4) reg[0];
+        dst[4*i + 1] = (half4) reg[1];
+        dst[4*i + 2] = (half4) reg[2];
+        dst[4*i + 3] = (half4) reg[3];
+    }
+}
+
+typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;
+
+template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
+template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
+template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
+template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
+template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;
+
 constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
 
 constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];
diff --git src/ggml-opencl/CMakeLists.txt src/ggml-opencl/CMakeLists.txt
index 1dc70717..72334d5c 100644
--- src/ggml-opencl/CMakeLists.txt
+++ src/ggml-opencl/CMakeLists.txt
@@ -202,6 +202,7 @@ set(GGML_OPENCL_KERNELS
     sqr
     sqrt
     ssm_conv
+    ssm_scan
     gated_delta_net
     sub
     sum_rows
diff --git src/ggml-opencl/ggml-opencl.cpp src/ggml-opencl/ggml-opencl.cpp
index 25790860..fbf7dadb 100644
--- src/ggml-opencl/ggml-opencl.cpp
+++ src/ggml-opencl/ggml-opencl.cpp
@@ -866,6 +866,9 @@ struct ggml_backend_opencl_context {
     // [size_idx][kda][tgpp] where size_idx: 0=S_V=16, 1=32, 2=64, 3=128; kda: 0 or 1.
     // tgpp 0 = TG variant (COLS_PER_LANE_GROUP=1), tgpp 1 = prefill variant (COLS_PER_LANE_GROUP=4).
     cl_kernel kernel_gated_delta_net_f32[4][2][2] = {};
+    cl_kernel kernel_ssm_scan_f32_mamba2_d128 = nullptr;
+    cl_kernel kernel_ssm_scan_f32_mamba2_d256 = nullptr;
+
     cl_kernel kernel_timestep_embedding;
     cl_kernel kernel_gemv_moe_q4_0_f32_ns, kernel_gemm_moe_q4_0_f32_ns, kernel_gemm_moe_q4_0_f32_ns_bin;
     cl_kernel kernel_gemm_moe_q8_0_f32_ns;
@@ -892,6 +895,7 @@ struct ggml_backend_opencl_context {
     cl_kernel kernel_gemm_moe_q4_0_q8_1_dp4a = nullptr;    // dp4a (int8) q4_0 MoE prefill GEMM
     cl_kernel kernel_moe_reorder_b;
     cl_kernel kernel_moe_histogram, kernel_moe_scan, kernel_moe_fill, kernel_moe_scatter;
+    cl_kernel kernel_moe_scatter_stable = nullptr;   // deterministic slot assignment
     cl_kernel kernel_moe_combine_f32 = nullptr;   // fused router-weight mul + cross-expert sum
     cl_kernel kernel_mul_mv_id_q4_0_f32_8x_flat;
     cl_kernel kernel_mul_mv_id_q8_0_f32, kernel_mul_mv_id_q8_0_f32_flat;
@@ -3154,6 +3158,24 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         GGML_LOG_CONT(".");
     }
 
+    // ssm_scan (Mamba-2 fused per-token recurrent step; d_state in {128, 256})
+    {
+#ifdef GGML_OPENCL_EMBED_KERNELS
+        const std::string kernel_src {
+            #include "ssm_scan.cl.h"
+        };
+#else
+        const std::string kernel_src = read_file("ssm_scan.cl");
+#endif
+        cl_program prog =
+            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
+
+        CL_CHECK((backend_ctx->kernel_ssm_scan_f32_mamba2_d128 = clCreateKernel(prog, "kernel_ssm_scan_f32_mamba2_d128", &err), err));
+        CL_CHECK((backend_ctx->kernel_ssm_scan_f32_mamba2_d256 = clCreateKernel(prog, "kernel_ssm_scan_f32_mamba2_d256", &err), err));
+        CL_CHECK(clReleaseProgram(prog));
+        GGML_LOG_CONT(".");
+    }
+
     // gated_delta_net: one kernel per (S_V, KDA, tgpp) triple.
     {
     #ifdef GGML_OPENCL_EMBED_KERNELS
@@ -4442,6 +4464,7 @@ static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
         CL_CHECK((backend_ctx->kernel_moe_scan = clCreateKernel(prog, "kernel_moe_scan", &err), err));
         CL_CHECK((backend_ctx->kernel_moe_fill = clCreateKernel(prog, "kernel_moe_fill", &err), err));
         CL_CHECK((backend_ctx->kernel_moe_scatter = clCreateKernel(prog, "kernel_moe_scatter", &err), err));
+        CL_CHECK((backend_ctx->kernel_moe_scatter_stable = clCreateKernel(prog, "kernel_moe_scatter_stable", &err), err));
         CL_CHECK(clReleaseProgram(prog));
         GGML_LOG_CONT(".");
     }
@@ -7301,6 +7324,23 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
                    (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
         case GGML_OP_SSM_CONV:
             return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
+        case GGML_OP_SSM_SCAN: {
+            // Mamba-2 fused per-token scan. Requires src3->ne[0] == 1 (scalar
+            // A per head); d_state in {128, 256}; all sources f32. Falls back
+            // to CPU otherwise (incl. Mamba-1 element-wise A).
+            for (int i = 0; i < 6; ++i) {
+                if (op->src[i]->type != GGML_TYPE_F32) {
+                    return false;
+                }
+            }
+            if (op->type != GGML_TYPE_F32) {
+                return false;
+            }
+            const int K = ggml_get_op_params_i32(op, 0);
+            const int d_state = (int) op->src[0]->ne[0];
+            const bool is_mamba2 = (op->src[3]->ne[0] == 1);
+            return is_mamba2 && (d_state == 128 || d_state == 256) && (K == 1);
+        }
         case GGML_OP_GATED_DELTA_NET:
             {
                 // Match the Vulkan backend: only F32 -> F32, S_v in {16, 32, 64, 128}.
@@ -7376,6 +7416,9 @@ static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_te
         case GGML_OP_DIAG_MASK_INF:
             return op->ne[3] == 1;
         case GGML_OP_ROPE: {
+            if (((const int32_t *) op->op_params)[15] != 0) {
+                return false; // FIXME: support ggml_rope_set_offset
+            }
             const int mode = ((const int32_t *) op->op_params)[2];
             const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
             const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
@@ -12257,6 +12300,103 @@ static void ggml_cl_mean(ggml_backend_t backend, const ggml_tensor * src0, const
     backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
 }
 
+static void ggml_cl_ssm_scan(ggml_backend_t backend, ggml_tensor * dst) {
+    const ggml_tensor * src0 = dst->src[0]; // s
+    const ggml_tensor * src1 = dst->src[1]; // x
+    const ggml_tensor * src2 = dst->src[2]; // dt
+    const ggml_tensor * src3 = dst->src[3]; // A
+    const ggml_tensor * src4 = dst->src[4]; // B
+    const ggml_tensor * src5 = dst->src[5]; // C
+    const ggml_tensor * src6 = dst->src[6]; // ids
+
+    GGML_ASSERT(src0 && src1 && src2 && src3 && src4 && src5 && src6 && dst);
+
+    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
+
+    ggml_tensor_extra_cl * e0 = (ggml_tensor_extra_cl *) src0->extra;
+    ggml_tensor_extra_cl * e1 = (ggml_tensor_extra_cl *) src1->extra;
+    ggml_tensor_extra_cl * e2 = (ggml_tensor_extra_cl *) src2->extra;
+    ggml_tensor_extra_cl * e3 = (ggml_tensor_extra_cl *) src3->extra;
+    ggml_tensor_extra_cl * e4 = (ggml_tensor_extra_cl *) src4->extra;
+    ggml_tensor_extra_cl * e5 = (ggml_tensor_extra_cl *) src5->extra;
+    ggml_tensor_extra_cl * e6 = (ggml_tensor_extra_cl *) src6->extra;
+    ggml_tensor_extra_cl * ed = (ggml_tensor_extra_cl *) dst->extra;
+
+    cl_ulong o0 = e0->offset + src0->view_offs;
+    cl_ulong o1 = e1->offset + src1->view_offs;
+    cl_ulong o2 = e2->offset + src2->view_offs;
+    cl_ulong o3 = e3->offset + src3->view_offs;
+    cl_ulong o4 = e4->offset + src4->view_offs;
+    cl_ulong o5 = e5->offset + src5->view_offs;
+    cl_ulong o6 = e6->offset + src6->view_offs;
+    cl_ulong od = ed->offset + dst->view_offs;
+
+    const int d_state  = (int) src0->ne[0];
+    const int head_dim = (int) src0->ne[1];
+    const int n_head   = (int) src1->ne[1];
+    const int n_group  = (int) src4->ne[1];
+    const int n_tokens = (int) src1->ne[2];
+    const int n_seqs   = (int) src1->ne[3];
+
+    // Mirror CPU ref: s_off = ggml_nelements(src1) * sizeof(float)
+    const cl_ulong s_off_bytes = (cl_ulong) ggml_nelements(src1) * sizeof(float);
+
+    cl_kernel kernel = (d_state == 128)
+        ? backend_ctx->kernel_ssm_scan_f32_mamba2_d128
+        : backend_ctx->kernel_ssm_scan_f32_mamba2_d256;
+    GGML_ASSERT(kernel != nullptr);
+
+    cl_ulong s0_nb2 = src0->nb[2];
+    cl_ulong s0_nb3 = src0->nb[3];
+    cl_ulong x_nb2  = src1->nb[2];
+    cl_ulong x_nb3  = src1->nb[3];
+    cl_ulong dt_nb1 = src2->nb[1];
+    cl_ulong dt_nb2 = src2->nb[2];
+    cl_ulong A_nb1  = src3->nb[1];
+    cl_ulong B_nb2  = src4->nb[2];
+    cl_ulong B_nb3  = src4->nb[3];
+    cl_ulong C_nb2  = src5->nb[2];
+    cl_ulong C_nb3  = src5->nb[3];
+
+    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &e0->data_device));
+    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &o0));
+    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &e1->data_device));
+    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &o1));
+    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &e2->data_device));
+    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &o2));
+    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &e3->data_device));
+    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &o3));
+    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_mem),   &e4->data_device));
+    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &o4));
+    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem),   &e5->data_device));
+    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &o5));
+    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem),   &e6->data_device));
+    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &o6));
+    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_mem),   &ed->data_device));
+    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &od));
+    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &s0_nb2));
+    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &s0_nb3));
+    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &x_nb2));
+    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &x_nb3));
+    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &dt_nb1));
+    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &dt_nb2));
+    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &A_nb1));
+    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &B_nb2));
+    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &B_nb3));
+    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong), &C_nb2));
+    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &C_nb3));
+    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &s_off_bytes));
+    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(int),      &head_dim));
+    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(int),      &n_head));
+    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(int),      &n_group));
+    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(int),      &n_tokens));
+
+    size_t global_work_size[] = { (size_t)n_head * head_dim * 64, (size_t)n_seqs, 1 };
+    size_t local_work_size[]  = { 64, 1, 1 };
+
+    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
+}
+
 static void ggml_cl_ssm_conv(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
     GGML_ASSERT(src0);
     GGML_ASSERT(src0->extra);
@@ -20725,18 +20865,42 @@ static void moe_router_reoerder(ggml_backend_t backend, const ggml_tensor * src,
     size_t fill_local_size[] = {64, 1, 1};
     backend_ctx->enqueue_ndrange_kernel(kernel, 3, fill_global_size, fill_local_size, src);
 
-    // Scatter
-    kernel = backend_ctx->kernel_moe_scatter;
-    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &original_router_buf));
-    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &post_router_buf));
-    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &emap_buf));
-    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &tile_offset_buf));
-    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &slot_counter_buf));
-    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne21));
-    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne20));
-    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &ne02));
+    // Scatter. The deterministic variant is the default: kernel_moe_scatter derives
+    // each token's slot from an atomic counter, so the packing inside an expert - and
+    // with it the output of the ragged prefill GEMM - changes from run to run. Set
+    // GGML_OPENCL_MOE_STABLE_SCATTER=0 to restore the atomic version.
+    static const bool stable_scatter = []{
+        const char * e = getenv("GGML_OPENCL_MOE_STABLE_SCATTER");
+        return !e || e[0] == '\0' || e[0] != '0';
+    }();
 
-    backend_ctx->enqueue_ndrange_kernel(kernel, 3, histogram_global_size, histogram_local_size, src);
+    if (stable_scatter) {
+        kernel = backend_ctx->kernel_moe_scatter_stable;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &original_router_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &post_router_buf));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &emap_buf));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &tile_offset_buf));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne21));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne20));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne02));
+
+        // one workgroup (one wave) per expert; each ranks its own tokens
+        size_t scatter_global_size[] = {64, (size_t)ne02};
+        size_t scatter_local_size[]  = {64, 1};
+        backend_ctx->enqueue_ndrange_kernel(kernel, 2, scatter_global_size, scatter_local_size, src);
+    } else {
+        kernel = backend_ctx->kernel_moe_scatter;
+        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &original_router_buf));
+        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &post_router_buf));
+        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &emap_buf));
+        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &tile_offset_buf));
+        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &slot_counter_buf));
+        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne21));
+        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne20));
+        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &ne02));
+
+        backend_ctx->enqueue_ndrange_kernel(kernel, 3, histogram_global_size, histogram_local_size, src);
+    }
 
     // [MOE_TILES] env-gated padding probe: read back total_tiles (= Sum_e
     // ceil(k_e/n_tile_size)) and compare to the ideal tile count for the real
@@ -24743,6 +24907,14 @@ bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor
             }
             func = ggml_cl_ssm_conv;
             break;
+        case GGML_OP_SSM_SCAN:
+            if (!any_on_device) {
+                return false;
+            }
+            // SSM_SCAN has 7 source tensors, so it cannot use the standard
+            // (src0, src1, dst) func signature. Dispatch directly and return.
+            ggml_cl_ssm_scan(backend, tensor);
+            return true;
         case GGML_OP_GATED_DELTA_NET:
             if (!any_on_device) {
                 return false;
diff --git src/ggml-opencl/kernels/flash_attn_f16.cl src/ggml-opencl/kernels/flash_attn_f16.cl
index fc58a22e..f9797d34 100644
--- src/ggml-opencl/kernels/flash_attn_f16.cl
+++ src/ggml-opencl/kernels/flash_attn_f16.cl
@@ -118,6 +118,17 @@ __kernel void flash_attn_f16(
     __local DATA_TYPE4 l_v[BLOCK_N][DV_VEC];
 
     for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
+#if WG_SIZE > FA_SG
+        // WAR on l_k/l_v: a thread that finishes the compute below early — either
+        // it skipped it (my_query_row >= n_q, the continue) or its subgroup simply
+        // ran ahead — wraps around and reloads the tiles while another subgroup is
+        // still reading them. Any WG that is exactly one lockstep subgroup
+        // (WG_SIZE == FA_SG) cannot diverge and hides this; a WG spanning multiple
+        // subgroups (Intel sg=32, or BLOCK_M > 64 on Adreno) corrupts the result.
+        // All threads reach this each iteration (no-op on the first), so it does
+        // not diverge with the continue. Compiled out when WG == one subgroup.
+        barrier(CLK_LOCAL_MEM_FENCE);
+#endif
         for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
             const int row = i / DK_VEC;
             const int col = i % DK_VEC;
diff --git src/ggml-opencl/kernels/flash_attn_f32.cl src/ggml-opencl/kernels/flash_attn_f32.cl
index 599877bd..5911524e 100644
--- src/ggml-opencl/kernels/flash_attn_f32.cl
+++ src/ggml-opencl/kernels/flash_attn_f32.cl
@@ -119,13 +119,15 @@ __kernel void flash_attn_f32(
     __local DATA_TYPE4 l_v[BLOCK_N][DV_VEC];
 
     for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
-#if FA_SG < 64
-        // WAR on l_k/l_v: threads with my_query_row >= n_q skip the compute below
-        // (continue) and would race ahead to reload the tiles while active threads
-        // still read them. A single 64-wide Adreno subgroup (WG == sg) runs lockstep
-        // and hides this; a WG that spans multiple narrower subgroups (Intel sg=32)
-        // corrupts the result. All threads reach this each iteration (no-op on the
-        // first), so it does not diverge with the continue. Compiled out at sg=64.
+#if WG_SIZE > FA_SG
+        // WAR on l_k/l_v: a thread that finishes the compute below early — either
+        // it skipped it (my_query_row >= n_q, the continue) or its subgroup simply
+        // ran ahead — wraps around and reloads the tiles while another subgroup is
+        // still reading them. Any WG that is exactly one lockstep subgroup
+        // (WG_SIZE == FA_SG) cannot diverge and hides this; a WG spanning multiple
+        // subgroups (Intel sg=32, or BLOCK_M > 64 on Adreno) corrupts the result.
+        // All threads reach this each iteration (no-op on the first), so it does
+        // not diverge with the continue. Compiled out when WG == one subgroup.
         barrier(CLK_LOCAL_MEM_FENCE);
 #endif
         for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
diff --git src/ggml-opencl/kernels/moe_sort_by_expert.cl src/ggml-opencl/kernels/moe_sort_by_expert.cl
index d9703429..d52d11aa 100644
--- src/ggml-opencl/kernels/moe_sort_by_expert.cl
+++ src/ggml-opencl/kernels/moe_sort_by_expert.cl
@@ -68,6 +68,79 @@ __kernel void kernel_moe_scatter(
     emap[tile_idx] = val;
 }
 
+// Deterministic replacement for kernel_moe_scatter.
+//
+// kernel_moe_scatter takes each token's slot from atomic_inc(slot_counter[expert]),
+// so the token -> slot packing inside an expert depends on which work-item wins the
+// atomic and changes from run to run. The ragged prefill GEMM path is sensitive to
+// that packing (the non-ragged path is not, since its padded slots alias slot 0 and
+// are overwritten last), which makes MoE prompt processing non-reproducible: the same
+// binary on the same prompt returns one of several outputs.
+//
+// Here the slot is the token's rank in flat (n, k) order among the tokens routed to
+// the same expert - a fixed function of the routing input. One workgroup per expert
+// walks the flat routing list in blocks of 64 and ranks its own tokens with a
+// workgroup scan, carrying a running count between blocks. Cost is one pass over the
+// routing list per expert; the list is a few KiB and stays in cache.
+__kernel void kernel_moe_scatter_stable(
+    __global const int * input,
+    __global int * post_router,
+    __global ushort * emap,
+    __global const int * tile_offset,
+    int N,
+    int topK,
+    uint n_experts
+) {
+    const int e   = get_group_id(1);
+    const int lid = get_local_id(0);
+    const int M   = N * topK;
+
+    __local int scan[64];
+    __local int running;
+
+    if (lid == 0) {
+        running = 0;
+    }
+    barrier(CLK_LOCAL_MEM_FENCE);
+
+    for (int base = 0; base < M; base += 64) {
+        const int j = base + lid;
+
+        int pred = 0;
+        if (j < M) {
+            const int n = j / topK;
+            const int k = j - n * topK;
+            pred = (input[n * (int)n_experts + k] == e) ? 1 : 0;
+        }
+
+        scan[lid] = pred;
+        barrier(CLK_LOCAL_MEM_FENCE);
+
+        // Hillis-Steele inclusive scan over the 64 lanes
+        for (int off = 1; off < 64; off <<= 1) {
+            int add = (lid >= off) ? scan[lid - off] : 0;
+            barrier(CLK_LOCAL_MEM_FENCE);
+            scan[lid] += add;
+            barrier(CLK_LOCAL_MEM_FENCE);
+        }
+
+        if (pred) {
+            const int local_slot = running + (scan[lid] - 1);   // exclusive rank
+            const int tile_idx   = tile_offset[e] + (local_slot >> 5);
+            const int lane       = local_slot & 31;
+
+            post_router[tile_idx * 32 + lane] = j;
+            emap[tile_idx] = (ushort)e;
+        }
+
+        barrier(CLK_LOCAL_MEM_FENCE);
+        if (lid == 63) {
+            running += scan[63];
+        }
+        barrier(CLK_LOCAL_MEM_FENCE);
+    }
+}
+
 __kernel void kernel_moe_fill(
     __global int * post_router,
     __global int * total_tiles,
diff --git src/ggml-opencl/kernels/ssm_scan.cl src/ggml-opencl/kernels/ssm_scan.cl
new file mode 100644
index 00000000..37698d12
--- /dev/null
+++ src/ggml-opencl/kernels/ssm_scan.cl
@@ -0,0 +1,216 @@
+// Mamba2 fused SSM scan kernel. One workgroup per (head, dim, seq); WG size =
+// 64 threads. Each thread owns c_factor = d_state/64 state elements in
+// private registers; the state stays resident across the n_tokens t-loop
+//
+// References:
+//   ggml/src/ggml-cuda/ssm-scan.cu:117 ssm_scan_f32_group
+//   ggml/src/ggml-cpu/ops.cpp:9368 ggml_compute_forward_ssm_scan_f32
+
+#pragma OPENCL EXTENSION cl_khr_fp16 : enable
+
+#ifdef cl_khr_subgroups
+#pragma OPENCL EXTENSION cl_khr_subgroups : enable
+#endif
+
+#if defined(cl_qcom_reqd_sub_group_size)
+#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
+#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
+#else
+#define REQD_SUBGROUP_SIZE_64
+#endif
+
+inline float softplus_f32(float x) {
+    return (x <= 20.0f) ? log(1.0f + exp(x)) : x;
+}
+
+// d_state = 128 (most Mamba-2 models, e.g. mamba2-2.7B, Codestral-Mamba).
+// WG = 64 threads, each holds 2 state elements (tid and tid+64).
+REQD_SUBGROUP_SIZE_64
+kernel void kernel_ssm_scan_f32_mamba2_d128(
+    global const char * src0_base, ulong src0_off,
+    global const char * src1_base, ulong src1_off,
+    global const char * src2_base, ulong src2_off,
+    global const char * src3_base, ulong src3_off,
+    global const char * src4_base, ulong src4_off,
+    global const char * src5_base, ulong src5_off,
+    global const char * src6_base, ulong src6_off,
+    global       char * dst_base,  ulong dst_off,
+    ulong s0_nb2, ulong s0_nb3,
+    ulong x_nb2,  ulong x_nb3,
+    ulong dt_nb1, ulong dt_nb2,
+    ulong A_nb1,
+    ulong B_nb2,  ulong B_nb3,
+    ulong C_nb2,  ulong C_nb3,
+    ulong s_off_bytes,
+    int   head_dim, int n_head, int n_group, int n_tokens
+) {
+    const int d_state = 128;
+
+    const int tid     = (int) get_local_id(0);
+    const int wg_x    = (int) get_group_id(0);
+    const int seq_id  = (int) get_group_id(1);
+
+    const int head_id = wg_x / head_dim;
+    const int dim_id  = wg_x - head_id * head_dim;
+    const int g       = head_id / (n_head / n_group);
+
+    src0_base += src0_off;
+    src1_base += src1_off;
+    src2_base += src2_off;
+    src3_base += src3_off;
+    src4_base += src4_off;
+    src5_base += src5_off;
+    src6_base += src6_off;
+    dst_base  += dst_off;
+
+    const int seq_slot = ((global const int *) src6_base)[seq_id];
+
+    const ulong state_base_off = (ulong)seq_slot * s0_nb3 + (ulong)head_id * s0_nb2
+                                + (ulong)dim_id * d_state * sizeof(float);
+    global const float * s0_warp = (global const float *)(src0_base + state_base_off);
+    const ulong state_out_off = (ulong)seq_id * s0_nb3 + (ulong)head_id * s0_nb2
+                              + (ulong)dim_id * d_state * sizeof(float);
+    global float * s_warp = (global float *)(dst_base + s_off_bytes + state_out_off);
+
+    global const char * x_seq  = src1_base + (ulong)seq_id * x_nb3;
+    global const char * dt_seq = src2_base + (ulong)seq_id * dt_nb2;
+    global const char * B_seq  = src4_base + (ulong)seq_id * B_nb3 + (ulong)g * d_state * sizeof(float);
+    global const char * C_seq  = src5_base + (ulong)seq_id * C_nb3 + (ulong)g * d_state * sizeof(float);
+
+    const ulong y_dim_total = (ulong)n_head * head_dim;
+    global float * y_seq = (global float *)dst_base
+                           + (ulong)seq_id * (ulong)n_tokens * y_dim_total;
+
+    const float A_val = ((global const float *)src3_base)[(ulong)head_id * A_nb1 / sizeof(float)];
+
+    // c_factor = 2: each thread owns 2 state elements (tid and tid+64).
+    float state0 = s0_warp[tid];
+    float state1 = s0_warp[tid + 64];
+
+    for (int t = 0; t < n_tokens; ++t) {
+        const float dt_h        = ((global const float *)(dt_seq + (ulong)t * dt_nb1))[head_id];
+        const float dt_softplus = softplus_f32(dt_h);
+        const float dA          = exp(dt_softplus * A_val);
+        const float x_val       = ((global const float *)(x_seq + (ulong)t * x_nb2))[(ulong)head_id * head_dim + dim_id];
+        const float x_dt        = x_val * dt_softplus;
+
+        const float B0 = ((global const float *)(B_seq + (ulong)t * B_nb2))[tid];
+        const float B1 = ((global const float *)(B_seq + (ulong)t * B_nb2))[tid + 64];
+        const float C0 = ((global const float *)(C_seq + (ulong)t * C_nb2))[tid];
+        const float C1 = ((global const float *)(C_seq + (ulong)t * C_nb2))[tid + 64];
+
+        state0 = state0 * dA + B0 * x_dt;
+        state1 = state1 * dA + B1 * x_dt;
+        const float partial = state0 * C0 + state1 * C1;
+
+        const float sum = sub_group_reduce_add(partial);
+        if (tid == 0) {
+            y_seq[(ulong)t * y_dim_total + (ulong)head_id * head_dim + dim_id] = sum;
+        }
+    }
+
+    s_warp[tid]      = state0;
+    s_warp[tid + 64] = state1;
+}
+
+// d_state = 256 (Falcon-H1). WG = 64 threads, each holds 4 state elements.
+REQD_SUBGROUP_SIZE_64
+kernel void kernel_ssm_scan_f32_mamba2_d256(
+    global const char * src0_base, ulong src0_off,
+    global const char * src1_base, ulong src1_off,
+    global const char * src2_base, ulong src2_off,
+    global const char * src3_base, ulong src3_off,
+    global const char * src4_base, ulong src4_off,
+    global const char * src5_base, ulong src5_off,
+    global const char * src6_base, ulong src6_off,
+    global       char * dst_base,  ulong dst_off,
+    ulong s0_nb2, ulong s0_nb3,
+    ulong x_nb2,  ulong x_nb3,
+    ulong dt_nb1, ulong dt_nb2,
+    ulong A_nb1,
+    ulong B_nb2,  ulong B_nb3,
+    ulong C_nb2,  ulong C_nb3,
+    ulong s_off_bytes,
+    int   head_dim, int n_head, int n_group, int n_tokens
+) {
+    const int d_state = 256;
+
+    const int tid     = (int) get_local_id(0);
+    const int wg_x    = (int) get_group_id(0);
+    const int seq_id  = (int) get_group_id(1);
+
+    const int head_id = wg_x / head_dim;
+    const int dim_id  = wg_x - head_id * head_dim;
+    const int g       = head_id / (n_head / n_group);
+
+    src0_base += src0_off;
+    src1_base += src1_off;
+    src2_base += src2_off;
+    src3_base += src3_off;
+    src4_base += src4_off;
+    src5_base += src5_off;
+    src6_base += src6_off;
+    dst_base  += dst_off;
+
+    const int seq_slot = ((global const int *) src6_base)[seq_id];
+
+    const ulong state_base_off = (ulong)seq_slot * s0_nb3 + (ulong)head_id * s0_nb2
+                                + (ulong)dim_id * d_state * sizeof(float);
+    global const float * s0_warp = (global const float *)(src0_base + state_base_off);
+    const ulong state_out_off = (ulong)seq_id * s0_nb3 + (ulong)head_id * s0_nb2
+                              + (ulong)dim_id * d_state * sizeof(float);
+    global float * s_warp = (global float *)(dst_base + s_off_bytes + state_out_off);
+
+    global const char * x_seq  = src1_base + (ulong)seq_id * x_nb3;
+    global const char * dt_seq = src2_base + (ulong)seq_id * dt_nb2;
+    global const char * B_seq  = src4_base + (ulong)seq_id * B_nb3 + (ulong)g * d_state * sizeof(float);
+    global const char * C_seq  = src5_base + (ulong)seq_id * C_nb3 + (ulong)g * d_state * sizeof(float);
+
+    const ulong y_dim_total = (ulong)n_head * head_dim;
+    global float * y_seq = (global float *)dst_base
+                           + (ulong)seq_id * (ulong)n_tokens * y_dim_total;
+
+    const float A_val = ((global const float *)src3_base)[(ulong)head_id * A_nb1 / sizeof(float)];
+
+    // c_factor = 4: each thread owns 4 state elements.
+    float state0 = s0_warp[tid];
+    float state1 = s0_warp[tid + 64];
+    float state2 = s0_warp[tid + 128];
+    float state3 = s0_warp[tid + 192];
+
+    for (int t = 0; t < n_tokens; ++t) {
+        const float dt_h        = ((global const float *)(dt_seq + (ulong)t * dt_nb1))[head_id];
+        const float dt_softplus = softplus_f32(dt_h);
+        const float dA          = exp(dt_softplus * A_val);
+        const float x_val       = ((global const float *)(x_seq + (ulong)t * x_nb2))[(ulong)head_id * head_dim + dim_id];
+        const float x_dt        = x_val * dt_softplus;
+
+        global const float * B_t = (global const float *)(B_seq + (ulong)t * B_nb2);
+        global const float * C_t = (global const float *)(C_seq + (ulong)t * C_nb2);
+
+        const float B0 = B_t[tid];
+        const float B1 = B_t[tid + 64];
+        const float B2 = B_t[tid + 128];
+        const float B3 = B_t[tid + 192];
+        const float C0 = C_t[tid];
+        const float C1 = C_t[tid + 64];
+        const float C2 = C_t[tid + 128];
+        const float C3 = C_t[tid + 192];
+
+        state0 = state0 * dA + B0 * x_dt;
+        state1 = state1 * dA + B1 * x_dt;
+        state2 = state2 * dA + B2 * x_dt;
+        state3 = state3 * dA + B3 * x_dt;
+        const float partial = state0 * C0 + state1 * C1 + state2 * C2 + state3 * C3;
+
+        const float sum = sub_group_reduce_add(partial);
+        if (tid == 0) {
+            y_seq[(ulong)t * y_dim_total + (ulong)head_id * head_dim + dim_id] = sum;
+        }
+    }
+
+    s_warp[tid]       = state0;
+    s_warp[tid + 64]  = state1;
+    s_warp[tid + 128] = state2;
+    s_warp[tid + 192] = state3;
+}
diff --git src/ggml-openvino/ggml-openvino.cpp src/ggml-openvino/ggml-openvino.cpp
index cac83a1b..e299e16c 100644
--- src/ggml-openvino/ggml-openvino.cpp
+++ src/ggml-openvino/ggml-openvino.cpp
@@ -1227,6 +1227,10 @@ static bool is_op_unsupported_case(const ggml_tensor * op) {
         const int32_t * op_params = op->op_params;
         const int n_dims = op_params[1];
         const int mode = op_params[2];
+        if (op_params[15] != 0) {
+            // FIXME: support ggml_rope_set_offset
+            return true;
+        }
         if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_IMROPE) {
             // GGML_LOG_WARN("OpenVINO backend does not support ROPE with mode %d\n", mode);
             return true;
diff --git src/ggml-rpc/ggml-rpc.cpp src/ggml-rpc/ggml-rpc.cpp
index e9de0d0a..9d480226 100644
--- src/ggml-rpc/ggml-rpc.cpp
+++ src/ggml-rpc/ggml-rpc.cpp
@@ -47,7 +47,7 @@ struct rpc_tensor {
     uint64_t data;
     char name[GGML_MAX_NAME];
 
-    char padding[4];
+    int32_t use_count;
 };
 
 static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");
@@ -447,7 +447,7 @@ static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
 
     // Avoid sending uninitialized data over the wire
     memset(result.name, 0, sizeof(result.name));
-    memset(result.padding, 0, sizeof(result.padding));
+    result.use_count = 0;
 
     snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
     return result;
@@ -675,7 +675,7 @@ static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
     // this is no-op because we don't have any async operations
 }
 
-static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
+static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
     if (tensor == nullptr) {
         return;
     }
@@ -684,10 +684,15 @@ static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors,
     }
     visited.insert(tensor);
     for (int i = 0; i < GGML_MAX_SRC; i++) {
-        add_tensor(tensor->src[i], tensors, visited);
+        add_tensor(tensor->src[i], cgraph, tensors, visited);
     }
-    add_tensor(tensor->view_src, tensors, visited);
-    tensors.push_back(serialize_tensor(tensor));
+    add_tensor(tensor->view_src, cgraph, tensors, visited);
+    rpc_tensor result = serialize_tensor(tensor);
+    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
+    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
+        result.use_count = cgraph->use_counts[hash_pos];
+    }
+    tensors.push_back(result);
 }
 
 static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
@@ -695,7 +700,7 @@ static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::ve
     std::vector<rpc_tensor> tensors;
     std::unordered_set<ggml_tensor*> visited;
     for (uint32_t i = 0; i < n_nodes; i++) {
-        add_tensor(cgraph->nodes[i], tensors, visited);
+        add_tensor(cgraph->nodes[i], cgraph, tensors, visited);
     }
     // serialization format:
     // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
@@ -1451,6 +1456,10 @@ bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
             GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
             return false;
         }
+        if (graph->nodes[i] != nullptr) {
+            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
+            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
+        }
     }
     ggml_status status = ggml_backend_graph_compute(backends[device], graph);
     GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
diff --git src/ggml-sycl/fwht.cpp src/ggml-sycl/fwht.cpp
new file mode 100644
index 00000000..2312b3d1
--- /dev/null
+++ src/ggml-sycl/fwht.cpp
@@ -0,0 +1,119 @@
+#include "fwht.hpp"
+
+#include <cmath>
+
+template <int N>
+static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
+                        const float scale, const sycl::nd_item<2> & item) {
+    const sycl::sub_group sg = item.get_sub_group();
+
+    const int64_t r = item.get_global_id(0);
+    if (r >= n_rows) {
+        return;
+    }
+
+    src += r * N;
+    dst += r * N;
+
+    constexpr int el_w = N / WARP_SIZE;
+    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");
+
+    float     reg[el_w];
+    const int lane = sg.get_local_linear_id();
+
+#pragma unroll
+    for (int i = 0; i < el_w; ++i) {
+        reg[i] = src[i * WARP_SIZE + lane] * scale;
+    }
+
+    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
+    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
+#pragma unroll
+    for (int h = 1; h < WARP_SIZE; h *= 2) {
+#pragma unroll
+        for (int j = 0; j < el_w; ++j) {
+            const float val  = reg[j];
+            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);
+
+            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
+        }
+    }
+
+    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
+    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
+#pragma unroll
+    for (int h = WARP_SIZE; h < N; h *= 2) {
+        const int step = h / WARP_SIZE;
+#pragma unroll
+        for (int j = 0; j < el_w; j += 2 * step) {
+#pragma unroll
+            for (int k = 0; k < step; ++k) {
+                const float x = reg[j + k];
+                const float y = reg[j + k + step];
+
+                reg[j + k]        = x + y;
+                reg[j + k + step] = x - y;
+            }
+        }
+    }
+
+#pragma unroll
+    for (int i = 0; i < el_w; ++i) {
+        dst[i * WARP_SIZE + lane] = reg[i];
+    }
+}
+
+template <int N>
+static void launch_fwht(const float * src, float * dst, const int64_t n_rows, const float scale,
+                        dpct::queue_ptr stream) {
+    constexpr int rows_per_block = 4;
+
+    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
+
+    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
+    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
+    const sycl::range<2> local(rows_per_block, WARP_SIZE);
+
+    stream->parallel_for(sycl::nd_range<2>(global, local),
+                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
+                             fwht_kernel<N>(src, dst, n_rows, scale, item);
+                         });
+}
+
+bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
+    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
+        return false;
+    }
+    if (!ggml_are_same_shape(src, dst)) {
+        return false;
+    }
+    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
+        return false;
+    }
+
+    const int     n    = (int) src->ne[0];
+    const int64_t rows = ggml_nrows(src);
+
+    const float *   src_d  = (const float *) src->data;
+    float *         dst_d  = (float *) dst->data;
+    dpct::queue_ptr stream = ctx.stream();
+
+    const float scale = 1.0f / std::sqrt((float) n);
+
+    switch (n) {
+        case 64:
+            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
+            return true;
+        case 128:
+            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
+            return true;
+        case 256:
+            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
+            return true;
+        case 512:
+            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
+            return true;
+        default:
+            return false;
+    }
+}
diff --git src/ggml-sycl/fwht.hpp src/ggml-sycl/fwht.hpp
new file mode 100644
index 00000000..cd238cfa
--- /dev/null
+++ src/ggml-sycl/fwht.hpp
@@ -0,0 +1,12 @@
+#ifndef GGML_SYCL_FWHT_HPP
+#define GGML_SYCL_FWHT_HPP
+
+#include "common.hpp"
+
+// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
+// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
+// shape is not one this can serve, in which case the caller must fall through to the
+// ordinary mat-mul dispatch.
+bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
+
+#endif  // GGML_SYCL_FWHT_HPP
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index 5416d4f0..c7434a6b 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -58,6 +58,7 @@
 #include "ggml-sycl/backend.hpp"
 #include "ggml-sycl/common.hpp"
 #include "ggml-sycl/element_wise.hpp"
+#include "ggml-sycl/fwht.hpp"
 #include "ggml-sycl/gemm.hpp"
 #include "ggml-sycl/getrows.hpp"
 #include "ggml-sycl/norm.hpp"
@@ -108,7 +109,14 @@ int g_ggml_sycl_enable_host_pinned_mem = 1;
 static ggml_sycl_device_info ggml_sycl_init() {
     ggml_sycl_device_info info = {};
 
-    info.device_count = dpct::dev_mgr::instance().device_count();
+    // Do not hard crash when there exists no SYCL devices.
+    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
+    try {
+        info.device_count = dpct::dev_mgr::instance().device_count();
+    } catch (sycl::exception const &exc) {
+        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
+        info.device_count = 0;
+    }
     if (info.device_count == 0) {
         GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
         return info;
@@ -4473,6 +4481,18 @@ static bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor *
 
 static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
     scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
+
+    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
+    // kernel.
+    //
+    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
+    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
+    // asserts GGML_OP_MUL_MAT for the same reason.
+    if (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
+        ggml_sycl_op_fwht(ctx, src1, dst)) {
+        return;
+    }
+
     const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
     int64_t min_compute_capability = INT_MAX;
 
@@ -6222,6 +6242,8 @@ static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, cons
         }
         case GGML_OP_ROPE:
         case GGML_OP_ROPE_BACK:
+            // FIXME: support ggml_rope_set_offset
+            return ((const int32_t *) op->op_params)[15] == 0;
         case GGML_OP_IM2COL:
         case GGML_OP_IM2COL_3D:
         case GGML_OP_UPSCALE:
diff --git src/ggml-vulkan/CMakeLists.txt src/ggml-vulkan/CMakeLists.txt
index 1dc6a145..e733ad5c 100644
--- src/ggml-vulkan/CMakeLists.txt
+++ src/ggml-vulkan/CMakeLists.txt
@@ -200,8 +200,11 @@ if (Vulkan_FOUND)
     set (_ggml_vk_header     "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp")
     set (_ggml_vk_input_dir  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders")
     set (_ggml_vk_output_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv")
+    set (_ggml_vk_generated_shader_files ${_ggml_vk_header})
 
     file(GLOB _ggml_vk_shader_files CONFIGURE_DEPENDS "${_ggml_vk_input_dir}/*.comp")
+    set_source_files_properties(${_ggml_vk_shader_files} PROPERTIES HEADER_FILE_ONLY TRUE)
+    target_sources(ggml-vulkan PRIVATE ${_ggml_vk_shader_files})
 
     # Because external projects do not provide source-level tracking,
     # the vulkan-shaders-gen sources need to be explicitly added to
@@ -241,8 +244,11 @@ if (Vulkan_FOUND)
             COMMENT "Generate vulkan shaders for ${file}"
         )
         target_sources(ggml-vulkan PRIVATE ${_ggml_vk_target_cpp})
+        list(APPEND _ggml_vk_generated_shader_files ${_ggml_vk_target_cpp})
     endforeach()
 
+    source_group("Vulkan shaders" FILES ${_ggml_vk_shader_files})
+    source_group("Generated Vulkan shaders" FILES ${_ggml_vk_generated_shader_files})
 else()
     message(WARNING "Vulkan not found")
 endif()
diff --git src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan.cpp
index 585e10d4..f6cbaecb 100644
--- src/ggml-vulkan/ggml-vulkan.cpp
+++ src/ggml-vulkan/ggml-vulkan.cpp
@@ -913,6 +913,7 @@ struct vk_device_struct {
     vk_pipeline pipeline_quantize_q8_1_x4;
 
     vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
+    vk_pipeline pipeline_dequant_transpose[GGML_TYPE_COUNT]; // fused dequant+transpose for FA quant-KV
     vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
     vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
     vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT];
@@ -962,6 +963,7 @@ struct vk_device_struct {
     vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_COUNT];
     vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_COUNT];
     vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
+    vk_pipeline pipeline_cpy_transpose_02_16, pipeline_cpy_transpose_02_32;
     // [src0 0=fp32,1=fp16][dst]
     vk_pipeline pipeline_set_rows_i32[2][GGML_TYPE_COUNT];
     vk_pipeline pipeline_set_rows_i64[2][GGML_TYPE_COUNT];
@@ -1644,6 +1646,7 @@ struct vk_op_rope_push_constants {
     uint32_t rope_mode;
     uint32_t nrows;
     uint32_t n_dims;
+    uint32_t n_offs;
     float freq_scale;
     float freq_base;
     float ext_factor;
@@ -3382,10 +3385,10 @@ static void ggml_vk_queue_command_pools_cleanup(vk_device& device) {
     // Arbitrary frequency to cleanup/reuse command buffers
     static constexpr uint32_t cleanup_frequency = 10;
 
-    if (device->compute_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
+    if (device->compute_queue && device->compute_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
         ggml_vk_command_pool_cleanup(device, device->compute_queue->cmd_pool);
     }
-    if (device->transfer_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
+    if (device->transfer_queue && device->transfer_queue->cmd_pool.buffers_in_use() >= cleanup_frequency) {
         ggml_vk_command_pool_cleanup(device, device->transfer_queue->cmd_pool);
     }
 }
@@ -5389,6 +5392,7 @@ static void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
@@ -5525,6 +5529,8 @@ static void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
 
     ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
+    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
 
     ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
     ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
@@ -8931,6 +8937,18 @@ static vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, const
         }
     }
 
+    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
+    bool transpose02 = dst && !contig && src->nb[2] == ggml_type_size(to) &&
+                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);
+
+    if (transpose02 && src->type == to) {
+        if (ggml_type_size(to) == 4) {
+            return ctx->device->pipeline_cpy_transpose_02_32;
+        } else if (ggml_type_size(to) == 2) {
+            return ctx->device->pipeline_cpy_transpose_02_16;
+        }
+    }
+
     if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F32) {
         if (contig) {
             return ctx->device->pipeline_contig_cpy_f32_f32;
@@ -10807,9 +10825,32 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
 
     const bool f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;
 
+    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
+    auto is_dense_kv_cache = [](const ggml_tensor * t) {
+        return t->nb[0] == ggml_type_size(t->type) &&
+               t->nb[2] == ggml_row_size(t->type, t->ne[0]) &&
+               t->nb[1] == t->nb[2] * t->ne[2] &&
+               t->nb[3] == t->nb[1] * t->ne[1];
+    };
+    const bool k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
+    const bool v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
+    const bool use_dequant_kv = k_quant && v_quant && neq1 >= 64 &&
+                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
+                                (uint64_t)ggml_nelements(k) * sizeof(ggml_fp16_t) <= ctx->device->properties.limits.maxStorageBufferRange &&
+                                (uint64_t)ggml_nelements(v) * sizeof(ggml_fp16_t) <= ctx->device->properties.limits.maxStorageBufferRange &&
+                                ctx->device->pipeline_dequant_transpose[k->type] != nullptr &&
+                                ctx->device->pipeline_dequant_transpose[v->type] != nullptr &&
+                                // coopmat2 path does not benefit from the f16 scratch
+                                !ctx->device->coopmat2 &&
+                                // Intel Xe1 regresses, see PR 25494
+                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
+                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
+    const ggml_type k_type_eff = use_dequant_kv ? GGML_TYPE_F16 : k->type;
+    const ggml_type v_type_eff = use_dequant_kv ? GGML_TYPE_F16 : v->type;
+
     // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
     // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
-    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k->type, v->type, f32acc);
+    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
     const uint32_t max_gqa = std::min(tuning_params.block_rows, 32u);
 
     if (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
@@ -10822,7 +10863,7 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
         workgroups_y /= gqa_ratio;
     }
 
-    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k->type, v->type, f32acc);
+    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);
 
     const uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_size(q->type));
     uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_size(k->type));
@@ -10836,6 +10877,17 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
         v_stride /= 4;
     }
 
+    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
+    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
+    if (use_dequant_kv) {
+        k_stride = HSK;
+        v_stride = HSV;
+        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * sizeof(ggml_fp16_t));
+        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * sizeof(ggml_fp16_t));
+        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * sizeof(ggml_fp16_t));
+        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * sizeof(ggml_fp16_t));
+    }
+
     const uint32_t alignment = tuning_params.block_cols;
     bool aligned = (KV % alignment) == 0 &&
                    // the "aligned" shader variant will forcibly align strides, for performance
@@ -10862,7 +10914,7 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
     bool use_mask_opt = mask && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.block_cols * 16
                         && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
     vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
-                                                                   mask != nullptr, use_mask_opt, logit_softcap != 0, k->type, v->type);
+                                                                   mask != nullptr, use_mask_opt, logit_softcap != 0, k_type_eff, v_type_eff);
 
     vk_pipeline pipeline = nullptr;
 
@@ -10966,6 +11018,34 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
     vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
     vk_subbuffer mask_opt_buf = use_mask_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
 
+    if (use_dequant_kv) {
+        const uint64_t fp = sizeof(ggml_fp16_t);
+        const uint64_t k_f16_sz = (uint64_t)ggml_nelements(k) * fp;
+        const uint64_t v_f16_sz = (uint64_t)ggml_nelements(v) * fp;
+        if (ctx->prealloc_size_x < k_f16_sz + v_f16_sz) {
+            ctx->prealloc_size_x = k_f16_sz + v_f16_sz;
+            ggml_vk_preallocate_buffers(ctx, subctx);
+        }
+        vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
+        vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
+        ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
+        ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
+        if (ctx->prealloc_x_need_sync) {
+            ggml_vk_sync_buffers(ctx, subctx);
+        }
+        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,        k_f16_sz };
+        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_f16_sz, v_f16_sz };
+        const uint32_t k_nel = (uint32_t)ggml_nelements(k);
+        const uint32_t v_nel = (uint32_t)ggml_nelements(v);
+        { const std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
+          ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
+        { const std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
+          ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
+        ggml_vk_sync_buffers(ctx, subctx);
+        k_buf = k_dst;
+        v_buf = v_dst;
+    }
+
     uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | n_head_log2;
 
     if (use_mask_opt)
@@ -10995,8 +11075,8 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
                                               (uint32_t)nev2, (uint32_t)nev3,
                                               nem1, nem2, nem3,
                                               q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
-                                              k_stride, (uint32_t)nbk2, (uint32_t)nbk3,
-                                              v_stride, (uint32_t)nbv2, (uint32_t)nbv3,
+                                              k_stride, nbk2_eff, nbk3_eff,
+                                              v_stride, nbv2_eff, nbv3_eff,
                                               scale, max_bias, logit_softcap,
                                               mask_n_head_log2, m0, m1,
                                               gqa_ratio, split_kv, split_k };
@@ -11038,6 +11118,10 @@ static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx
                                     {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf},
                                     pc, { workgroups_x, workgroups_y, workgroups_z });
     }
+
+    if (use_dequant_kv) {
+        ctx->prealloc_x_need_sync = true;
+    }
 }
 
 static vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
@@ -12192,7 +12276,16 @@ static void ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, co
                 elements = { ne, 1, 1 };
             }
 
-            if (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
+            if (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
+                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
+                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
+                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
+                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
+                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
+                elements[0] = std::min(elements[0], ctx->device->properties.limits.maxComputeWorkGroupCount[0]);
+                elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
+                elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
+            } else if (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                 pipeline == ctx->device->pipeline_cpy_transpose_16) {
                 // 32x32 tiles
                 elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
@@ -13120,6 +13213,7 @@ static uint32_t ggml_vk_rms_partials_size(ggml_backend_vk_context * ctx, const g
 static vk_op_rope_push_constants ggml_vk_make_rope_constants(const ggml_tensor *dst, const ggml_tensor *src0, const bool has_ff, bool backprop, const uint32_t set_rows_stride) {
     const int n_dims        = ((const int32_t *) dst->op_params)[1];
     const int mode          = ((const int32_t *) dst->op_params)[2];
+    const int n_offs        = ((const int32_t *) dst->op_params)[15];
     // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
     const int n_ctx_orig    = ((const int32_t *) dst->op_params)[4];
     const float freq_base   = ((const float *)   dst->op_params)[5];
@@ -13149,7 +13243,7 @@ static vk_op_rope_push_constants ggml_vk_make_rope_constants(const ggml_tensor *
     uint32_t nb13 = dst->nb[3] / ggml_type_size(dst->type);
 
     vk_op_rope_push_constants rope {
-        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, freq_scale,
+        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
         freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
         { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, set_rows_stride,
 
@@ -19195,6 +19289,10 @@ static void ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph *
                     tensor_clone = ggml_rope_ext_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                 }
             }
+            const int n_offs = ((int32_t *) tensor->op_params)[15];
+            if (n_offs != 0) {
+                tensor_clone = ggml_rope_set_offset(tensor_clone, n_offs);
+            }
         } else if (tensor->op == GGML_OP_UNARY) {
             switch (ggml_get_unary_op(tensor)) {
             case GGML_UNARY_OP_EXP:
diff --git src/ggml-vulkan/vulkan-shaders/copy_transpose_02.comp src/ggml-vulkan/vulkan-shaders/copy_transpose_02.comp
new file mode 100644
index 00000000..5a3d66da
--- /dev/null
+++ src/ggml-vulkan/vulkan-shaders/copy_transpose_02.comp
@@ -0,0 +1,61 @@
+#version 450
+
+#include "types.glsl"
+#include "generic_unary_head.glsl"
+
+// workgroup does 32x32 tile, but uses 32x8 threads
+#define TILE_DIM 32
+layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in;
+
+// +1 padding avoids shared-memory bank conflicts on the transposed read
+shared uint sh[TILE_DIM][TILE_DIM + 1];
+
+void iter(uvec3 wg_id) {
+    const uint tile_i0 = wg_id.x;   // tiles dst ne10 (== src ne00)
+    const uint tile_i2 = wg_id.y;   // tiles dst ne12 (== src ne02)
+
+    const uint tid_col = gl_LocalInvocationID.x;
+    const uint tid_row = gl_LocalInvocationID.y;
+
+    const uint i1 = wg_id.z % p.ne11;
+    const uint i3 = wg_id.z / p.ne11;
+    const uint i01 = i1;
+    const uint i03 = i3;
+
+    [[unroll]] for (uint y = 0; y < 4; ++y) {
+        const uint i00 = tile_i0 * TILE_DIM + tid_row + 8 * y;
+        const uint i02 = tile_i2 * TILE_DIM + tid_col;
+        if (i00 < p.ne00 && i01 < p.ne01 && i02 < p.ne02 && i03 < p.ne03) {
+            const uint src_idx = i00 * p.nb00 + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
+            sh[tid_row + 8 * y][tid_col] = uint(data_a[get_aoffset() + src_idx]);
+        }
+    }
+
+    barrier();
+
+    [[unroll]] for (uint y = 0; y < 4; ++y) {
+        const uint i0 = tile_i0 * TILE_DIM + tid_col;
+        const uint i2 = tile_i2 * TILE_DIM + tid_row + 8 * y;
+        if (i0 < p.ne10 && i1 < p.ne11 && i2 < p.ne12 && i3 < p.ne13) {
+            const uint dst_idx = i0 * p.nb10 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
+            data_d[get_doffset() + dst_idx] = D_TYPE(sh[tid_col][tid_row + 8 * y]);
+        }
+    }
+}
+
+#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
+
+void main() {
+    bool need_barrier = false;
+    for (uint z = gl_WorkGroupID.z; z < p.ne11 * p.ne13; z += gl_NumWorkGroups.z) {
+        for (uint y = gl_WorkGroupID.y; y < CEIL_DIV(p.ne12, TILE_DIM); y += gl_NumWorkGroups.y) {
+            for (uint x = gl_WorkGroupID.x; x < CEIL_DIV(p.ne10, TILE_DIM); x += gl_NumWorkGroups.x) {
+                if (need_barrier) {
+                    barrier();
+                }
+                need_barrier = true;
+                iter(uvec3(x, y, z));
+            }
+        }
+    }
+}
diff --git src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp
index 10844ddf..3b3fbbe8 100644
--- src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp
+++ src/ggml-vulkan/vulkan-shaders/dequant_q8_0.comp
@@ -18,7 +18,18 @@ void main() {
         return;
     }
 
+#ifdef DEQUANT_TRANSPOSE
+    // read [HS, NH, KV, NS], write [HS, KV, NH, NS]
+    const uint HS = p.M, NH = p.K, KVn = p.stride_a;
+    const uint e0 = ib * 32;
+    const uint b_idx = (e0 % HS)
+                     + ((e0 / (HS * NH)) % KVn) * HS
+                     + ((e0 / HS) % NH) * (HS * KVn)
+                     + (e0 / (HS * NH * KVn)) * (HS * KVn * NH)
+                     + 16 * il;
+#else
     const uint b_idx = 1024*i + 32*ir + 16*il;
+#endif
 
     const float d = float(data_a[ib].d);
 
diff --git src/ggml-vulkan/vulkan-shaders/flash_attn.comp src/ggml-vulkan/vulkan-shaders/flash_attn.comp
index 6c264c78..0c1b6d06 100644
--- src/ggml-vulkan/vulkan-shaders/flash_attn.comp
+++ src/ggml-vulkan/vulkan-shaders/flash_attn.comp
@@ -121,13 +121,13 @@ void main() {
         const uint buf_ib = r * qf_stride + d / 8;
         const uint buf_iqs = d % 8;
 
-        FLOAT_TYPEV4 vals = is_in_bounds ? FLOAT_TYPEV4(data_qv4[q_offset / 4 + (i * Br + r) * q_stride / 4 + d] * p.scale) : FLOAT_TYPEV4(0.0f);
-        const FLOAT_TYPEV4 abs_vals = abs(vals);
+        vec4 vals = is_in_bounds ? data_qv4[q_offset / 4 + (i * Br + r) * q_stride / 4 + d] * p.scale : vec4(0.0f);
+        const vec4 abs_vals = abs(vals);
 
-        const FLOAT_TYPE thread_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));
-        const FLOAT_TYPE amax = subgroupClusteredMax(thread_max, 8);
-        const FLOAT_TYPE qd = amax / FLOAT_TYPE(127.0);
-        const FLOAT_TYPE qd_inv = qd != FLOAT_TYPE(0.0) ? FLOAT_TYPE(1.0) / qd : FLOAT_TYPE(0.0);
+        const float thread_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));
+        const float amax = subgroupClusteredMax(thread_max, 8);
+        const float qd = amax / 127.0f;
+        const float qd_inv = qd != 0.0f ? 1.0f / qd : 0.0f;
         vals = round(vals * qd_inv);
 
         Qf[buf_ib].qs[buf_iqs] = pack32(i8vec4(vals));
@@ -136,11 +136,11 @@ void main() {
         // the row-sum scaled by qd, used in k_dot_correction.
         if (FaTypeK == FA_TYPE_Q8_0) {
             if (buf_iqs == 0) {
-                Qf[buf_ib].ds = FLOAT_TYPEV2(qd, 0.0);
+                Qf[buf_ib].ds = FLOAT_TYPEV2(qd, 0.0f);
             }
         } else {
-            const FLOAT_TYPE thread_sum = vals.x + vals.y + vals.z + vals.w;
-            const FLOAT_TYPE sum = subgroupClusteredAdd(thread_sum, 8);
+            const float thread_sum = vals.x + vals.y + vals.z + vals.w;
+            const float sum = subgroupClusteredAdd(thread_sum, 8);
 
             if (buf_iqs == 0) {
                 Qf[buf_ib].ds = FLOAT_TYPEV2(qd, sum * qd);
diff --git src/ggml-vulkan/vulkan-shaders/rope_funcs.glsl src/ggml-vulkan/vulkan-shaders/rope_funcs.glsl
index 03358793..feb55b20 100644
--- src/ggml-vulkan/vulkan-shaders/rope_funcs.glsl
+++ src/ggml-vulkan/vulkan-shaders/rope_funcs.glsl
@@ -50,19 +50,21 @@ void rope_norm(const uint i0, const uint i1, const uint i2, const uint i3, rope_
     }
     idst += p.d_offset;
 
-    if (i0 >= p.n_dims) {
+    if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
         rope_data_d[idst + 0] = ROPE_D_TYPE(rope_data_a[ix + 0]);
         rope_data_d[idst + 1] = ROPE_D_TYPE(rope_data_a[ix + 1]);
 
         return;
     }
 
-    const float theta_base = rope_data_pos[i2] * pow(p.theta_scale, i0/2.0f);
+    const uint iw = i0 - p.n_offs; // relative idx
 
-    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;
+    const float theta_base = rope_data_pos[i2] * pow(p.theta_scale, iw/2.0f);
+
+    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;
 
     float cos_theta, sin_theta;
-    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);
+    rope_yarn(theta_base / freq_factor, iw, cos_theta, sin_theta, p);
 
     const float x0 = float(rope_data_a[ix + 0]);
     const float x1 = float(rope_data_a[ix + 1]);
@@ -87,25 +89,28 @@ void rope_neox(const uint i0, const uint i1, const uint i2, const uint i3, rope_
     }
     idst += p.d_offset;
 
-    if (i0 >= p.n_dims) {
+    if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
         rope_data_d[idst + i0/2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 0]);
         rope_data_d[idst + i0/2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 1]);
 
         return;
     }
 
-    const float theta_base = rope_data_pos[i2] * pow(p.theta_scale, i0/2.0f);
+    const uint iw = i0 - p.n_offs; // relative idx
 
-    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;
+    const float theta_base = rope_data_pos[i2] * pow(p.theta_scale, iw/2.0f);
+
+    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;
 
     float cos_theta, sin_theta;
-    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);
+    rope_yarn(theta_base / freq_factor, iw, cos_theta, sin_theta, p);
 
-    const float x0 = float(rope_data_a[ix + 0]);
-    const float x1 = float(rope_data_a[ix + p.n_dims/2]);
+    // idst/ix point at channel i0/2; the first channel of the rotated pair is p.n_offs + iw/2 = i0/2 + p.n_offs/2
+    const float x0 = float(rope_data_a[ix + p.n_offs/2 + 0]);
+    const float x1 = float(rope_data_a[ix + p.n_offs/2 + p.n_dims/2]);
 
-    rope_data_d[idst + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
-    rope_data_d[idst + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
+    rope_data_d[idst + p.n_offs/2 + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
+    rope_data_d[idst + p.n_offs/2 + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
 }
 
 
@@ -125,53 +130,56 @@ void rope_multi(const uint i0, const uint i1, const uint i2, const uint i3, rope
     }
     idst += p.d_offset;
 
-    if (i0 >= p.n_dims) {
+    if (i0 < p.n_offs || i0 >= p.n_offs + p.n_dims) {
         rope_data_d[idst + i0/2 + 0] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 0]);
         rope_data_d[idst + i0/2 + 1] = ROPE_D_TYPE(rope_data_a[ix + i0/2 + 1]);
 
         return;
     }
 
+    const uint iw = i0 - p.n_offs; // relative idx
+
     const int sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
     const int sec_w = p.sections[1] + p.sections[0];
-    const uint sector = (i0 / 2) % sect_dims;
+    const uint sector = (iw / 2) % sect_dims;
 
     float theta_base = 0.0;
     if (p.is_imrope != 0) {
         if (sector % 3 == 1 && sector < 3 * p.sections[1]) {
-            theta_base = rope_data_pos[i2 + p.ne02 * 1]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2 + p.ne02 * 1]*pow(p.theta_scale, iw/2.0f);
         } else if (sector % 3 == 2 && sector < 3 * p.sections[2]) {
-            theta_base = rope_data_pos[i2 + p.ne02 * 2]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2 + p.ne02 * 2]*pow(p.theta_scale, iw/2.0f);
         } else if (sector % 3 == 0 && sector < 3 * p.sections[0]) {
-            theta_base = rope_data_pos[i2]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2]*pow(p.theta_scale, iw/2.0f);
         } else {
-            theta_base = rope_data_pos[i2 + p.ne02 * 3]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2 + p.ne02 * 3]*pow(p.theta_scale, iw/2.0f);
         }
     } else {
         if (sector < p.sections[0]) {
-            theta_base = rope_data_pos[i2]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2]*pow(p.theta_scale, iw/2.0f);
         }
         else if (sector >= p.sections[0] && sector < sec_w) {
-            theta_base = rope_data_pos[i2 + p.ne02 * 1]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2 + p.ne02 * 1]*pow(p.theta_scale, iw/2.0f);
         }
         else if (sector >= sec_w && sector < sec_w + p.sections[2]) {
-            theta_base = rope_data_pos[i2 + p.ne02 * 2]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2 + p.ne02 * 2]*pow(p.theta_scale, iw/2.0f);
         }
         else if (sector >= sec_w + p.sections[2]) {
-            theta_base = rope_data_pos[i2 + p.ne02 * 3]*pow(p.theta_scale, i0/2.0f);
+            theta_base = rope_data_pos[i2 + p.ne02 * 3]*pow(p.theta_scale, iw/2.0f);
         }
     }
 
-    const float freq_factor = p.has_ff != 0 ? rope_data_ff[i0/2] : 1.0f;
+    const float freq_factor = p.has_ff != 0 ? rope_data_ff[iw/2] : 1.0f;
 
     float cos_theta, sin_theta;
-    rope_yarn(theta_base / freq_factor, i0, cos_theta, sin_theta, p);
+    rope_yarn(theta_base / freq_factor, iw, cos_theta, sin_theta, p);
 
-    const float x0 = float(rope_data_a[ix + 0]);
-    const float x1 = float(rope_data_a[ix + p.n_dims/2]);
+    // idst/ix point at channel i0/2; the first channel of the rotated pair is p.n_offs + iw/2 = i0/2 + p.n_offs/2
+    const float x0 = float(rope_data_a[ix + p.n_offs/2 + 0]);
+    const float x1 = float(rope_data_a[ix + p.n_offs/2 + p.n_dims/2]);
 
-    rope_data_d[idst + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
-    rope_data_d[idst + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
+    rope_data_d[idst + p.n_offs/2 + 0]          = ROPE_D_TYPE(x0*cos_theta - x1*sin_theta);
+    rope_data_d[idst + p.n_offs/2 + p.n_dims/2] = ROPE_D_TYPE(x0*sin_theta + x1*cos_theta);
 }
 
 void rope_vision(const uint i0, const uint i1, const uint i2, const uint i3, rope_params p) {
diff --git src/ggml-vulkan/vulkan-shaders/rope_params.glsl src/ggml-vulkan/vulkan-shaders/rope_params.glsl
index 3602485b..b88a73fc 100644
--- src/ggml-vulkan/vulkan-shaders/rope_params.glsl
+++ src/ggml-vulkan/vulkan-shaders/rope_params.glsl
@@ -5,6 +5,7 @@ struct rope_params {
     uint rope_mode;
     uint nrows;
     uint n_dims;
+    uint n_offs;
     float freq_scale;
     float freq_base;
     float ext_factor;
diff --git src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
index 6c9f76af..caa0c889 100644
--- src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
+++ src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
@@ -780,6 +780,10 @@ void process_shaders() {
         if (tname != "f16" && tname != "bf16") {
             string_to_spv("dequant_" + tname, "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float16_t"}}));
         }
+        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
+        if (tname == "q8_0") {
+            string_to_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
+        }
 
         shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";
 
@@ -826,6 +830,8 @@ void process_shaders() {
 
     string_to_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
     string_to_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
+    string_to_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
+    string_to_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
 
     for (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
         string_to_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
diff --git src/ggml-webgpu/ggml-webgpu-shader-lib.hpp src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
index 0604e1c2..7a67ccf4 100644
--- src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
+++ src/ggml-webgpu/ggml-webgpu-shader-lib.hpp
@@ -954,10 +954,11 @@ struct ggml_webgpu_mul_mat_vec_pipeline_key {
     int       vectorized;
     uint32_t  num_cols;
     bool      use_mmvq;
+    bool      src_overlap;
 
     bool operator==(const ggml_webgpu_mul_mat_vec_pipeline_key & other) const {
         return src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
-               num_cols == other.num_cols && use_mmvq == other.use_mmvq;
+               num_cols == other.num_cols && use_mmvq == other.use_mmvq && src_overlap == other.src_overlap;
     }
 };
 
@@ -969,6 +970,7 @@ struct ggml_webgpu_mul_mat_vec_pipeline_key_hash {
         ggml_webgpu_hash_combine(seed, key.vectorized);
         ggml_webgpu_hash_combine(seed, key.num_cols);
         ggml_webgpu_hash_combine(seed, key.use_mmvq);
+        ggml_webgpu_hash_combine(seed, key.src_overlap);
         return seed;
     }
 };
@@ -977,6 +979,7 @@ struct ggml_webgpu_mul_mat_vec_shader_decisions {
     uint32_t wg_size;
     uint32_t outputs_per_wg;
     uint32_t vec_size;
+    bool     src_overlap = false;
 };
 
 struct ggml_webgpu_quantize_q8_pipeline_key {
@@ -998,10 +1001,11 @@ struct ggml_webgpu_mul_mat_pipeline_key {
     ggml_type src1_type;
     int       vectorized;
     int       use_subgroup_matrix;
+    bool      src_overlap;
 
     bool operator==(const ggml_webgpu_mul_mat_pipeline_key & other) const {
         return src0_type == other.src0_type && src1_type == other.src1_type && vectorized == other.vectorized &&
-               use_subgroup_matrix == other.use_subgroup_matrix;
+               use_subgroup_matrix == other.use_subgroup_matrix && src_overlap == other.src_overlap;
     }
 };
 
@@ -1012,6 +1016,7 @@ struct ggml_webgpu_mul_mat_pipeline_key_hash {
         ggml_webgpu_hash_combine(seed, key.src1_type);
         ggml_webgpu_hash_combine(seed, key.vectorized);
         ggml_webgpu_hash_combine(seed, key.use_subgroup_matrix);
+        ggml_webgpu_hash_combine(seed, key.src_overlap);
         return seed;
     }
 };
@@ -1034,6 +1039,7 @@ struct ggml_webgpu_mul_mat_shader_decisions {
     uint32_t subgroup_matrix_n;
 
     uint32_t mul_mat_wg_size;
+    bool     src_overlap = false;
 };
 
 /** MUL_MAT_ID **/
@@ -1950,7 +1956,7 @@ class ggml_webgpu_shader_lib {
         return quantize_q8_pipelines[key];
     }
 
-    webgpu_pipeline get_mul_mat_vec_pipeline(const ggml_webgpu_shader_lib_context & context) {
+    webgpu_pipeline get_mul_mat_vec_pipeline(const ggml_webgpu_shader_lib_context & context, bool src_overlap) {
         ggml_webgpu_mul_mat_vec_pipeline_key key = {};
         key.src0_type                            = context.src0->type;
         key.src1_type                            = context.src1->type;
@@ -1961,6 +1967,7 @@ class ggml_webgpu_shader_lib {
         key.num_cols   = context.dst->ne[1];
         key.use_mmvq =
             ggml_webgpu_can_use_mmvq(context.src0, context.src1, context.supports_dot_product, context.vendor);
+        key.src_overlap = src_overlap;
 
         auto it = mul_mat_vec_pipelines.find(key);
         if (it != mul_mat_vec_pipelines.end()) {
@@ -2068,6 +2075,11 @@ class ggml_webgpu_shader_lib {
             defines.push_back("Q8_1_T");
         }
 
+        if (key.src_overlap) {
+            defines.push_back("SRC_OVERLAP");
+            variant += "_src_overlap";
+        }
+
         defines.push_back(std::string("WG_SIZE=") + std::to_string(wg_size));
         defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));
         defines.push_back(context.supports_subgroups ? "USE_SUBGROUP_REDUCTION" : "USE_WORKGROUP_REDUCTION");
@@ -2089,7 +2101,7 @@ class ggml_webgpu_shader_lib {
         return mul_mat_vec_pipelines[key];
     }
 
-    webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context & context) {
+    webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context & context, bool src_overlap) {
         ggml_webgpu_mul_mat_pipeline_key key = {};
         key.src0_type                        = context.src0->type;
         key.src1_type                        = context.src1->type;
@@ -2098,6 +2110,7 @@ class ggml_webgpu_shader_lib {
                                       1 :
                                       0;
         key.use_subgroup_matrix = context.supports_subgroup_matrix;
+        key.src_overlap         = src_overlap;
 
         auto it = mul_mat_fast_pipelines.find(key);
         if (it != mul_mat_fast_pipelines.end()) {
@@ -2216,6 +2229,11 @@ class ggml_webgpu_shader_lib {
             variant += "_vectorized";
         }
 
+        if (key.src_overlap) {
+            defines.push_back("SRC_OVERLAP");
+            variant += "_src_overlap";
+        }
+
         if (!key.use_subgroup_matrix) {
             defines.push_back("WORKGROUP_SIZE_M=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_M) + "u");
             defines.push_back("WORKGROUP_SIZE_N=" + std::to_string(WEBGPU_MUL_MAT_WG_SIZE_N) + "u");
diff --git src/ggml-webgpu/ggml-webgpu.cpp src/ggml-webgpu/ggml-webgpu.cpp
index 394aeeda..4367f9a6 100644
--- src/ggml-webgpu/ggml-webgpu.cpp
+++ src/ggml-webgpu/ggml-webgpu.cpp
@@ -1628,48 +1628,65 @@ static webgpu_encoded_op ggml_webgpu_mul_mat(webgpu_context & ctx,
     // Get or create pipeline
     webgpu_pipeline                   pipeline;
     std::vector<webgpu_dispatch_desc> dispatches;
+    const bool src_overlap = ggml_webgpu_tensor_binding_overlap(ctx->global_ctx, src0, src1) && !use_mmvq;
 
     if (use_mat_vec) {
         if (use_mmvq) {
             ggml_webgpu_quantize_q8_dispatch(ctx, src0, src1, dst, dispatches);
         }
-        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx);
+        pipeline = ctx->shader_lib->get_mul_mat_vec_pipeline(shader_lib_ctx, src_overlap);
     } else {
-        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx);
+        pipeline = ctx->shader_lib->get_mul_mat_fast_pipeline(shader_lib_ctx, src_overlap);
+    }
+
+    uint32_t offset_src0   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type));
+    uint32_t offset_src1   = (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type));
+    size_t   merged_offset = 0;
+    size_t   merged_size   = 0;
+    if (src_overlap) {
+        const ggml_webgpu_merged_binding_range merged_range =
+            ggml_webgpu_tensor_merged_binding_range(ctx, { src0, src1 });
+        merged_offset = merged_range.offset;
+        merged_size   = merged_range.size;
+        offset_src0   = ggml_webgpu_tensor_merged_element_offset(src0, merged_range);
+        offset_src1   = ggml_webgpu_tensor_merged_element_offset(src1, merged_range);
     }
 
     // Build params
-    std::vector<uint32_t> params = {
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src0) / ggml_type_size(src0->type)),
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, src1) / ggml_type_size(src1->type)),
-        (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
-        (uint32_t) dst->ne[0],
-        (uint32_t) dst->ne[1],
-        (uint32_t) src0->ne[0],
-        (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
-        (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
-        (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
-        (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
-        (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
-        (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
-        (uint32_t) src0->ne[2],
-        (uint32_t) src0->ne[3],
-        (uint32_t) (src1->ne[2] / src0->ne[2]),
-        (uint32_t) (src1->ne[3] / src0->ne[3])
-    };
+    std::vector<uint32_t> params = { offset_src0,
+                                     offset_src1,
+                                     (uint32_t) (ggml_webgpu_tensor_misalignment(ctx, dst) / ggml_type_size(dst->type)),
+                                     (uint32_t) dst->ne[0],
+                                     (uint32_t) dst->ne[1],
+                                     (uint32_t) src0->ne[0],
+                                     (uint32_t) (src0->nb[1] / ggml_type_size(src0->type)),
+                                     (uint32_t) (src1->nb[1] / ggml_type_size(src1->type)),
+                                     (uint32_t) (src0->nb[2] / ggml_type_size(src0->type)),
+                                     (uint32_t) (src1->nb[2] / ggml_type_size(src1->type)),
+                                     (uint32_t) (src0->nb[3] / ggml_type_size(src0->type)),
+                                     (uint32_t) (src1->nb[3] / ggml_type_size(src1->type)),
+                                     (uint32_t) src0->ne[2],
+                                     (uint32_t) src0->ne[3],
+                                     (uint32_t) (src1->ne[2] / src0->ne[2]),
+                                     (uint32_t) (src1->ne[3] / src0->ne[3]) };
 
     // Build bind group entries
     std::vector<wgpu::BindGroupEntry> entries = {};
-
-    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
     if (use_mmvq) {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
         auto & mmvq_qq8_entry = dispatches[0].bind_group_entries[1];
         entries.push_back(ggml_webgpu_make_bind_group_entry(1, ggml_webgpu_tensor_buf(dst), mmvq_qq8_entry.offset,
                                                             mmvq_qq8_entry.size));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
+    } else if (src_overlap) {
+        entries.push_back(
+            ggml_webgpu_make_bind_group_entry(0, ggml_webgpu_tensor_buf(src0), merged_offset, merged_size));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, dst));
     } else {
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 0, src0));
         entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 1, src1));
+        entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
     }
-    entries.push_back(ggml_webgpu_make_tensor_bind_group_entry(ctx, 2, dst));
 
     // Calculate workgroup dimensions
     uint32_t       wg_x           = 1;
@@ -4455,7 +4472,9 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
             supports_op = (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32) && ggml_is_contiguous_rows(src0);
             break;
         case GGML_OP_ROPE:
-            supports_op = op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
+            // FIXME: support ggml_rope_set_offset
+            supports_op =
+                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ((const int32_t *) op->op_params)[15] == 0;
             break;
         case GGML_OP_GLU:
             switch (ggml_get_glu_op(op)) {
diff --git src/ggml-webgpu/wgsl-shaders/common_decls.tmpl src/ggml-webgpu/wgsl-shaders/common_decls.tmpl
index b0cf2853..4a500e4e 100644
--- src/ggml-webgpu/wgsl-shaders/common_decls.tmpl
+++ src/ggml-webgpu/wgsl-shaders/common_decls.tmpl
@@ -1,3 +1,7 @@
+#ifndef SRC0
+#define SRC0 src0
+#endif
+
 #ifdef BYTE_HELPERS
 fn get_byte(value: u32, index: u32) -> u32 {
     return (value >> (index * 8)) & 0xFF;
@@ -46,7 +50,7 @@ fn load_f16_as_f32_at_src(byte_offset: u32) -> f32 {
 
 #ifdef DECLARE_BYTE_LOADERS_SRC0
 fn load_u16_at_src0(byte_offset: u32) -> u32 {
-    let word = src0[byte_offset / 4u];
+    let word = SRC0[byte_offset / 4u];
     let shift = (byte_offset & 0x2u) * 8u;
     return (word >> shift) & 0xFFFFu;
 }
@@ -55,14 +59,14 @@ fn load_u16_at_src0(byte_offset: u32) -> u32 {
 // Caller extracts the 16-bit half it needs via & 0xFFFFu or >> 16u.
 // this is used in k-quants for better performance
 fn load_u32_at_src0_aligned(byte_offset: u32) -> u32 {
-    return src0[(byte_offset & ~3u) / 4u];
+    return SRC0[(byte_offset & ~3u) / 4u];
 }
 
 fn load_u32_at_src0(byte_offset: u32) -> u32 {
     let word_idx = byte_offset / 4u;
     let shift = (byte_offset & 0x3u) * 8u;
-    let lo = src0[word_idx];
-    let hi = src0[word_idx + 1u];
+    let lo = SRC0[word_idx];
+    let hi = SRC0[word_idx + 1u];
     let shifted = (lo >> shift) | (hi << (32u - shift));
     return select(shifted, lo, shift == 0u);
 }
@@ -73,7 +77,7 @@ fn load_f16_at_src0(byte_offset: u32) -> f16 {
 }
 
 fn load_f16_as_f32_at_src0(byte_offset: u32) -> f32 {
-    let word = src0[byte_offset / 4u];
+    let word = SRC0[byte_offset / 4u];
     let shift = (byte_offset & 0x2u) * 8u;
     let d_bits = (word >> shift) & 0xFFFFu;
     return unpack2x16float(d_bits)[0];
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl
index 13996ab5..44b6bb71 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl
@@ -1,3 +1,10 @@
+#ifndef SRC0
+#define SRC0 src0
+#endif
+#ifndef SRC1
+#define SRC1 src1
+#endif
+
 #ifdef VEC
 #define VEC_SIZE 4
 #define SHMEM_TYPE vec4<f16>
@@ -39,7 +46,7 @@ fn init_shmem_src0(thread_id: u32, batch_offset: u32, offset_m: u32, k_outer: u3
         let src0_idx = batch_offset + global_m * params.stride_01 + global_k;
         let src0_val = select( // taking a slight performance hit to avoid oob
             SRC0_TYPE(0.0),
-            src0[src0_idx/VEC_SIZE],
+            SRC0[src0_idx/VEC_SIZE],
             global_m < params.m && global_k < params.k);
         store_shmem(SHMEM_TYPE(src0_val), elem_idx);
     }
@@ -57,7 +64,7 @@ fn init_shmem_src1(thread_id: u32, batch_offset: u32, offset_n: u32, k_outer: u3
         let src1_idx = batch_offset + global_n * params.stride_11 + global_k;
         let src1_val = select(
             SRC1_TYPE(0.0),
-            src1[src1_idx/VEC_SIZE],
+            SRC1[src1_idx/VEC_SIZE],
             global_n < params.n && global_k < params.k);
         store_shmem(SHMEM_TYPE(src1_val), TILE_SRC0_SHMEM + elem_idx);
     }
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl
index 98bbdeb8..0e17fae1 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl
@@ -1,8 +1,12 @@
 enable f16;
 
 #define DECLARE_BYTE_LOADERS_SRC0
-#include "common_decls.tmpl"
 
+#ifdef SRC_OVERLAP
+#define SRC0 merged_src
+#define SRC1 merged_src
+#endif
+#include "common_decls.tmpl"
 #include "mul_mat_decls.tmpl"
 
 #ifdef VEC
@@ -36,11 +40,17 @@ struct MulMatParams {
     broadcast3: u32
 };
 
+#ifdef SRC_OVERLAP
+@group(0) @binding(0) var<storage, read_write> merged_src: array<SRC0_TYPE>;
+#define DST_BINDING 1
+#else
 @group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
 @group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
-@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)
+#define DST_BINDING 2
+#endif
 
-@group(0) @binding(3) var<uniform> params: MulMatParams;
+@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)
+@group(0) @binding(DST_BINDING + 1) var<uniform> params: MulMatParams;
 
 fn get_local_n(thread_id: u32) -> u32 {
     return thread_id / WORKGROUP_SIZE_M;
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_subgroup_matrix.wgsl src/ggml-webgpu/wgsl-shaders/mul_mat_subgroup_matrix.wgsl
index d86a72ce..35998a9b 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_subgroup_matrix.wgsl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_subgroup_matrix.wgsl
@@ -4,6 +4,10 @@ enable subgroups;
 enable chromium_experimental_subgroup_matrix;
 
 #define DECLARE_BYTE_LOADERS_SRC0
+#ifdef SRC_OVERLAP
+#define SRC0 merged_src
+#define SRC1 merged_src
+#endif
 #include "common_decls.tmpl"
 
 #include "mul_mat_decls.tmpl"
@@ -48,11 +52,17 @@ struct MulMatParams {
 };
 
 // SRC0_TYPE and SRC1_TYPE are defined in mul_mat_decls, which is included
+#ifdef SRC_OVERLAP
+@group(0) @binding(0) var<storage, read_write> merged_src: array<SRC0_TYPE>;
+#define DST_BINDING 1
+#else
 @group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
 @group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
-@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)
+#define DST_BINDING 2
+#endif
 
-@group(0) @binding(3) var<uniform> params: MulMatParams;
+@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)
+@group(0) @binding(DST_BINDING + 1) var<uniform> params: MulMatParams;
 
 const WG_M_SG_TILE_SIZE = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
 const WG_N_SG_TILE_SIZE = SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl
index ebdf0951..1781a6c7 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl
@@ -7,6 +7,11 @@ enable f16;
 requires packed_4x8_integer_dot_product;
 #endif
 
+#ifdef SRC_OVERLAP
+#define SRC0 merged_src
+#define SRC1 merged_src
+#endif
+
 #define DECLARE_BYTE_LOADERS_SRC0
 #include "common_decls.tmpl"
 
@@ -35,17 +40,22 @@ struct MulMatParams {
     broadcast3: u32
 };
 
+#if defined(MMVQ)
 @group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>;
-
-#ifdef MMVQ
 @group(0) @binding(1) var<storage, read_write> src1q: array<q8_1>;
+#define DST_BINDING 2
+#elif defined(SRC_OVERLAP)
+@group(0) @binding(0) var<storage, read_write> merged_src: array<SRC0_TYPE>;
+#define DST_BINDING 1
 #else
+@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>;
 @group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>;
+#define DST_BINDING 2
 #endif
 
-@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
+@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<f32>;
 // "mul_mat_vec_acc.tmpl" requires params.k, params.m, params.stride_01
-@group(0) @binding(3) var<uniform> params: MulMatParams;
+@group(0) @binding(DST_BINDING + 1) var<uniform> params: MulMatParams;
 
 // Flattened as [row][thread] to keep each row's reduction contiguous in memory.
 var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;
diff --git src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl
index 8fd0d190..864b4bd2 100644
--- src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl
+++ src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl
@@ -1,3 +1,10 @@
+#ifndef SRC0
+#define SRC0 src0
+#endif
+#ifndef SRC1
+#define SRC1 src1
+#endif
+
 #ifdef U32_DEQUANT_HELPERS
 #define SRC0_TYPE u32
 
@@ -43,13 +50,13 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
     for (var k = thread_id; k < k_vec; k += WG_SIZE) {
         var x_vals: array<SRC1_TYPE, NUM_COLS>;
         for (var col = 0u;col < NUM_COLS;col += 1) {
-            x_vals[col] = src1[src1_idx_base_vec + col * (params.stride_11 / VEC_SIZE) + k];
+            x_vals[col] = SRC1[src1_idx_base_vec + col * (params.stride_11 / VEC_SIZE) + k];
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
             let output_row = row_base + row;
             if (output_row < params.m) {
                 let src0_idx = (src0_batch_offset + output_row * params.stride_01) / VEC_SIZE + k;
-                let w = src0[src0_idx];
+                let w = SRC0[src0_idx];
                 for (var col = 0u;col < NUM_COLS;col += 1) {
                     acc[col][row] += inner_dot(w, x_vals[col]);
                 }
@@ -76,7 +83,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -116,8 +123,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4] = f32(src1[x_base + col * params.stride_11 + i + 16]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4] = f32(SRC1[x_base + col * params.stride_11 + i + 16]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -160,8 +167,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4] = f32(src1[x_base + col * params.stride_11 + i + 16]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4] = f32(SRC1[x_base + col * params.stride_11 + i + 16]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -205,8 +212,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4] = f32(src1[x_base + col * params.stride_11 + i + 16]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4] = f32(SRC1[x_base + col * params.stride_11 + i + 16]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -253,8 +260,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4] = f32(src1[x_base + col * params.stride_11 + i + 16]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4] = f32(SRC1[x_base + col * params.stride_11 + i + 16]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -302,7 +309,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -347,7 +354,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -409,10 +416,10 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 4u; i++) {
-                x_block[col][i]       = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4u]  = f32(src1[x_base + col * params.stride_11 + 32u + i]);
-                x_block[col][i + 8u]  = f32(src1[x_base + col * params.stride_11 + 64u + i]);
-                x_block[col][i + 12u] = f32(src1[x_base + col * params.stride_11 + 96u + i]);
+                x_block[col][i]       = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4u]  = f32(SRC1[x_base + col * params.stride_11 + 32u + i]);
+                x_block[col][i + 8u]  = f32(SRC1[x_base + col * params.stride_11 + 64u + i]);
+                x_block[col][i + 12u] = f32(SRC1[x_base + col * params.stride_11 + 96u + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -518,8 +525,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 8u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 8u] = f32(src1[x_base + col * params.stride_11 + 32u + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 8u] = f32(SRC1[x_base + col * params.stride_11 + 32u + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -610,10 +617,10 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         for (var col = 0u; col < NUM_COLS;col += 1) {
             let col_base = x_base + col * params.stride_11;
             for (var i = 0u; i < 4u; i++) {
-                x_block[col][i]       = f32(src1[col_base + i]);
-                x_block[col][i + 4u]  = f32(src1[col_base + 32u + i]);
-                x_block[col][i + 8u]  = f32(src1[col_base + 128u + i]);
-                x_block[col][i + 12u] = f32(src1[col_base + 160u + i]);
+                x_block[col][i]       = f32(SRC1[col_base + i]);
+                x_block[col][i + 4u]  = f32(SRC1[col_base + 32u + i]);
+                x_block[col][i + 8u]  = f32(SRC1[col_base + 128u + i]);
+                x_block[col][i + 12u] = f32(SRC1[col_base + 160u + i]);
             }
         }
 
@@ -713,10 +720,10 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         for (var col = 0u; col < NUM_COLS;col += 1) {
             let col_base = x_base + col * params.stride_11;
             for (var i = 0u; i < 4u; i++) {
-                x_block[col][i]       = f32(src1[col_base + i]);
-                x_block[col][i + 4u]  = f32(src1[col_base + 32u + i]);
-                x_block[col][i + 8u]  = f32(src1[col_base + 128u + i]);
-                x_block[col][i + 12u] = f32(src1[col_base + 160u + i]);
+                x_block[col][i]       = f32(SRC1[col_base + i]);
+                x_block[col][i + 4u]  = f32(SRC1[col_base + 32u + i]);
+                x_block[col][i + 8u]  = f32(SRC1[col_base + 128u + i]);
+                x_block[col][i + 12u] = f32(SRC1[col_base + 160u + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -823,10 +830,10 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         for (var col = 0u; col < NUM_COLS;col += 1) {
             let col_base = x_base + col * params.stride_11;
             for (var l = 0u; l < 4u; l++) {
-                x_block[col][l]       = f32(src1[col_base + l]);
-                x_block[col][l + 4u]  = f32(src1[col_base + 32u + l]);
-                x_block[col][l + 8u]  = f32(src1[col_base + 64u + l]);
-                x_block[col][l + 12u] = f32(src1[col_base + 96u + l]);
+                x_block[col][l]       = f32(SRC1[col_base + l]);
+                x_block[col][l + 4u]  = f32(SRC1[col_base + 32u + l]);
+                x_block[col][l + 8u]  = f32(SRC1[col_base + 64u + l]);
+                x_block[col][l + 12u] = f32(SRC1[col_base + 96u + l]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -899,7 +906,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -960,7 +967,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1039,7 +1046,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1101,7 +1108,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1168,7 +1175,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1234,7 +1241,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1302,7 +1309,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1367,8 +1374,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2u; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4u] = f32(src1[x_base + col * params.stride_11 + i + 16u]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4u] = f32(SRC1[x_base + col * params.stride_11 + i + 16u]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1418,7 +1425,7 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, 16>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < 16u; i++) {
-                x_block[col][i] = f32(src1[x_base + col * params.stride_11 + i]);
+                x_block[col][i] = f32(SRC1[x_base + col * params.stride_11 + i]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1476,8 +1483,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 4] = f32(src1[x_base + col * params.stride_11 + i + 16]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 4] = f32(SRC1[x_base + col * params.stride_11 + i + 16]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
@@ -1521,8 +1528,8 @@ fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src
         var x_block: array<array<f32, ELEMS_PER_THREAD>, NUM_COLS>;
         for (var col = 0u; col < NUM_COLS;col += 1) {
             for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
-                x_block[col][i]     = f32(src1[x_base + col * params.stride_11 + i]);
-                x_block[col][i + 8] = f32(src1[x_base + col * params.stride_11 + i + 8]);
+                x_block[col][i]     = f32(SRC1[x_base + col * params.stride_11 + i]);
+                x_block[col][i + 8] = f32(SRC1[x_base + col * params.stride_11 + i + 8]);
             }
         }
         for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
diff --git src/ggml-zendnn/CMakeLists.txt src/ggml-zendnn/CMakeLists.txt
index 87d721f6..6e393d6b 100644
--- src/ggml-zendnn/CMakeLists.txt
+++ src/ggml-zendnn/CMakeLists.txt
@@ -86,6 +86,6 @@ endif()
 
 target_link_libraries(ggml-zendnn PRIVATE m pthread)
 
-if (GGML_OPENMP)
-    target_link_libraries(ggml-zendnn PRIVATE OpenMP::OpenMP_CXX)
+if (GGML_OPENMP_ENABLED)
+    target_link_libraries(ggml-zendnn PRIVATE ${GGML_OPENMP_TARGET_CXX})
 endif()
diff --git src/ggml.c src/ggml.c
index d0d369c4..1a60fec7 100644
--- src/ggml.c
+++ src/ggml.c
@@ -4200,7 +4200,7 @@ static struct ggml_tensor * ggml_rope_impl(
 
     struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
 
-    int32_t params[15] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
+    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
     memcpy(params +  5, &freq_base,    sizeof(float));
     memcpy(params +  6, &freq_scale,   sizeof(float));
     memcpy(params +  7, &ext_factor,   sizeof(float));
@@ -4212,6 +4212,8 @@ static struct ggml_tensor * ggml_rope_impl(
     } else {
         memset(params + 11, 0,         sizeof(int32_t) * GGML_MROPE_SECTIONS);
     }
+    params[15] = 0; // n_offs, set via ggml_rope_set_offset()
+
     ggml_set_op_params(result, params, sizeof(params));
 
     result->op     = GGML_OP_ROPE;
@@ -4422,6 +4424,20 @@ struct ggml_tensor * ggml_rope_multi_back(
     result->op = GGML_OP_ROPE_BACK;
     return result;
 }
+
+struct ggml_tensor * ggml_rope_set_offset(
+        struct ggml_tensor  * a,
+        int                   n_offs) {
+    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
+    GGML_ASSERT(n_offs >= 0);
+
+    const int32_t mode = ggml_get_op_params_i32(a, 2);
+    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);
+
+    ggml_set_op_params_i32(a, 15, n_offs);
+    return a;
+}
+
 // ggml_clamp
 
 struct ggml_tensor * ggml_clamp(
