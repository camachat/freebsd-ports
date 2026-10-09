diff --git include/ggml-backend.h include/ggml-backend.h
index 5ef80da3..38197285 100644
--- include/ggml-backend.h
+++ include/ggml-backend.h
@@ -317,6 +317,15 @@ extern "C" {
     //
     typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
 
+    // Callback while copying input weights of a split
+    // if the user returns false the scheduler simply copies the entire weight
+    // the callback is called only for input weights in host buffers
+    // the callback is called after all non-weight inputs of the split have been copied
+    // `src` is the tensor in the previous split
+    // `dst` is the copy of `src` in the split
+    // `graph` is the compute graph of the split
+    typedef bool (*ggml_backend_sched_copy_callback)(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst, struct ggml_cgraph * graph, void * user_data);
+
     // Initialize a backend scheduler, backends with low index are given priority over backends with high index
     GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
     GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
@@ -355,6 +364,9 @@ extern "C" {
     // Set a callback to be called for each resulting node during graph compute
     GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
 
+    // Set a callback to be called when the inputs weights of a split are being copied
+    GGML_API void                 ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data);
+
     //
     // Meta backend
     //
diff --git include/ggml-rpc.h include/ggml-rpc.h
index 1f8cb790..482bd366 100644
--- include/ggml-rpc.h
+++ include/ggml-rpc.h
@@ -6,7 +6,7 @@
 extern "C" {
 #endif
 
-#define RPC_PROTO_MAJOR_VERSION    7
+#define RPC_PROTO_MAJOR_VERSION    8
 #define RPC_PROTO_MINOR_VERSION    0
 #define RPC_PROTO_PATCH_VERSION    0
 
diff --git src/ggml-backend-meta.cpp src/ggml-backend-meta.cpp
index 0394433c..8ed5f4eb 100644
--- src/ggml-backend-meta.cpp
+++ src/ggml-backend-meta.cpp
@@ -869,7 +869,12 @@ static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
         ggml_backend_meta_split_state split_state;
         switch (tensor->op) {
             case GGML_OP_NONE: {
-                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
+                if (tensor->view_src != nullptr) {
+                    // full-tensor view created with ggml_view_tensor, transparent for the split state
+                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
+                } else {
+                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
+                }
             } break;
             case GGML_OP_DUP: {
                 split_state = handle_generic(src_ss, /*scalar_only =*/ true);
diff --git src/ggml-backend.cpp src/ggml-backend.cpp
index 5bdade6d..ff3f0799 100644
--- src/ggml-backend.cpp
+++ src/ggml-backend.cpp
@@ -966,6 +966,9 @@ struct ggml_backend_sched {
     ggml_backend_sched_eval_callback callback_eval;
     void * callback_eval_user_data;
 
+    ggml_backend_sched_copy_callback callback_copy;
+    void * callback_copy_user_data;
+
     char * context_buffer;
     size_t context_buffer_size;
 
@@ -1799,14 +1802,58 @@ static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
     return true;
 }
 
+static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
+    return t->buffer != NULL &&
+        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
+        ggml_backend_buffer_is_host(t->buffer);
+}
+
+static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {
+    const int split_backend_id = split->backend_id;
+    ggml_backend_t split_backend = sched->backends[split_backend_id];
+    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
+    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
+
+    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
+        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
+        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
+            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
+        } else {
+            ggml_backend_synchronize(split_backend);
+        }
+        ggml_backend_tensor_copy(input, input_cpy);
+        return;
+    }
+
+    // wait for the split backend to finish using the input before overwriting it
+    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
+        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
+    } else {
+        ggml_backend_synchronize(split_backend);
+    }
+
+    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
+        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
+        return;
+    }
+
+    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
+    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
+    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
+        ggml_backend_synchronize(input_backend);
+        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
+            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
+        } else {
+            ggml_backend_synchronize(split_backend);
+        }
+        ggml_backend_tensor_copy(input, input_cpy);
+    }
+}
+
 static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
     GGML_ASSERT(sched);
     struct ggml_backend_sched_split * splits = sched->splits;
 
-    ggml_tensor * prev_ids_tensor = nullptr;
-    std::vector<int32_t> ids;
-    std::vector<ggml_bitset_t> used_ids;
-
     int prev_backend_id = -1;
 
     for (int split_id = 0; split_id < sched->n_splits; split_id++) {
@@ -1825,129 +1872,15 @@ static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t s
         }
 
         // copy the input tensors to the split backend
+        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
         for (int input_id = 0; input_id < split->n_inputs; input_id++) {
-            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
-            struct ggml_tensor * input = split->inputs[input_id];
-            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
-
-            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
-                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
-                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
-                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
-                } else {
-                    ggml_backend_synchronize(split_backend);
-                }
-                ggml_backend_tensor_copy(input, input_cpy);
-            } else {
-                // wait for the split backend to finish using the input before overwriting it
-                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
-                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
-                } else {
-                    ggml_backend_synchronize(split_backend);
-                }
-
-                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
-                ggml_tensor * node = split->graph.nodes[0];
-                if (split->graph.n_nodes > 0 &&
-                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
-                    ggml_backend_buffer_is_host(input->buffer) && (
-                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
-                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
-                    )) {
-
-                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
-                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];
-
-                    ggml_backend_synchronize(input_backend);
-
-                    // get the ids
-                    ggml_tensor * ids_tensor = node->src[2];
-                    ggml_backend_t ids_backend = split_backend;
-
-                    if (ggml_nelements(ids_tensor) == 0) {
-                        continue;
-                    }
-
-                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
-                    // in that case, we use the original ids tensor
-                    for (int i = input_id + 1; i < split->n_inputs; i++) {
-                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
-                            ids_tensor = split->inputs[i];
-                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
-                            break;
-                        }
-                    }
-
-                    if (ids_tensor != prev_ids_tensor) {
-                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
-                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
-                        ggml_backend_synchronize(ids_backend);
-
-                        // find the used experts
-                        used_ids.clear();
-                        used_ids.resize(ggml_bitset_size(n_expert));
-                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
-                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
-                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
-                                GGML_ASSERT(id >= 0 && id < n_expert);
-                                ggml_bitset_set(used_ids.data(), id);
-                            }
-                        }
-
-                        prev_ids_tensor = ids_tensor;
-                    }
-
-                    // group consecutive experts and copy them together
-                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
-                        const size_t expert_offset = first_id * expert_size;
-                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
-                        const size_t padding = std::min<size_t>(expert_size, 512);
-                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;
-
-                        ggml_backend_tensor_set_async(split_backend,
-                            input_cpy,
-                            (const uint8_t *)input->data + expert_offset, expert_offset,
-                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
-                            // this is necessary for MMQ in the CUDA backend
-                            expert_size_copy + padding_end);
-                    };
-
-                    int id = 0;
-                    while (!ggml_bitset_get(used_ids.data(), id)) {
-                        id++;
-                    }
-                    int32_t first_id = id;
-                    int32_t last_id = first_id;
-
-                    for (++id; id < n_expert; ++id) {
-                        if (!ggml_bitset_get(used_ids.data(), id)) {
-                            continue;
-                        }
-
-                        if (id == last_id + 1) {
-                            last_id = id;
-                            continue;
-                        }
-
-                        copy_experts(first_id, last_id);
-
-                        first_id = id;
-                        last_id = id;
-                    }
-                    copy_experts(first_id, last_id);
-                } else {
-                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
-                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
-                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
-                        ggml_backend_synchronize(input_backend);
-                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
-                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
-                        } else {
-                            ggml_backend_synchronize(split_backend);
-                        }
-                        ggml_backend_tensor_copy(input, input_cpy);
-                    }
-                }
+            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
+                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
+            }
+        }
+        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
+            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
+                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
             }
         }
 
@@ -2204,6 +2137,12 @@ void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backe
     sched->callback_eval_user_data = user_data;
 }
 
+void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
+    GGML_ASSERT(sched);
+    sched->callback_copy = callback;
+    sched->callback_copy_user_data = user_data;
+}
+
 int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
     GGML_ASSERT(sched);
     return sched->n_splits;
diff --git src/ggml-cpu/ops.cpp src/ggml-cpu/ops.cpp
index 16a916d2..c7e7c47a 100644
--- src/ggml-cpu/ops.cpp
+++ src/ggml-cpu/ops.cpp
@@ -6063,18 +6063,18 @@ static void ggml_compute_forward_clamp_f32(
     const int n  = ggml_nrows(src0);
     const int nc = src0->ne[0];
 
-    const size_t nb00 = src0->nb[0];
-    const size_t nb01 = src0->nb[1];
-
-    const size_t nb0 = dst->nb[0];
-    const size_t nb1 = dst->nb[1];
+    GGML_TENSOR_UNARY_OP_LOCALS
 
     GGML_ASSERT( nb0 == sizeof(float));
     GGML_ASSERT(nb00 == sizeof(float));
 
     for (int j = ith; j < n; j += nth) {
-        float * dst_ptr  = (float *) ((char *)  dst->data + j*nb1);
-        float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
+        const int64_t i1 = j % ne01;
+        const int64_t i2 = (j / ne01) % ne02;
+        const int64_t i3 = j / (ne01*ne02);
+
+        float * dst_ptr  = (float *) ((char *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
+        float * src0_ptr = (float *) ((char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
 
         for (int i = 0; i < nc; i++) {
             dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
@@ -6099,18 +6099,18 @@ static void ggml_compute_forward_clamp_f16(
     const int n  = ggml_nrows(src0);
     const int nc = src0->ne[0];
 
-    const size_t nb00 = src0->nb[0];
-    const size_t nb01 = src0->nb[1];
-
-    const size_t nb0 = dst->nb[0];
-    const size_t nb1 = dst->nb[1];
+    GGML_TENSOR_UNARY_OP_LOCALS
 
     GGML_ASSERT( nb0 == sizeof(ggml_fp16_t));
     GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
 
     for (int j = ith; j < n; j += nth) {
-        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *)  dst->data + j*nb1);
-        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + j*nb01);
+        const int64_t i1 = j % ne01;
+        const int64_t i2 = (j / ne01) % ne02;
+        const int64_t i3 = j / (ne01*ne02);
+
+        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
+        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
 
         for (int i = 0; i < nc; i++) {
             float v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
diff --git src/ggml-cuda/argsort.cu src/ggml-cuda/argsort.cu
index f6a850dd..101afeba 100644
--- src/ggml-cuda/argsort.cu
+++ src/ggml-cuda/argsort.cu
@@ -2,7 +2,8 @@
 
 #ifdef GGML_CUDA_USE_CUB
 #    include <cub/cub.cuh>
-#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
+    // strided_iterator was added in CCCL 3.1
+#    if (CCCL_MAJOR_VERSION > 3 || (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION >= 1))
 #        define STRIDED_ITERATOR_AVAILABLE
 #        include <cuda/iterator>
 #    endif
@@ -27,21 +28,21 @@ static __global__ void init_offsets(int * offsets, const int ncols, const int nr
 }
 #endif  // STRIDED_ITERATOR_AVAILABLE
 
-#ifdef GGML_CUDA_USE_CUB
-
-// returns the suggested maximum number of rows to process during one argsort_f32_i32_cuda_cub() call
-int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows) {
-    // perform argsort in chunks up to approximately this size (currently 64MB)
+// returns the suggested maximum number of rows to process at once, given the temporary buffer bytes per row
+int ggml_cuda_chunk_nrows(const size_t row_bytes, const int64_t nrows) {
+    // process rows in chunks up to approximately this size (currently 64MB)
     // to avoid excessive temporary buffers memory usage
     const int chunk_bytes = 1 << 26;
 
     // calculate how many rows will fit in one chunk (must be at least one)
-    const int chunk_nrows = std::max((int) (chunk_bytes / nb01), 1);
+    const int chunk_nrows = std::max((int) (chunk_bytes / row_bytes), 1);
 
     // limit the resulting amount to total nrows
     return std::min((int64_t) chunk_nrows, nrows);
 }
 
+#ifdef GGML_CUDA_USE_CUB
+
 void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                               const float *    x,
                               int *            dst,
@@ -289,7 +290,7 @@ void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
         return;
     }
 
-    const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);
+    const int chunk_nrows = ggml_cuda_chunk_nrows(src0->nb[1], nrows);
 
     ggml_cuda_pool & pool = ctx.pool();
 
diff --git src/ggml-cuda/argsort.cuh src/ggml-cuda/argsort.cuh
index c9adfcb9..86df5fda 100644
--- src/ggml-cuda/argsort.cuh
+++ src/ggml-cuda/argsort.cuh
@@ -4,8 +4,9 @@
 
 void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
 
+int ggml_cuda_chunk_nrows(const size_t row_bytes, const int64_t nrows);
+
 #ifdef GGML_CUDA_USE_CUB
-int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t nb01, const int64_t nrows);
 void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                               const float *    x,
                               int *            dst,
diff --git src/ggml-cuda/clamp.cu src/ggml-cuda/clamp.cu
index fe415e7f..2727b7ec 100644
--- src/ggml-cuda/clamp.cu
+++ src/ggml-cuda/clamp.cu
@@ -4,21 +4,42 @@ static __device__ __forceinline__ float op_clamp(float x, float min, float max)
     return fminf(fmaxf(x, min), max);
 }
 
+// src and dst may be views: rows are contiguous, dims 1..3 follow the strides (in elements).
 template <class T>
-static __global__ void op_clamp_kernel(const T * x, T * dst, const T min, const T max, const int k) {
-    const int i = blockDim.x*blockIdx.x + threadIdx.x;
+static __global__ void op_clamp_kernel(const T * x, T * dst, const T min, const T max, const uint32_t k,
+        const uint3 ne0, const uint3 ne1, const uint3 ne2,
+        const uint32_t s01, const uint32_t s02, const uint32_t s03,
+        const uint32_t s1,  const uint32_t s2,  const uint32_t s3) {
+    const uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
 
     if (i >= k) {
         return;
     }
 
-    dst[i] = (T)op_clamp((float)x[i], (float)min, (float)max);
+    const uint2 d0 = fast_div_modulo(i,    ne0); // <i / ne0, i0>
+    const uint2 d1 = fast_div_modulo(d0.x, ne1); // <i / (ne0*ne1), i1>
+    const uint2 d2 = fast_div_modulo(d1.x, ne2); // <i3, i2>
+
+    const size_t i_src = d0.y + size_t(d1.y)*s01 + size_t(d2.y)*s02 + size_t(d2.x)*s03;
+    const size_t i_dst = d0.y + size_t(d1.y)*s1  + size_t(d2.y)*s2  + size_t(d2.x)*s3;
+
+    dst[i_dst] = (T)op_clamp((float)x[i_src], (float)min, (float)max);
 }
 
 template <class T>
-static void clamp_cuda(const T * x, T * dst, const T min, const T max, const int k, cudaStream_t stream) {
-    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
-    op_clamp_kernel<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
+static void clamp_cuda(const T * x, T * dst, const T min, const T max, const ggml_tensor * src0, const ggml_tensor * t, cudaStream_t stream) {
+    const int64_t k  = ggml_nelements(src0);
+    const size_t  ts = sizeof(T);
+    GGML_ASSERT(k <= std::numeric_limits<uint32_t>::max());
+
+    const uint3 ne0 = init_fastdiv_values(src0->ne[0]);
+    const uint3 ne1 = init_fastdiv_values(src0->ne[1]);
+    const uint3 ne2 = init_fastdiv_values(src0->ne[2]);
+
+    const int64_t num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
+    op_clamp_kernel<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, (uint32_t) k, ne0, ne1, ne2,
+        src0->nb[1]/ts, src0->nb[2]/ts, src0->nb[3]/ts,
+        t->nb[1]/ts,    t->nb[2]/ts,    t->nb[3]/ts);
 }
 
 
@@ -31,6 +52,7 @@ void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
     GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
     GGML_ASSERT(src0->type == dst->type);
+    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst));
 
     float min;
     float max;
@@ -38,8 +60,8 @@ void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     memcpy(&max, (float *) dst->op_params + 1, sizeof(float));
 
     if (src0->type == GGML_TYPE_F16) {
-        clamp_cuda((const half *)src0_d, (half *)dst_d, (half)min, (half)max, ggml_nelements(src0), stream);
+        clamp_cuda((const half *)src0_d, (half *)dst_d, (half)min, (half)max, src0, dst, stream);
     } else {
-        clamp_cuda((const float *)src0_d, (float *)dst_d, (float)min, (float)max, ggml_nelements(src0), stream);
+        clamp_cuda((const float *)src0_d, (float *)dst_d, (float)min, (float)max, src0, dst, stream);
     }
 }
diff --git src/ggml-cuda/fwht.cu src/ggml-cuda/fwht.cu
index 67eea594..d69e2ed8 100644
--- src/ggml-cuda/fwht.cu
+++ src/ggml-cuda/fwht.cu
@@ -2,6 +2,9 @@
 #include "convert.cuh"
 #include "fwht.cuh"
 
+// wide FWHT blocks use one row per thread block with this many threads
+#define GGML_CUDA_FWHT_BLOCK_NT 256
+
 template <int N, typename T>
 __launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
 __global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
@@ -59,6 +62,87 @@ __global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, cons
     }
 }
 
+// Wide blocks: one row per thread block instead of per warp, so each thread keeps N/NT
+// values rather than N/32. Stages below the warp width still shuffle, those up to the
+// block width go through shared memory, and the rest stay in registers.
+template <int N, int NT, typename T>
+__launch_bounds__(NT, 1)
+__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale) {
+    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
+    constexpr int NE        = N / NT;
+    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");
+
+    __shared__ float s[N];
+
+    const int64_t r = blockIdx.x;
+    if (r >= n_rows) {
+        return;
+    }
+
+    src += r * N;
+    dst += r * N;
+
+    const int tid  = threadIdx.x;
+    const int lane = tid % warp_size;
+
+    ggml_cuda_pdl_sync();
+
+    float reg[NE];
+#pragma unroll
+    for (int i = 0; i < NE; ++i) {
+        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
+    }
+
+    // stages within a warp: partner differs in the lane bits
+#pragma unroll
+    for (int h = 1; h < warp_size; h *= 2) {
+#pragma unroll
+        for (int j = 0; j < NE; j++) {
+            const float val  = reg[j];
+            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
+            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
+        }
+    }
+
+    // stages across warps: partner differs in the thread-index bits above the lane
+#pragma unroll
+    for (int h = warp_size; h < NT; h *= 2) {
+#pragma unroll
+        for (int j = 0; j < NE; j++) {
+            s[j * NT + tid] = reg[j];
+        }
+        __syncthreads();
+#pragma unroll
+        for (int j = 0; j < NE; j++) {
+            const float val  = reg[j];
+            const float val2 = s[j * NT + (tid ^ h)];
+            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
+        }
+        __syncthreads();
+    }
+
+    // stages above the block width: partner is another register of the same thread
+#pragma unroll
+    for (int h = NT; h < N; h *= 2) {
+        const int step = h / NT;
+#pragma unroll
+        for (int j = 0; j < NE; j += 2 * step) {
+#pragma unroll
+            for (int k = 0; k < step; k++) {
+                const float x = reg[j + k];
+                const float y = reg[j + k + step];
+                reg[j + k]        = x + y;
+                reg[j + k + step] = x - y;
+            }
+        }
+    }
+
+#pragma unroll
+    for (int i = 0; i < NE; ++i) {
+        dst[i * NT + tid] = reg[i];
+    }
+}
+
 template <typename T>
 static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
     const int     n    = src->ne[0];
@@ -94,7 +178,34 @@ static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_t
             ggml_cuda_kernel_launch(fwht_cuda<512, T>, launch_params, src_d, dst_d, rows, scale);
             return true;
         default:
-            return false;
+            break;
+    }
+
+    // wide blocks: one row per thread block
+    {
+        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;
+
+        dim3 grid_dims_w(rows, 1, 1);
+        dim3 block_dims_w(nt, 1, 1);
+        const ggml_cuda_kernel_launch_params launch_params_w =
+            ggml_cuda_kernel_launch_params(grid_dims_w, block_dims_w, 0, stream);
+
+        switch (n) {
+            case 1024:
+                ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
+                return true;
+            case 2048:
+                ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
+                return true;
+            case 4096:
+                ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
+                return true;
+            case 8192:
+                ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
+                return true;
+            default:
+                return false;
+        }
     }
 }
 
diff --git src/ggml-cuda/gated_delta_net.cu src/ggml-cuda/gated_delta_net.cu
index 1b431a72..b5825706 100644
--- src/ggml-cuda/gated_delta_net.cu
+++ src/ggml-cuda/gated_delta_net.cu
@@ -1,7 +1,9 @@
 #include "gated_delta_net.cuh"
 #include "ggml-cuda/common.cuh"
 
-template <int S_v, bool KDA, bool keep_rs_t>
+constexpr int gdn_cols_per_warp = 4;
+
+template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
 __global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
 gated_delta_net_cuda(const float * q,
                                      const float * k,
@@ -30,9 +32,19 @@ gated_delta_net_cuda(const float * q,
                                      int           K) {
     const uint32_t h_idx    = blockIdx.x;
     const uint32_t sequence = blockIdx.y;
-    // each warp owns one column, using warp-level primitives to reduce across rows
-    const int      lane     = threadIdx.x;
-    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;
+
+    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
+    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
+    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
+    // one state column and reduces within itself
+    constexpr int lanes_per_col = warp_size / cols_per_warp;
+    constexpr int rows_per_lane = S_v / lanes_per_col;
+    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");
+
+    const int lane        = threadIdx.x;
+    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
+    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
+    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;
 
     const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
     const uint32_t iq3 = fastdiv(sequence, rq3_magic);
@@ -47,16 +59,13 @@ gated_delta_net_cuda(const float * q,
     curr_state += state_in_offset + col * S_v;
     attn_data += (sequence * n_tokens * H + h_idx) * S_v;
 
-    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
-    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
-    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
     float         s_shard[rows_per_lane];
     // state is stored transposed: M[col][i] = S[i][col], row col is contiguous
 
     ggml_cuda_pdl_sync();
 #pragma unroll
     for (int r = 0; r < rows_per_lane; r++) {
-        const int i = r * warp_size + lane;
+        const int i = r * lanes_per_col + lane_in_col;
         s_shard[r]  = curr_state[i];
     }
 
@@ -76,7 +85,7 @@ gated_delta_net_cuda(const float * q,
         float q_reg[rows_per_lane];
 #pragma unroll
         for (int r = 0; r < rows_per_lane; r++) {
-            const int i = r * warp_size + lane;
+            const int i = r * lanes_per_col + lane_in_col;
             k_reg[r] = k_t[i];
             q_reg[r] = q_t[i];
         }
@@ -90,7 +99,7 @@ gated_delta_net_cuda(const float * q,
             for (int r = 0; r < rows_per_lane; r++) {
                 kv_shard += s_shard[r] * k_reg[r];
             }
-            float kv_col = warp_reduce_sum<warp_size>(kv_shard);
+            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);
 
             // delta[col] = (v[col] - g * kv[col]) * beta
             float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
@@ -104,9 +113,9 @@ gated_delta_net_cuda(const float * q,
                 attn_partial += s_shard[r] * q_reg[r];
             }
 
-            float attn_col = warp_reduce_sum<warp_size>(attn_partial);
+            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);
 
-            if (lane == 0) {
+            if (lane_in_col == 0) {
                 attn_data[col] = attn_col * scale;
             }
         } else {
@@ -114,11 +123,11 @@ gated_delta_net_cuda(const float * q,
             float kv_shard = 0.0f;
 #pragma unroll
             for (int r = 0; r < rows_per_lane; r++) {
-                const int i = r * warp_size + lane;
+                const int i = r * lanes_per_col + lane_in_col;
                 kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
             }
 
-            float kv_col = warp_reduce_sum<warp_size>(kv_shard);
+            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);
 
             // delta[col] = (v[col] - kv[col]) * beta
             float delta_col = (v_t[col] - kv_col) * beta_val;
@@ -128,14 +137,14 @@ gated_delta_net_cuda(const float * q,
             float attn_partial = 0.0f;
 #pragma unroll
             for (int r = 0; r < rows_per_lane; r++) {
-                const int i = r * warp_size + lane;
+                const int i = r * lanes_per_col + lane_in_col;
                 s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                 attn_partial += s_shard[r] * q_reg[r];
             }
 
-            float attn_col = warp_reduce_sum<warp_size>(attn_partial);
+            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);
 
-            if (lane == 0) {
+            if (lane_in_col == 0) {
                 attn_data[col] = attn_col * scale;
             }
         }
@@ -150,7 +159,7 @@ gated_delta_net_cuda(const float * q,
                 float * curr_state = state + target_slot * state_slot_stride;
 #pragma unroll
                 for (int r = 0; r < rows_per_lane; r++) {
-                    const int i = r * warp_size + lane;
+                    const int i = r * lanes_per_col + lane_in_col;
                     curr_state[col * S_v + i] = s_shard[r];
                 }
             }
@@ -160,7 +169,7 @@ gated_delta_net_cuda(const float * q,
     if constexpr (!keep_rs_t) {
 #pragma unroll
         for (int r = 0; r < rows_per_lane; r++) {
-            const int i          = r * warp_size + lane;
+            const int i          = r * lanes_per_col + lane_in_col;
             state[col * S_v + i] = s_shard[r];
         }
     }
@@ -179,8 +188,16 @@ static void launch_gated_delta_net(
         float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
     //TODO: Add chunked kernel for even faster pre-fill
     const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
-    const int num_warps = 4;
-    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
+    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
+    // SMs without a CTA, so small head counts keep the device filled
+    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
+    const int cols_per_warp = gdn_cols_per_warp;
+    int num_warps = 4;
+    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
+        num_warps /= 2;
+    }
+    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
+    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
     dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
 
     const uint3 neqk1_magic = init_fastdiv_values(neqk1);
diff --git src/ggml-cuda/ggml-cuda.cu src/ggml-cuda/ggml-cuda.cu
index e068116f..8e0c7658 100644
--- src/ggml-cuda/ggml-cuda.cu
+++ src/ggml-cuda/ggml-cuda.cu
@@ -762,7 +762,8 @@ static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer
 
         if (padded_size > original_size) {
             ggml_cuda_set_device(ctx->device);
-            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
+            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
+            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
         }
     }
     return GGML_STATUS_SUCCESS;
@@ -1409,13 +1410,13 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
     using traits = batched_mul_mat_traits<compute_type>;
     using cuda_t = typename traits::cuda_type;
 
-    GGML_ASSERT(ggml_is_contiguous(dst));
-
-    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
-    // As long as dst is contiguous this does not matter though.
+    // F32 chunks can have padding between rows to preserve the original destination stride.
+    GGML_ASSERT(ggml_is_contiguous(dst) ||
+        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));
 
     GGML_TENSOR_BINARY_OP_LOCALS
 
+    const int64_t ldc = nb1 / sizeof(float);
     const int64_t ne_dst = ggml_nelements(dst);
     cudaStream_t main_stream = ctx.stream();
     cublasHandle_t cublas_h = ctx.cublas_handle();
@@ -1545,14 +1546,14 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
                     ne01, ne11, ne10,
                     (const float *) alpha, (const float *) src0_ptr, s01,
                                            (const float *) src1_ptr, s11,
-                    (const float *) beta,  (float       *)  dst_ptr, ne0));
+                    (const float *) beta,  (float       *)  dst_ptr, ldc));
     } else if (ne12 == 1 && ne13 == 1) {
         CUBLAS_CHECK(
             cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     ne01, ne11, ne10,
                     alpha, src0_ptr, cu_data_type_a, s01,
                            src1_ptr, cu_data_type_b, s11,
-                    beta,   dst_ptr, cu_data_type,   ne0,
+                    beta,   dst_ptr, cu_data_type,   ldc,
                     cu_compute_type,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
     } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
@@ -1567,7 +1568,7 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
                 ne01, ne11, ne10,
                 alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                        src1_ptr, cu_data_type_b, s11, smb,     // strideB
-                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
+                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                 ne12*ne13,
                 cu_compute_type,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
@@ -1605,7 +1606,7 @@ static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const
                 ne01, ne11, ne10,
                 alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                        (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
-                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
+                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                 ne23,
                 cu_compute_type,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
@@ -1658,6 +1659,32 @@ static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml
         }
     }
 
+    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
+    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);
+
+    if (compute_type == GGML_TYPE_F32 &&
+            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
+            src0_f32_size > max_src0_convert_size) {
+        const size_t f32_row_size = src0_f32_size / src0->ne[1];
+        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));
+
+        if (rows_per_chunk < src0->ne[1]) {
+            ggml_tensor src0_chunk = *src0;
+            ggml_tensor dst_chunk = *dst;
+
+            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
+                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
+                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];
+
+                dst_chunk.ne[0] = src0_chunk.ne[1];
+                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];
+
+                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
+            }
+            return;
+        }
+    }
+
     switch (compute_type) {
         case GGML_TYPE_F32:
             ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
@@ -5290,9 +5317,7 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
                     if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                         return false;
                     }
-                    // TODO: should become:
-                    //return ggml_is_contiguous_rows(op->src[0]);
-                    return ggml_is_contiguous(op->src[0]);
+                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                 default:
                     return false;
             }
@@ -5583,11 +5608,12 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
         case GGML_OP_SQRT:
         case GGML_OP_SIN:
         case GGML_OP_COS:
-        case GGML_OP_CLAMP:
         case GGML_OP_LOG:
             return true;
         case GGML_OP_SCALE:
             return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
+        case GGML_OP_CLAMP:
+            return ggml_is_contiguous_rows(op->src[0]);
         case GGML_OP_ADD:
         case GGML_OP_SUB:
         case GGML_OP_MUL:
@@ -5633,7 +5659,7 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
             return max_bias == 0.0f;
         }
         case GGML_OP_ROLL:
-            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
+            if(op->src[0]->type == GGML_TYPE_F32) {
                 return true;
             }
             return false;
@@ -5663,11 +5689,7 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
         case GGML_OP_SUM:
             return ggml_is_contiguous_rows(op->src[0]);
         case GGML_OP_TOP_K:
-#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
-            return true;
-#else
-            return op->src[0]->ne[0] <= 1024;
-#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
+            return op->src[0]->ne[0] <= INT_MAX;
         case GGML_OP_ARGSORT:
 #ifndef GGML_CUDA_USE_CUB
             {
@@ -5679,7 +5701,7 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
                 return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
             }
 #else
-            return true;
+            return op->src[0]->ne[0] <= INT_MAX;
 #endif
         case GGML_OP_SUM_ROWS:
             return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
diff --git src/ggml-cuda/lightning-indexer.cu src/ggml-cuda/lightning-indexer.cu
index 57cb1c1d..429d2e20 100644
--- src/ggml-cuda/lightning-indexer.cu
+++ src/ggml-cuda/lightning-indexer.cu
@@ -239,6 +239,14 @@ static __global__ void lightning_indexer_kernel_wmma(
 // tokens scored per block by the tile kernel
 #define LIGHTNING_INDEXER_TILE_TOKENS 8
 
+// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
+// at 28 KB and the queries of four heads do not fit there next to the key tile
+#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
+#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
+#else
+#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
+#endif
+
 // TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
 // thanks to that one warp operating on float4 processes whole indexer K/Q vectors
 // 32 * 4 = 128 (N_EMBD)
@@ -406,9 +414,11 @@ static __global__ void lightning_indexer_kernel_tile(
     constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
     constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
     constexpr int N_EMBD_H2         = N_EMBD / 2;
+    constexpr int HEADS_PER_PASS    = N_HEAD < LIGHTNING_INDEXER_TILE_HEADS_PER_PASS ? N_HEAD : LIGHTNING_INDEXER_TILE_HEADS_PER_PASS;
 
     static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
     static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");
+    static_assert(N_HEAD % HEADS_PER_PASS == 0, "head passes must cover the heads");
 
     const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
     const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
@@ -417,7 +427,7 @@ static __global__ void lightning_indexer_kernel_tile(
 
     // the row padding keeps the keys of consecutive threads in distinct banks
     __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
-    __shared__ float2 q_shared[N_HEAD][TOKENS_PER_BLOCK][N_EMBD_H2];
+    __shared__ float2 q_shared[HEADS_PER_PASS][TOKENS_PER_BLOCK][N_EMBD_H2];
     __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];
 
     // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero
@@ -451,22 +461,7 @@ static __global__ void lightning_indexer_kernel_tile(
         k_shared[r][2*c4 + 1] = hi;
     }
 
-    // phase 2 - stage the queries and weights of every head, tokens past n_batch are zero
-
-#pragma unroll
-    for (int i = tid; i < N_HEAD * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
-        const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
-        const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
-        const int c4 = i % (N_EMBD / 4);
-
-        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
-        if (start_batch + r < n_batch) {
-            v = *(const float4 *) ((const char *) Q + h*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
-        }
-
-        q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
-        q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
-    }
+    // phase 2 - stage the weights of every head, tokens past n_batch are zero
 
     if (tid < N_HEAD * TOKENS_PER_BLOCK) {
         const int h = tid / TOKENS_PER_BLOCK;
@@ -475,33 +470,60 @@ static __global__ void lightning_indexer_kernel_tile(
             ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
     }
 
-    __syncthreads();
-
-    // phase 3 - float products of the widened keys for every head, ReLU, weight
-
     const int kl = tid % KEY_LANES;
     const int tl = tid / KEY_LANES;
 
     float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };
 
-#pragma unroll 8
-    for (int c = 0; c < N_EMBD_H2; ++c) {
-        float2 k_val[KEYS_PER_THREAD];
 #pragma unroll
-        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
-            k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
+    for (int h0 = 0; h0 < N_HEAD; h0 += HEADS_PER_PASS) {
+        // the previous pass is fully consumed before its queries are replaced
+        if (h0 > 0) {
+            __syncthreads();
         }
+
+        // phase 3 - stage the queries of the heads of this pass, tokens past n_batch are zero
+
 #pragma unroll
-        for (int h = 0; h < N_HEAD; ++h) {
-            const float2 q_val = q_shared[h][tl][c];
+        for (int i = tid; i < HEADS_PER_PASS * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
+            const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
+            const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
+            const int c4 = i % (N_EMBD / 4);
+
+            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
+            if (start_batch + r < n_batch) {
+                v = *(const float4 *) ((const char *) Q + (h0 + h)*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
+            }
+
+            q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
+            q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
+        }
+
+        __syncthreads();
+
+        // phase 4 - float products of the widened keys for the heads of this pass
+
+#pragma unroll 8
+        for (int c = 0; c < N_EMBD_H2; ++c) {
+            float2 k_val[KEYS_PER_THREAD];
 #pragma unroll
             for (int j = 0; j < KEYS_PER_THREAD; ++j) {
-                qk[h][j] = fmaf(k_val[j].x, q_val.x, qk[h][j]);
-                qk[h][j] = fmaf(k_val[j].y, q_val.y, qk[h][j]);
+                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
+            }
+#pragma unroll
+            for (int h = 0; h < HEADS_PER_PASS; ++h) {
+                const float2 q_val = q_shared[h][tl][c];
+#pragma unroll
+                for (int j = 0; j < KEYS_PER_THREAD; ++j) {
+                    qk[h0 + h][j] = fmaf(k_val[j].x, q_val.x, qk[h0 + h][j]);
+                    qk[h0 + h][j] = fmaf(k_val[j].y, q_val.y, qk[h0 + h][j]);
+                }
             }
         }
     }
 
+    // phase 5 - ReLU, weight, add the mask and write, consecutive threads write consecutive keys
+
     float score[KEYS_PER_THREAD] = { 0.0f };
 
 #pragma unroll
@@ -512,8 +534,6 @@ static __global__ void lightning_indexer_kernel_tile(
         }
     }
 
-    // phase 4 - add the mask and write, consecutive threads write consecutive keys
-
     const int i_batch = start_batch + tl;
     if (i_batch >= n_batch) {
         return;
@@ -677,8 +697,6 @@ void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor *
             LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
             GGML_ABORT("fatal error");
         }
-#ifndef GGML_USE_MUSA
-    // MUSA archs 21 and 22 cap static shared memory at 28 KB, below what the tile kernel stages
     } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
         // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
         constexpr int WARPS_PER_BLOCK = 8;
@@ -698,9 +716,8 @@ void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor *
         LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
         LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
         GGML_ABORT("fatal error");
-#endif // GGML_USE_MUSA
     } else if (n_embd == 128 && n_head == 4) {
-        // a batch smaller than a token tile, or MUSA, use vector kernel
+        // a batch smaller than a token tile, use vector kernel
         constexpr int K_VECS_PER_WARP = 8;
         constexpr int WARPS_PER_BLOCK = 8;
         constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;
diff --git src/ggml-cuda/mmq.cu src/ggml-cuda/mmq.cu
index 43603bb8..027a7e60 100644
--- src/ggml-cuda/mmq.cu
+++ src/ggml-cuda/mmq.cu
@@ -141,7 +141,10 @@ void ggml_cuda_mul_mat_q(
     GGML_TENSOR_BINARY_OP_LOCALS;
 
     cudaStream_t stream = ctx.stream();
-    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
+
+    const int    id    = ggml_cuda_get_device();
+    const int    cc    = ggml_cuda_info().devices[id].cc;
+    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
 
     const size_t ts_src0 = ggml_type_size(src0->type);
     const size_t ts_src1 = ggml_type_size(src1->type);
@@ -176,7 +179,7 @@ void ggml_cuda_mul_mat_q(
     const int64_t s03 = src0->nb[3] / ts_src0;
     const int64_t s3  =  dst->nb[3] / ts_dst;
 
-    const bool fallback = ne01 % 128 != 0;
+    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);
 
     const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);
 
@@ -184,9 +187,52 @@ void ggml_cuda_mul_mat_q(
     const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
     const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;
 
+    int J_best        = 0;
+    int nthreads_best = 0;
+    {
+        int64_t ncols_opt = ne11;
+        if (ids) {
+            const int64_t n_expert_used = ids->ne[0];
+            ncols_opt = ne12;
+
+            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
+            // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
+            if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
+                ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
+            }
+        }
+
+        int ntiles_J_best = INT_MAX;
+
+        for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
+            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc, prec_src1);
+            if (config.type == GGML_TYPE_COUNT) {
+                continue;
+            }
+
+            if (mmq_get_nbytes_shared(config, cc) > smpbo) {
+                continue;
+            }
+
+            const int ntiles_x = (ncols_opt + config.J - 1) / config.J;
+
+            if (ntiles_x < ntiles_J_best) {
+                J_best = J;
+                nthreads_best = config.nthreads;
+                ntiles_J_best = ntiles_x;
+            }
+        }
+    }
+    GGML_ASSERT(J_best > 0);
+
+    // A tile of size J can read in at most J - 1 extra columns.
+    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
+    const size_t src1_load_chunk_size = nthreads_best * sizeof(int);
+    const size_t src1_q8_1_padding = ((J_best * sizeof(block_q8_1_mmq) + src1_load_chunk_size - 1) / src1_load_chunk_size)
+        * src1_load_chunk_size;
+
     if (!ids) {
-        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
-            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
+        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
         ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
         ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
         if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
@@ -223,7 +269,7 @@ void ggml_cuda_mul_mat_q(
             ne00, ne01, ne1, s01, ne11, s1,
             ne02, ne12, s02, s12, s2,
             ne03, ne13, s03, s13, s3,
-            ne1, ne1};
+            ne1, J_best};
         ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
         return;
     }
@@ -237,7 +283,7 @@ void ggml_cuda_mul_mat_q(
     GGML_ASSERT(ne1 == n_expert_used);
 
     ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
-    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
+    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.
     ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
 
     // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
@@ -254,8 +300,7 @@ void ggml_cuda_mul_mat_q(
         CUDA_CHECK(cudaGetLastError());
     }
 
-    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
-        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) * sizeof(block_q8_1_mmq);
+    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
     ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
     ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
     if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
@@ -296,13 +341,6 @@ void ggml_cuda_mul_mat_q(
                                          ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
     const int64_t s13 = ne12*s12;
 
-    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
-    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
-    int64_t ncols_opt = ne12;
-    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
-        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
-    }
-
     // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
     const mmq_args args = {
         src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
@@ -310,7 +348,7 @@ void ggml_cuda_mul_mat_q(
         ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
         ne02, ne02, s02, s12, s2,
         ne03, ne13, s03, s13, s3,
-        ne12, ncols_opt};
+        ne12, J_best};
 
     ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
 }
diff --git src/ggml-cuda/mmq.cuh src/ggml-cuda/mmq.cuh
index 0cd31d91..198aab4b 100644
--- src/ggml-cuda/mmq.cuh
+++ src/ggml-cuda/mmq.cuh
@@ -208,7 +208,7 @@ struct ggml_cuda_mmq_config {
         static_assert((nthreads_) %  32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                \
         static_assert(                          (occupancy_)      <=   8, "bad occupancy");                                               \
         static_assert((I_)        %  32 == 0,                             "bad I");                                                       \
-        static_assert((J_)        %   8 == 0,                             "bad J");                                                       \
+        static_assert((J_)        %   8 == 0 && (J_)              <= 128, "bad J");                                                       \
         static_assert((K_vram_)   % 256 == 0,                             "bad K_vram");                                                  \
         return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
     }                                                                                                                                     \
@@ -295,6 +295,8 @@ static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_t
     GGML_UNUSED_VARS(type, J, fallback, prec_src1);
 }
 
+// FIXME all of the host functions are missing prec_src1, this can lead to inconsitent behavior.
+
 static __host__ int ggml_cuda_mmq_get_type(const ggml_type type, const int J, const bool fallback, const int cc) {
     return ggml_cuda_mmq_get_config(type, J, fallback, cc).type;
 }
@@ -369,15 +371,8 @@ static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, in
     return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_sram_layout(type, J, fallback, prec_src1));
 }
 
-static __host__ int ggml_cuda_mmq_get_J_max(const ggml_type type, const bool fallback, const int cc, const int64_t ne11) {
-    int ret = std::min(ne11, int64_t(512));
-    ret -= ret % 8;
-    for (;ret > 0; ret -= 8) {
-        if (ggml_cuda_mmq_get_config(type, ret, fallback, cc).type != GGML_TYPE_COUNT) {
-            return ret;
-        }
-    }
-    return ret;
+static __host__ bool ggml_cuda_mmq_needs_fallback(const int64_t nrows_x) {
+    return nrows_x % 128 != 0;
 }
 
 static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(ggml_type type, int J, bool fallback) {
@@ -1390,7 +1385,7 @@ struct mmq_args {
     int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
     int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
     int64_t ncols_max;
-    int64_t ncols_opt; // value to optimize the tile size against, launch grid still uses ncols_max
+    int J_best; // Tile width in ne11(dense)/ne12(MoE) direction to use for optimal performance.
 };
 
 static size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, const int cc) {
@@ -1484,32 +1479,7 @@ static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & a
 
 template <ggml_type type, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
 void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
-    const int    id    = ggml_cuda_get_device();
-    const int    cc    = ggml_cuda_info().devices[id].cc;
-    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
-
-    int J_best        = 0;
-    int ntiles_J_best = INT_MAX;
-
-    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
-        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
-        if (config.type == GGML_TYPE_COUNT) {
-            continue;
-        }
-
-        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
-            continue;
-        }
-
-        const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;
-
-        if (ntiles_x < ntiles_J_best) {
-            J_best = J;
-            ntiles_J_best = ntiles_x;
-        }
-    }
-
-    switch (J_best) {
+    switch (args.J_best) {
         case   8:
             launch_mul_mat_q<type,   8, fallback, prec_src1>(ctx, args, stream);
             break;
@@ -1559,7 +1529,7 @@ void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
             launch_mul_mat_q<type, 128, fallback, prec_src1>(ctx, args, stream);
             break;
         default:
-            fprintf(stderr, "J_best=%d\n", J_best);
+            fprintf(stderr, "J_best=%d\n", args.J_best);
             GGML_ABORT("fatal error");
             break;
     }
@@ -1567,11 +1537,11 @@ void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
 
 template <ggml_type type, ggml_prec prec_src1 = GGML_PREC_Q8>
 void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
-    if (args.nrows_x % 128 == 0) {
-        constexpr bool fallback = false;
+    if (ggml_cuda_mmq_needs_fallback(args.nrows_x)) {
+        constexpr bool fallback = true;
         mul_mat_q_switch_J<type, fallback, prec_src1>(ctx, args, stream);
     } else {
-        constexpr bool fallback = true;
+        constexpr bool fallback = false;
         mul_mat_q_switch_J<type, fallback, prec_src1>(ctx, args, stream);
     }
 }
diff --git src/ggml-cuda/norm.cu src/ggml-cuda/norm.cu
index 5543307b..d804b933 100644
--- src/ggml-cuda/norm.cu
+++ src/ggml-cuda/norm.cu
@@ -3,38 +3,46 @@
 
 template <int block_size>
 static __global__ void norm_f32(
-        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
-        const int64_t stride_sample, const float eps) {
-    const int nrows     = gridDim.x;
-    const int nchannels = gridDim.y;
+        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
+        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
+    const int nrows = gridDim.x;
+    const int row   = blockIdx.x;
+    const int tid   = threadIdx.x;
 
-    const int row       = blockIdx.x;
-    const int channel   = blockIdx.y;
-    const int sample    = blockIdx.z;
-    const int tid       = threadIdx.x;
+    extern __shared__ float2 s_sum2[];
 
-    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
-    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
+    ggml_cuda_pdl_sync();
 
-    float2 mean_var = make_float2(0.0f, 0.0f);
+    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
+    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
+        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
+            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
+            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;
 
-    ggml_cuda_pdl_sync();
-    for (int col = tid; col < ncols; col += block_size) {
-        const float xi = x[col];
-        mean_var.x += xi;
-        mean_var.y += xi * xi;
-    }
+            float2 mean_var = make_float2(0.0f, 0.0f);
 
-    // sum up partial sums
-    extern __shared__ float2 s_sum2[];
-    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);
+            for (int col = tid; col < ncols; col += block_size) {
+                const float xi = xc[col];
+                mean_var.x += xi;
+                mean_var.y += xi * xi;
+            }
 
-    const float mean = mean_var.x / ncols;
-    const float var = mean_var.y / ncols - mean * mean;
-    const float inv_std = rsqrtf(var + eps);
+            // sum up partial sums
+            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);
 
-    for (int col = tid; col < ncols; col += block_size) {
-        dst[col] = (x[col] - mean) * inv_std;
+            const float mean = mean_var.x / ncols;
+            const float var = mean_var.y / ncols - mean * mean;
+            const float inv_std = rsqrtf(var + eps);
+
+            for (int col = tid; col < ncols; col += block_size) {
+                dstc[col] = (xc[col] - mean) * inv_std;
+            }
+
+            if constexpr (block_size > WARP_SIZE) {
+                // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
+                __syncthreads();
+            }
+        }
     }
 }
 
@@ -77,6 +85,8 @@ template <int block_size, bool do_multiply = false, bool do_add = false, bool do
 static __global__ void rms_norm_f32(const float * x,
                                     float *       dst,
                                     const int     ncols,
+                                    const int     nchannels,
+                                    const int     nsamples,
                                     const int64_t stride_row,
                                     const int64_t stride_channel,
                                     const int64_t stride_sample,
@@ -99,61 +109,71 @@ static __global__ void rms_norm_f32(const float * x,
                                     const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                     const float   scale_out            = 1.0f) {
     ggml_cuda_pdl_lc();
-    const int nrows     = gridDim.x;
-    const int nchannels = gridDim.y;
-
-    const int row       = blockIdx.x;
-    const int channel   = blockIdx.y;
-    const int sample    = blockIdx.z;
-    const int tid       = threadIdx.x;
+    const int nrows = gridDim.x;
+    const int row   = blockIdx.x;
+    const int tid   = threadIdx.x;
 
     static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
     static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");
 
-    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
-    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
-
-    if constexpr (do_multiply) {
-        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
-        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
-        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
-        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
-    }
-
-    if constexpr (do_add) {
-        const int add_row     = fastmodulo(row, add_nrows_packed);
-        const int add_channel = fastmodulo(channel, add_nchannels_packed);
-        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
-        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
-    }
-
-    float tmp = 0.0f; // partial sum for thread in warp
-
-    ggml_cuda_pdl_sync();
-    for (int col = tid; col < ncols; col += block_size) {
-        const float xi = x[col];
-        tmp += xi * xi;
-    }
-
-    // sum up partial sums
     extern __shared__ float s_sum[];
-    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
 
-    const float mean = tmp / ncols;
-    const float scale = rsqrtf(mean + eps);
+    ggml_cuda_pdl_sync();
 
-    for (int col = tid; col < ncols; col += block_size) {
-        if constexpr (do_multiply && do_add) {
-            const int mul_col = fastmodulo(col, mul_ncols_packed);
-            const int add_col = fastmodulo(col, add_ncols_packed);
-            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
-        } else if constexpr (do_multiply) {
-            const int mul_col = fastmodulo(col, mul_ncols_packed);
-            dst[col]          = scale * x[col] * mul[mul_col];
-        } else if constexpr (do_scale) {
-            dst[col] = scale_out * (scale * x[col]);
-        } else {
-            dst[col] = scale * x[col];
+    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
+    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
+        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
+            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
+            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;
+
+            [[maybe_unused]] const float * mulc = nullptr;
+            if constexpr (do_multiply) {
+                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
+                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
+                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
+                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
+            }
+
+            [[maybe_unused]] const float * addc = nullptr;
+            if constexpr (do_add) {
+                const int add_row     = fastmodulo(row, add_nrows_packed);
+                const int add_channel = fastmodulo(channel, add_nchannels_packed);
+                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
+                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
+            }
+
+            float tmp = 0.0f; // partial sum for thread in warp
+
+            for (int col = tid; col < ncols; col += block_size) {
+                const float xi = xc[col];
+                tmp += xi * xi;
+            }
+
+            // sum up partial sums
+            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
+
+            const float mean = tmp / ncols;
+            const float scale = rsqrtf(mean + eps);
+
+            for (int col = tid; col < ncols; col += block_size) {
+                if constexpr (do_multiply && do_add) {
+                    const int mul_col = fastmodulo(col, mul_ncols_packed);
+                    const int add_col = fastmodulo(col, add_ncols_packed);
+                    dstc[col]         = scale * xc[col] * mulc[mul_col] + addc[add_col];
+                } else if constexpr (do_multiply) {
+                    const int mul_col = fastmodulo(col, mul_ncols_packed);
+                    dstc[col]         = scale * xc[col] * mulc[mul_col];
+                } else if constexpr (do_scale) {
+                    dstc[col] = scale_out * (scale * xc[col]);
+                } else {
+                    dstc[col] = scale * xc[col];
+                }
+            }
+
+            if constexpr (block_size > WARP_SIZE) {
+                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
+                __syncthreads();
+            }
         }
     }
 }
@@ -247,50 +267,57 @@ static __global__ void rms_norm_back_f32(
 
 template <int block_size>
 static __global__ void l2_norm_f32(
-        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
-        const int64_t stride_sample, const float eps) {
-    const int nrows     = gridDim.x;
-    const int nchannels = gridDim.y;
+        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
+        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
+    const int nrows = gridDim.x;
+    const int row   = blockIdx.x;
+    const int tid   = threadIdx.x;
 
-    const int row       = blockIdx.x;
-    const int channel   = blockIdx.y;
-    const int sample    = blockIdx.z;
-    const int tid       = threadIdx.x;
+    extern __shared__ float s_sum[];
 
-    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
-    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
+    ggml_cuda_pdl_sync();
 
-    float tmp = 0.0f; // partial sum for thread in warp
+    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
+    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
+        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
+            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
+            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;
 
-    ggml_cuda_pdl_sync();
-    for (int col = tid; col < ncols; col += block_size) {
-        const float xi = x[col];
-        tmp += xi * xi;
-    }
+            float tmp = 0.0f; // partial sum for thread in warp
 
-    // sum up partial sums
-    extern __shared__ float s_sum[];
-    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
-    ggml_cuda_pdl_lc();
+            for (int col = tid; col < ncols; col += block_size) {
+                const float xi = xc[col];
+                tmp += xi * xi;
+            }
 
-    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
-    const float scale = rsqrtf(fmaxf(tmp, eps * eps));
+            // sum up partial sums
+            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
 
-    for (int col = tid; col < ncols; col += block_size) {
-        dst[col] = scale * x[col];
+            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
+            const float scale = rsqrtf(fmaxf(tmp, eps * eps));
+
+            for (int col = tid; col < ncols; col += block_size) {
+                dstc[col] = scale * xc[col];
+            }
+
+            if constexpr (block_size > WARP_SIZE) {
+                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
+                __syncthreads();
+            }
+        }
     }
 }
 
 static void norm_f32_cuda(
         const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
         const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
-    const dim3 blocks_num(nrows, nchannels, nsamples);
+    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
     if (ncols < 1024) {
         const dim3 block_dims(WARP_SIZE, 1, 1);
-        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
+        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
     } else {
         const dim3 block_dims(1024, 1, 1);
-        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
+        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
     }
 }
 
@@ -310,19 +337,19 @@ static void rms_norm_f32_cuda(
         const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
         const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
         const float scale_out = 1.0f) {
-    const dim3 blocks_num(nrows, nchannels, nsamples);
+    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
     if (ncols < 1024) {
         const dim3 block_dims(256, 1, 1);
         const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
         ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, do_scale>, launch_params,
-            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
+            x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
         // underlying cudaLaunchKernelEx does not support default params
         nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
         nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
     } else {
         const dim3 block_dims(1024, 1, 1);
         const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
-        ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, do_scale>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
+        ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, do_scale>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
         // underlying cudaLaunchKernelEx does not support default params
         nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
         nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
@@ -356,7 +383,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
                                   const uint32_t add_nsamples,
                                   const float    eps,
                                   cudaStream_t   stream) {
-    const dim3 blocks_num(nrows, nchannels, nsamples);
+    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
     if (mul == nullptr) {
         rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
         return;
@@ -370,7 +397,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
             const dim3 block_dims(256, 1, 1);
             const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
             ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
-                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
+                x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 // underlying cudaLaunchKernelEx does not support default params
             nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);
@@ -378,7 +405,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
             const dim3 block_dims(1024, 1, 1);
             const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
             ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
-                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
+                x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 // underlying cudaLaunchKernelEx does not support default params
             nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);
@@ -397,7 +424,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
             const dim3 block_dims(256, 1, 1);
             const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
             ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
-                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
+                x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                 add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                 add_nchannels_packed, add_nsamples_packed, 1.0f);
@@ -405,7 +432,7 @@ static void rms_norm_mul_f32_cuda(const float *  x,
             const dim3 block_dims(1024, 1, 1);
             const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
             ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
-                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
+                x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                 mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                 add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                 add_nchannels_packed, add_nsamples_packed, 1.0f);
@@ -426,15 +453,15 @@ static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float *
 static void l2_norm_f32_cuda(
         const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
         const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
-    const dim3 blocks_num(nrows, nchannels, nsamples);
+    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
     if (ncols < 1024) {
         const dim3 block_dims(WARP_SIZE, 1, 1);
         const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
-        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
+        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
     } else {
         const dim3 block_dims(1024, 1, 1);
         const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
-        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
+        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
     }
 }
 
diff --git src/ggml-cuda/pad.cu src/ggml-cuda/pad.cu
index 31cd00f7..e39a6bf0 100644
--- src/ggml-cuda/pad.cu
+++ src/ggml-cuda/pad.cu
@@ -15,49 +15,52 @@ static __global__ void pad_f32(const float * src, size_t s00, size_t s01, size_t
     // blockIdx.z: i3*ne2+i2
     // blockIdx.y: i1
     // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
-    // gridDim.y:  ne1
+    // gridDim.y and gridDim.z are capped at 65535, blocks stride over larger ne1 and ne2*ne3
     int i0 = threadIdx.x + blockIdx.x * blockDim.x;
-    int i1 = blockIdx.y;
-    int i2 = blockIdx.z % ne2;
-    int i3 = blockIdx.z / ne2;
-
-    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
+    if (i0 >= ne0) {
         return;
     }
 
-    const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;
-
-    if (!circular) {
-        if ((i0 >= lp0 && i0 < ne0 - rp0) && (i1 >= lp1 && i1 < ne1 - rp1) && (i2 >= lp2 && i2 < ne2 - rp2) &&
-            (i3 >= lp3 && i3 < ne3 - rp3)) {
-            const int64_t i00  = i0 - lp0;
-            const int64_t i01  = i1 - lp1;
-            const int64_t i02  = i2 - lp2;
-            const int64_t i03  = i3 - lp3;
-
-            const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
-
-            dst[dst_idx] = src[src_idx];
-        } else {
-            dst[dst_idx] = 0.0f;
+    for (int i1 = blockIdx.y; i1 < ne1; i1 += gridDim.y) {
+        for (int i23 = blockIdx.z; i23 < ne2 * ne3; i23 += gridDim.z) {
+            int i2 = i23 % ne2;
+            int i3 = i23 / ne2;
+
+            const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;
+
+            if (!circular) {
+                if ((i0 >= lp0 && i0 < ne0 - rp0) && (i1 >= lp1 && i1 < ne1 - rp1) && (i2 >= lp2 && i2 < ne2 - rp2) &&
+                    (i3 >= lp3 && i3 < ne3 - rp3)) {
+                    const int64_t i00  = i0 - lp0;
+                    const int64_t i01  = i1 - lp1;
+                    const int64_t i02  = i2 - lp2;
+                    const int64_t i03  = i3 - lp3;
+
+                    const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
+
+                    dst[dst_idx] = src[src_idx];
+                } else {
+                    dst[dst_idx] = 0.0f;
+                }
+            }
+            // circular means on a torus, so x and y wrap around
+            else {
+                const int64_t ne00 = ne0 - lp0 - rp0;
+                const int64_t ne01 = ne1 - lp1 - rp1;
+                const int64_t ne02 = ne2 - lp2 - rp2;
+                const int64_t ne03 = ne3 - lp3 - rp3;
+
+                const int64_t i00 = wrap_around(i0 - lp0, ne00);
+                const int64_t i01 = wrap_around(i1 - lp1, ne01);
+                const int64_t i02 = wrap_around(i2 - lp2, ne02);
+                const int64_t i03 = wrap_around(i3 - lp3, ne03);
+
+                const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
+
+                dst[dst_idx] = src[src_idx];
+            }
         }
     }
-    // circular means on a torus, so x and y wrap around
-    else {
-        const int64_t ne00 = ne0 - lp0 - rp0;
-        const int64_t ne01 = ne1 - lp1 - rp1;
-        const int64_t ne02 = ne2 - lp2 - rp2;
-        const int64_t ne03 = ne3 - lp3 - rp3;
-
-        const int64_t i00 = wrap_around(i0 - lp0, ne00);
-        const int64_t i01 = wrap_around(i1 - lp1, ne01);
-        const int64_t i02 = wrap_around(i2 - lp2, ne02);
-        const int64_t i03 = wrap_around(i3 - lp3, ne03);
-
-        const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
-
-        dst[dst_idx] = src[src_idx];
-    }
 }
 
 
@@ -67,7 +70,7 @@ static void pad_f32_cuda(const float * src, size_t s00, size_t s01, size_t s02,
     const int ne0, const int ne1, const int ne2, const int ne3,
     const bool circular, cudaStream_t stream) {
     int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
-    dim3 gridDim(num_blocks, ne1, ne2 * ne3);
+    dim3 gridDim(num_blocks, std::min(ne1, 65535), std::min(ne2 * ne3, 65535));
     pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, s00, s01, s02, s03, dst,
                                                          lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                                          ne0, ne1, ne2, ne3, circular);
diff --git src/ggml-cuda/pool2d.cu src/ggml-cuda/pool2d.cu
index c6d51e4d..83a49753 100644
--- src/ggml-cuda/pool2d.cu
+++ src/ggml-cuda/pool2d.cu
@@ -50,6 +50,65 @@ static  __global__ void pool2d_nchw_kernel(
     o_ptr[cur_oh * ow + cur_ow] = res;
 }
 
+template <typename Ti, typename To>
+static __global__ void pool2d_nchw_kernel_warp(
+        const int ih, const int iw, const int oh, const int ow,
+        const int kh, const int kw, const int sh, const int sw,
+        const int ph, const int pw, const int parallel_elements,
+        const Ti * __restrict__ src, To * __restrict__ dst, const enum ggml_op_pool op) {
+    const int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
+    const int lane     = threadIdx.x % WARP_SIZE;
+    if (warp_id >= parallel_elements) {
+        return;
+    }
+
+    const int I_HW = ih * iw;
+    const int O_HW = oh * ow;
+    const int nc     = warp_id / O_HW;
+    const int cur_oh = warp_id % O_HW / ow;
+    const int cur_ow = warp_id % O_HW % ow;
+    const Ti* i_ptr = src + nc * I_HW;
+
+    const int start_h = cur_oh * sh - ph;
+    const int bh = max(0, start_h);
+    const int eh = min(ih, start_h + kh);
+    const int start_w = cur_ow * sw - pw;
+    const int bw = max(0, start_w);
+    const int ew = min(iw, start_w + kw);
+
+    const int win_w     = ew - bw;
+    const int win_elems = (eh - bh) * win_w;
+    const To scale = 1. / (kh * kw);
+
+    To res;
+    switch (op) {
+        case GGML_OP_POOL_AVG: res = 0; break;
+        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
+        default: res = 0; assert(false);
+    }
+
+    for (int t = lane; t < win_elems; t += WARP_SIZE) {
+        const int i = bh + t / win_w;
+        const int j = bw + t % win_w;
+        const Ti cur = i_ptr[i * iw + j];
+        switch (op) {
+            case GGML_OP_POOL_AVG: res += cur * scale; break;
+            case GGML_OP_POOL_MAX: res = max(res, (To)cur); break;
+            default: break;
+        }
+    }
+
+#pragma unroll
+    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
+        const To other = __shfl_xor_sync(0xFFFFFFFF, res, offset, WARP_SIZE);
+        res = (op == GGML_OP_POOL_MAX) ? max(res, other) : res + other;
+    }
+
+    if (lane == 0) {
+        dst[nc * O_HW + cur_oh * ow + cur_ow] = res;
+    }
+}
+
 static void pool2d_nchw_kernel_f32_f32_cuda(
         const int ih, const int iw, const int oh, const int ow,
         const int kh, const int kw, const int sh, const int sw,
@@ -57,6 +116,13 @@ static void pool2d_nchw_kernel_f32_f32_cuda(
         const float * src, float * dst, const enum ggml_op_pool op,
         cudaStream_t stream) {
 
+    if (kh * kw >= CUDA_POOL2D_WARP_KERNEL_MIN_WINDOW) {
+        const int warps_per_block = CUDA_POOL2D_BLOCK_SIZE / WARP_SIZE;
+        const int num_blocks = (parallel_elements + warps_per_block - 1) / warps_per_block;
+        pool2d_nchw_kernel_warp<<<num_blocks, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
+        return;
+    }
+
     const int num_blocks = (parallel_elements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
     dim3 block_nums(num_blocks);
     pool2d_nchw_kernel<<<block_nums, CUDA_POOL2D_BLOCK_SIZE, 0, stream>>>(ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, parallel_elements, src, dst, op);
diff --git src/ggml-cuda/pool2d.cuh src/ggml-cuda/pool2d.cuh
index 7841292b..e2c7fb44 100644
--- src/ggml-cuda/pool2d.cuh
+++ src/ggml-cuda/pool2d.cuh
@@ -1,5 +1,6 @@
 #include "common.cuh"
 
 #define CUDA_POOL2D_BLOCK_SIZE 256
+#define CUDA_POOL2D_WARP_KERNEL_MIN_WINDOW 32
 
 void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
diff --git src/ggml-cuda/roll.cu src/ggml-cuda/roll.cu
index a339dfc1..c9bbe76f 100644
--- src/ggml-cuda/roll.cu
+++ src/ggml-cuda/roll.cu
@@ -17,6 +17,10 @@ static __global__ void roll_f32_cuda(const float * __restrict__ src,
                                      const int64_t ne01,
                                      const int64_t ne02,
                                      const int64_t ne03,
+                                     const int64_t nb00,
+                                     const int64_t nb01,
+                                     const int64_t nb02,
+                                     const int64_t nb03,
                                      const int     s0,
                                      const int     s1,
                                      const int     s2,
@@ -39,7 +43,7 @@ static __global__ void roll_f32_cuda(const float * __restrict__ src,
     const int64_t d3 = wrap_index(i3 - s3, ne03);
 
     dst[i3 * (ne00 * ne01 * ne02) + i2 * (ne01 * ne00) + i1 * ne00 + i0] =
-        src[d3 * (ne00 * ne01 * ne02) + d2 * (ne01 * ne00) + d1 * ne00 + d0];
+        src[(d3 * nb03 + d2 * nb02 + d1 * nb01 + d0 * nb00) / sizeof(float)];
 }
 
 void ggml_cuda_op_roll(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
@@ -63,5 +67,5 @@ void ggml_cuda_op_roll(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     int64_t num_blocks = (sz + CUDA_ROLL_BLOCK_SIZE - 1) / CUDA_ROLL_BLOCK_SIZE;
 
     roll_f32_cuda<<<num_blocks, CUDA_ROLL_BLOCK_SIZE, 0, stream>>>(
-        src0_d, dst_d, ne00, ne01, ne02, ne03, s0, s1, s2, s3);
+        src0_d, dst_d, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, s0, s1, s2, s3);
 }
diff --git src/ggml-cuda/rope.cu src/ggml-cuda/rope.cu
index e546fb65..b4fca778 100644
--- src/ggml-cuda/rope.cu
+++ src/ggml-cuda/rope.cu
@@ -709,7 +709,7 @@ void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope
 // one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
 template <int block_size, bool has_ff, typename D>
 static __global__ void rms_norm_mul_rope_f32(
-        const float * x, D * dst, const int ncols,
+        const float * x, D * dst, const int ncols, const int nchannels, const int nsamples,
         const int64_t s01, const int64_t s02, const int64_t s03,
         const int64_t s1, const int64_t s2, const int64_t s3,
         const float eps,
@@ -724,66 +724,76 @@ static __global__ void rms_norm_mul_rope_f32(
         const int64_t * row_indices, const int set_rows_stride,
         const bool is_neox) {
     ggml_cuda_pdl_lc();
-    const int row     = blockIdx.x;
-    const int channel = blockIdx.y;
-    const int sample  = blockIdx.z;
-    const int tid     = threadIdx.x;
-
-    x += sample*s03 + channel*s02 + row*s01;
-
-    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
-    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
-    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
-    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;
-
-    float tmp = 0.0f;
-
-    ggml_cuda_pdl_sync();
-    for (int col = tid; col < ncols; col += block_size) {
-        const float xi = x[col];
-        tmp += xi * xi;
-    }
+    const int row = blockIdx.x;
+    const int tid = threadIdx.x;
 
     extern __shared__ float s_sum[];
-    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
-
-    const float scale = rsqrtf(tmp/ncols + eps);
-
-    int64_t idst = sample*s3 + channel*s2 + row*s1;
-    if (set_rows_stride != 0) {
-        idst = row*s1 + row_indices[channel]*set_rows_stride;
-    }
-    dst += idst;
-
-    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
-        int ix0;
-        int ix1;
-        if (is_neox && i0 < n_dims) {
-            ix0 = i0/2;
-            ix1 = i0/2 + n_dims/2;
-        } else {
-            ix0 = i0 + 0;
-            ix1 = i0 + 1;
-        }
 
-        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
-        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];
+    ggml_cuda_pdl_sync();
 
-        if (i0 >= n_dims) {
-            dst[ix0] = ggml_cuda_cast<D>(x0);
-            dst[ix1] = ggml_cuda_cast<D>(x1);
-            continue;
+    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
+    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
+        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
+            const float * xc = x + sample*s03 + channel*s02 + row*s01;
+
+            const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
+            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
+            const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
+            const float * mulc = mul + mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;
+
+            float tmp = 0.0f;
+
+            for (int col = tid; col < ncols; col += block_size) {
+                const float xi = xc[col];
+                tmp += xi * xi;
+            }
+
+            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
+
+            const float scale = rsqrtf(tmp/ncols + eps);
+
+            int64_t idst = sample*s3 + channel*s2 + row*s1;
+            if (set_rows_stride != 0) {
+                idst = row*s1 + row_indices[channel]*set_rows_stride;
+            }
+            D * dstc = dst + idst;
+
+            for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
+                int ix0;
+                int ix1;
+                if (is_neox && i0 < n_dims) {
+                    ix0 = i0/2;
+                    ix1 = i0/2 + n_dims/2;
+                } else {
+                    ix0 = i0 + 0;
+                    ix1 = i0 + 1;
+                }
+
+                const float x0 = scale * xc[ix0] * mulc[fastmodulo(ix0, mul_ncols_packed)];
+                const float x1 = scale * xc[ix1] * mulc[fastmodulo(ix1, mul_ncols_packed)];
+
+                if (i0 >= n_dims) {
+                    dstc[ix0] = ggml_cuda_cast<D>(x0);
+                    dstc[ix1] = ggml_cuda_cast<D>(x1);
+                    continue;
+                }
+
+                const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
+                const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;
+
+                float cos_theta;
+                float sin_theta;
+                rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
+
+                dstc[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
+                dstc[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
+            }
+
+            if constexpr (block_size > WARP_SIZE) {
+                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
+                __syncthreads();
+            }
         }
-
-        const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
-        const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;
-
-        float cos_theta;
-        float sin_theta;
-        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
-
-        dst[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
-        dst[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
     }
 }
 
@@ -806,7 +816,7 @@ static void rms_norm_mul_rope_cuda(
         const bool is_neox, cudaStream_t stream) {
     GGML_ASSERT(ncols % 2 == 0);
 
-    const dim3 blocks_num(nrows, nchannels, nsamples);
+    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
 
     const float theta_scale = powf(freq_base, -2.0f/n_dims);
 
@@ -820,13 +830,13 @@ static void rms_norm_mul_rope_cuda(
         const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
         if (freq_factors == nullptr) {
             ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
-                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
+                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                 mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                 freq_factors, row_indices, set_rows_stride, is_neox);
         } else {
             ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
-                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
+                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                 mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                 freq_factors, row_indices, set_rows_stride, is_neox);
@@ -836,13 +846,13 @@ static void rms_norm_mul_rope_cuda(
         const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
         if (freq_factors == nullptr) {
             ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
-                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
+                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                 mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                 freq_factors, row_indices, set_rows_stride, is_neox);
         } else {
             ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
-                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
+                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                 mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                 n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                 freq_factors, row_indices, set_rows_stride, is_neox);
diff --git src/ggml-cuda/top-k.cu src/ggml-cuda/top-k.cu
index 3ffbba83..b0f60999 100644
--- src/ggml-cuda/top-k.cu
+++ src/ggml-cuda/top-k.cu
@@ -1,6 +1,29 @@
 #include "argsort.cuh"
 #include "top-k.cuh"
 
+// Adjusted implementation thresholds from #28547, can be overridden at build time
+#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
+#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
+// not measured on HIP/MUSA, keep the old split
+#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
+#    else
+#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
+#    endif
+#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
+
+#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
+#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
+#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
+
+// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
+#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
+#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
+#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
+#    else
+#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
+#    endif
+#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
+
 #ifdef GGML_CUDA_USE_CUB
 #    include <cub/cub.cuh>
 // DeviceTopK has a race condition before CCCL 3.4.3.
@@ -14,6 +37,15 @@ using namespace cub;
 #    endif  // CCCL >= 3.4.3
 #endif      // GGML_CUDA_USE_CUB
 
+// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
+#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
+#    ifdef CUB_TOP_K_AVAILABLE
+#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
+#    else
+#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
+#    endif
+#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD
+
 #ifdef CUB_TOP_K_AVAILABLE
 
 static void top_k_cub(ggml_cuda_pool & pool,
@@ -40,7 +72,7 @@ static void top_k_cub(ggml_cuda_pool & pool,
                          ncols, k, env));
 }
 
-#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
+#endif                            // CUB_TOP_K_AVAILABLE
 
 static int next_power_of_2(int x) {
     int n = 1;
@@ -50,10 +82,6 @@ static int next_power_of_2(int x) {
     return n;
 }
 
-#endif                            // CUB_TOP_K_AVAILABLE
-
-#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
-
 static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
     const uint32_t bits = __float_as_uint(value);
     const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
@@ -95,7 +123,7 @@ static __global__ void top_k_radix_histogram(
     __syncthreads();
 
     const top_k_radix_state state = states[row];
-    for (int col = row_block * BLOCK_SIZE + tid;
+    for (int64_t col = row_block * BLOCK_SIZE + tid;
          col < ncols;
          col += blocks_per_row * BLOCK_SIZE) {
         const uint32_t key = top_k_float_to_ordered(row_src[col]);
@@ -165,7 +193,7 @@ static __global__ void top_k_radix_gather(
     int * row_dst = dst + (size_t) row * k;
     top_k_radix_state * state = &states[row];
 
-    for (int col = row_block * BLOCK_SIZE + tid;
+    for (int64_t col = row_block * BLOCK_SIZE + tid;
          col < ncols;
          col += blocks_per_row * BLOCK_SIZE) {
         const uint32_t key = top_k_float_to_ordered(row_src[col]);
@@ -183,36 +211,72 @@ static __global__ void top_k_radix_gather(
 
 static void top_k_radix_cuda(
         ggml_cuda_pool & pool,
-        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
+        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
     constexpr int BLOCK_SIZE = 256;
     constexpr int RADIX_BITS = 8;
     constexpr int NBINS = 1 << RADIX_BITS;
-    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);
+    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
+
+    // chunk the rows to bound the histogram memory to 64 MB
+    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);
 
-    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
-    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
+    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
+    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
     top_k_radix_state * states = states_alloc.get();
     int * histograms = histograms_alloc.get();
 
-    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);
+    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
+        const int iter_nrows = std::min(chunk_nrows, nrows - i);
+
+        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);
+
+        const dim3 row_grid(blocks_per_row * iter_nrows);
+        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
+            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
+                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
+                    src, states, histograms, ncols, blocks_per_row, shift);
+            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
+                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
+        }
 
-    const dim3 row_grid(blocks_per_row * nrows);
-    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
-        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
+        top_k_radix_reset_counters
+            <<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows);
+        top_k_radix_gather<BLOCK_SIZE>
             <<<row_grid, BLOCK_SIZE, 0, stream>>>(
-                src, states, histograms, ncols, blocks_per_row, shift);
-        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
-            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
-    }
+                src, dst, states, ncols, k, blocks_per_row);
 
-    top_k_radix_reset_counters
-        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
-    top_k_radix_gather<BLOCK_SIZE>
-        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
-            src, dst, states, ncols, k, blocks_per_row);
+        src += (size_t) ncols * iter_nrows;
+        dst += (size_t) k     * iter_nrows;
+    }
 }
 
-#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
+static void top_k_argsort_cuda(
+        ggml_cuda_pool & pool,
+        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
+    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);
+
+    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
+    int * tmp = tmp_alloc.get();
+
+    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
+        const int iter_nrows = std::min(chunk_nrows, nrows - i);
+
+        if (use_cub) {
+#ifdef GGML_CUDA_USE_CUB
+            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
+#else
+            GGML_ABORT("CUB is not available");
+#endif // GGML_CUDA_USE_CUB
+        } else {
+            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
+        }
+        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
+                                     cudaMemcpyDeviceToDevice, stream));
+
+        src += (size_t) ncols * iter_nrows;
+        dst += (size_t) k     * iter_nrows;
+    }
+}
 
 void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const ggml_tensor * src0   = dst->src[0];
@@ -229,51 +293,45 @@ void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const int64_t    nrows = ggml_nrows(src0);
     const int64_t    k     = dst->ne[0];
     ggml_cuda_pool & pool  = ctx.pool();
-#ifdef CUB_TOP_K_AVAILABLE
-    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
-    // https://github.com/NVIDIA/cccl/issues/6391
-    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
-    for (int i = 0; i < nrows; i++) {
-        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
-    }
-#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
-    // Fall back to argsort + copy
-    const int    ncols_pad      = next_power_of_2(ncols);
-    const size_t shared_mem     = ncols_pad * sizeof(int);
-    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
-    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
-    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);
 
-    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
-    int *                     tmp_dst = temp_dst_alloc.get();
+    const int device = ggml_cuda_get_device();
 
-    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
-        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);
-
-        if (use_bitonic) {
-            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
-        } else {
-            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
+#ifdef CUB_TOP_K_AVAILABLE
+    // a single row always uses DeviceTopK if available
+    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
+#else
+    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
+#endif // CUB_TOP_K_AVAILABLE
+    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
+                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
+                                  nrows <= ggml_cuda_info().devices[device].nsm;
+
+    if (bitonic_short || bitonic_few_rows) {
+        // the padded row must fit in shared memory
+        const int ncols_pad = next_power_of_2(ncols);
+        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
+            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
+            return;
         }
-        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
-                                     cudaMemcpyDeviceToDevice, stream));
-
-        src0_d += ncols * iter_nrows;
-        dst_d  += k     * iter_nrows;
     }
-#else                             // GGML_CUDA_USE_CUB
-#if defined(GGML_USE_HIP)
-    if (ncols > 1024) {
+
+    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
         top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
+        return;
+    }
+
+#ifdef CUB_TOP_K_AVAILABLE
+    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
+    for (int64_t i = 0; i < nrows; i++) {
+        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
+    }
+#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
+    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
+        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
     } else {
-#endif // defined(GGML_USE_HIP)
-        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
-        int *                     tmp_dst = temp_dst_alloc.get();
-        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
-        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
-                                     cudaMemcpyDeviceToDevice, stream));
-#if defined(GGML_USE_HIP)
+        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
     }
-#endif // defined(GGML_USE_HIP)
-#endif
+#else                             // GGML_CUDA_USE_CUB
+    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
+#endif                            // CUB_TOP_K_AVAILABLE
 }
diff --git src/ggml-cuda/unary.cu src/ggml-cuda/unary.cu
index 84788c5d..436fdfff 100644
--- src/ggml-cuda/unary.cu
+++ src/ggml-cuda/unary.cu
@@ -134,24 +134,67 @@ static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
     ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
 }
 
+template <float (*op)(float), typename T>
+static __global__ void unary_op_kernel_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03) {
+    ggml_cuda_pdl_lc();
+    const int i = blockDim.x*blockIdx.x + threadIdx.x;
+
+    if (i >= k) {
+        return;
+    }
+
+    int64_t rem = i;
+    const int64_t i0 = rem % ne00; rem /= ne00;
+    const int64_t i1 = rem % ne01; rem /= ne01;
+    const int64_t i2 = rem % ne02;
+    const int64_t i3 = rem / ne02;
+    const size_t src_byte_offset = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
+    const T * src_ptr = (const T *)((const char *)x + src_byte_offset);
+
+    ggml_cuda_pdl_sync();
+    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(*src_ptr)));
+}
+
+template <float (*op)(float), typename T>
+static void unary_cuda_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03, cudaStream_t stream) {
+    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
+    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
+    ggml_cuda_kernel_launch(unary_op_kernel_strided<op, T>, launch_params, x, dst, k, ne00,ne01,ne02,nb00,nb01,nb02,nb03);
+}
+
 template <float (*op)(float)>
 void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
     const ggml_tensor * src0 = dst->src[0];
     const void * src0_d = src0->data;
     void * dst_d = dst->data;
-    cudaStream_t stream = ctx.stream();
 
-    GGML_ASSERT(ggml_is_contiguous(src0));
+    cudaStream_t stream = ctx.stream();
 
     GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
     GGML_ASSERT(src0->type == dst->type);
 
-    if (src0->type == GGML_TYPE_F16) {
-        unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
-    } else if (src0->type == GGML_TYPE_BF16) {
-        unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
+    if (ggml_is_contiguous(src0)) {
+        if (src0->type == GGML_TYPE_F16) {
+            unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
+        } else if (src0->type == GGML_TYPE_BF16) {
+            unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
+        } else {
+            unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
+        }
     } else {
-        unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
+        if (src0->type == GGML_TYPE_F16) {
+            unary_cuda_strided<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0),
+                src0->ne[0], src0->ne[1], src0->ne[2],
+                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
+        } else if (src0->type == GGML_TYPE_BF16) {
+            unary_cuda_strided<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0),
+                src0->ne[0], src0->ne[1], src0->ne[2],
+                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
+        } else {
+            unary_cuda_strided<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0),
+                src0->ne[0], src0->ne[1], src0->ne[2],
+                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
+        }
     }
 }
 
@@ -547,8 +590,8 @@ void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
 
     GGML_ASSERT(ggml_is_contiguous(src0));
 
-    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
-    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
+    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
+    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
     GGML_ASSERT(src0->type == dst->type);
 
     const float alpha_n = ggml_get_op_params_f32(dst, 1);
@@ -558,6 +601,8 @@ void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
 
     if (src0->type == GGML_TYPE_F16) {
         xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
+    } else if (src0->type == GGML_TYPE_BF16) {
+        xielu_cuda((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
     } else {
         xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
     }
diff --git src/ggml-hexagon/ggml-hexagon.cpp src/ggml-hexagon/ggml-hexagon.cpp
index 45459447..f3ca7312 100644
--- src/ggml-hexagon/ggml-hexagon.cpp
+++ src/ggml-hexagon/ggml-hexagon.cpp
@@ -1,3 +1,5 @@
+#define _USE_MATH_DEFINES
+
 #include <assert.h>
 #include <inttypes.h>
 #include <stdio.h>
@@ -24,6 +26,10 @@
 #include <cmath>
 #include <initializer_list>
 
+#ifndef M_LOG2E
+#    define M_LOG2E 1.44269504088896340736
+#endif
+
 #ifdef _WIN32
 #    define WIN32_LEAN_AND_MEAN
 #    ifndef NOMINMAX
@@ -47,6 +53,7 @@
 
 #define GGML_COMMON_IMPL_CPP
 #include "ggml-backend-impl.h"
+#include "ggml-alloc.h"
 #include "ggml-common.h"
 #include "ggml-hexagon.h"
 #include "ggml-impl.h"
@@ -60,10 +67,13 @@
 #include "htp/get-rows-ops.h"
 #include "htp/set-rows-ops.h"
 #include "htp/softmax-ops.h"
+#include "htp/pool-ops.h"
 #include "htp/rope-ops.h"
 #include "htp/ssm-conv.h"
 #include "htp/gated-delta-net-ops.h"
 #include "htp/argsort-ops.h"
+#include "htp/concat-ops.h"
+#include "htp/cpy-ops.h"
 #include "htp_iface.h"
 #include "htp-drv.h"
 
@@ -92,15 +102,19 @@ static size_t opt_ndev    = 1;
 static size_t opt_nhvx    = 0; // use all
 static int    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
 static size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
-static size_t opt_mbuf    = 1ul * 1024 * 1024 * 1024; // max buffer size
 static int    opt_etm     = 0;
 static int    opt_verbose = 0;
 static int    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
 static bool   opt_hostbuf = false;
 static bool   opt_dma64   = false;
 
+static size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
+static size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
+static size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)
+
 static int    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
 static int    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
+static int    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
 static int    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
 static int    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
 static int    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction
@@ -111,7 +125,7 @@ static int    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD
 static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };
 
 static int opt_opbatch  = 1280; // max number of ops in a batch
-static int opt_opqueue  = 32;   // max number of pending batches
+static int opt_opqueue  = 8;   // max number of pending batches
 static int opt_optrace  = 0;    // trace buffer size per thread (0 means default)
 static int opt_oppoll   = 0;    // polling for batch completions
 static int opt_opfusion = 1;    // enable/disable op fusion
@@ -384,10 +398,30 @@ static void ggml_hexagon_precompute_sort_params(
     struct htp_sort_kernel_params * kparams
 );
 
+static void ggml_hexagon_precompute_pool_2d_params(
+    const struct ggml_hexagon_session * sess,
+    const struct ggml_tensor * src0,
+    const struct ggml_tensor * dst,
+    struct htp_pool_2d_kernel_params * kparams,
+    bool is_pool_1d
+);
+
+static bool ggml_hexagon_precompute_concat_params(
+    const struct ggml_hexagon_session * sess,
+    const struct ggml_tensor * op,
+    struct htp_concat_kernel_params * kparams
+);
+
+static bool ggml_hexagon_precompute_cpy_params(
+    const struct ggml_hexagon_session * sess,
+    const struct ggml_tensor * op,
+    struct htp_copy_kernel_params * kparams
+);
+
 static void ggml_hexagon_precompute_fused_mmnx_params(
     const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0,
-    const struct ggml_tensor * src1,
+    const struct ggml_tensor * act,
     int32_t n_weights,
     struct htp_mm_kernel_params * kparams
 );
@@ -395,7 +429,7 @@ static void ggml_hexagon_precompute_fused_mmnx_params(
 static void ggml_hexagon_precompute_fused_mmidnx_params(
     const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0,
-    const struct ggml_tensor * src1,
+    const struct ggml_tensor * act,
     const struct ggml_tensor * dst,
     int32_t n_weights,
     struct htp_mm_kernel_params * kparams
@@ -412,6 +446,10 @@ static bool ggml_hexagon_precompute_allreduce_params(
     struct htp_allreduce_kernel_params * kparams
 );
 
+static bool ggml_hexagon_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride);
+static bool ggml_hexagon_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
+static ggml_tensor ggml_hexagon_tensor_collapse_rows(const struct ggml_tensor * t);
+
 static bool mm_is_hmx_eligible(const ggml_tensor * t);
 static htp_op_code op_remap_to_htp(const ggml_tensor * t);
 static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams);
@@ -444,6 +482,13 @@ static inline bool ggml_hexagon_tensors_overlap(const struct ggml_tensor * a, co
     return a0 < b1 && b0 < a1;
 }
 
+static inline bool ggml_hexagon_can_row_partition(const struct ggml_tensor * t) {
+    if (t->ne[1] > 1 && (t->nb[1] & 127) != 0) return false;
+    if (t->ne[2] > 1 && (t->nb[2] & 127) != 0) return false;
+    if (t->ne[3] > 1 && (t->nb[3] & 127) != 0) return false;
+    return true;
+}
+
 struct htp_opnode;
 
 struct ggml_hexagon_opbatch;
@@ -2779,8 +2824,30 @@ static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffe
     GGML_UNUSED(buft);
 }
 
+static size_t parse_size(const char * str, size_t default_unit = 1024 * 1024) {
+    if (!str || str[0] == '\0') {
+        return 0;
+    }
+    char * end = NULL;
+    double val = strtod(str, &end);
+    if (val < 0) {
+        return 0;
+    }
+    if (end && *end) {
+        while (*end == ' ') end++;
+        if (*end == 'k' || *end == 'K') {
+            return (size_t) (val * 1024);
+        } else if (*end == 'm' || *end == 'M') {
+            return (size_t) (val * 1024 * 1024);
+        } else if (*end == 'g' || *end == 'G') {
+            return (size_t) (val * 1024 * 1024 * 1024);
+        }
+    }
+    return (size_t) (val * default_unit);
+}
+
 static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
-    return opt_mbuf;
+    return opt_mbuf_dyn;
     GGML_UNUSED(buft);
 }
 
@@ -2794,25 +2861,199 @@ static bool ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_ty
     GGML_UNUSED(buft);
 }
 
+struct ggml_backend_hexagon_alloc_buffer_n_plan_item {
+    size_t size;
+    int    first;
+    int    last;
+};
+
+using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;
+
+static const char * ggml_hexagon_kv_layer_suffix(const struct ggml_tensor * t) {
+    if (strncmp(t->name, "cache_", 6) != 0) {
+        return NULL;
+    }
+    const char * p = strstr(t->name, "_l");
+    if (!p || !isdigit((unsigned char)p[2])) {
+        return NULL;
+    }
+    return p;
+}
+
+struct ggml_backend_hexagon_alloc_unit {
+    size_t size;
+    int    first;
+    int    last;
+};
+
+static ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
+        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
+    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;
+
+    const size_t alignment = ggml_backend_buft_get_alignment(buft);
+    const size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;
+
+    std::vector<ggml_backend_hexagon_alloc_unit> units;
+
+    int i = 0;
+    while (i < n_tensors) {
+        struct ggml_tensor * t = tensors[i];
+        size_t unit_size = 0;
+        int unit_first = i;
+        int unit_last = i + 1;
+
+        if (t->data == NULL && t->view_src == NULL) {
+            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
+        }
+
+        const char * layer_suffix = ggml_hexagon_kv_layer_suffix(t);
+
+        while (unit_last < n_tensors) {
+            struct ggml_tensor * next = tensors[unit_last];
+
+            if (next->view_src != NULL) {
+                unit_last++;
+                continue;
+            }
+
+            if (layer_suffix != NULL) {
+                const char * next_suffix = ggml_hexagon_kv_layer_suffix(next);
+                if (next_suffix != NULL && strcmp(layer_suffix, next_suffix) == 0) {
+                    if (next->data == NULL) {
+                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
+                    }
+                    unit_last++;
+                    continue;
+                }
+            }
+
+            break;
+        }
+
+        units.push_back({ unit_size, unit_first, unit_last });
+        i = unit_last;
+    }
+
+    size_t cur_buf_size  = 0;
+    int    cur_buf_first = 0;
+
+    for (const auto & unit : units) {
+        if (unit.size == 0) {
+            continue;
+        }
+
+        if (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
+            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
+            cur_buf_size  = 0;
+            cur_buf_first = unit.first;
+        }
+
+        cur_buf_size += unit.size;
+    }
+
+    if (cur_buf_size > 0) {
+        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
+    }
+
+    return plan;
+}
+
+static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
+        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
+    const ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);
+
+    std::vector<ggml_backend_buffer_t> buffers;
+    buffers.reserve(plan.size());
+
+    for (const auto & item : plan) {
+        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
+        if (buffer == NULL) {
+            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
+            for (ggml_backend_buffer_t b : buffers) {
+                ggml_backend_buffer_free(b);
+            }
+            return NULL;
+        }
+
+        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);
+
+        struct ggml_tensor * t_failed = NULL;
+        for (int j = item.first; j < item.last; j++) {
+            struct ggml_tensor * t = tensors[j];
+            if (t->data == NULL) {
+                if (t->view_src == NULL) {
+                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
+                        t_failed = t;
+                        break;
+                    }
+                } else if (t->buffer == NULL) {
+                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
+                        t_failed = t;
+                        break;
+                    }
+                }
+            } else {
+                if (t->view_src != NULL && t->buffer == NULL) {
+                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
+                        t_failed = t;
+                        break;
+                    }
+                }
+            }
+        }
+        if (t_failed != NULL) {
+            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
+            for (ggml_backend_buffer_t b : buffers) {
+                ggml_backend_buffer_free(b);
+            }
+            ggml_backend_buffer_free(buffer);
+            return NULL;
+        }
+
+        buffers.push_back(buffer);
+    }
+
+    if (buffers.empty()) {
+        return NULL;
+    }
+
+    if (buffers.size() == 1) {
+        return buffers[0];
+    }
+
+    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
+}
+
+static size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
+        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
+    const ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);
+
+    size_t total = 0;
+    for (const auto & item : plan) {
+        total += item.size;
+    }
+    return total;
+}
+
 static ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
     /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
     /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
-    /* .alloc_buffer_n      = */ NULL,
+    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
     /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
     /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
     /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
-    /* .get_alloc_size_n    = */ NULL,
+    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
     /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
 };
 
 static ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
     /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
     /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
-    /* .alloc_buffer_n      = */ NULL,
+    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
     /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
     /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
     /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
-    /* .get_alloc_size_n    = */ NULL,
+    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
     /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
 };
 
@@ -3410,6 +3651,10 @@ struct ggml_hexagon_opbatch {
             return false;
         }
 
+        if (orig_kparams->collapse != kparams.collapse) {
+            return false;
+        }
+
         const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
         const bool can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
         if (!can_fuse) return false;
@@ -3493,8 +3738,20 @@ struct ggml_hexagon_opbatch {
                 return false;
             }
 
+            const struct htp_mm_kernel_params * orig_kparams = (const struct htp_mm_kernel_params *) last_node.kernel_params;
+            const bool collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
+            if (orig_kparams->collapse && !collapse) {
+                return false;
+            }
+
             struct htp_mm_kernel_params kparams;
-            ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
+            if (collapse) {
+                const ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
+                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
+                kparams.collapse = 1;
+            } else {
+                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
+            }
             if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                 return false;
             }
@@ -3543,9 +3800,23 @@ struct ggml_hexagon_opbatch {
             const ggml_tensor * w0 = last_node.src0();
             const ggml_tensor * x  = last_node.src1();
             const ggml_tensor * w1 = node.src0();
+            const ggml_tensor * dst_0 = last_node.dst();
+            const ggml_tensor * dst_1 = node.dst();
+
+            const struct htp_mm_kernel_params * orig_kparams = (const struct htp_mm_kernel_params *) last_node.kernel_params;
+            const bool collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
+            if (orig_kparams->collapse && !collapse) {
+                return false;
+            }
 
             struct htp_mm_kernel_params kparams;
-            ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
+            if (collapse) {
+                const ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
+                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
+                kparams.collapse = 1;
+            } else {
+                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
+            }
             if (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                 return false;
             }
@@ -3559,9 +3830,6 @@ struct ggml_hexagon_opbatch {
                 return false;
             }
 
-            const ggml_tensor * dst_0 = last_node.dst();
-            const ggml_tensor * dst_1 = node.dst();
-
             last_node.opcode = HTP_OP_MUL_MAT_NX;
             last_node.name   = "MUL_MAT_NX";
             last_node.inputs.clear();
@@ -4206,6 +4474,11 @@ void ggml_hexagon_session::enqueue_cpy(const ggml_tensor * src, ggml_tensor * ds
     if (with_fence) {
         cpy_node.name = "CPY+FENCE";
     }
+    const bool ok = ggml_hexagon_precompute_cpy_params(this, node, (struct htp_copy_kernel_params *) cpy_node.kernel_params);
+    const auto * kparams = (const struct htp_copy_kernel_params *) cpy_node.kernel_params;
+    if (ok && !with_fence && kparams->total_elems == 0) {
+        return;
+    }
     this->enqueue_op(cpy_node);
 }
 
@@ -4831,15 +5104,48 @@ static bool ggml_hexagon_flash_attn_is_hmx_eligible(
         return false;
     }
 
-    // Fall back to HVX for small token counts if head dimension is small (DK <= 128)
-    const uint32_t neq1 = q->ne[1];
-    if (DK <= 128 && neq1 < 5) {
-        return false;
+    GGML_UNUSED(sinks);
+
+    // Explicit force mode
+    if (opt_fa_select > 2) {
+        return true;
     }
 
-    return true;
+    const uint32_t M = q->ne[1];
 
-    GGML_UNUSED(sinks);
+    // Prefill or batched decode
+    if (M > 1) {
+        return true;
+    }
+
+    // Compute-bound head dim
+    if (DK >= 256) {
+        return true;
+    }
+
+    const uint32_t n_head = q->ne[2];
+    const uint32_t n_kv_heads = k->ne[2];
+    const uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
+    const uint32_t S = k->ne[1];
+
+    // Tile alignment for 32-row HMX tiles
+    const bool is_tile_aligned = (G > 0 && (32 % G == 0));
+    if (!is_tile_aligned) {
+        if (sess->n_threads >= 6) {
+            return false;
+        }
+        return S >= 1024;
+    }
+
+    // Context depth crossover
+    uint32_t s_cross = 512;
+    if (DK <= 64) {
+        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
+    } else {
+        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
+    }
+
+    return S >= s_cross;
 }
 
 static bool ggml_hexagon_precompute_flash_attn_params(
@@ -4857,7 +5163,6 @@ static bool ggml_hexagon_precompute_flash_attn_params(
     const struct ggml_tensor * k    = op->src[1];
     const struct ggml_tensor * v    = op->src[2];
     const struct ggml_tensor * mask = op->src[3];
-    const struct ggml_tensor * dst  = op;
 
     const uint32_t neq0 = q->ne[0];  // head_dim (DK)
     const uint32_t neq1 = q->ne[1];  // n_tokens
@@ -4888,8 +5193,8 @@ static bool ggml_hexagon_precompute_flash_attn_params(
     kparams->max_bias = max_bias;
     kparams->logit_softcap = logit_softcap;
 
-    kparams->is_q_fp32 = (q->type == GGML_TYPE_F32) ? 1 : 0;
-    kparams->is_dst_fp32 = (dst->type == GGML_TYPE_F32) ? 1 : 0;
+    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
+    kparams->flags      = 0;
     kparams->G = G;
 
     const uint32_t n_head = q->ne[2];
@@ -4906,9 +5211,16 @@ static bool ggml_hexagon_precompute_flash_attn_params(
         const uint32_t DK_pad = hex_round_up(DK, 64);
         const uint32_t DV_pad = hex_round_up(DV, 64);
         size_t Br = 0, Bc = 0;
-        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, kparams->is_q_fp32 != 0, sinks != nullptr, n_head);
+        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
         if (ret == 0) {
             kparams->kernel_type = HTP_FA_KERNEL_HMX;
+            if (logit_softcap == 0.0f) {
+                kparams->scale = scale * (float) M_LOG2E;
+                kparams->logit_softcap = 0.0f;
+            } else {
+                kparams->scale = scale;
+                kparams->logit_softcap = logit_softcap * (float) M_LOG2E;
+            }
             kparams->Br = Br;
             kparams->Bc = Bc;
             kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
@@ -4916,7 +5228,7 @@ static bool ggml_hexagon_precompute_flash_attn_params(
 
             kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
             kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
-            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, kparams->is_q_fp32 != 0, sinks != nullptr, n_head);
+            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
 
             const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
             kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes
@@ -4943,11 +5255,11 @@ static bool ggml_hexagon_precompute_flash_attn_params(
     kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
     kparams->n_threads = sess->n_threads;
 
-    const size_t size_q_row_padded = hex_round_up(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2), 128);
+    const size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
     const size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
     const size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);
 
-    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, kparams->is_q_fp32 != 0, mask != nullptr, sinks != nullptr, n_head, sess->n_threads);
+    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);
 
     kparams->u.hvx.size_q_row_padded = size_q_row_padded;
     kparams->u.hvx.size_k_row_padded = size_k_row_padded;
@@ -5102,7 +5414,7 @@ static bool ggml_hexagon_matmul_is_hmx_eligible(
     bool is_matmul_id,
     bool is_batched
 ) {
-    if (src1->type != GGML_TYPE_F32) {
+    if (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
         return false;
     }
 
@@ -5111,8 +5423,8 @@ static bool ggml_hexagon_matmul_is_hmx_eligible(
     const int ne12  = src1->ne[2];
     const int wtype = src0->type;
 
-    // HMX weight tile requires N to be 32-aligned.
-    if (ne01_padded % 32 != 0) {
+    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
+    if (ne01_padded % 32 != 0 && is_matmul_id) {
         return false;
     }
 
@@ -5151,7 +5463,7 @@ static bool ggml_hexagon_matmul_is_hmx_eligible(
 static bool ggml_hexagon_precompute_hmx_mm_params(
     const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0,
-    const struct ggml_tensor * src1,
+    const struct ggml_tensor * act,
     const struct ggml_tensor * dst,
     int wtype,
     int ne00_padded,
@@ -5167,9 +5479,22 @@ static bool ggml_hexagon_precompute_hmx_mm_params(
     struct htp_mm_kernel_params * kparams
 ) {
     const int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
-    const bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(ne11);
     const int n_threads = (int)sess->n_threads;
-    const int ne10 = src1->ne[0];
+    const int ne10 = act->ne[0];
+
+    int m_for_solver = ne11;
+    int m_for_solver_padded = ne11_padded;
+    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
+    if (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
+        // when dst is null, padded dims are used for estimate which are 128-byte aligned
+        const bool dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : true;
+        const bool act_row_split = ggml_hexagon_can_row_partition(act);
+        if (dst_row_split && act_row_split) {
+            m_for_solver = (ne11 + (int) sess->mdev.count - 1) / (int) sess->mdev.count;
+            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
+        }
+    }
+    const bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(m_for_solver);
 
     const bool is_batched_val = is_matmul_id ? false : is_batched;
     const int group_size = (ne02 > 0 ? ne12 / ne02 : 1);
@@ -5182,15 +5507,25 @@ static bool ggml_hexagon_precompute_hmx_mm_params(
 
     if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
         // Try grouped path first
-        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, ne11, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
+        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
             use_grouped = true;
         }
     }
 
     if (!use_grouped) {
         // Fallback to simple 2D path (group_size = 1)
-        const int m_id_rows = (dst && is_matmul_id) ? (int) ((size_t) dst->ne[1] * dst->ne[2]) : 0;
-        if (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, ne11_padded, ne11, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
+        int m_id_rows = 0;
+        if (dst && is_matmul_id) {
+            const int n_experts = ne02 > 0 ? ne02 : 1;
+            const size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
+            int m_per_expert = (int) ((total_expert_rows + n_experts - 1) / n_experts);
+            if (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
+                m_per_expert = (m_per_expert + (int) sess->mdev.count - 1) / (int) sess->mdev.count;
+            }
+            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
+        }
+        const uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
+        if (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
             return false;
         }
     }
@@ -5203,14 +5538,14 @@ static bool ggml_hexagon_precompute_hmx_mm_params(
     kparams->n_act_threads = act_threads_selected;
     kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
     kparams->aligned_tile_size = aligned_tile_size;
-    kparams->src1_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+    kparams->act_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
     kparams->vtcm_size = vtcm_size;
     kparams->vtcm_src0_size = 0;
     kparams->div_n_act_threads = init_fastdiv_values(act_threads_selected);
     kparams->div_ne00_padded   = init_fastdiv_values(ne00_padded);
-    kparams->vtcm_src1_size = 0;
-    kparams->vtcm_src2_size = (int32_t) src2_size;
-    kparams->vtcm_dst_size = 0;
+    kparams->vtcm_act_size     = 0;
+    kparams->vtcm_bias_size    = (int32_t) src2_size;
+    kparams->vtcm_dst_size     = 0;
 
     if (is_batched && !is_matmul_id) {
         kparams->kernel_type = HTP_MM_KERNEL_HMX_F16_BATCHED;
@@ -5247,6 +5582,9 @@ static void ggml_hexagon_precompute_hvx_mm_params(
     kparams->n_hmx = 0;
     kparams->n_threads = sess->n_threads;
 
+    GGML_UNUSED(ne02);
+    GGML_UNUSED(ne03);
+
     const bool is_quant = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
     const int src1_nrows = ne11 * ne12 * ne13;
 
@@ -5259,7 +5597,7 @@ static void ggml_hexagon_precompute_hvx_mm_params(
 
         if (is_matmul_id) {
             kparams->kernel_type   = (src1_nrows < (int) sess->n_threads) ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
-            kparams->src1_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+            kparams->act_row_size  = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
 
             struct htp_mm_hvx_vtcm_layout L;
             uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
@@ -5267,7 +5605,7 @@ static void ggml_hexagon_precompute_hvx_mm_params(
             for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                 htp_mm_hvx_vtcm_layout_build(
                     &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
-                    0, src0->nb[1], kparams->src1_row_size, 0, d, true, false
+                    0, src0->nb[1], kparams->act_row_size, 0, d, true, false
                 );
                 if (L.total_bytes <= vtcm_budget) {
                     best_n_prefetch = d;
@@ -5281,13 +5619,14 @@ static void ggml_hexagon_precompute_hvx_mm_params(
             kparams->n_prefetch     = best_n_prefetch;
             kparams->vtcm_size      = L.total_bytes;
             kparams->vtcm_src0_size = L.src0_bytes;
-            kparams->vtcm_src1_size = L.src1_bytes;
+            kparams->vtcm_act_size  = L.act_bytes;
+            kparams->vtcm_bias_size = 0;
             kparams->vtcm_dst_size  = L.dst_bytes;
             goto done_quant;
         } else {
             bool try_tiled = (k_align && opt_mm_select >= 1);
             if (try_tiled) {
-                kparams->src1_row_size = htp_mm_weight_has_offset(wtype)
+                kparams->act_row_size = htp_mm_weight_has_offset(wtype)
                                        ? htp_mm_q8_1_tiled_row_size(ne10)
                                        : htp_mm_q8_0_tiled_row_size(ne10);
                 if (src1_nrows < (int) sess->n_threads) {
@@ -5319,8 +5658,8 @@ static void ggml_hexagon_precompute_hvx_mm_params(
                     kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                     kparams->vtcm_size      = L.total_bytes;
                     kparams->vtcm_src0_size = L.src0_bytes;
-                    kparams->vtcm_src1_size = L.src1_bytes;
-                    kparams->vtcm_src2_size = L.src2_bytes;
+                    kparams->vtcm_act_size  = L.act_bytes;
+                    kparams->vtcm_bias_size = L.bias_bytes;
                     kparams->vtcm_dst_size  = L.dst_bytes;
                     goto done_quant;
                 }
@@ -5341,11 +5680,11 @@ static void ggml_hexagon_precompute_hvx_mm_params(
                 &L, &m_chunk)) {
             kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
             kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
-            kparams->src1_row_size = hex_round_up(ne10 * 2, 128);
+            kparams->act_row_size = hex_round_up(ne10 * 2, 128);
             kparams->vtcm_size = L.total_bytes;
             kparams->vtcm_src0_size = L.src0_bytes;
-            kparams->vtcm_src1_size = L.src1_bytes;
-            kparams->vtcm_src2_size = L.src2_bytes;
+            kparams->vtcm_act_size  = L.act_bytes;
+            kparams->vtcm_bias_size = L.bias_bytes;
             kparams->vtcm_dst_size = L.dst_bytes;
             kparams->n_prefetch = 16;
             return;
@@ -5363,11 +5702,11 @@ static void ggml_hexagon_precompute_hvx_mm_params(
                 &L, &m_chunk)) {
             kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
             kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
-            kparams->src1_row_size = hex_round_up(ne10 * 4, 128);
+            kparams->act_row_size = hex_round_up(ne10 * 4, 128);
             kparams->vtcm_size = L.total_bytes;
             kparams->vtcm_src0_size = L.src0_bytes;
-            kparams->vtcm_src1_size = L.src1_bytes;
-            kparams->vtcm_src2_size = L.src2_bytes;
+            kparams->vtcm_act_size  = L.act_bytes;
+            kparams->vtcm_bias_size = L.bias_bytes;
             kparams->vtcm_dst_size = L.dst_bytes;
             kparams->n_prefetch = 16;
             return;
@@ -5404,6 +5743,9 @@ static void ggml_hexagon_precompute_matmul_params_impl(
     const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : ne00;
     const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : ne01;
     const int ne11_padded = hex_round_up(ne11, 32);
+    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
+    // even when the tensor itself is ragged.
+    const int  ne01_tiled  = hex_round_up(ne01_padded, 32);
 
     const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
     const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);
@@ -5413,7 +5755,7 @@ static void ggml_hexagon_precompute_matmul_params_impl(
     // Check HMX eligibility and try precomputing HMX parameters
     bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
     if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, ne01_padded, is_matmul_id, is_batched)) {
-        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_padded, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
+        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
             goto finalize;
         }
     }
@@ -5429,6 +5771,66 @@ finalize:
     kparams->div_ne12     = init_fastdiv_values(ne12);
 }
 
+// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
+static bool ggml_hexagon_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride) {
+    size_t s = 0, next = 0;
+    for (int i = 1; i < GGML_MAX_DIMS; i++) {
+        if (ne[i] == 1) continue;
+        if (s == 0) { s = nb[i]; next = s * ne[i]; continue; }
+        if (nb[i] != next) return false;
+        next *= ne[i];
+    }
+    *stride = s ? s : nb[1];
+    return true;
+}
+
+// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
+static bool ggml_hexagon_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
+    size_t s1, sd;
+    return (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
+           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
+           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
+           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
+}
+
+static bool ggml_hexagon_matmul_add_can_collapse(
+    const struct ggml_tensor * src0,
+    const struct ggml_tensor * src1,
+    const struct ggml_tensor * src2,
+    const struct ggml_tensor * dst
+) {
+    if (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
+        return false;
+    }
+    if (!src2) {
+        return true;
+    }
+    const int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
+    if (src2_nrows == 1) {
+        return true;
+    }
+    size_t s2;
+    return src2->nb[0] == ggml_type_size(src2->type) &&
+           src2->ne[0] == dst->ne[0] &&
+           src2->ne[1] == dst->ne[1] &&
+           src2->ne[2] == dst->ne[2] &&
+           src2->ne[3] == dst->ne[3] &&
+           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
+}
+
+static ggml_tensor ggml_hexagon_tensor_collapse_rows(const struct ggml_tensor * t) {
+    size_t stride = 0;
+    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
+    ggml_tensor c = *t;
+    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
+    c.ne[2] = 1;
+    c.ne[3] = 1;
+    c.nb[1] = stride;
+    c.nb[2] = c.nb[1] * c.ne[1];
+    c.nb[3] = c.nb[2];
+    return c;
+}
+
 static void ggml_hexagon_precompute_matmul_params(
     const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0,
@@ -5436,6 +5838,13 @@ static void ggml_hexagon_precompute_matmul_params(
     const struct ggml_tensor * dst,
     struct htp_mm_kernel_params * kparams
 ) {
+    if (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
+        const ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
+        const ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
+        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
+        kparams->collapse = 1;
+        return;
+    }
     ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
 }
 
@@ -5447,6 +5856,18 @@ static void ggml_hexagon_precompute_fused_matmul_add_params(
     const struct ggml_tensor * dst,
     struct htp_mm_kernel_params * kparams
 ) {
+    if (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
+        const ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
+        const ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
+        const ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
+                                           ? ggml_hexagon_tensor_collapse_rows(src2)
+                                           : (src2 ? *src2 : ggml_tensor{});
+        const struct ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
+        const size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
+        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
+        kparams->collapse = 1;
+        return;
+    }
     const size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
     ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
 }
@@ -5878,68 +6299,63 @@ static void ggml_hexagon_precompute_ssm_conv_params(
 
     const uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
     const uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);
-    kparams->d_inner_per_thread = d_inner_per_thread;
 
-    kparams->src0_row_size_aligned = hex_round_up(ncs * sizeof(float), 128);
-    kparams->src1_row_size_aligned = hex_round_up(d_conv * sizeof(float), 128);
-    kparams->dst_row_size_aligned  = hex_round_up(d_inner * sizeof(float), 128);
+    const uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
+    const uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
+    const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;
+
+    uint32_t vtcm_src0_per_thread = 0;
+    uint32_t vtcm_dst_per_thread  = 0;
 
     if (n_t == 1) {
         kparams->d_inner_tile = d_inner_per_thread;
 
-        const uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
-        const uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
-        const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;
-
-        const uint32_t src0_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
-        const uint32_t src0_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
-        const uint32_t vtcm_src0_per_thread = src0_raw_bytes + src0_T_bytes;
+        const uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128);
+        const uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
+        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;
 
-        const uint32_t vtcm_dst_per_thread = hex_round_up(d_inner_per_thread * sizeof(float), 128);
-
-        kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
-        kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
-        kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;
-
-        kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
-        kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
-        kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
-        kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
+        const uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * sizeof(float), 128);
+        vtcm_dst_per_thread = 2 * dst_tile_bytes;
     } else {
-        const uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
-        const uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
-        const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;
-
         const size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));
-        const size_t avail_for_src0 = vtcm_budget > vtcm_src1_per_thread ? vtcm_budget - vtcm_src1_per_thread : (128 * 1024);
 
-        uint32_t d_inner_tile = (uint32_t)((avail_for_src0 / 2) / (ncs * sizeof(float) + n_t * sizeof(float) + 1));
+        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
+        // one 32-channel block at a time
+        const uint32_t src0_block_T    = hex_round_up(ncs * 32 * sizeof(float), 128);
+        const size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
+        const size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);
+
+        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
+        target_max_tile = (std::max)(target_max_tile, 32u);
+        target_max_tile = (std::min)(target_max_tile, 128u);
+
+        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * sizeof(float)));
         d_inner_tile = (d_inner_tile / 32) * 32;
         if (d_inner_tile == 0) {
             d_inner_tile = 32;
         }
+        if (d_inner_tile > target_max_tile) {
+            d_inner_tile = target_max_tile;
+        }
         if (d_inner_tile > d_inner_per_thread) {
             d_inner_tile = d_inner_per_thread;
         }
         kparams->d_inner_tile = d_inner_tile;
 
-        const uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * sizeof(float), 128) + 128;
-        const uint32_t src0_tile_T   = hex_round_up(ncs * d_inner_tile * sizeof(float), 128);
-        const uint32_t vtcm_src0_per_thread = src0_tile_raw + src0_tile_T;
+        const uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * sizeof(float), 128);
+        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;
 
-        const uint32_t vtcm_dst_per_thread = hex_round_up(d_inner_tile * n_t * sizeof(float), 128);
-
-        kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
-        kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
-        kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;
-
-        kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
-        kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
-        kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
-        kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
+        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * sizeof(float), 128);
     }
 
-    kparams->div_n_threads = init_fastdiv_values(n_threads);
+    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
+    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
+    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;
+
+    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
+    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
+    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
+    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
 }
 
 static void ggml_hexagon_precompute_gated_delta_net_params(
@@ -6063,10 +6479,302 @@ static void ggml_hexagon_precompute_sort_params(
     kparams->n_slots           = (int32_t) layout.n_slots;
 }
 
+static void ggml_hexagon_pool_interior_range(
+    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
+    uint32_t * ox_lo, uint32_t * ox_hi) {
+    const uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
+    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;
+
+    const int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
+    if (numer_hi < 0) {
+        *ox_hi = 0;
+    } else {
+        const uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
+        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
+    }
+    if (*ox_hi < *ox_lo) {
+        *ox_hi = *ox_lo;
+    }
+}
+
+static bool ggml_hexagon_pool_shape_fits(
+    const struct ggml_tensor * src0,
+    const struct ggml_tensor * dst
+) {
+    return ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
+}
+
+static void ggml_hexagon_precompute_pool_2d_params(
+    const struct ggml_hexagon_session * sess,
+    const struct ggml_tensor * src0,
+    const struct ggml_tensor * dst,
+    struct htp_pool_2d_kernel_params * kparams,
+    bool is_pool_1d
+) {
+    memset(kparams, 0, sizeof(*kparams));
+
+    const uint32_t src_x = (uint32_t) src0->ne[0];
+    const uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
+    const uint32_t dst_x = (uint32_t) dst->ne[0];
+    const uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
+    const uint32_t src_plane_bytes = src_x * src_y * sizeof(float);
+    const uint32_t dst_plane_bytes = dst_x * dst_y * sizeof(float);
+    const uint32_t planes = (uint32_t) (is_pool_1d
+        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
+        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
+    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);
+
+    struct htp_pool_vtcm_layout layout;
+    const bool ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
+    GGML_ASSERT(ok);
+
+    kparams->src_x = src_x;
+    kparams->src_y = src_y;
+    kparams->dst_x = dst_x;
+    kparams->dst_y = dst_y;
+    kparams->kernel_x = (uint32_t) ggml_get_op_params_i32(dst, 1);
+    kparams->kernel_y = is_pool_1d ? 1 : (uint32_t) ggml_get_op_params_i32(dst, 2);
+    kparams->stride_x = (uint32_t) ggml_get_op_params_i32(dst, is_pool_1d ? 2 : 3);
+    kparams->stride_y = is_pool_1d ? 1 : (uint32_t) ggml_get_op_params_i32(dst, 4);
+    kparams->pad_x = ggml_get_op_params_i32(dst, is_pool_1d ? 3 : 5);
+    kparams->pad_y = is_pool_1d ? 0 : ggml_get_op_params_i32(dst, 6);
+    kparams->src_plane_bytes = src_plane_bytes;
+    kparams->dst_plane_bytes = dst_plane_bytes;
+    kparams->src_plane_bytes_aligned = (uint32_t) layout.src_spad_half_size;
+    kparams->dst_plane_bytes_aligned = (uint32_t) layout.dst_spad_half_size;
+    kparams->n_threads = n_threads;
+    kparams->planes = planes;
+    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
+    // Fast HVX path requires exact tiling and a supported kernel width.
+    kparams->fast_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
+                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
+                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
+    kparams->global_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
+                            kparams->stride_x == kparams->kernel_x &&
+                            kparams->stride_y == kparams->kernel_y &&
+                            kparams->kernel_x == kparams->src_x &&
+                            kparams->kernel_y == kparams->src_y &&
+                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
+    kparams->block_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
+                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
+                           kparams->stride_x == kparams->kernel_x &&
+                           kparams->kernel_x >= 32) ? 1 : 0;
+    kparams->avg_divide_count = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
+    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
+                                     kparams->stride_x, kparams->pad_x,
+                                     &kparams->ox_lo, &kparams->ox_hi);
+
+    const bool narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
+    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
+    kparams->inv_kernel_area = 1.0f / (float) (kparams->kernel_x * kparams->kernel_y);
+}
+
+static bool ggml_hexagon_precompute_concat_params(
+    const struct ggml_hexagon_session * sess,
+    const struct ggml_tensor * op,
+    struct htp_concat_kernel_params * kparams
+) {
+    memset(kparams, 0, sizeof(*kparams));
+    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;
+
+    const struct ggml_tensor * src0 = op->src[0];
+    const struct ggml_tensor * src1 = op->src[1];
+    const struct ggml_tensor * dst  = op;
+
+    if (!src0 || !src1 || !dst) {
+        return false;
+    }
+
+    int dim = ((const int32_t *) op->op_params)[0];
+    if (dim < 0 || dim >= GGML_MAX_DIMS) {
+        return false;
+    }
+    kparams->dim = dim;
+
+    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
+        return false;
+    }
+    if (src0->type != dst->type || src1->type != dst->type) {
+        return false;
+    }
+
+    const uint32_t type_size = ggml_type_size(dst->type);
+
+    for (int d = 0; d < GGML_MAX_DIMS; d++) {
+        const int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
+        if (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
+            return false;
+        }
+    }
+
+    const bool dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);
+
+    if (dma_strides_ok) {
+        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
+        kparams->n_threads   = 1;
+        return true;
+    }
+
+    const bool is_src1_transposed = (src1->nb[0] > src1->nb[1]);
+    const bool is_src0_transposed = (src0->nb[0] > src0->nb[1]);
+    const bool transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);
+
+    if (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
+        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {
+
+        const uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
+        struct htp_concat_transposed_vtcm_layout layout;
+        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);
+
+        if (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
+            return false;
+        }
+
+        kparams->kernel_type           = HTP_CONCAT_KERNEL_TRANSPOSED;
+        kparams->n_threads             = n_threads;
+        kparams->vtcm_size             = layout.total_bytes;
+        kparams->spad0_size_per_thread = layout.src0_spad_size_per_thread;
+        kparams->spad1_size_per_thread = layout.src1_spad_size_per_thread;
+        return true;
+    }
+
+    return false;
+}
+
+static bool ggml_hexagon_precompute_cpy_params(
+    const struct ggml_hexagon_session * sess,
+    const struct ggml_tensor * op,
+    struct htp_copy_kernel_params * kparams
+) {
+    memset(kparams, 0, sizeof(*kparams));
+    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;
+
+    const struct ggml_tensor * src0 = op->src[0];
+    const struct ggml_tensor * dst  = op;
+
+    if (!src0 || !dst) {
+        return false;
+    }
+
+    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
+        return false;
+    }
+    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
+        return false;
+    }
+
+    const int64_t nelem_src = ggml_nelements(src0);
+    const int64_t nelem_dst = ggml_nelements(dst);
+    if (nelem_src != nelem_dst || nelem_src < 0) {
+        return false;
+    }
+
+    const uint32_t src_type_size = ggml_type_size(src0->type);
+    const uint32_t dst_type_size = ggml_type_size(dst->type);
+
+    kparams->src0_type_size = (uint8_t) src_type_size;
+    kparams->dst_type_size  = (uint8_t) dst_type_size;
+    kparams->total_elems    = (uint32_t) nelem_src;
+
+    if (nelem_src == 0) {
+        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
+        return true;
+    }
+
+    if (nelem_src == 1) {
+        if (src0->type == dst->type) {
+            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
+            return true;
+        }
+        if ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
+            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
+            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
+            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
+            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
+            return true;
+        }
+        return false;
+    }
+
+    const bool sametype = (src0->type == dst->type);
+
+    bool same_extents = true;
+    for (int d = 0; d < GGML_MAX_DIMS; d++) {
+        if (src0->ne[d] != dst->ne[d]) {
+            same_extents = false;
+            break;
+        }
+    }
+
+    const bool transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
+                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
+                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
+    const bool sameshape  = same_extents && !transposed;
+
+    const bool src_is_contiguous = ggml_is_contiguous(src0);
+    const bool dst_is_contiguous = ggml_is_contiguous(dst);
+
+    if (sametype) {
+        if (src_is_contiguous && dst_is_contiguous) {
+            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
+            return true;
+        }
+
+        if (sameshape) {
+            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
+            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
+            return true;
+        }
+
+        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
+        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
+        kparams->u.reshape.div_ne0            = init_fastdiv_values((uint32_t) dst->ne[0]);
+        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_values((uint32_t) (dst->ne[1] * dst->ne[0]));
+        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_values((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
+        kparams->u.reshape.div_ne00           = init_fastdiv_values((uint32_t) src0->ne[0]);
+        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_values((uint32_t) (src0->ne[1] * src0->ne[0]));
+        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_values((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
+        return true;
+    }
+
+    if (!sameshape) {
+        return false;
+    }
+
+    const bool valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
+                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
+                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
+                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
+    if (!valid_conversion) {
+        return false;
+    }
+
+    const uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
+    struct htp_copy_convert_vtcm_layout layout;
+    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], src_type_size, dst_type_size, n_threads);
+
+    if (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
+        return false;
+    }
+
+    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
+    kparams->total_rows              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
+    kparams->n_threads               = (uint8_t) n_threads;
+    kparams->vtcm_size               = layout.total_bytes;
+    kparams->u.convert.src0_buf_size = layout.src0_buf_size;
+    kparams->u.convert.dst_buf_size  = layout.dst_buf_size;
+    kparams->u.convert.spad0_size_per_thread = layout.spad0_size_per_thread;
+    kparams->u.convert.spad1_size_per_thread = layout.spad1_size_per_thread;
+    kparams->u.convert.div_ne01      = init_fastdiv_values((uint32_t) src0->ne[1]);
+    kparams->u.convert.div_ne02_ne01 = init_fastdiv_values((uint32_t) (src0->ne[2] * src0->ne[1]));
+
+    return true;
+}
+
 static void ggml_hexagon_precompute_fused_mmnx_params(
     const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0, // W0
-    const struct ggml_tensor * src1, // x
+    const struct ggml_tensor * act,  // x
     int32_t n_weights,
     struct htp_mm_kernel_params * kparams
 ) {
@@ -6078,23 +6786,24 @@ static void ggml_hexagon_precompute_fused_mmnx_params(
     const int ne02 = src0->ne[2];
     const int ne03 = src0->ne[3];
 
-    const int ne10 = src1->ne[0];
-    const int ne11 = src1->ne[1];
-    const int ne12 = src1->ne[2];
-    const int ne13 = src1->ne[3];
+    const int ne10 = act->ne[0];
+    const int ne11 = act->ne[1];
+    const int ne12 = act->ne[2];
+    const int ne13 = act->ne[3];
 
     const int wtype = src0->type;
     const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
     const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : ne00;
     const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : ne01;
     const int ne11_padded = hex_round_up(ne11, 32);
+    const int ne01_tiled  = hex_round_up(ne01_padded, 32);
 
     const size_t vtcm_budget = sess->vtcm_size;
     const bool is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);
 
     bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
-    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, nullptr, ne01_padded, false, is_batched)) {
-        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, nullptr, wtype, ne00_padded, ne01_padded, ne02, ne11, ne12, ne11_padded, false, is_batched, 0, vtcm_budget, kparams)) {
+    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, ne01_padded, false, is_batched)) {
+        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, false, is_batched, 0, vtcm_budget, kparams)) {
             kparams->n_weights = n_weights;
             goto finalize;
         }
@@ -6106,20 +6815,20 @@ static void ggml_hexagon_precompute_fused_mmnx_params(
     }
 
     {
-        const int src1_nrows = ne11 * ne12 * ne13;
-        const size_t src1_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+        const int act_nrows = ne11 * ne12 * ne13;
+        const size_t act_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
         const size_t src0_row_size = src0->nb[1];
 
         uint32_t best_n_prefetch = 16;
 
         if (is_repack) {
-            const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
+            const uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
             best_n_prefetch = 2;
             for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                 struct htp_mm_hvx_vtcm_layout L;
                 htp_mm_hvx_vtcm_layout_build(
-                    &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
-                    0, src0_row_size, src1_row_size, 0, d, false, true
+                    &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
+                    0, src0_row_size, act_row_size, 0, d, false, true
                 );
                 if (L.total_bytes <= sess->vtcm_size) {
                     best_n_prefetch = d;
@@ -6133,14 +6842,16 @@ static void ggml_hexagon_precompute_fused_mmnx_params(
 
         // Test tiled first
         htp_mm_hvx_vtcm_layout_build(
-            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
-            0, src0_row_size, src1_row_size, 0, best_n_prefetch, false, true
+            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
+            0, src0_row_size, act_row_size, 0, best_n_prefetch, false, true
         );
 
         if (try_tiled && L.total_bytes <= sess->vtcm_size) {
-            kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
+            kparams->kernel_type    = HTP_MM_KERNEL_HVX_QUANT_ROW;
+            kparams->act_row_size   = act_row_size;
             kparams->vtcm_src0_size = L.src0_bytes;
-            kparams->vtcm_src1_size = L.src1_bytes;
+            kparams->vtcm_act_size  = L.act_bytes;
+            kparams->vtcm_bias_size = 0;
             kparams->vtcm_dst_size  = L.dst_bytes;
             kparams->vtcm_size      = L.total_bytes;
             kparams->n_prefetch     = best_n_prefetch;
@@ -6162,12 +6873,12 @@ finalize:
 static void ggml_hexagon_precompute_fused_mmidnx_params(
     const struct ggml_hexagon_session * sess,
     const struct ggml_tensor * src0, // W0
-    const struct ggml_tensor * src1, // x
+    const struct ggml_tensor * act,  // x
     const struct ggml_tensor * dst,  // dst0
     int32_t n_weights,
     struct htp_mm_kernel_params * kparams
 ) {
-    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
+    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
     kparams->n_weights = n_weights;
 }
 
@@ -6505,6 +7216,76 @@ static bool ggml_hexagon_supported_argmax(const struct ggml_hexagon_session * se
     GGML_UNUSED(sess);
 }
 
+static bool ggml_hexagon_supported_pool_2d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
+    const struct ggml_tensor * src0 = op->src[0];
+    const int32_t * params = op->op_params;
+
+    if (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
+        return false;
+    }
+    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
+        return false;
+    }
+    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
+        return false;
+    }
+    if (!ggml_hexagon_pool_shape_fits(src0, op)) {
+        return false;
+    }
+
+    const int32_t kernel_x = params[1];
+    const int32_t kernel_y = params[2];
+    const int32_t stride_x = params[3];
+    const int32_t stride_y = params[4];
+    const int32_t pad_x    = params[5];
+    const int32_t pad_y    = params[6];
+
+    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
+    if (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
+        pad_x < 0 || pad_y < 0) {
+        return false;
+    }
+
+    const uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
+    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);
+
+    struct htp_pool_vtcm_layout layout;
+    return htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
+                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
+                                 n_threads, sess->vtcm_size);
+}
+
+static bool ggml_hexagon_supported_pool_1d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
+    const struct ggml_tensor * src0 = op->src[0];
+    const int32_t * params = op->op_params;
+
+    if (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
+        return false;
+    }
+    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
+        return false;
+    }
+    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
+        return false;
+    }
+    if (!ggml_hexagon_pool_shape_fits(src0, op)) {
+        return false;
+    }
+
+    const int32_t kernel = params[1];
+    const int32_t stride = params[2];
+    const int32_t pad    = params[3];
+    if (kernel <= 0 || stride <= 0 || pad < 0) {
+        return false;
+    }
+
+    const uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
+    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);
+
+    struct htp_pool_vtcm_layout layout;
+    return htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
+}
+
 static bool ggml_hexagon_supported_activations(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
     const struct ggml_tensor * src0 = op->src[0];
     const struct ggml_tensor * src1 = op->src[1];
@@ -6651,7 +7432,7 @@ static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session *
     const struct ggml_tensor * src1 = op->src[1]; // indices
     const struct ggml_tensor * dst  = op;
 
-    if (src0->type == GGML_TYPE_Q4_0 && src0->view_src) {
+    if ((src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) && src0->view_src) {
         return false;
     }
 
@@ -6660,7 +7441,7 @@ static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session *
     if (src0_base->buffer && ggml_backend_buffer_is_hexagon(src0_base->buffer) && src0_base->extra) {
         const auto * extra = (const ggml_hexagon_tensor_extra *) src0_base->extra;
         is_repacked = (extra->flags & GGML_HEXAGON_TENSOR_REPACK) != 0;
-        if (is_repacked && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0) {
+        if (is_repacked && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_Q8_0) {
             return false;
         }
     }
@@ -6671,7 +7452,7 @@ static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session *
         return false;
     }
 
-    if (src0->type == GGML_TYPE_Q4_0 && src0->buffer && !is_repacked) {
+    if ((src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) && src0->buffer && ggml_backend_buffer_get_size(src0->buffer) != 0 && !is_repacked) {
         return false;
     }
 
@@ -6680,7 +7461,11 @@ static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session *
     }
 
     if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
-        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_I32) {
+        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_I32) {
+        return false;
+    }
+
+    if ((src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) && (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || src0->ne[0] % QK_K)) {
         return false;
     }
 
@@ -6709,8 +7494,8 @@ static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session *
         return false;
     }
 
-    // Q4_0 has no raw fallback. Mark only accepted tensors for repacking.
-    if (src0->type == GGML_TYPE_Q4_0 && !src0->buffer) {
+    // Tiled quantized weights have no raw fallback. Mark only accepted tensors for repacking.
+    if ((src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) && !src0->buffer) {
         sess->needs_repack.insert(src0);
     }
 
@@ -7056,12 +7841,15 @@ static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
         case GGML_OP_ADD_ID:          return HTP_OP_ADD_ID;
         case GGML_OP_SUB:             return HTP_OP_SUB;
         case GGML_OP_DIV:             return HTP_OP_DIV;
+        case GGML_OP_DUP:
         case GGML_OP_CPY:             return HTP_OP_CPY;
         case GGML_OP_CONT:            return HTP_OP_CPY;
         case GGML_OP_GET_ROWS:        return HTP_OP_GET_ROWS;
         case GGML_OP_SET_ROWS:        return HTP_OP_SET_ROWS;
         case GGML_OP_SUM:             return HTP_OP_SUM;
         case GGML_OP_SUM_ROWS:        return HTP_OP_SUM_ROWS;
+        case GGML_OP_POOL_2D:         return HTP_OP_POOL_2D;
+        case GGML_OP_POOL_1D:         return HTP_OP_POOL_1D;
         case GGML_OP_ARGSORT:         return HTP_OP_ARGSORT;
         case GGML_OP_TOP_K:           return HTP_OP_TOP_K;
         case GGML_OP_ARGMAX:          return HTP_OP_ARGMAX;
@@ -7093,7 +7881,7 @@ static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
             switch (ggml_get_unary_op(t)) {
                 case GGML_UNARY_OP_SILU:       return HTP_OP_UNARY_SILU;
                 case GGML_UNARY_OP_GELU:       return HTP_OP_UNARY_GELU;
-                case GGML_UNARY_OP_GELU_QUICK: return HTP_OP_UNARY_GELU;
+                case GGML_UNARY_OP_GELU_QUICK: return HTP_OP_UNARY_GELU_QUICK;
                 case GGML_UNARY_OP_GELU_ERF:   return HTP_OP_UNARY_GELU_ERF;
                 case GGML_UNARY_OP_SIGMOID:    return HTP_OP_UNARY_SIGMOID;
                 case GGML_UNARY_OP_NEG:        return HTP_OP_UNARY_NEG;
@@ -7137,6 +7925,14 @@ static bool mm_is_hmx_eligible(const ggml_tensor * t) {
     const ggml_tensor * src0 = t->src[0];
     const ggml_tensor * src1 = t->src[1];
 
+    if (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
+        const ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
+        const int wtype = src0->type;
+        const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
+        const int ne01_padded = is_repack ? hex_round_up(src0->ne[1], 32) : src0->ne[1];
+        return ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, ne01_padded, false, false);
+    }
+
     const int wtype = src0->type;
     const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
     const bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
@@ -7176,13 +7972,15 @@ static bool is_mergeable_mul_mat(const ggml_tensor * t) {
 
     const ggml_tensor * src0 = t->src[0];
     const ggml_tensor * src1 = t->src[1];
-    if (src1->type != GGML_TYPE_F32) return false;
     if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
 
     if (mm_is_hmx_eligible(t)) {
         return ggml_hexagon_is_hmx_weight_type(src0->type);
     }
 
+    // HVX path requires F32 activations and repacked weights (except Q6_K)
+    if (src1->type != GGML_TYPE_F32) return false;
+
     return ggml_hexagon_is_repack_type(src0->type) && src0->type != GGML_TYPE_Q6_K;
 }
 
@@ -7340,6 +8138,25 @@ static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, gg
                     node.opcode == HTP_OP_TOP_K,
                     (struct htp_sort_kernel_params *) node.kernel_params
                 );
+            } else if (node.opcode == HTP_OP_POOL_2D || node.opcode == HTP_OP_POOL_1D) {
+                ggml_hexagon_precompute_pool_2d_params(
+                    sess, node.node->src[0], node.dst(),
+                    (struct htp_pool_2d_kernel_params *)node.kernel_params,
+                    node.opcode == HTP_OP_POOL_1D);
+            } else if (node.opcode == HTP_OP_CONCAT) {
+                ggml_hexagon_precompute_concat_params(sess,
+                    node.node,
+                    (struct htp_concat_kernel_params *) node.kernel_params
+                );
+            } else if (node.opcode == HTP_OP_CPY || node.opcode == HTP_OP_CPY_FENCE) {
+                const bool ok = ggml_hexagon_precompute_cpy_params(sess,
+                    node.node,
+                    (struct htp_copy_kernel_params *) node.kernel_params
+                );
+                const auto * kparams = (const struct htp_copy_kernel_params *) node.kernel_params;
+                if (ok && node.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
+                    continue;
+                }
             }
             computed_nodes.push_back(std::move(node));
         }
@@ -7928,8 +8745,8 @@ static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev
 }
 
 static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
-    *free  = 0;
-    *total = *free;
+    *free  = opt_mbuf_total;
+    *total = opt_mbuf_total;
 
     GGML_UNUSED(dev);
 }
@@ -7968,49 +8785,13 @@ static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_ty
 }
 
 static bool ggml_hexagon_supported_cpy(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
-    GGML_UNUSED(sess);
-
-    const struct ggml_tensor * src0 = op->src[0];
-    const struct ggml_tensor * dst  = op;
-
-    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
-        src0->type != GGML_TYPE_I32) return false;
-    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 &&
-        dst->type != GGML_TYPE_I32) return false;
-
-    const bool is_scalar  = (ggml_nelements(src0) == 1 && ggml_nelements(dst) == 1);
-    const bool sametype   = (src0->type == dst->type);
-    const bool transposed = !is_scalar && (ggml_is_transposed(src0) || ggml_is_transposed(dst));
-    const bool sameshape  = is_scalar || (!transposed && ggml_are_same_shape(src0, dst));
-
-    if (src0->type == GGML_TYPE_I32 || dst->type == GGML_TYPE_I32) {
-        if (!sameshape) return false;
-        if (sametype) return true;
-        if ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
-            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32)) {
-            return true;
-        }
-        return false;
-    }
-
-    // can handle any shape and any same-type (pretty slow if reshaping is required)
-    if (sametype) return true;
-
-    // cannot handle re-shaping and type conversion at the same time
-    if (!sameshape) return false;
-
-    return true;
+    struct htp_copy_kernel_params kparams;
+    return ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
 }
 
 static bool ggml_hexagon_supported_cont(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
-    GGML_UNUSED(sess);
-    const struct ggml_tensor * src0 = op->src[0];
-
-    // CONT is same-type only and supports F32, F16, and I32.
-    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
-        src0->type != GGML_TYPE_I32) return false;
-
-    return true;
+    struct htp_copy_kernel_params kparams;
+    return ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
 }
 
 static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
@@ -8037,23 +8818,8 @@ static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * se
 }
 
 static bool ggml_hexagon_supported_concat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
-    int dim = ((const int32_t *) op->op_params)[0];
-    if (dim < 0 || dim >= GGML_MAX_DIMS) {
-        return false;
-    }
-
-    for (int i = 0; i < GGML_MAX_SRC; ++i) {
-        const struct ggml_tensor * src = op->src[i];
-        if (!src) {
-            continue;
-        }
-        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32 && src->type != GGML_TYPE_F16) {
-            return false;
-        }
-    }
-
-    return true;
-    GGML_UNUSED(sess);
+    struct htp_concat_kernel_params kparams;
+    return ggml_hexagon_precompute_concat_params(sess, op, &kparams);
 }
 
 static bool ggml_hexagon_supported_fill(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
@@ -8157,6 +8923,14 @@ static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, cons
             supp = ggml_hexagon_supported_argmax(sess, op);
             break;
 
+        case GGML_OP_POOL_2D:
+            supp = ggml_hexagon_supported_pool_2d(sess, op);
+            break;
+
+        case GGML_OP_POOL_1D:
+            supp = ggml_hexagon_supported_pool_1d(sess, op);
+            break;
+
         case GGML_OP_SOFT_MAX:
             supp = ggml_hexagon_supported_softmax(sess, op);
             break;
@@ -8215,6 +8989,7 @@ static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, cons
             supp = ggml_hexagon_supported_get_rows(sess, op);
             break;
 
+        case GGML_OP_DUP:
         case GGML_OP_CPY:
             supp = ggml_hexagon_supported_cpy(sess, op);
             break;
@@ -8683,6 +9458,7 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
     const char * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
     const char * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
     const char * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
+    const char * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
     const char * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
     const char * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
     const char * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
@@ -8719,7 +9495,7 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
     size_t MiB = 1024 * 1024;
 
     // Update vmem default
-    opt_vmem = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
+    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
     opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);
 
     auto RE_ICASE = std::regex_constants::icase;
@@ -8737,12 +9513,30 @@ static void ggml_hexagon_init(ggml_backend_reg * reg) {
     opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
     opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
     opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
-    opt_gdn_select = str_gdn_select ? atoi(str_gdn_select)                 : opt_gdn_select;
-    opt_ar_select = str_ar_select ? atoi(str_ar_select)                   : opt_ar_select;
-    opt_ar_scatter = str_ar_scatter ? atoi(str_ar_scatter)                : opt_ar_scatter;
-    opt_mbuf      = str_mbuf     ? strtoul(str_mbuf, NULL, 0) * MiB       : opt_mbuf;
-    opt_vmem      = str_vmem     ? strtoul(str_vmem, NULL, 0) * MiB       : opt_vmem;
-    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf) != 0                 : opt_hostbuf;
+    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
+    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
+    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
+    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;
+
+    if (str_mbuf) {
+        const char * p = str_mbuf;
+        for (int idx = 0; idx < 3 && p && *p; idx++) {
+            while (*p == ' ') p++;
+            const char * comma = strchr(p, ',');
+            size_t len = comma ? (size_t)(comma - p) : strlen(p);
+            while (len > 0 && p[len - 1] == ' ') len--;
+            if (len > 0) {
+                std::string token(p, len);
+                if (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
+                if (idx == 1) opt_mbuf_static = parse_size(token.c_str());
+                if (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
+            }
+            if (!comma) break;
+            p = comma + 1;
+        }
+    }
+    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
+    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;
 
     // Parse device configuration
     const char * str_devices  = getenv("GGML_HEXAGON_DEVICES");
diff --git src/ggml-hexagon/htp/CMakeLists.txt src/ggml-hexagon/htp/CMakeLists.txt
index 821f08c0..787badf0 100644
--- src/ggml-hexagon/htp/CMakeLists.txt
+++ src/ggml-hexagon/htp/CMakeLists.txt
@@ -44,6 +44,7 @@ add_library(${HTP_LIB} SHARED
     argsort-ops.c
     im2col-ops.c
     roll-ops.c
+    pool-ops.c
     allreduce-ops.c
 )
 
diff --git src/ggml-hexagon/htp/concat-ops.c src/ggml-hexagon/htp/concat-ops.c
index 4dc14639..f7dda140 100644
--- src/ggml-hexagon/htp/concat-ops.c
+++ src/ggml-hexagon/htp/concat-ops.c
@@ -1,11 +1,13 @@
+#include "concat-ops.h"
 #include "dma-queue.h"
 #include "hex-common.h"
-#include "hex-cpy-dma.h"
+#include "dma-copy.h"
 #include "hex-fastdiv.h"
 #include "hex-profile.h"
 #include "hexagon_protos.h"
 #include "hexagon_types.h"
 #include "htp-ctx.h"
+#include "htp-fence.h"
 #include "htp-ops.h"
 #include "htp-tensor.h"
 #include "htp-vtcm.h"
@@ -16,15 +18,14 @@
 
 struct htp_concat_context {
     struct htp_ops_context * octx;
-    uint32_t dim;
-    uint32_t nrows_per_thread;
-    uint32_t row_start;
-    uint32_t nrows;
-    uint32_t elem_start;
-    uint32_t nelems;
-    uint32_t nplanes;
-    struct fastdiv_values div_ne0;
-    struct fastdiv_values div_ne1;
+    uint8_t * spad0_base;
+    uint8_t * spad1_base;
+    uint32_t  spad0_size_per_thread;
+    uint32_t  spad1_size_per_thread;
+    uint32_t  row_start;
+    uint32_t  nrows;
+    uint32_t  nrows_per_thread;
+    uint32_t  nplanes;
     struct fastdiv_values div_ne2;
 };
 
@@ -52,8 +53,8 @@ static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void *
 
     dma_queue * dma_q = octx->ctx->dma[ith];
 
-    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
-    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
+    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
+    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;
 
     const uint32_t block_i = 32;
     const uint32_t spad1_stride = block_i * sizeof(float);
@@ -127,6 +128,7 @@ static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void *
         p = np;
         i = ni;
     }
+    dma_queue_flush(dma_q);
 }
 
 static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
@@ -147,8 +149,8 @@ static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void *
 
     dma_queue * dma_q = octx->ctx->dma[ith];
 
-    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
-    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
+    uint8_t * spad0_base = cctx->spad0_base + ith * cctx->spad0_size_per_thread;
+    uint8_t * spad1_base = cctx->spad1_base + ith * cctx->spad1_size_per_thread;
 
     const uint32_t block_i = 64;
     const uint32_t spad1_stride = block_i * sizeof(__fp16);
@@ -222,219 +224,123 @@ static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void *
         p = np;
         i = ni;
     }
+    dma_queue_flush(dma_q);
 }
 
-static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
-    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
-    struct htp_ops_context * octx = cctx->octx;
-
-    const struct htp_tensor * src0 = octx->src[0];
-    const struct htp_tensor * src1 = octx->src[1];
-    const struct htp_tensor * dst  = octx->dst;
-
-    const int dim = cctx->dim;
-    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
-
-    const uint32_t ne[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};
-
-    // Per-device element range aligned to prevent false sharing
-    const uint32_t elem_start = cctx->elem_start;
-    const uint32_t nelems     = cctx->nelems;
-    const uint32_t chunk_size = fastdiv(nelems + nth - 1, &octx->n_threads_div);
-
-    const uint32_t start_idx = MIN(elem_start + ith * chunk_size, elem_start + nelems);
-    const uint32_t end_idx   = MIN(start_idx + chunk_size, elem_start + nelems);
-
-    // Naive scalar element-wise copy
-    for (uint32_t idx = start_idx; idx < end_idx; idx++) {
-        uint32_t idx_div_ne0 = fastdiv(idx, &cctx->div_ne0);
-        uint32_t i0 = idx - idx_div_ne0 * ne[0];
-
-        uint32_t idx_div_ne01 = fastdiv(idx_div_ne0, &cctx->div_ne1);
-        uint32_t i1 = idx_div_ne0 - idx_div_ne01 * ne[1];
-
-        uint32_t idx_div_ne012 = fastdiv(idx_div_ne01, &cctx->div_ne2);
-        uint32_t i2 = idx_div_ne01 - idx_div_ne012 * ne[2];
-        uint32_t i3 = idx_div_ne012;
-
-        uint8_t * dst_ptr = (uint8_t *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];
-
-        uint32_t idx_dim = 0;
-        if (dim == 0) idx_dim = i0;
-        else if (dim == 1) idx_dim = i1;
-        else if (dim == 2) idx_dim = i2;
-        else if (dim == 3) idx_dim = i3;
-
-        const struct htp_tensor * src = (idx_dim < src0->ne[dim]) ? src0 : src1;
-
-        uint32_t s0 = i0;
-        uint32_t s1 = i1;
-        uint32_t s2 = i2;
-        uint32_t s3 = i3;
-
-        if (dim == 0 && src == src1) s0 -= src0->ne[0];
-        if (dim == 1 && src == src1) s1 -= src0->ne[1];
-        if (dim == 2 && src == src1) s2 -= src0->ne[2];
-        if (dim == 3 && src == src1) s3 -= src0->ne[3];
-
-        uint8_t * src_ptr = (uint8_t *)src->data + s3 * src->nb[3] + s2 * src->nb[2] + s1 * src->nb[1] + s0 * src->nb[0];
-
-        if (type_size == 4) {
-            *(float*)dst_ptr = *(float*)src_ptr;
-        } else {
-            *(__fp16*)dst_ptr = *(__fp16*)src_ptr;
-        }
-    }
-}
-
-static bool concat_dma(struct htp_ops_context * octx, int dim, uint32_t type_size) {
-    if (dim < 0 || dim >= HTP_OP_MAX_DIMS) {
-        return false;
-    }
-
+static int concat_regular(struct htp_ops_context * octx, int dim, uint32_t type_size) {
     const struct htp_tensor * src0 = octx->src[0];
     const struct htp_tensor * src1 = octx->src[1];
     const struct htp_tensor * dst  = octx->dst;
 
-    // Not partitioned across devices: the row/element-split paths handle that.
-    if (octx->ctx->mdev.count > 1 ||
-        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_I32) ||
-        src0->type != dst->type || src1->type != dst->type || src0->nb[0] != type_size || src1->nb[0] != type_size ||
-        dst->nb[0] != type_size || (size_t) dst->ne[0] * type_size > DMA_MAX_SIZE_24B ||
-        dst->nb[1] > DMA_MAX_STRIDE_24B || src0->nb[1] > DMA_MAX_STRIDE_24B || src1->nb[1] > DMA_MAX_STRIDE_24B) {
-        return false;
-    }
-
-    for (int d = 0; d < HTP_OP_MAX_DIMS; d++) {
-        const uint32_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
-        if (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
-            return false;
-        }
-    }
-
-    // The two views of dst, shaped like the sources.
     struct htp_tensor view0 = *dst;
     struct htp_tensor view1 = *dst;
     for (int d = 0; d < HTP_OP_MAX_DIMS; d++) {
         view0.ne[d] = src0->ne[d];
         view1.ne[d] = src1->ne[d];
     }
-    view1.data += (uint64_t) src0->ne[dim] * dst->nb[dim];
+    view1.data += src0->ne[dim] * dst->nb[dim];
+
+    const uint32_t total_rows_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
+    const uint32_t total_rows_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];
+
+    uint32_t rstart0 = 0, nrows0 = total_rows_0;
+    uint32_t rstart1 = 0, nrows1 = total_rows_1;
+
+    if (octx->ctx->mdev.count > 1) {
+        const struct htp_tensor_mdev_range range0 = htp_tensor_mdev_partition(
+            total_rows_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        rstart0 = range0.start;
+        nrows0  = range0.count;
+
+        const struct htp_tensor_mdev_range range1 = htp_tensor_mdev_partition(
+            total_rows_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        rstart1 = range1.start;
+        nrows1  = range1.count;
+    }
 
     dma_queue * q = octx->ctx->dma[0];
 
-    cpy_dma_sametype_sameshape(q, &view0, src0, type_size);
-    cpy_dma_sametype_sameshape(q, &view1, src1, type_size);
+    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_size, rstart0, nrows0);
+    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_size, rstart1, nrows1);
     dma_queue_flush(q);
-    return true;
+
+    return HTP_STATUS_OK;
 }
 
-int op_concat(struct htp_ops_context * octx) {
-    int dim = octx->op_params[0];
-    if (dim < 0 || dim >= HTP_OP_MAX_DIMS) {
-        return HTP_STATUS_NO_SUPPORT;
+static int concat_transposed(struct htp_ops_context * octx, const struct htp_concat_kernel_params * kparams, uint32_t type_size) {
+    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
+        return HTP_STATUS_INVAL_PARAMS;
     }
 
-    const struct htp_tensor * src0 = octx->src[0];
-    const struct htp_tensor * src1 = octx->src[1];
-    const struct htp_tensor * dst  = octx->dst;
+    const struct htp_tensor * dst = octx->dst;
 
-    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
-    bool is_src1_transposed  = (src1->nb[0] > src1->nb[1]);
-    bool is_src0_transposed  = (src0->nb[0] > src0->nb[1]);
+    const uint32_t total_rows = dst->ne[1];
+    uint32_t row_start = 0;
+    uint32_t nrows     = total_rows;
+    if (octx->ctx->mdev.count > 1) {
+        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        row_start = range.start;
+        nrows     = range.count;
+    }
 
-    if (concat_dma(octx, dim, type_size)) {
+    if (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
         return HTP_STATUS_OK;
     }
 
-    uint32_t n_threads = octx->n_threads;
-    struct htp_concat_context cctx;
-    cctx.octx = octx;
-    cctx.dim = dim;
-    cctx.div_ne0 = init_fastdiv_values(dst->ne[0]);
-    cctx.div_ne1 = init_fastdiv_values(dst->ne[1]);
-    cctx.div_ne2 = init_fastdiv_values(dst->ne[2]);
-
-    void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;
-
-    const bool rows_ok = src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size;
-
-    if (dim == 0 && is_src1_transposed && !is_src0_transposed && rows_ok) {
-        const uint32_t total_rows = dst->ne[1];
-        const size_t dst_data_row_size = dst->ne[0] * type_size;
-        uint32_t row_start = 0;
-        uint32_t nrows     = total_rows;
-        if (octx->ctx->mdev.count > 1) {
-            uint32_t rows_per_chunk = 0;
-            htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
-            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
-            row_start = range.start;
-            nrows     = range.count;
-        }
-
-        if (nrows == 0) {
-            return HTP_STATUS_OK;
-        }
-
-        cctx.row_start = row_start;
-        cctx.nrows     = nrows;
-        cctx.nplanes   = dst->ne[2] * dst->ne[3];
-
-        uint32_t block_i = (type_size == 4) ? 32 : 64;
-
-        cctx.nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
+    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
+        return HTP_STATUS_VTCM_TOO_SMALL;
+    }
 
-        // Allocate VTCM
-        uint32_t spad1_stride = block_i * type_size;
+    const uint32_t n_threads = octx->n_threads;
 
-        uint32_t src1_ne0_padded = hex_round_up(src1->ne[0], block_i);
-        // src0 row is right-aligned to VLEN so the gathered src1 part starts aligned
-        uint32_t spad0_row_bytes = hex_round_up(src0->ne[0] * type_size, VLEN) + src1_ne0_padded * type_size;
+    // layout precomputed on host; kept for reference:
+    // struct htp_concat_transposed_vtcm_layout layout;
+    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);
 
-        octx->src0_spad.size_per_thread = block_i * spad0_row_bytes;
-        octx->src1_spad.size_per_thread = src1_ne0_padded * spad1_stride;
+    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
 
-        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
-        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;
+    struct htp_concat_context cctx;
+    cctx.octx                  = octx;
+    cctx.spad0_base            = vtcm_base;
+    cctx.spad1_base            = vtcm_base + n_threads * kparams->spad0_size_per_thread;
+    cctx.spad0_size_per_thread = kparams->spad0_size_per_thread;
+    cctx.spad1_size_per_thread = kparams->spad1_size_per_thread;
+    cctx.row_start             = row_start;
+    cctx.nrows                 = nrows;
+    cctx.nplanes               = dst->ne[2] * dst->ne[3];
+    cctx.div_ne2               = init_fastdiv_values(dst->ne[2]);
+    cctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
+
+    work_queue_func_t worker_func = (type_size == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
+    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
+    return HTP_STATUS_OK;
+}
 
-        if (octx->src0_spad.size + octx->src1_spad.size > octx->ctx->vtcm_size) {
-            return HTP_STATUS_VTCM_TOO_SMALL;
-        }
+int op_concat(struct htp_ops_context * octx) {
+    const struct htp_concat_kernel_params * kparams = (const struct htp_concat_kernel_params *) octx->kernel_params;
+    const struct htp_tensor * dst = octx->dst;
+    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
 
-        octx->src0_spad.data = octx->ctx->vtcm_base;
-        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
-        octx->src0_spad.src  = NULL;
-        octx->src1_spad.src  = NULL;
+    int status = HTP_STATUS_OK;
+    switch (kparams->kernel_type) {
+        case HTP_CONCAT_KERNEL_REGULAR:
+            status = concat_regular(octx, kparams->dim, type_size);
+            break;
 
-        if (type_size == 4) {
-            worker_func = concat_2d_f32_transposed;
-        } else {
-            worker_func = concat_2d_f16_transposed;
-        }
-    } else {
-        if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
-            return HTP_STATUS_NO_SUPPORT;
-        }
+        case HTP_CONCAT_KERNEL_TRANSPOSED:
+            status = concat_transposed(octx, kparams, type_size);
+            break;
 
-        const uint32_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
-        uint32_t elem_start = 0;
-        uint32_t nelems     = total_elements;
-        if (octx->ctx->mdev.count > 1) {
-            const uint32_t elems_per_chunk = HEX_L2_LINE_SIZE / type_size;
-            const bool can_split = htp_tensor_mdev_data_aligned(dst) && htp_tensor_is_contiguous(dst, type_size) && !htp_tensor_is_permuted(dst);
-            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_elements, can_split ? elems_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
-            elem_start = range.start;
-            nelems     = range.count;
-        }
+        default:
+            status = HTP_STATUS_NO_SUPPORT;
+            break;
+    }
 
-        if (nelems == 0) {
-            return HTP_STATUS_OK;
-        }
+    htp_ops_context_set_status(octx, status);
 
-        cctx.elem_start = elem_start;
-        cctx.nelems     = nelems;
+    if (octx->ctx->mdev.count > 1) {
+        htp_mdev_group_barrier(octx);
     }
 
-    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
-    return HTP_STATUS_OK;
+    return octx->status;
 }
diff --git src/ggml-hexagon/htp/concat-ops.h src/ggml-hexagon/htp/concat-ops.h
new file mode 100644
index 00000000..1d721844
--- /dev/null
+++ src/ggml-hexagon/htp/concat-ops.h
@@ -0,0 +1,53 @@
+#ifndef HTP_CONCAT_OPS_H
+#define HTP_CONCAT_OPS_H
+
+#include "hex-common.h"
+#include <stdint.h>
+
+enum htp_concat_kernel_type {
+    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
+    HTP_CONCAT_KERNEL_REGULAR     = 1,
+    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
+};
+
+struct htp_concat_kernel_params {
+    uint8_t  kernel_type;
+    uint8_t  dim;
+    uint8_t  n_threads;
+    uint8_t  pad;
+
+    uint32_t vtcm_size;
+    uint32_t spad0_size_per_thread;
+    uint32_t spad1_size_per_thread;
+};
+
+#if defined(__cplusplus)
+static_assert(sizeof(struct htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
+#else
+_Static_assert(sizeof(struct htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
+#endif
+
+struct htp_concat_transposed_vtcm_layout {
+    uint32_t src0_spad_size_per_thread;
+    uint32_t src1_spad_size_per_thread;
+    uint32_t total_bytes;
+};
+
+static inline void htp_concat_transposed_vtcm_layout_build(
+    struct htp_concat_transposed_vtcm_layout * layout,
+    uint32_t src0_ne0,
+    uint32_t src1_ne0,
+    uint32_t type_size,
+    uint32_t n_threads) {
+
+    uint32_t block_i = (type_size == 4) ? 32 : 64;
+    uint32_t spad1_stride = block_i * type_size;
+    uint32_t src1_ne0_padded = hex_round_up(src1_ne0, block_i);
+    uint32_t spad0_row_bytes = hex_round_up(src0_ne0 * type_size, 128) + src1_ne0_padded * type_size;
+
+    layout->src0_spad_size_per_thread = block_i * spad0_row_bytes;
+    layout->src1_spad_size_per_thread = src1_ne0_padded * spad1_stride;
+    layout->total_bytes = n_threads * (layout->src0_spad_size_per_thread + layout->src1_spad_size_per_thread);
+}
+
+#endif // HTP_CONCAT_OPS_H
diff --git src/ggml-hexagon/htp/cpy-ops.c src/ggml-hexagon/htp/cpy-ops.c
index df38d3ee..d366acb7 100644
--- src/ggml-hexagon/htp/cpy-ops.c
+++ src/ggml-hexagon/htp/cpy-ops.c
@@ -11,7 +11,8 @@
 
 #define GGML_COMMON_DECL_C
 #include "ggml-common.h"
-#include "hex-cpy-dma.h"
+#include "cpy-ops.h"
+#include "dma-copy.h"
 #include "htp-ctx.h"
 #include "htp-fence.h"
 #include "htp-ops.h"
@@ -19,34 +20,19 @@
 #include "hvx-utils.h"
 
 struct htp_copy_context {
-    struct htp_ops_context * octx;
+    struct htp_ops_context *              octx;
+    const struct htp_copy_kernel_params * kparams;
 
-    uint32_t          src0_type_size;
-    uint32_t          src0_block_size;
+    uint32_t row_start;
+    uint32_t nrows;
+    uint32_t src0_nrows_per_thread;
 
-    uint32_t          dst_type_size;
-    uint32_t          dst_block_size;
+    uint32_t elem_start;
+    uint32_t nelem;
+    uint32_t elem_per_thread;
 
-    uint32_t          src0_blocks_per_row;
-    uint32_t          dst_blocks_per_row;
-
-    uint32_t          elem_start;
-    uint32_t          nelem;
-    uint32_t          elem_per_thread;
-
-    uint32_t          src0_nrows_per_thread;
-    uint32_t          row_start;
-    uint32_t          nrows;
-
-    struct fastdiv_values div_ne01;
-    struct fastdiv_values div_ne02_ne01;
-
-    struct fastdiv_values div_ne0;
-    struct fastdiv_values div_ne1_ne0;
-    struct fastdiv_values div_ne2_ne1_ne0;
-    struct fastdiv_values div_ne00;
-    struct fastdiv_values div_ne01_ne00;
-    struct fastdiv_values div_ne02_ne01_ne00;
+    uint8_t * vtcm_src0;
+    uint8_t * vtcm_dst;
 };
 
 #define cpy_preamble                              \
@@ -73,52 +59,6 @@ struct htp_copy_context {
     const uint32_t  nb2 = dst->nb[2];             \
     const uint32_t  nb3 = dst->nb[3];
 
-#define DEFINE_CPY_SAMESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                                           \
-static void cpy_thread_##NAME##_sameshape(unsigned int nth, unsigned int ith, void * data) {                       \
-    struct htp_copy_context * ct = (struct htp_copy_context *) data;                                               \
-    struct htp_ops_context * octx = ct->octx;                                                                      \
-    cpy_preamble;                                                                                                  \
-    const uint32_t dr  = ct->src0_nrows_per_thread;                                                                \
-    const uint32_t ir0 = ct->row_start + dr * ith;                                                                 \
-    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                                                 \
-    if (ir0 >= ir1) return;                                                                                        \
-    const bool contiguous = htp_tensor_is_contiguous(src0, ELEM_SIZE) && htp_tensor_is_contiguous(dst, ELEM_SIZE); \
-    if (contiguous) {                                                                                              \
-        dma_queue * dma_q = octx->ctx->dma[ith];                                                                   \
-        dma_addr_t dst_addr  = dst->data  + ir0 * ne00 * ELEM_SIZE;                                                \
-        dma_addr_t src0_addr = src0->data + ir0 * ne00 * ELEM_SIZE;                                                \
-        cpy_dma_sametype_reshape_contig(dma_q, dst_addr, src0_addr, (ir1 - ir0) * ne00 * ELEM_SIZE);               \
-        dma_queue_flush(dma_q);                                                                                    \
-        return;                                                                                                    \
-    }                                                                                                              \
-    const uint32_t ne02_ne01 = ne02 * ne01;                                                                        \
-    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);                                                               \
-    uint32_t rem = ir0 - i03 * ne02_ne01;                                                                          \
-    uint32_t i02 = fastdiv(rem, &ct->div_ne01);                                                                    \
-    uint32_t i01 = rem - i02 * ne01;                                                                               \
-    uint8_t * dst_ptr  = (uint8_t *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;                                   \
-    uint8_t * src0_ptr = (uint8_t *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;                                  \
-    for (uint32_t r = ir0; r < ir1; r++) {                                                                         \
-        hex_l2fetch(src0_ptr, ne00 * ELEM_SIZE, nb01, 2);                                                          \
-        hvx_copy_uu(dst_ptr, src0_ptr, ne00, ELEM_SIZE);                                                           \
-        dst_ptr  += nb1;                                                                                           \
-        src0_ptr += nb01;                                                                                          \
-        if (++i01 == ne01) {                                                                                       \
-            i01 = 0;                                                                                               \
-            if (++i02 == ne02) {                                                                                   \
-                i02 = 0;                                                                                           \
-                i03++;                                                                                             \
-            }                                                                                                      \
-            dst_ptr  = (uint8_t *) dst->data  + i02*nb2  + i03*nb3;                                                \
-            src0_ptr = (uint8_t *) src0->data + i02*nb02 + i03*nb03;                                               \
-        }                                                                                                          \
-    }                                                                                                              \
-}
-
-DEFINE_CPY_SAMESHAPE(f32,  float, 4)
-DEFINE_CPY_SAMESHAPE(f16, __fp16, 2)
-DEFINE_CPY_SAMESHAPE(i32, int32_t, 4)
-
 #define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                                \
 static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void * data) {            \
     struct htp_copy_context * ct = (struct htp_copy_context *) data;                                  \
@@ -129,52 +69,45 @@ static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void
     const uint32_t th_end   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
     if (th_start >= th_end) return;                                                                   \
                                                                                                       \
-    if (htp_tensor_is_contiguous(src0, ELEM_SIZE) && htp_tensor_is_contiguous(dst, ELEM_SIZE)) {      \
-        dma_queue * dma_q = octx->ctx->dma[ith];                                                      \
-        dma_addr_t dst_addr  = dst->data  + th_start * ELEM_SIZE;                                     \
-        dma_addr_t src0_addr = src0->data + th_start * ELEM_SIZE;                                     \
-        cpy_dma_sametype_reshape_contig(dma_q, dst_addr, src0_addr, (th_end - th_start) * ELEM_SIZE); \
-        dma_queue_flush(dma_q);                                                                       \
-        return;                                                                                       \
-    }                                                                                                 \
+    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                       \
     const uint32_t ne01_ne00      = ne01 * ne00;                                                      \
     const uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
     const uint32_t ne1_ne0        = ne1 * ne0;                                                        \
     const uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                       \
+    const struct htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
     uint32_t e = th_start;                                                                            \
-    uint32_t i13 = fastdiv(e, &ct->div_ne2_ne1_ne0);                                                  \
+    uint32_t i13 = fastdiv(e, &rsh->div_ne2_ne1_ne0);                                                 \
     uint32_t rem = e - i13 * ne2_ne1_ne0;                                                             \
-    uint32_t i12 = fastdiv(rem, &ct->div_ne1_ne0);                                                    \
+    uint32_t i12 = fastdiv(rem, &rsh->div_ne1_ne0);                                                   \
     uint32_t rem2 = rem - i12 * ne1_ne0;                                                              \
-    uint32_t i11 = fastdiv(rem2, &ct->div_ne0);                                                       \
+    uint32_t i11 = fastdiv(rem2, &rsh->div_ne0);                                                      \
     uint32_t i10 = rem2 - i11 * ne0;                                                                  \
                                                                                                       \
-    uint32_t i03 = fastdiv(e, &ct->div_ne02_ne01_ne00);                                               \
+    uint32_t i03 = fastdiv(e, &rsh->div_ne02_ne01_ne00);                                              \
     uint32_t rem_s = e - i03 * ne02_ne01_ne00;                                                        \
-    uint32_t i02 = fastdiv(rem_s, &ct->div_ne01_ne00);                                                \
+    uint32_t i02 = fastdiv(rem_s, &rsh->div_ne01_ne00);                                               \
     uint32_t rem2_s = rem_s - i02 * ne01_ne00;                                                        \
-    uint32_t i01 = fastdiv(rem2_s, &ct->div_ne00);                                                    \
+    uint32_t i01 = fastdiv(rem2_s, &rsh->div_ne00);                                                   \
     uint32_t i00 = rem2_s - i01 * ne00;                                                               \
                                                                                                       \
-    char * dst_ptr        = (char *)       dst->data  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;     \
-    const char * src0_ptr = (const char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;    \
+    dma_addr_t dst_addr  = dst->data  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
+    dma_addr_t src0_addr = src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                       \
     const bool rows_contig = (nb00 == ELEM_SIZE) && (nb0 == ELEM_SIZE);                               \
                                                                                                       \
     while (e < th_end) {                                                                              \
-        uint32_t run = 1;                                                                             \
+        const uint32_t run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                             \
         if (rows_contig) {                                                                            \
-            run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                                        \
-            hvx_copy_uu((uint8_t *) dst_ptr, (const uint8_t *) src0_ptr, run, ELEM_SIZE);             \
+            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, run * ELEM_SIZE);             \
         } else {                                                                                      \
-            *((ELEM_TYPE *) dst_ptr) = *((const ELEM_TYPE *) src0_ptr);                               \
+            dma_cpy_push_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_SIZE, run);           \
         }                                                                                             \
         e += run;                                                                                     \
                                                                                                       \
-        dst_ptr += run * nb0;                                                                         \
-        i10     += run;                                                                               \
+        dst_addr += run * nb0;                                                                        \
+        i10      += run;                                                                              \
         if (i10 == ne0) {                                                                             \
             i10 = 0;                                                                                  \
             if (++i11 == ne1) {                                                                       \
@@ -184,11 +117,11 @@ static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void
                     i13++;                                                                            \
                 }                                                                                     \
             }                                                                                         \
-            dst_ptr = (char *) dst->data + i11*nb1 + i12*nb2 + i13*nb3;                               \
+            dst_addr = dst->data + i11*nb1 + i12*nb2 + i13*nb3;                                       \
         }                                                                                             \
                                                                                                       \
-        src0_ptr += run * nb00;                                                                       \
-        i00      += run;                                                                              \
+        src0_addr += run * nb00;                                                                      \
+        i00       += run;                                                                             \
         if (i00 == ne00) {                                                                            \
             i00 = 0;                                                                                  \
             if (++i01 == ne01) {                                                                      \
@@ -198,371 +131,350 @@ static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void
                     i03++;                                                                            \
                 }                                                                                     \
             }                                                                                         \
-            src0_ptr = (const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;                    \
+            src0_addr = src0->data + i01*nb01 + i02*nb02 + i03*nb03;                                  \
         }                                                                                             \
     }                                                                                                 \
+    dma_queue_flush(dma_q);                                                                           \
 }
 
 DEFINE_CPY_RESHAPE(f32,  float, 4)
 DEFINE_CPY_RESHAPE(f16, __fp16, 2)
 DEFINE_CPY_RESHAPE(i32, int32_t, 4)
 
-static void cpy_thread_f16_f32_sameshape(unsigned int nth, unsigned int ith, void * data) {
-    struct htp_copy_context * ct = (struct htp_copy_context *) data;
-    struct htp_ops_context * octx = ct->octx;
-    cpy_preamble;
-
-    const uint32_t dr  = ct->src0_nrows_per_thread;
-    const uint32_t ir0 = ct->row_start + dr * ith;
-    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);
-    if (ir0 >= ir1) return;
-
-    const uint32_t ne02_ne01 = ne02 * ne01;
-    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);
-    uint32_t rem = ir0 - i03 * ne02_ne01;
-    uint32_t i02 = fastdiv(rem, &ct->div_ne01);
-    uint32_t i01 = rem - i02 * ne01;
-
-    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
-    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
-
-    for (uint32_t r = ir0; r < ir1; r++) {
-        hex_l2fetch(src0_ptr, ne00 * sizeof(float), nb01, 2);
-        hvx_copy_f16_f32_uu(dst_ptr, src0_ptr, ne00);
-        dst_ptr  += nb1;
-        src0_ptr += nb01;
-        if (++i01 == ne01) {
-            i01 = 0;
-            if (++i02 == ne02) {
-                i02 = 0;
-                i03++;
-            }
-            dst_ptr  = (uint8_t*) dst->data  + i02*nb2  + i03*nb3;
-            src0_ptr = (uint8_t*) src0->data + i02*nb02 + i03*nb03;
-        }
-    }
+#define DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
+static void cpy_thread_##NAME##_sameshape(unsigned int nth, unsigned int ith, void * data) { \
+    struct htp_copy_context * ct = (struct htp_copy_context *) data;                         \
+    struct htp_ops_context * octx = ct->octx;                                                \
+    cpy_preamble;                                                                            \
+                                                                                             \
+    const uint32_t dr  = ct->src0_nrows_per_thread;                                          \
+    const uint32_t ir0 = ct->row_start + dr * ith;                                           \
+    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
+    if (ir0 >= ir1) return;                                                                  \
+    const uint32_t nrows_thread = ir1 - ir0;                                                 \
+                                                                                             \
+    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
+    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
+                                                                                             \
+    const struct htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
+    const uint32_t src0_buf_size = cvt->src0_buf_size;                                       \
+    const uint32_t dst_buf_size  = cvt->dst_buf_size;                                        \
+    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * cvt->spad0_size_per_thread;             \
+    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * cvt->spad1_size_per_thread;             \
+    const uint32_t src0_row_size = ne00 * ct->kparams->src0_type_size;                       \
+    const uint32_t dst_row_size  = ne00 * ct->kparams->dst_type_size;                        \
+                                                                                             \
+    const uint32_t ne02_ne01 = ne02 * ne01;                                                  \
+    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
+    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
+    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
+    uint32_t i01 = rem - i02 * ne01;                                                         \
+                                                                                             \
+    uint32_t f_i01 = i01, f_i02 = i02, f_i03 = i03;                                          \
+    dma_addr_t f_src0_addr = src0->data + f_i01*nb01 + f_i02*nb02 + f_i03*nb03;              \
+                                                                                             \
+    uint32_t c_i01 = i01, c_i02 = i02, c_i03 = i03;                                          \
+    dma_addr_t c_dst_addr = dst->data + c_i01*nb1 + c_i02*nb2 + c_i03*nb3;                   \
+                                                                                             \
+    for (uint32_t r = 0; r < nrows_thread && r < 2; ++r) {                                   \
+        uint8_t * src_spad = vtcm_src0_base + r * src0_buf_size;                             \
+        uint8_t * dst_spad = vtcm_dst_base  + r * dst_buf_size;                              \
+        dma_queue_push(dma_q, dma_make_data(dst->data, dst_spad),                            \
+                       dst_row_size, dst_buf_size, dst_row_size, 0);                         \
+        dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                          \
+                       src0_buf_size, src0_row_size, src0_row_size, 1);                      \
+        f_src0_addr += nb01;                                                                 \
+        if (++f_i01 == ne01) {                                                               \
+            f_i01 = 0;                                                                       \
+            if (++f_i02 == ne02) {                                                           \
+                f_i02 = 0;                                                                   \
+                f_i03++;                                                                     \
+            }                                                                                \
+            f_src0_addr = src0->data + f_i02*nb02 + f_i03*nb03;                              \
+        }                                                                                    \
+    }                                                                                        \
+                                                                                             \
+    for (uint32_t r = 0; r < nrows_thread; ++r) {                                            \
+        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;               \
+        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;               \
+                                                                                             \
+        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                     \
+        CONV_FUNC(dst_spad, src_spad, ne00);                                                 \
+        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                      \
+                                                                                             \
+        dma_queue_push(dma_q, dma_make_data(c_dst_addr, dst_spad),                           \
+                       dst_row_size, dst_buf_size, dst_row_size, 1);                         \
+        c_dst_addr += nb1;                                                                   \
+        if (++c_i01 == ne01) {                                                               \
+            c_i01 = 0;                                                                       \
+            if (++c_i02 == ne02) {                                                           \
+                c_i02 = 0;                                                                   \
+                c_i03++;                                                                     \
+            }                                                                                \
+            c_dst_addr = dst->data + c_i02*nb2 + c_i03*nb3;                                  \
+        }                                                                                    \
+                                                                                             \
+        if (r + 2 < nrows_thread) {                                                          \
+            dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                      \
+                           src0_buf_size, src0_row_size, src0_row_size, 1);                  \
+            f_src0_addr += nb01;                                                             \
+            if (++f_i01 == ne01) {                                                           \
+                f_i01 = 0;                                                                   \
+                if (++f_i02 == ne02) {                                                       \
+                    f_i02 = 0;                                                               \
+                    f_i03++;                                                                 \
+                }                                                                            \
+                f_src0_addr = src0->data + f_i02*nb02 + f_i03*nb03;                          \
+            }                                                                                \
+        }                                                                                    \
+    }                                                                                        \
+    dma_queue_flush(dma_q);                                                                  \
 }
 
-static void cpy_thread_f32_f16_sameshape(unsigned int nth, unsigned int ith, void * data) {
-    struct htp_copy_context * ct = (struct htp_copy_context *) data;
-    struct htp_ops_context * octx = ct->octx;
-    cpy_preamble;
-
-    const uint32_t dr  = ct->src0_nrows_per_thread;
-    const uint32_t ir0 = ct->row_start + dr * ith;
-    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);
-    if (ir0 >= ir1) return;
-
-    const uint32_t ne02_ne01 = ne02 * ne01;
-    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);
-    uint32_t rem = ir0 - i03 * ne02_ne01;
-    uint32_t i02 = fastdiv(rem, &ct->div_ne01);
-    uint32_t i01 = rem - i02 * ne01;
-
-    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
-    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
-
-    for (uint32_t r = ir0; r < ir1; r++) {
-        hex_l2fetch(src0_ptr, ne00 * sizeof(__fp16), nb01, 2);
-        hvx_copy_f32_f16_uu(dst_ptr, src0_ptr, ne00);
-        dst_ptr  += nb1;
-        src0_ptr += nb01;
-        if (++i01 == ne01) {
-            i01 = 0;
-            if (++i02 == ne02) {
-                i02 = 0;
-                i03++;
-            }
-            dst_ptr  = (uint8_t*) dst->data  + i02*nb2  + i03*nb3;
-            src0_ptr = (uint8_t*) src0->data + i02*nb02 + i03*nb03;
-        }
+DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
+DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
+DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
+DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)
+
+static int cpy_scalar(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
+    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
+        return HTP_STATUS_OK;
     }
-}
 
-static void cpy_thread_i32_f32_sameshape(unsigned int nth, unsigned int ith, void * data) {
-    struct htp_copy_context * ct = (struct htp_copy_context *) data;
-    struct htp_ops_context * octx = ct->octx;
-    cpy_preamble;
-
-    const uint32_t dr  = ct->src0_nrows_per_thread;
-    const uint32_t ir0 = ct->row_start + dr * ith;
-    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);
-    if (ir0 >= ir1) return;
-
-    const uint32_t ne02_ne01 = ne02 * ne01;
-    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);
-    uint32_t rem = ir0 - i03 * ne02_ne01;
-    uint32_t i02 = fastdiv(rem, &ct->div_ne01);
-    uint32_t i01 = rem - i02 * ne01;
-
-    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
-    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
-
-    for (uint32_t r = ir0; r < ir1; r++) {
-        hex_l2fetch(src0_ptr, ne00 * sizeof(float), nb01, 2);
-        const float * restrict src_row = (const float *) src0_ptr;
-        int32_t * restrict dst_row = (int32_t *) dst_ptr;
-        for (uint32_t i = 0; i < ne00; i++) {
-            dst_row[i] = (int32_t) src_row[i];
-        }
-        dst_ptr  += nb1;
-        src0_ptr += nb01;
-        if (++i01 == ne01) {
-            i01 = 0;
-            if (++i02 == ne02) {
-                i02 = 0;
-                i03++;
-            }
-            dst_ptr  = (uint8_t*) dst->data  + i02*nb2  + i03*nb3;
-            src0_ptr = (uint8_t*) src0->data + i02*nb02 + i03*nb03;
-        }
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst  = octx->dst;
+
+    if (src0->type == dst->type) {
+        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->data, src0->data, kparams->src0_type_size);
+        dma_queue_flush(octx->ctx->dma[0]);
+        return HTP_STATUS_OK;
     }
+
+    dma_queue * dma_q = octx->ctx->dma[0];
+    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) octx->ctx->vtcm_base;
+    dma_addr_t d_vtcm = s_vtcm + VLEN;
+    const uint32_t s_size = kparams->src0_type_size;
+    const uint32_t d_size = kparams->dst_type_size;
+
+    dma_queue_push(dma_q, dma_make_data(s_vtcm, src0->data), s_size, s_size, s_size, 1);
+    dma_queue_pop(dma_q);
+
+    uint8_t * s_ptr = (uint8_t *) octx->ctx->vtcm_base;
+    uint8_t * d_ptr = s_ptr + VLEN;
+
+    const HVX_Vector v_src = hvx_vmem(s_ptr);
+    HVX_Vector v_dst;
+
+    if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
+        v_dst = Q6_Vw_equals_Vsf(v_src);
+    } else if (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
+        v_dst = Q6_Vsf_equals_Vw(v_src);
+    } else if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
+        v_dst = hvx_vec_f32_to_f16(v_src, v_src);
+    } else if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
+        v_dst = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src));
+    } else {
+        return HTP_STATUS_NO_SUPPORT;
+    }
+
+    hvx_vmem(d_ptr) = v_dst;
+
+    dma_queue_push(dma_q, dma_make_data(dst->data, d_vtcm), d_size, d_size, d_size, 1);
+    dma_queue_flush(dma_q);
+    return HTP_STATUS_OK;
 }
 
-static void cpy_thread_f32_i32_sameshape(unsigned int nth, unsigned int ith, void * data) {
-    struct htp_copy_context * ct = (struct htp_copy_context *) data;
-    struct htp_ops_context * octx = ct->octx;
-    cpy_preamble;
-
-    const uint32_t dr  = ct->src0_nrows_per_thread;
-    const uint32_t ir0 = ct->row_start + dr * ith;
-    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);
-    if (ir0 >= ir1) return;
-
-    const uint32_t ne02_ne01 = ne02 * ne01;
-    uint32_t i03 = fastdiv(ir0, &ct->div_ne02_ne01);
-    uint32_t rem = ir0 - i03 * ne02_ne01;
-    uint32_t i02 = fastdiv(rem, &ct->div_ne01);
-    uint32_t i01 = rem - i02 * ne01;
-
-    uint8_t* dst_ptr  = (uint8_t*) dst->data  + i01*nb1  + i02*nb2  + i03*nb3;
-    uint8_t* src0_ptr = (uint8_t*) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
-
-    for (uint32_t r = ir0; r < ir1; r++) {
-        hex_l2fetch(src0_ptr, ne00 * sizeof(int32_t), nb01, 2);
-        const int32_t * restrict src_row = (const int32_t *) src0_ptr;
-        float * restrict dst_row = (float *) dst_ptr;
-        for (uint32_t i = 0; i < ne00; i++) {
-            dst_row[i] = (float) src_row[i];
-        }
-        dst_ptr  += nb1;
-        src0_ptr += nb01;
-        if (++i01 == ne01) {
-            i01 = 0;
-            if (++i02 == ne02) {
-                i02 = 0;
-                i03++;
-            }
-            dst_ptr  = (uint8_t*) dst->data  + i02*nb2  + i03*nb3;
-            src0_ptr = (uint8_t*) src0->data + i02*nb02 + i03*nb03;
-        }
+static int cpy_1d_contig(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst  = octx->dst;
+
+    uint32_t elem_start = 0;
+    uint32_t nelem      = kparams->total_elems;
+
+    if (octx->ctx->mdev.count > 1) {
+        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
+            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        elem_start = range.start;
+        nelem      = range.count;
     }
+
+    if (nelem > 0) {
+        dma_queue * q = octx->ctx->dma[0];
+        const uint32_t type_size = kparams->src0_type_size;
+        dma_addr_t dst_addr  = dst->data  + elem_start * type_size;
+        dma_addr_t src0_addr = src0->data + elem_start * type_size;
+        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_size);
+        dma_queue_flush(q);
+    }
+
+    return HTP_STATUS_OK;
 }
 
-static int exec_cpy(struct htp_ops_context * octx, bool * use_dma) {
-    cpy_preamble;
-    *use_dma = false;
+static int cpy_sameshape_sametype(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst  = octx->dst;
 
-    const uint32_t total_elems_src = ne00 * ne01 * ne02 * ne03;
-    const uint32_t total_elems_dst = ne0 * ne1 * ne2 * ne3;
-    if (total_elems_src == 1 && total_elems_dst == 1) {
-        if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
-            ((int32_t *) dst->data)[0] = (int32_t) (((const float *) src0->data)[0]);
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
-            ((float *) dst->data)[0] = (float) (((const int32_t *) src0->data)[0]);
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_I32) {
-            ((int32_t *) dst->data)[0] = ((const int32_t *) src0->data)[0];
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F32) {
-            ((float *) dst->data)[0] = ((const float *) src0->data)[0];
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F16) {
-            ((__fp16 *) dst->data)[0] = ((const __fp16 *) src0->data)[0];
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
-            ((__fp16 *) dst->data)[0] = (__fp16) (((const float *) src0->data)[0]);
-            return HTP_STATUS_OK;
-        }
-        if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
-            ((float *) dst->data)[0] = (float) (((const __fp16 *) src0->data)[0]);
-            return HTP_STATUS_OK;
-        }
+    uint32_t row_start = 0;
+    uint32_t nrows     = kparams->total_rows;
+
+    if (octx->ctx->mdev.count > 1) {
+        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
+            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        row_start = range.start;
+        nrows     = range.count;
     }
 
-    struct htp_copy_context ct;
-    ct.octx = octx;
+    if (nrows > 0) {
+        dma_queue * q = octx->ctx->dma[0];
+        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_size, row_start, nrows);
+        dma_queue_flush(q);
+    }
 
-    switch (src0->type) {
-    case HTP_TYPE_F32: ct.src0_type_size = 4; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
-    case HTP_TYPE_F16: ct.src0_type_size = 2; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
-    case HTP_TYPE_I32: ct.src0_type_size = 4; ct.src0_block_size = 1; ct.src0_blocks_per_row = ne00 / 1; break;
-    default:
+    return HTP_STATUS_OK;
+}
+
+static int cpy_sameshape_convert(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst  = octx->dst;
+
+    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
         return HTP_STATUS_NO_SUPPORT;
     }
 
-    switch (dst->type) {
-    case HTP_TYPE_F32: ct.dst_type_size = 4; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
-    case HTP_TYPE_F16: ct.dst_type_size = 2; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
-    case HTP_TYPE_I32: ct.dst_type_size = 4; ct.dst_block_size = 1; ct.dst_blocks_per_row = ne0 / 1; break;
-    default:
-        return HTP_STATUS_NO_SUPPORT;
+    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
+        return HTP_STATUS_INVAL_PARAMS;
     }
 
-    const bool sametype   = (src0->type == dst->type);
-    const bool transposed = (nb00 > nb01) || (nb0 > nb1) ||
-                            (nb00 != ct.src0_type_size) || (nb0 != ct.dst_type_size) ||
-                            (nb01 < ne00 * ct.src0_type_size) || (nb1 < ne0 * ct.dst_type_size);
-    const bool sameshape  = !transposed && (ne00 == ne0 && ne01 == ne1 && ne02 == ne2 && ne03 == ne3);
+    uint32_t row_start = 0;
+    uint32_t nrows     = kparams->total_rows;
 
-    const uint32_t n_threads = octx->n_threads;
+    if (octx->ctx->mdev.count > 1) {
+        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
+            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        row_start = range.start;
+        nrows     = range.count;
+    }
 
-    const bool src_is_contiguous = htp_tensor_is_contiguous(src0, ct.src0_type_size);
-    const bool dst_is_contiguous = htp_tensor_is_contiguous(dst, ct.dst_type_size);
+    if (nrows == 0) {
+        return HTP_STATUS_OK;
+    }
 
-    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
-        if (!sametype) {
-            return HTP_STATUS_NO_SUPPORT;
-        }
-        if (!sameshape && !(src_is_contiguous && dst_is_contiguous && octx->ctx->mdev.count <= 1)) {
-            return HTP_STATUS_NO_SUPPORT;
-        }
+    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
+        return HTP_STATUS_VTCM_TOO_SMALL;
     }
 
-    if (sameshape) {
-        const uint32_t total_rows = ne01 * ne02 * ne03;
-        const uint32_t row_size   = ne00 * ct.dst_type_size;
+    const uint32_t n_threads = octx->n_threads;
+    const struct htp_copy_convert_params * cvt = &kparams->u.convert;
 
-        ct.div_ne01      = init_fastdiv_values(ne01);
-        ct.div_ne02_ne01 = init_fastdiv_values(ne02 * ne01);
+    struct htp_copy_context ct;
+    ct.octx                  = octx;
+    ct.kparams               = kparams;
+    ct.row_start             = row_start;
+    ct.nrows                 = nrows;
+    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
+
+    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
+    ct.vtcm_src0 = vtcm_base;
+    ct.vtcm_dst  = vtcm_base + (size_t) n_threads * cvt->spad0_size_per_thread;
+
+    work_queue_func_t copy_fun = NULL;
+    if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
+        copy_fun = cpy_thread_f16_f32_sameshape;
+    } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
+        copy_fun = cpy_thread_f32_f16_sameshape;
+    } else if (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
+        copy_fun = cpy_thread_i32_f32_sameshape;
+    } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
+        copy_fun = cpy_thread_f32_i32_sameshape;
+    } else {
+        return HTP_STATUS_NO_SUPPORT;
+    }
 
-        uint32_t row_start = 0;
-        uint32_t nrows     = total_rows;
+    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
+    return HTP_STATUS_OK;
+}
 
-        if (octx->ctx->mdev.count > 1) {
-            const uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
-            const bool can_split = htp_tensor_mdev_data_aligned(dst) && dst_is_contiguous;
-            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, can_split ? rows_per_chunk : 0,
-                                                               octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
-            row_start = range.start;
-            nrows     = range.count;
-        }
+static int cpy_reshape(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst  = octx->dst;
 
-        if (nrows == 0) {
-            return HTP_STATUS_OK;
-        }
+    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
+        return HTP_STATUS_NO_SUPPORT;
+    }
 
-        ct.row_start = row_start;
-        ct.nrows     = nrows;
-        ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
-
-        if (sametype && (octx->ctx->mdev.count <= 1 || htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
-            if (octx->ctx->mdev.idx == 0) {
-                *use_dma = true;
-                cpy_dma_sametype_sameshape(octx->ctx->dma[0], dst, src0, ct.src0_type_size);
-                dma_queue_flush(octx->ctx->dma[0]);
-            }
-        } else {
-            work_queue_func_t copy_fun = NULL;
-            if (sametype) {
-                switch (src0->type) {
-                    case HTP_TYPE_F32: copy_fun = cpy_thread_f32_sameshape; break;
-                    case HTP_TYPE_F16: copy_fun = cpy_thread_f16_sameshape; break;
-                    case HTP_TYPE_I32: copy_fun = cpy_thread_i32_sameshape; break;
-                    default: return HTP_STATUS_NO_SUPPORT;
-                }
-            } else if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
-                copy_fun = cpy_thread_f16_f32_sameshape;
-            } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
-                copy_fun = cpy_thread_f32_f16_sameshape;
-            } else if (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
-                copy_fun = cpy_thread_i32_f32_sameshape;
-            } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
-                copy_fun = cpy_thread_f32_i32_sameshape;
-            } else {
-                return HTP_STATUS_NO_SUPPORT;
-            }
-            work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
-        }
-    } else if (sametype) {
-        const uint32_t total_elems = ne0 * ne1 * ne2 * ne3;
-        const uint32_t elems_per_line = (ct.dst_type_size == 4) ? 32 : 64;
-
-        if (octx->ctx->mdev.count <= 1 && dst_is_contiguous && src_is_contiguous) {
-            *use_dma = true;
-            cpy_dma_sametype_reshape_contig(octx->ctx->dma[0], dst->data, src0->data, total_elems * ct.dst_type_size);
-            dma_queue_flush(octx->ctx->dma[0]);
-            return HTP_STATUS_OK;
-        }
+    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
+        return HTP_STATUS_INVAL_PARAMS;
+    }
 
-        ct.div_ne0            = init_fastdiv_values(ne0);
-        ct.div_ne1_ne0        = init_fastdiv_values(ne1 * ne0);
-        ct.div_ne2_ne1_ne0    = init_fastdiv_values(ne2 * ne1 * ne0);
-        ct.div_ne00           = init_fastdiv_values(ne00);
-        ct.div_ne01_ne00      = init_fastdiv_values(ne01 * ne00);
-        ct.div_ne02_ne01_ne00 = init_fastdiv_values(ne02 * ne01 * ne00);
-
-        uint32_t elem_start = 0;
-        uint32_t nelem      = total_elems;
-
-        if (octx->ctx->mdev.count > 1) {
-            const bool can_split = htp_tensor_mdev_data_aligned(dst) && dst_is_contiguous;
-            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_elems, can_split ? elems_per_line : 0,
-                                                               octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
-            elem_start = range.start;
-            nelem      = range.count;
-        }
+    uint32_t elem_start = 0;
+    uint32_t nelem      = kparams->total_elems;
 
-        if (nelem == 0) {
-            return HTP_STATUS_OK;
-        }
+    if (octx->ctx->mdev.count > 1) {
+        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
+            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        elem_start = range.start;
+        nelem      = range.count;
+    }
 
-        ct.elem_start      = elem_start;
-        ct.nelem           = nelem;
-        ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);
+    if (nelem == 0) {
+        return HTP_STATUS_OK;
+    }
 
-        work_queue_func_t copy_fun = NULL;
-        switch (src0->type) {
-            case HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; break;
-            case HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; break;
-            case HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; break;
-            default: return HTP_STATUS_NO_SUPPORT;
-        }
-        work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
-    } else {
-        return HTP_STATUS_NO_SUPPORT;
+    const uint32_t n_threads = octx->n_threads;
+
+    struct htp_copy_context ct;
+    ct.octx            = octx;
+    ct.kparams         = kparams;
+    ct.elem_start      = elem_start;
+    ct.nelem           = nelem;
+    ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);
+
+    work_queue_func_t copy_fun = NULL;
+    switch (src0->type) {
+        case HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; break;
+        case HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; break;
+        case HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; break;
+        default: return HTP_STATUS_NO_SUPPORT;
     }
 
+    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
     return HTP_STATUS_OK;
 }
 
 int op_cpy(struct htp_ops_context * octx) {
-    bool use_dma = false;
-    int status = exec_cpy(octx, &use_dma);
+    const struct htp_copy_kernel_params * kparams = (const struct htp_copy_kernel_params *) octx->kernel_params;
+    int status = HTP_STATUS_OK;
+
+    switch (kparams->kernel_type) {
+        case HTP_COPY_KERNEL_SCALAR:
+            status = cpy_scalar(octx, kparams);
+            break;
+        case HTP_COPY_KERNEL_1D_CONTIG:
+            status = cpy_1d_contig(octx, kparams);
+            break;
+        case HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
+            status = cpy_sameshape_sametype(octx, kparams);
+            break;
+        case HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
+            status = cpy_sameshape_convert(octx, kparams);
+            break;
+        case HTP_COPY_KERNEL_RESHAPE:
+            status = cpy_reshape(octx, kparams);
+            break;
+        default:
+            status = HTP_STATUS_NO_SUPPORT;
+            break;
+    }
 
     htp_ops_context_set_status(octx, status);
 
-    if (octx->op == HTP_OP_CPY_FENCE) {
-        if (!use_dma) {
-            htp_flush_dirty_ranges(octx->ctx);
-        }
-
+    if (octx->ctx->mdev.count > 1) {
         htp_mdev_group_barrier(octx);
+    }
 
+    if (octx->op == HTP_OP_CPY_FENCE) {
         if (octx->ctx->mdev.idx == 0) {
             const struct htp_tensor * sync = octx->src[1];
-            if (htp_tensor_is_extended(sync)) {
-                return HTP_STATUS_NO_SUPPORT;
-            }
             const uint32_t seq = (uint32_t) octx->op_params[0];
             atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;
             htp_fence_write(sync_fence, seq, octx->status);
diff --git src/ggml-hexagon/htp/cpy-ops.h src/ggml-hexagon/htp/cpy-ops.h
new file mode 100644
index 00000000..4cf7dc13
--- /dev/null
+++ src/ggml-hexagon/htp/cpy-ops.h
@@ -0,0 +1,79 @@
+#ifndef HTP_CPY_OPS_H
+#define HTP_CPY_OPS_H
+
+#include "hex-common.h"
+#include "hex-fastdiv.h"
+#include <stdint.h>
+
+enum htp_copy_kernel_type {
+    HTP_COPY_KERNEL_UNSUPPORTED        = 0,
+    HTP_COPY_KERNEL_1D_CONTIG          = 1,
+    HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE = 2,
+    HTP_COPY_KERNEL_SAMESHAPE_CONVERT  = 3,
+    HTP_COPY_KERNEL_RESHAPE            = 4,
+    HTP_COPY_KERNEL_SCALAR             = 5,
+};
+
+struct htp_copy_convert_params {
+    uint32_t              src0_buf_size;
+    uint32_t              dst_buf_size;
+    uint32_t              spad0_size_per_thread;
+    uint32_t              spad1_size_per_thread;
+    struct fastdiv_values div_ne01;
+    struct fastdiv_values div_ne02_ne01;
+};
+
+struct htp_copy_reshape_params {
+    struct fastdiv_values div_ne0;
+    struct fastdiv_values div_ne1_ne0;
+    struct fastdiv_values div_ne2_ne1_ne0;
+    struct fastdiv_values div_ne00;
+    struct fastdiv_values div_ne01_ne00;
+    struct fastdiv_values div_ne02_ne01_ne00;
+};
+
+struct htp_copy_kernel_params {
+    uint8_t  kernel_type;
+    uint8_t  src0_type_size;
+    uint8_t  dst_type_size;
+    uint8_t  n_threads;
+
+    uint32_t total_elems;
+    uint32_t total_rows;
+    uint32_t vtcm_size;
+
+    union {
+        struct htp_copy_convert_params convert;
+        struct htp_copy_reshape_params reshape;
+    } u;
+};
+
+struct htp_copy_convert_vtcm_layout {
+    uint32_t src0_buf_size;
+    uint32_t dst_buf_size;
+    uint32_t spad0_size_per_thread;
+    uint32_t spad1_size_per_thread;
+    uint32_t total_bytes;
+};
+
+static inline void htp_copy_convert_vtcm_layout_build(
+    struct htp_copy_convert_vtcm_layout * layout,
+    uint32_t ne00,
+    uint32_t src_type_size,
+    uint32_t dst_type_size,
+    uint32_t n_threads) {
+
+    layout->src0_buf_size = hex_round_up(ne00 * src_type_size, 256);
+    layout->dst_buf_size  = hex_round_up(ne00 * dst_type_size, 256);
+    layout->spad0_size_per_thread = 2 * layout->src0_buf_size;
+    layout->spad1_size_per_thread = 2 * layout->dst_buf_size;
+    layout->total_bytes = n_threads * (layout->spad0_size_per_thread + layout->spad1_size_per_thread);
+}
+
+#if defined(__cplusplus)
+static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
+#else
+_Static_assert(sizeof(struct htp_copy_kernel_params) <= 128, "htp_copy_kernel_params is too large for kernel_params blob");
+#endif
+
+#endif // HTP_CPY_OPS_H
diff --git src/ggml-hexagon/htp/dma-copy.h src/ggml-hexagon/htp/dma-copy.h
new file mode 100644
index 00000000..b2c91a13
--- /dev/null
+++ src/ggml-hexagon/htp/dma-copy.h
@@ -0,0 +1,169 @@
+#ifndef HTP_DMA_COPY_H
+#define HTP_DMA_COPY_H
+
+// DDR<->DDR DMA copies of same-type, same-shape tensors with arbitrary strides.
+// Used by CPY for the copy itself and by CONCAT, which is two such copies into
+// two views of its destination. Every helper only pushes descriptors; the
+// caller flushes the queue when it needs the data.
+
+#include "dma-queue.h"
+#include "hex-common.h"
+#include "htp-tensor.h"
+
+#include <stddef.h>
+#include <stdint.h>
+
+// Contiguous byte run, as 1d transfers of at most DMA_SAFE_CHUNK_SIZE each.
+static inline void dma_cpy_sametype_reshape_contig(dma_queue * dma_q,
+                                                   dma_addr_t  dst,
+                                                   dma_addr_t  src0,
+                                                   uint32_t    total_bytes) {
+    if (total_bytes == 0) {
+        return;
+    }
+
+    const uint32_t max_chunk = DMA_SAFE_CHUNK_SIZE;
+    while (total_bytes > 0) {
+        const uint32_t chunk = MIN(total_bytes, max_chunk);
+        if (!dma_queue_push(dma_q, dma_make_data(dst, src0), chunk, chunk, chunk, /*nrows=*/1)) {
+            dma_queue_flush(dma_q);
+            dma_queue_push(dma_q, dma_make_data(dst, src0), chunk, chunk, chunk, /*nrows=*/1);
+        }
+        dst += chunk;
+        src0 += chunk;
+        total_bytes -= chunk;
+    }
+}
+
+// One 2d transfer, split at the 16-bit nrows field.
+static inline void dma_cpy_push_2d_chunked(dma_queue * dma_q,
+                                           dma_addr_t  dst,
+                                           dma_addr_t  src,
+                                           size_t      dst_stride,
+                                           size_t      src_stride,
+                                           size_t      row_size,
+                                           uint32_t    nrows) {
+    if (row_size == 0 || nrows == 0) {
+        return;
+    }
+
+    while (nrows > 0) {
+        const uint32_t cur_rows = MIN(nrows, DMA_MAX_NROWS);
+        if (!dma_queue_push(dma_q, dma_make_data(dst, src), dst_stride, src_stride, row_size, cur_rows)) {
+            dma_queue_flush(dma_q);
+            dma_queue_push(dma_q, dma_make_data(dst, src), dst_stride, src_stride, row_size, cur_rows);
+        }
+        dst += cur_rows * dst_stride;
+        src += cur_rows * src_stride;
+        nrows -= cur_rows;
+    }
+}
+
+// Copy a range of rows [row_start, row_start + nrows) from src0 into dst:
+// same type, same ne[], any nb[] above dim 0, dim 0 dense on both sides (nb[0] == elem_size).
+static inline void dma_cpy_sametype_sameshape_range(dma_queue *               dma_q,
+                                                    const struct htp_tensor * dst,
+                                                    const struct htp_tensor * src0,
+                                                    uint32_t                  elem_size,
+                                                    uint32_t                  row_start,
+                                                    uint32_t                  nrows) {
+    if (nrows == 0) {
+        return;
+    }
+
+    const uint32_t ne00 = src0->ne[0];
+    const uint32_t ne01 = src0->ne[1];
+    const uint32_t ne02 = src0->ne[2];
+    const uint32_t ne03 = src0->ne[3];
+
+    if (ne00 == 0 || ne01 == 0 || ne02 == 0 || ne03 == 0) {
+        return;
+    }
+
+    const uint32_t nb01 = src0->nb[1];
+    const uint32_t nb02 = src0->nb[2];
+    const uint32_t nb03 = src0->nb[3];
+
+    const uint32_t nb1 = dst->nb[1];
+    const uint32_t nb2 = dst->nb[2];
+    const uint32_t nb3 = dst->nb[3];
+
+    const bool contiguous = htp_tensor_is_contiguous(src0, elem_size) && htp_tensor_is_contiguous(dst, elem_size);
+
+    if (contiguous) {
+        dma_cpy_sametype_reshape_contig(dma_q,
+                                        dst->data  + (dma_addr_t) row_start * ne00 * elem_size,
+                                        src0->data + (dma_addr_t) row_start * ne00 * elem_size,
+                                        nrows * ne00 * elem_size);
+        return;
+    }
+
+    // The single-descriptor path flattens (i01,i02,i03) into one row index, so every
+    // row must sit at a constant stride: nb01 on the source, nb1 on the destination.
+    // Walk the outer dims and require each to continue that progression. A dim of
+    // extent 1 spans no rows, so it is skipped -- but its own stride must NOT then be
+    // used to justify the next dim's stride, which is what comparing nb03 against
+    // ne02*nb02 did: ggml leaves the stride of an extent-1 dim meaningless, so a view
+    // could pass the check while its rows were nowhere near that stride.
+    uint32_t exp_src          = ne01 * nb01;
+    uint32_t exp_dst          = ne01 * nb1;
+    bool     contiguous_outer = true;
+    if (ne02 != 1) {
+        contiguous_outer = contiguous_outer && (nb02 == exp_src) && (nb2 == exp_dst);
+    }
+    exp_src *= ne02;
+    exp_dst *= ne02;
+    if (ne03 != 1) {
+        contiguous_outer = contiguous_outer && (nb03 == exp_src) && (nb3 == exp_dst);
+    }
+
+    if (contiguous_outer) {
+        dma_cpy_push_2d_chunked(dma_q,
+                                dst->data  + (dma_addr_t) row_start * nb1,
+                                src0->data + (dma_addr_t) row_start * nb01,
+                                nb1, nb01, ne00 * elem_size, nrows);
+        return;
+    }
+
+    const uint32_t ne02_ne01 = ne02 * ne01;
+    uint32_t i03 = row_start / ne02_ne01;
+    uint32_t rem = row_start - i03 * ne02_ne01;
+    uint32_t i02 = rem / ne01;
+    uint32_t i01 = rem - i02 * ne01;
+
+    dma_addr_t cur_dst  = dst->data  + (dma_addr_t) i01 * nb1  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
+    dma_addr_t cur_src0 = src0->data + (dma_addr_t) i01 * nb01 + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
+
+    uint32_t r = row_start;
+    const uint32_t row_end = row_start + nrows;
+    while (r < row_end) {
+        uint32_t cur_rows = MIN(row_end - r, ne01 - i01);
+        dma_cpy_push_2d_chunked(dma_q, cur_dst, cur_src0, nb1, nb01, ne00 * elem_size, cur_rows);
+        r   += cur_rows;
+        i01 += cur_rows;
+        if (i01 == ne01) {
+            i01 = 0;
+            if (++i02 == ne02) {
+                i02 = 0;
+                i03++;
+            }
+            cur_dst  = dst->data  + (dma_addr_t) i02 * nb2  + (dma_addr_t) i03 * nb3;
+            cur_src0 = src0->data + (dma_addr_t) i02 * nb02 + (dma_addr_t) i03 * nb03;
+        } else {
+            cur_dst  += cur_rows * nb1;
+            cur_src0 += cur_rows * nb01;
+        }
+    }
+}
+
+// Copy src0 into dst: same type, same ne[], any nb[] above dim 0, dim 0 dense on
+// both sides (nb[0] == elem_size).
+static inline void dma_cpy_sametype_sameshape(dma_queue *               dma_q,
+                                              const struct htp_tensor * dst,
+                                              const struct htp_tensor * src0,
+                                              uint32_t                  elem_size) {
+    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
+    dma_cpy_sametype_sameshape_range(dma_q, dst, src0, elem_size, 0, total_rows);
+}
+
+#endif /* HTP_DMA_COPY_H */
diff --git src/ggml-hexagon/htp/dma-queue.c src/ggml-hexagon/htp/dma-queue.c
index 464e4b84..ef61d2d2 100644
--- src/ggml-hexagon/htp/dma-queue.c
+++ src/ggml-hexagon/htp/dma-queue.c
@@ -161,9 +161,9 @@ bool dma_queue_push_fallback_contig(dma_queue * q, dma_data ddata, size_t total)
     while (rem_bytes > 0) {
         const uint32_t cur_bytes = MIN(rem_bytes, DMA_SAFE_CHUNK_SIZE);
         dma_data cur_data = dma_make_data(cur_dst, cur_src);
-        if (!dma_ring_push_single_1d(r1, cur_data, cur_bytes)) {
+        if (!dma_ring_push_single_contig(r1, cur_data, cur_bytes)) {
             dma_ring_flush(r1);
-            dma_ring_push_single_1d(r1, cur_data, cur_bytes);
+            dma_ring_push_single_contig(r1, cur_data, cur_bytes);
         }
         cur_dst   += cur_bytes;
         cur_src   += cur_bytes;
diff --git src/ggml-hexagon/htp/dma-queue.h src/ggml-hexagon/htp/dma-queue.h
index a736eb76..9b774bdd 100644
--- src/ggml-hexagon/htp/dma-queue.h
+++ src/ggml-hexagon/htp/dma-queue.h
@@ -107,6 +107,14 @@ typedef struct {
 #define DMA_MAX_STRIDE_24B     0x00FFFFFFu    // 24-bit HW descriptor limit for strides (16MB - 1)
 #define DMA_SAFE_CHUNK_SIZE    0x00F00000u    // ~15MB safe contiguous chunk size
 
+#if __HVX_ARCH__ < 75
+#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_16B
+#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_16B
+#else
+#define DMA_MAX_2D_ROW_SIZE    DMA_MAX_SIZE_24B
+#define DMA_MAX_2D_STRIDE      DMA_MAX_STRIDE_24B
+#endif
+
 #define DMA_FALLBACK_CAPACITY  16u            // descriptors in secondary fallback ring
 
 typedef struct dma_ring_s dma_ring;
@@ -216,13 +224,10 @@ static inline bool dma_ring_push_single_1d(dma_ring * r, dma_data ddata, size_t
 
 static inline bool dma_ring_push_single_2d(dma_ring * r, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
 #if __HVX_ARCH__ > 79
+    assert(!((ddata.src | ddata.dst) >> 40) || nrows == 0);
     const uint32_t src_hi = (uint32_t) (ddata.src >> 32);
     const uint32_t dst_hi = (uint32_t) (ddata.dst >> 32);
     const bool is_ext     = (src_hi | dst_hi) != 0;
-
-    if (is_ext && ((ddata.src >> 40) || (ddata.dst >> 40))) {
-        return false;
-    }
 #endif
 
     if (((r->push_idx + 1) & r->idx_mask) == r->pop_idx) {
@@ -284,6 +289,16 @@ static inline bool dma_ring_push_single_2d(dma_ring * r, dma_data ddata, size_t
     return true;
 }
 
+#if __HVX_ARCH__ < 75
+static inline bool dma_ring_push_single_contig(dma_ring * r, dma_data ddata, size_t size) {
+    return dma_ring_push_single_1d(r, ddata, size);
+}
+#else
+static inline bool dma_ring_push_single_contig(dma_ring * r, dma_data ddata, size_t size) {
+    return dma_ring_push_single_2d(r, ddata, size, size, size, 1);
+}
+#endif
+
 static inline dma_data dma_ring_pop(dma_ring * r) {
     dma_data ddata = { 0 };
 
@@ -374,57 +389,36 @@ static inline uint32_t dma_queue_capacity(dma_queue * q) {
     return dma_ring_capacity(q->ring0);
 }
 
-#if __HVX_ARCH__ < 75
-
-static inline bool dma_queue_push(dma_queue *q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
-    // Fast path: everything fits in 16 bits
-    if (nrows == 0 || __builtin_expect(
-            nrows      <= DMA_MAX_NROWS &&
-            row_size   <= DMA_MAX_SIZE_16B &&
-            src_stride <= DMA_MAX_STRIDE_16B &&
-            dst_stride <= DMA_MAX_STRIDE_16B, 1)) {
-        return dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
+static inline bool dma_queue_push(dma_queue * q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
+    if (__builtin_expect(nrows == 0, 0)) {
+        return dma_ring_push_single_1d(q->ring0, ddata, 0);
     }
 
-    // Contiguous block: 1D DMA mode supports up to 24-bit size (16MB)
-    if (nrows == 1 || (row_size == src_stride && row_size == dst_stride)) {
-        size_t total = row_size * nrows;
-        if (total <= DMA_MAX_SIZE_24B) {
-            return dma_ring_push_single_1d(q->ring0, ddata, total);
+    // 1. Hot path: Contiguous or single-row (80-90% of calls)
+    if (nrows == 1 || !((row_size ^ src_stride) | (row_size ^ dst_stride))) {
+        const size_t total = row_size * nrows;
+        if (__builtin_expect(total <= DMA_MAX_SIZE_24B, 1)) {
+            return dma_ring_push_single_contig(q->ring0, ddata, total);
         }
         return dma_queue_push_fallback_contig(q, ddata, total);
     }
 
-    // Row count overflow with 16-bit strides: chunk 2D descriptors via fallback ring
-    if (row_size <= DMA_MAX_SIZE_16B && src_stride <= DMA_MAX_STRIDE_16B && dst_stride <= DMA_MAX_STRIDE_16B) {
-        return dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
-    }
-
-    // Stride or row_size overflow: row-by-row 1D via fallback ring
-    return dma_queue_push_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
-}
-
-#else // HVX_ARCH >= 75
-
-static inline bool dma_queue_push(dma_queue *q, dma_data ddata, size_t dst_stride, size_t src_stride, size_t row_size, size_t nrows) {
-    if (nrows == 0 || __builtin_expect(
-            nrows      <= DMA_MAX_NROWS &&
-            row_size   <= DMA_MAX_SIZE_24B &&
-            src_stride <= DMA_MAX_STRIDE_24B &&
-            dst_stride <= DMA_MAX_STRIDE_24B, 1)) {
+    // 2. Hot path: Standard strided 2D (10-20% of calls)
+    if (__builtin_expect(nrows <= DMA_MAX_NROWS &&
+                         (row_size | src_stride | dst_stride) <= DMA_MAX_2D_ROW_SIZE, 1)) {
         return dma_ring_push_single_2d(q->ring0, ddata, dst_stride, src_stride, row_size, nrows);
     }
 
-    // Contiguous block exceeding 24 bits
-    if (nrows == 1 || (row_size == src_stride && row_size == dst_stride)) {
-        size_t total = row_size * nrows;
-        return dma_queue_push_fallback_contig(q, ddata, total);
+    // 3. Cold path: Descriptor chunking fallbacks (< 0.1%)
+#if __HVX_ARCH__ < 75
+    if (row_size <= DMA_MAX_SIZE_16B && (src_stride | dst_stride) <= DMA_MAX_STRIDE_16B) {
+        return dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
     }
-
+    return dma_queue_push_fallback_1d(q, ddata, dst_stride, src_stride, row_size, nrows);
+#else
     return dma_queue_push_fallback_2d(q, ddata, dst_stride, src_stride, row_size, nrows);
-}
-
 #endif
+}
 
 static inline void dma_sync_read(dma_queue * dma_q, void * dst, dma_addr_t src, size_t bytes) {
     const uint32_t b = (uint32_t) bytes;
diff --git src/ggml-hexagon/htp/flash-attn-ops.c src/ggml-hexagon/htp/flash-attn-ops.c
index f079f738..2ee55feb 100644
--- src/ggml-hexagon/htp/flash-attn-ops.c
+++ src/ggml-hexagon/htp/flash-attn-ops.c
@@ -1869,8 +1869,8 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
     factx.Bc             = kparams->Bc;
     factx.g_br           = kparams->u.hmx.g_br;
     factx.n_kv_blocks    = kparams->n_kv_blocks;
-    factx.is_q_fp32      = (kparams->is_q_fp32 != 0);
-    factx.is_dst_fp32    = (kparams->is_dst_fp32 != 0);
+    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
+    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
     factx.pipeline       = (kparams->u.hmx.pipeline != 0);
     factx.mask_broadcast = (kparams->u.hmx.mask_broadcast != 0);
     if (mask) {
@@ -1879,13 +1879,12 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
     }
 
     factx.has_softcap   = (kparams->logit_softcap != 0.0f);
-    if (!factx.has_softcap) {
-        factx.scale = (__fp16) (kparams->scale * EXP_LOG2E_F);  // log2(e)
-    } else {
-        factx.scale = (__fp16) kparams->scale;
-    }
+    factx.scale         = (__fp16) kparams->scale;
     factx.max_bias      = kparams->max_bias;
-    factx.logit_softcap = factx.has_softcap ? (__fp16) (kparams->logit_softcap * EXP_LOG2E_F) : 0;
+    factx.logit_softcap = 0;
+    if (factx.has_softcap) {
+        factx.logit_softcap = (__fp16) kparams->logit_softcap;
+    }
 
     factx.n_head_log2 = kparams->n_head_log2;
     factx.m0          = kparams->m0;
@@ -1898,22 +1897,36 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
     const uint32_t n_threads = factx.n_threads;
     const uint32_t G = factx.G;
 
-    // Multi-device: split Q blocks across devices
+    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
+    // shard), falling back to Q-block (token) split when heads don't divide evenly.
     const uint32_t n_q_blocks = (neq1 + Br - 1) / Br;
-    uint32_t q_start_min = 0;
-    uint32_t q_start_max = neq1;
+    uint32_t q_start_min  = 0;
+    uint32_t q_start_max  = neq1;
+    uint32_t kv_head_min  = 0;
+    uint32_t kv_head_max  = n_kv_heads;
 
     if (octx->ctx->mdev.count > 1) {
-        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_q_blocks, htp_tensor_mdev_data_aligned(dst) ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
-        const uint32_t block_start = range.start;
-        const uint32_t block_end   = range.start + range.count;
+        const uint32_t mdev_count = octx->ctx->mdev.count;
+        const uint32_t mdev_idx   = octx->ctx->mdev.idx;
+        const uint32_t dst_e_size = (dst->type == HTP_TYPE_F32) ? sizeof(float) : sizeof(__fp16);
+        const bool can_split      = htp_tensor_can_row_partition(dst, dst_e_size);
+
+        if (kparams->head_split && can_split && n_kv_heads >= mdev_count && n_kv_heads % mdev_count == 0) {
+            const uint32_t kv_per_core = n_kv_heads / mdev_count;
+            kv_head_min = mdev_idx * kv_per_core;
+            kv_head_max = kv_head_min + kv_per_core;
+        } else {
+            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_q_blocks, can_split ? 1 : 0, mdev_idx, mdev_count, &octx->ctx->mdev.count_div);
+            const uint32_t block_start = range.start;
+            const uint32_t block_end   = range.start + range.count;
 
-        if (block_start >= block_end) {
-            return HTP_STATUS_OK;
-        }
+            if (block_start >= block_end) {
+                return HTP_STATUS_OK;
+            }
 
-        q_start_min = block_start * Br;
-        q_start_max = MIN(block_end * Br, neq1);
+            q_start_min = block_start * Br;
+            q_start_max = MIN(block_end * Br, neq1);
+        }
     }
 
     // ======== VTCM allocation (GQA-aware) ========
@@ -2032,7 +2045,7 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
             const size_t   g_br_actual = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
             const size_t   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_ROWS;
 
-            for (uint32_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
+            for (uint32_t kv_head = kv_head_min; kv_head < kv_head_max; ++kv_head) {
                 const uint32_t ik2 = kv_head;
                 const uint32_t ik3 = fastdiv(ib3, &kparams->broadcast_rk3);
                 const uint32_t iv2 = kv_head;
@@ -2040,7 +2053,7 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
 
                 // 1. Push Q and KV DMAs for the very first iteration.
                 // Subsequent iterations are enqueued early at the end of the previous iteration.
-                if (ib3 == 0 && q_start == q_start_min && kv_head == 0) {
+                if (ib3 == 0 && q_start == q_start_min && kv_head == kv_head_min) {
                     const dma_addr_t q_ptr = q->data + q_start * q->nb[1] +
                                             (kv_head * factx.G) * q->nb[2] + ib3 * q->nb[3];
                     const size_t q_row_bytes = q_transposed ? n_rows_q * q_row_bytes_trans_factor : q_row_bytes_untransposed;
@@ -2358,8 +2371,8 @@ int hmx_flash_attn_ext(struct htp_ops_context * octx) {
                 uint32_t next_kv_head = kv_head + 1;
                 uint32_t next_q_start = q_start;
                 uint32_t next_ib3     = ib3;
-                if (next_kv_head >= n_kv_heads) {
-                    next_kv_head = 0;
+                if (next_kv_head >= kv_head_max) {
+                    next_kv_head = kv_head_min;
                     next_q_start = q_start + Br;
                     if (next_q_start >= q_start_max) {
                         next_q_start = q_start_min;
@@ -2478,7 +2491,7 @@ int op_flash_attn_ext(struct htp_ops_context * octx) {
         factx.src3_div3 = kparams->src3_div3;
     }
 
-    factx.is_q_fp32 = (kparams->is_q_fp32 != 0);
+    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
     factx.size_q_row_padded = kparams->u.hvx.size_q_row_padded;
     factx.size_k_row_padded = kparams->u.hvx.size_k_row_padded;
     factx.size_v_row_padded = kparams->u.hvx.size_v_row_padded;
@@ -2488,7 +2501,10 @@ int op_flash_attn_ext(struct htp_ops_context * octx) {
     factx.scale = kparams->scale;
     factx.max_bias = kparams->max_bias;
     factx.has_softcap = (kparams->logit_softcap != 0.0f);
-    factx.logit_softcap = factx.has_softcap ? (__fp16) kparams->logit_softcap : 0;
+    factx.logit_softcap = 0;
+    if (factx.has_softcap) {
+        factx.logit_softcap = (__fp16) kparams->logit_softcap;
+    }
 
     factx.n_head_log2 = kparams->n_head_log2;
     factx.m0          = kparams->m0;
@@ -2512,10 +2528,25 @@ int op_flash_attn_ext(struct htp_ops_context * octx) {
     uint32_t qrows      = total_qrows;
 
     if (octx->ctx->mdev.count > 1) {
-        const bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0);
-        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_qrows, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
-        qrow_start = range.start;
-        qrows      = range.count;
+        const uint32_t mdev_count = octx->ctx->mdev.count;
+        const uint32_t mdev_idx   = octx->ctx->mdev.idx;
+        const uint32_t n_kv_heads = k->ne[2];
+        const uint32_t dst_e_size = (dst->type == HTP_TYPE_F32) ? sizeof(float) : sizeof(__fp16);
+        const bool can_split      = htp_tensor_can_row_partition(dst, dst_e_size);
+
+        // head range is contiguous in flat row space only when neq3 == 1
+        if (kparams->head_split && can_split && neq3 == 1 && n_kv_heads >= mdev_count && n_kv_heads % mdev_count == 0) {
+            const uint32_t G              = kparams->G;
+            const uint32_t kv_per_core    = n_kv_heads / mdev_count;
+            const uint32_t heads_per_core = kv_per_core * G;
+            const uint32_t head_start     = mdev_idx * heads_per_core;
+            qrow_start = head_start * neq1;
+            qrows      = heads_per_core * neq1;
+        } else {
+            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_qrows, can_split ? 1 : 0, mdev_idx, mdev_count, &octx->ctx->mdev.count_div);
+            qrow_start = range.start;
+            qrows      = range.count;
+        }
     }
 
     if (qrows == 0) {
diff --git src/ggml-hexagon/htp/flash-attn-ops.h src/ggml-hexagon/htp/flash-attn-ops.h
index 22bb8c53..04538842 100644
--- src/ggml-hexagon/htp/flash-attn-ops.h
+++ src/ggml-hexagon/htp/flash-attn-ops.h
@@ -34,8 +34,8 @@ enum htp_fa_kernel_type {
 
 struct htp_fa_kernel_params {
     uint8_t  kernel_type;        // enum htp_fa_kernel_type
-    uint8_t  is_q_fp32;          // 1 = Q type is F32, 0 = F16
-    uint8_t  is_dst_fp32;        // 1 = dst type is F32, 0 = F16
+    uint8_t  head_split;         // 1 = partition by KV heads in multicore, 0 = token partition
+    uint8_t  flags;              // reserved
     uint8_t  n_threads;          // Number of threads to run
 
     // Common parameters
diff --git src/ggml-hexagon/htp/get-rows-ops.c src/ggml-hexagon/htp/get-rows-ops.c
index f354b813..90f38fa5 100644
--- src/ggml-hexagon/htp/get-rows-ops.c
+++ src/ggml-hexagon/htp/get-rows-ops.c
@@ -217,7 +217,7 @@ GET_ROWS_THREAD_DT_FN(q8_0, Q8_0_BYTES, int32_t, { compute_get_rows_q8_0((float
 GET_ROWS_THREAD_DT_FN(q8_0, Q8_0_BYTES, int64_t, { compute_get_rows_q8_0((float *)dst_spad, src_spad, cur_elems); })
 
 
-static __attribute__((noinline)) void compute_get_rows_tiled(float * dst, const uint8_t * tile, uint32_t row, bool q4) {
+static __attribute__((noinline)) void compute_get_rows_tiled(float * dst, const uint8_t * tile, uint32_t row, bool q4, bool q4_k) {
     const HVX_VectorPred first2 = Q6_Q_vsetq_R(2);
     const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);
     HVX_Vector vq = Q6_V_vzero();
@@ -235,7 +235,9 @@ static __attribute__((noinline)) void compute_get_rows_tiled(float * dst, const
         const HVX_Vector lo = Q6_V_vand_VV(vq, Q6_Vb_vsplat_R(0x0F));
         const HVX_Vector hi = Q6_Vub_vlsr_VubR(vq, 4);
         vq = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
-        vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));
+        if (!q4_k) {
+            vq = Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(8));
+        }
     } else {
         for (int group = 7; group >= 0; --group) {
             const HVX_Vector v = Q6_V_vror_VR(hvx_vmem(tile + group * VLEN), 2 * row);
@@ -245,14 +247,46 @@ static __attribute__((noinline)) void compute_get_rows_tiled(float * dst, const
         }
     }
     const HVX_Vector scales = hvx_vmem(tile + (q4 ? 512 : 1024));
-    const HVX_Vector scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * row));
+    const HVX_Vector scale_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, (q4_k ? 4 : 2) * row));
     const HVX_Vector scale = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hf));
     const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(vq);
     const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
-    const HVX_Vector values = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
+    HVX_Vector values = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
+    if (q4_k) {
+        const HVX_Vector offset_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 4 * row + 2));
+        const HVX_Vector offset = Q6_V_lo_W(hvx_vec_f16_to_f32(offset_hf));
+        values = hvx_vec_add_f32_f32(values, offset);
+    }
     *(HVX_Vector *) dst = values;
 }
 
+static __attribute__((noinline)) void compute_get_rows_q6_k(float * dst, const uint8_t * tile, uint32_t row) {
+    const HVX_VectorPred first4 = Q6_Q_vsetq_R(4);
+    const HVX_VectorPred first16 = Q6_Q_vsetq_R(16 * sizeof(float));
+    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
+    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);
+    HVX_Vector vq = Q6_V_vzero();
+
+    for (int group = 7; group >= 0; --group) {
+        const HVX_Vector lo_plane = Q6_V_vror_VR(hvx_vmem(tile + (group >> 1) * VLEN), 4 * row);
+        const HVX_Vector hi_plane = Q6_V_vror_VR(hvx_vmem(tile + 512 + (group >> 2) * VLEN), 4 * row);
+        const HVX_Vector lo = (group & 1) ? Q6_Vub_vlsr_VubR(lo_plane, 4) : Q6_V_vand_VV(lo_plane, mask_0f);
+        const HVX_Vector hi = Q6_Vub_vlsr_VubR(hi_plane, 2 * (group & 3));
+        const HVX_Vector packed = Q6_V_vor_VV(lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(hi, mask_03), 4));
+        vq = Q6_V_vmux_QVV(first4, packed, Q6_V_vror_VR(vq, VLEN - 4));
+    }
+
+    const HVX_Vector scales = hvx_vmem(tile + 768);
+    const HVX_Vector scale_lo_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 2 * row));
+    const HVX_Vector scale_hi_hf = hvx_vec_repl_f16(Q6_V_vror_VR(scales, 64 + 2 * row));
+    const HVX_Vector scale_lo = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_lo_hf));
+    const HVX_Vector scale_hi = Q6_V_lo_W(hvx_vec_f16_to_f32(scale_hi_hf));
+    const HVX_Vector scale = Q6_V_vmux_QVV(first16, scale_lo, scale_hi);
+    const HVX_VectorPair p16 = Q6_Wh_vunpack_Vb(Q6_Vb_vsub_VbVb(vq, Q6_Vb_vsplat_R(32)));
+    const HVX_VectorPair p32 = Q6_Ww_vunpack_Vh(Q6_V_lo_W(p16));
+    *(HVX_Vector *) dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(p32)), scale);
+}
+
 struct get_rows_tiled_task {
     dma_addr_t tile_src_base;
     dma_addr_t dst_data;
@@ -315,7 +349,9 @@ static void get_rows_thread_tiled(unsigned int nth, unsigned int ith, void * dat
     const uint32_t tile_size   = grctx->tile_size;
     const uint32_t tile_stride = grctx->tile_stride;
     const uint32_t dst_bytes   = ne00 * sizeof(float);
-    const bool is_q4 = (octx->src[0]->type == HTP_TYPE_Q4_0);
+    const bool is_q4 = octx->src[0]->type == HTP_TYPE_Q4_0 || octx->src[0]->type == HTP_TYPE_Q4_K;
+    const bool is_q4_k = octx->src[0]->type == HTP_TYPE_Q4_K;
+    const bool is_q6_k = octx->src[0]->type == HTP_TYPE_Q6_K;
 
     for (uint32_t step = 0, spad_idx = 0; step < ir1 - ir0 && spad_idx < 2; ++step, ++spad_idx) {
         const uint32_t i = ir0 + step;
@@ -343,7 +379,11 @@ static void get_rows_thread_tiled(unsigned int nth, unsigned int ith, void * dat
         for (uint32_t k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
             const uint8_t * tile = src_spad + k_tile * tile_stride;
             float * dst_block = dst_spad + k_tile * HTP_MM_HMX_TILE_N_COLS;
-            compute_get_rows_tiled(dst_block, tile, task.row, is_q4);
+            if (is_q6_k) {
+                compute_get_rows_q6_k(dst_block, tile, task.row);
+            } else {
+                compute_get_rows_tiled(dst_block, tile, task.row, is_q4, is_q4_k);
+            }
         }
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
 
@@ -369,10 +409,12 @@ int op_get_rows(struct htp_ops_context * octx) {
     const struct htp_get_rows_kernel_params * kparams = (const struct htp_get_rows_kernel_params *) octx->kernel_params;
 
     if (octx->src[0]->type != HTP_TYPE_F32 &&
-         octx->src[0]->type != HTP_TYPE_F16 &&
-         octx->src[0]->type != HTP_TYPE_Q4_0 &&
-         octx->src[0]->type != HTP_TYPE_Q8_0 &&
-         octx->src[0]->type != HTP_TYPE_I32) {
+        octx->src[0]->type != HTP_TYPE_F16 &&
+        octx->src[0]->type != HTP_TYPE_Q4_0 &&
+        octx->src[0]->type != HTP_TYPE_Q4_K &&
+        octx->src[0]->type != HTP_TYPE_Q6_K &&
+        octx->src[0]->type != HTP_TYPE_Q8_0 &&
+        octx->src[0]->type != HTP_TYPE_I32) {
         return HTP_STATUS_NO_SUPPORT;
     }
 
@@ -426,7 +468,7 @@ int op_get_rows(struct htp_ops_context * octx) {
     grctx.task_start = task_start;
     grctx.tasks = tasks;
     grctx.tasks_per_thread = octx->ctx->mdev.count == 1 ? kparams->tasks_per_thread : fastdiv(tasks + n_threads - 1, &octx->n_threads_div);
-    grctx.tile_size = octx->src[0]->type == HTP_TYPE_Q4_0 ? HTP_MM_WEIGHT_TILE_SIZE_Q4_0 : HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
+    grctx.tile_size = htp_mm_get_weight_tile_size(octx->src[0]->type);
     grctx.tile_stride = (grctx.tile_size + 127) & ~127;
     grctx.index_i32 = octx->src[1]->type == HTP_TYPE_I32;
 
diff --git src/ggml-hexagon/htp/get-rows-ops.h src/ggml-hexagon/htp/get-rows-ops.h
index 06ca1ea7..69fa5446 100644
--- src/ggml-hexagon/htp/get-rows-ops.h
+++ src/ggml-hexagon/htp/get-rows-ops.h
@@ -55,7 +55,7 @@ static inline void htp_get_rows_vtcm_layout_build(
     }
 
     if (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
-        const size_t tile_size   = type == HTP_TYPE_Q4_0 ? HTP_MM_WEIGHT_TILE_SIZE_Q4_0 : HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
+        const size_t tile_size   = htp_mm_get_weight_tile_size(type);
         const size_t tile_stride = (tile_size + 127) & ~127;
         const uint32_t n_k_tiles = ne00 / HTP_MM_HMX_TILE_N_COLS;
         const size_t row_tiles_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
diff --git src/ggml-hexagon/htp/hex-utils.h src/ggml-hexagon/htp/hex-utils.h
index 853f1c1b..abb44da2 100644
--- src/ggml-hexagon/htp/hex-utils.h
+++ src/ggml-hexagon/htp/hex-utils.h
@@ -38,21 +38,15 @@ static inline void hex_l2fetch_block(const void * addr, size_t size) {
 }
 
 #define HEX_L2_LINE_SIZE           128
-#define HEX_L2_BLOCK_SIZE          (HEX_L2_LINE_SIZE * 4) // flush granularity (lines per loop iteration)
+#define HEX_L2_BLOCK_SIZE          (HEX_L2_LINE_SIZE * 4) // flush granularity (chunks per thread)
 #define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
 #define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)
 
 static inline void hex_l2flush(void * addr, size_t size) {
+    if (size == 0) return;
     const uint32_t s = ((uint32_t) addr) & ~(HEX_L2_LINE_SIZE - 1);
     const uint32_t e = (((uint32_t) addr) + size + HEX_L2_LINE_SIZE - 1) & ~(HEX_L2_LINE_SIZE - 1);
-    const uint32_t eb = s + ((e - s) & ~(HEX_L2_BLOCK_SIZE - 1));
-    for (uint32_t i = s; i < eb; i += HEX_L2_BLOCK_SIZE) {
-        Q6_dccleaninva_A((void *) (i + HEX_L2_LINE_SIZE * 0));
-        Q6_dccleaninva_A((void *) (i + HEX_L2_LINE_SIZE * 1));
-        Q6_dccleaninva_A((void *) (i + HEX_L2_LINE_SIZE * 2));
-        Q6_dccleaninva_A((void *) (i + HEX_L2_LINE_SIZE * 3));
-    }
-    for (uint32_t i = eb; i < e; i += HEX_L2_LINE_SIZE) {
+    for (uint32_t i = s; i < e; i += HEX_L2_LINE_SIZE) {
         Q6_dccleaninva_A((void *) i);
     }
 }
diff --git src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
index 7d7455d0..40404bf1 100644
--- src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
+++ src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h
@@ -631,17 +631,40 @@ static void dequantize_tiled_weight_to_fp16_task_q6_k(
         HVX_Vector v_scale_k16 = Q6_V_lo_W(Q6_W_vshuff_VVR(v_sc_k16, v_sc_k16, -2));
 
         #pragma unroll
-        for (int g = 0; g < 8; g++) {
+        for (int g = 0; g < 8; g += 4) {
             const HVX_Vector v_scale = (g < 4) ? v_scale_k0 : v_scale_k16;
 
-            HVX_Vector     v_q   = unpack_q6_k_group(vptr, g, mask_0f, mask_03, i32);
-            HVX_VectorPair vp16  = Q6_Wh_vunpack_Vb(v_q);
-            HVX_VectorPair vp_k  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16), Q6_V_lo_W(vp16), -4);
-
-            hvx_vmem(dst_ptr + (2 * g + 0) * 64) =
-                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k)), v_scale));
-            hvx_vmem(dst_ptr + (2 * g + 1) * 64) =
-                Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k)), v_scale));
+            HVX_Vector v_q0 = unpack_q6_k_group(vptr, g + 0, mask_0f, mask_03, i32);
+            HVX_Vector v_q1 = unpack_q6_k_group(vptr, g + 1, mask_0f, mask_03, i32);
+            HVX_Vector v_q2 = unpack_q6_k_group(vptr, g + 2, mask_0f, mask_03, i32);
+            HVX_Vector v_q3 = unpack_q6_k_group(vptr, g + 3, mask_0f, mask_03, i32);
+
+            HVX_VectorPair vp16_0 = Q6_Wh_vunpack_Vb(v_q0);
+            HVX_VectorPair vp16_1 = Q6_Wh_vunpack_Vb(v_q1);
+            HVX_VectorPair vp16_2 = Q6_Wh_vunpack_Vb(v_q2);
+            HVX_VectorPair vp16_3 = Q6_Wh_vunpack_Vb(v_q3);
+            HVX_VectorPair vp_k0  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_0), Q6_V_lo_W(vp16_0), -4);
+            HVX_VectorPair vp_k1  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_1), Q6_V_lo_W(vp16_1), -4);
+            HVX_VectorPair vp_k2  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_2), Q6_V_lo_W(vp16_2), -4);
+            HVX_VectorPair vp_k3  = Q6_W_vdeal_VVR(Q6_V_hi_W(vp16_3), Q6_V_lo_W(vp16_3), -4);
+
+            HVX_Vector v_out00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k0)), v_scale));
+            HVX_Vector v_out01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k0)), v_scale));
+            HVX_Vector v_out10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k1)), v_scale));
+            HVX_Vector v_out11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k1)), v_scale));
+            HVX_Vector v_out20 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k2)), v_scale));
+            HVX_Vector v_out21 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k2)), v_scale));
+            HVX_Vector v_out30 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(vp_k3)), v_scale));
+            HVX_Vector v_out31 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(vp_k3)), v_scale));
+
+            hvx_vmem(dst_ptr + (2 * g + 0) * 64) = v_out00;
+            hvx_vmem(dst_ptr + (2 * g + 1) * 64) = v_out01;
+            hvx_vmem(dst_ptr + (2 * g + 2) * 64) = v_out10;
+            hvx_vmem(dst_ptr + (2 * g + 3) * 64) = v_out11;
+            hvx_vmem(dst_ptr + (2 * g + 4) * 64) = v_out20;
+            hvx_vmem(dst_ptr + (2 * g + 5) * 64) = v_out21;
+            hvx_vmem(dst_ptr + (2 * g + 6) * 64) = v_out30;
+            hvx_vmem(dst_ptr + (2 * g + 7) * 64) = v_out31;
         }
     }
 }
@@ -745,7 +768,7 @@ void convert_f16_weight_to_fp16_tiles_task(
                 const uint8_t *r0 = state->src + row0 * state->row_stride;
                 const uint8_t *r1 = state->src + row1 * state->row_stride;
 
-                HVX_Vector v0 = hvx_vmemu((const __fp16 *)(r0 + byte_off));
+                HVX_Vector v0 = (row0 < state->n_cols) ? hvx_vmemu((const __fp16 *)(r0 + byte_off)) : Q6_V_vzero();
                 HVX_Vector v1 = (row1 < state->n_cols) ? hvx_vmemu((const __fp16 *)(r1 + byte_off)) : Q6_V_vzero();
 
                 Q6_vscatter_QRMVwV(q_mask64, (size_t)tile_base, HTP_MM_HMX_TILE_SIZE - 1, v_off, v0);
@@ -788,7 +811,7 @@ void quantize_f32_weight_to_fp16_tiles_task(
                 const uint8_t *r0 = state->src + row0 * state->row_stride;
                 const uint8_t *r1 = state->src + row1 * state->row_stride;
 
-                HVX_Vector v0_f32 = hvx_vmem((const float *)(r0 + byte_off));
+                HVX_Vector v0_f32 = (row0 < state->n_cols) ? hvx_vmem((const float *)(r0 + byte_off)) : Q6_V_vzero();
                 HVX_Vector v1_f32 = (row1 < state->n_cols) ? hvx_vmem((const float *)(r1 + byte_off)) : Q6_V_vzero();
 
                 HVX_Vector v_out = hvx_vec_f32_to_f16(v0_f32, v1_f32);
@@ -988,9 +1011,7 @@ static void transfer_output_chunk_fp16_to_fp32_col_chunk(
     uint32_t src2_stride,
     uint32_t dst_cols
 ) {
-    assert(c_len % HTP_MM_HMX_TILE_N_COLS == 0);
-    assert(total_n_cols % HTP_MM_HMX_TILE_N_COLS == 0);
-    const size_t tile_row_stride = (total_n_cols / HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
+    const size_t tile_row_stride = hmx_ceil_div(total_n_cols, HTP_MM_HMX_TILE_N_COLS) * HTP_MM_HMX_TILE_N_ELMS;
 
     const HVX_Vector one = hvx_vec_splat_f16(1.0);
 
@@ -1137,6 +1158,73 @@ static void transfer_activation_row_pair_fp32_to_fp16(
     }
 }
 
+// F16-input variant of transfer_activation_row_pair_fp32_to_fp16, for F16 activation
+// (src1). Same shape as the F16 Q-prep in hmx-fa-kernels.h: one 128-byte load carries 64 f16
+// = two tile columns, and Q6_W_vshuff_VVR interleaves the two rows straight into the HMX tile
+// layout, so no F32 round-trip is needed. Rows are only 64-byte aligned when k_block is an odd
+// multiple of the tile width, hence the unaligned load type.
+static void transfer_activation_row_pair_f16_to_f16(__fp16 * restrict vtcm_dst,
+                                                    const __fp16 * restrict row0,
+                                                    const __fp16 * restrict row1,
+                                                    uint32_t r,
+                                                    uint32_t k_block,
+                                                    uint32_t k_valid,
+                                                    bool     row0_valid,
+                                                    bool     row1_valid) {
+    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
+    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx
+
+    const uint32_t n_tile_cols = k_block / HTP_MM_HMX_TILE_N_COLS;
+    __fp16 * restrict tile_row = vtcm_dst + (size_t) r0 * n_tile_cols * HTP_MM_HMX_TILE_N_ELMS;
+
+    const HVX_UVector * pv0 = row0_valid ? (const HVX_UVector *) row0 : NULL;
+    const HVX_UVector * pv1 = row1_valid ? (const HVX_UVector *) row1 : NULL;
+
+    uint32_t c = 0;
+    for (; c + 64 <= k_valid; c += 64) {
+        HVX_Vector     v0 = pv0 ? pv0[c / 64] : Q6_V_vzero();
+        HVX_Vector     v1 = pv1 ? pv1[c / 64] : Q6_V_vzero();
+        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
+
+        uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
+
+        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
+        HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
+
+        tile0[r1 / 2] = Q6_V_lo_W(vp);
+        tile1[r1 / 2] = Q6_V_hi_W(vp);
+    }
+    // Tail: fewer than 64 valid columns left, plus the k_valid..k_block padding that HMX will
+    // still multiply, so it has to be written as zeros.
+    for (; c < k_block; c += 64) {
+        HVX_Vector v0 = Q6_V_vzero();
+        HVX_Vector v1 = Q6_V_vzero();
+
+        if (c < k_valid) {
+            uint32_t       rem  = k_valid - c;  // 1..63 valid f16 lanes
+            HVX_VectorPred mask = Q6_Q_vsetq2_R(rem * sizeof(__fp16));
+            if (pv0) {
+                v0 = Q6_V_vmux_QVV(mask, pv0[c / 64], Q6_V_vzero());
+            }
+            if (pv1) {
+                v1 = Q6_V_vmux_QVV(mask, pv1[c / 64], Q6_V_vzero());
+            }
+        }
+
+        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
+
+        uint32_t c0 = c / HTP_MM_HMX_TILE_N_COLS;
+
+        HVX_Vector * tile0 = (HVX_Vector *) (tile_row + (size_t) c0 * HTP_MM_HMX_TILE_N_ELMS);
+        tile0[r1 / 2]      = Q6_V_lo_W(vp);
+
+        if (c0 + 1 < n_tile_cols) {
+            HVX_Vector * tile1 = (HVX_Vector *) (tile_row + (size_t) (c0 + 1) * HTP_MM_HMX_TILE_N_ELMS);
+            tile1[r1 / 2]      = Q6_V_hi_W(vp);
+        }
+    }
+}
+
 static void transfer_activation_row_pair_fp32_to_fp16_col_chunk(
         __fp16 *restrict vtcm_dst,
         const float *restrict row0, // offset by c_first
diff --git src/ggml-hexagon/htp/hmx-utils.h src/ggml-hexagon/htp/hmx-utils.h
index ad295cb7..1952aaa2 100644
--- src/ggml-hexagon/htp/hmx-utils.h
+++ src/ggml-hexagon/htp/hmx-utils.h
@@ -73,19 +73,20 @@ static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
         for (uint32_t r = start_row; r < end_row; r += 2) {
             const uint32_t   ct             = r / HMX_FP16_TILE_N_ROWS;
             const uint32_t   local_r        = r % HMX_FP16_TILE_N_ROWS;
+            const bool       row0_valid     = r < n_cols;
             const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
             const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
             const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
 
             __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
-            const uint8_t * p0 = (const uint8_t *) (vtcm_src + r * src_stride);
+            const uint8_t * p0 = row0_valid ? (const uint8_t *) (vtcm_src + r * src_stride) : NULL;
             const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;
 
-            assert(hex_is_aligned(p0, 128));
-            assert(hex_is_aligned(p1, 128));
+            assert(!p0 || hex_is_aligned(p0, 128));
+            assert(!p1 || hex_is_aligned(p1, 128));
             assert(c_byte_step % 128 == 0);
 
-            if (p1) {
+            if (p0 && p1) {
                 for (uint32_t i = 0; i < n_c_iters; ++i) {
                     HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
                     HVX_Vector v1 = hvx_vmem(p1); p1 += c_byte_step;
@@ -96,9 +97,12 @@ static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
             } else {
                 const HVX_Vector vzero = Q6_V_vzero();
                 for (uint32_t i = 0; i < n_c_iters; ++i) {
-                    HVX_Vector v0 = hvx_vmem(p0); p0 += c_byte_step;
+                    HVX_Vector v0 = p0 ? hvx_vmem(p0) : vzero;
+                    if (p0) p0 += c_byte_step;
+                    HVX_Vector v1 = p1 ? hvx_vmem(p1) : vzero;
+                    if (p1) p1 += c_byte_step;
                     Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off0, v0);
-                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, vzero);
+                    Q6_vscatter_RMVwV((size_t) tile_base, pair_region, v_off1, v1);
                     tile_base += dst_step;
                 }
             }
@@ -113,15 +117,16 @@ static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
         for (uint32_t r = start_row; r < end_row; r += 2) {
             const uint32_t   ct             = r / HMX_FP16_TILE_N_ROWS;
             const uint32_t   local_r        = r % HMX_FP16_TILE_N_ROWS;
+            const bool       row0_valid     = r < n_cols;
             const bool       next_row_valid = (r + 1) < end_row && (r + 1) < n_cols;
             const HVX_Vector v_off0         = Q6_Vw_vadd_VwVw(v_scat_base, Q6_V_vsplat_R(local_r * 4));
             const HVX_Vector v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_step);
 
             __fp16 * tile_base = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
-            const uint8_t * p0 = (const uint8_t *) (vtcm_src + r * src_stride);
+            const uint8_t * p0 = row0_valid ? (const uint8_t *) (vtcm_src + r * src_stride) : NULL;
             const uint8_t * p1 = next_row_valid ? (const uint8_t *) (vtcm_src + (r + 1) * src_stride) : NULL;
 
-            if (p1) {
+            if (p0 && p1) {
                 for (uint32_t i = 0; i < n_c_iters; ++i) {
                     HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
                     HVX_Vector v1 = hvx_vmemu(p1); p1 += c_byte_step;
@@ -132,9 +137,12 @@ static inline void hmx_interleave_rows_to_tiles(__fp16 * restrict vtcm_dst,
             } else {
                 const HVX_Vector vzero = Q6_V_vzero();
                 for (uint32_t i = 0; i < n_c_iters; ++i) {
-                    HVX_Vector v0 = hvx_vmemu(p0); p0 += c_byte_step;
+                    HVX_Vector v0 = p0 ? hvx_vmemu(p0) : vzero;
+                    if (p0) p0 += c_byte_step;
+                    HVX_Vector v1 = p1 ? hvx_vmemu(p1) : vzero;
+                    if (p1) p1 += c_byte_step;
                     Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off0, v0);
-                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, vzero);
+                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_base, single_region, v_off1, v1);
                     tile_base += dst_step;
                 }
             }
diff --git src/ggml-hexagon/htp/htp-ctx.h src/ggml-hexagon/htp/htp-ctx.h
index f9f682be..87a1f4c7 100644
--- src/ggml-hexagon/htp/htp-ctx.h
+++ src/ggml-hexagon/htp/htp-ctx.h
@@ -173,6 +173,8 @@ int op_solve_tri(struct htp_ops_context * octx);
 int op_gated_delta_net(struct htp_ops_context * octx);
 int op_pad(struct htp_ops_context * octx);
 int op_im2col(struct htp_ops_context * octx);
+int op_pool_2d(struct htp_ops_context * octx);
+int op_pool_1d(struct htp_ops_context * octx);
 int op_allreduce(struct htp_ops_context * octx);
 int op_roll(struct htp_ops_context * octx);
 
diff --git src/ggml-hexagon/htp/htp-ops.h src/ggml-hexagon/htp/htp-ops.h
index a6a3bb85..5d2571fe 100644
--- src/ggml-hexagon/htp/htp-ops.h
+++ src/ggml-hexagon/htp/htp-ops.h
@@ -115,6 +115,9 @@ enum htp_op_code {
     HTP_OP_ARGMAX,
     HTP_OP_UNARY_GELU_ERF,
     HTP_OP_GLU_GEGLU_ERF,
+    HTP_OP_POOL_2D,
+    HTP_OP_POOL_1D,
+    HTP_OP_UNARY_GELU_QUICK,
 
     HTP_OP_INVALID
 };
diff --git src/ggml-hexagon/htp/hvx-copy.h src/ggml-hexagon/htp/hvx-copy.h
index a3e33c3b..08eb7cd6 100644
--- src/ggml-hexagon/htp/hvx-copy.h
+++ src/ggml-hexagon/htp/hvx-copy.h
@@ -44,7 +44,7 @@ static inline void hvx_splat_f32_u(void * restrict dst, float v, uint32_t n) {
 }
 
 static inline void hvx_splat_f16_a(void * restrict dst, _Float16 v, uint32_t n) {
-    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
+    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
 }
 
 static inline void hvx_splat_f16_u(void * restrict dst, _Float16 v, uint32_t n) {
@@ -106,42 +106,42 @@ static inline void hvx_copy_uu(uint8_t * restrict dst, const uint8_t * restrict
     hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
 }
 
-// copy n fp16 elements : source and destination are aligned to HVX Vector (128)
+// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
 static inline void hvx_copy_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_aa(dst, src, n, sizeof(__fp16));
 }
 
-// copy n fp16 elements : source is aligned, destination is potentially unaligned
+// copy n fp16 elements : destination is aligned, source is unaligned
 static inline void hvx_copy_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_au(dst, src, n, sizeof(__fp16));
 }
 
-// copy n fp16 elements : source is aligned, destination is potentially unaligned
+// copy n fp16 elements : destination is unaligned, source is aligned
 static inline void hvx_copy_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_ua(dst, src, n, sizeof(__fp16));
 }
 
-// copy n fp16 elements : source is aligned, destination is potentially unaligned
+// copy n fp16 elements : destination and source are unaligned
 static inline void hvx_copy_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_uu(dst, src, n, sizeof(__fp16));
 }
 
-// copy n fp32 elements : source and destination are aligned to HVX Vector (128)
+// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
 static inline void hvx_copy_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_aa(dst, src, n, sizeof(float));
 }
 
-// copy n fp32 elements : source is aligned, destination is unaligned
-static inline void hvx_copy_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
-    hvx_copy_ua(dst, src, n, sizeof(float));
-}
-
-// copy n fp32 elements : source is unaligned, destination is aligned
+// copy n fp32 elements : destination is aligned, source is unaligned
 static inline void hvx_copy_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_au(dst, src, n, sizeof(float));
 }
 
-// copy n fp32 elements : source is unaligned, destination unaligned
+// copy n fp32 elements : destination is unaligned, source is aligned
+static inline void hvx_copy_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    hvx_copy_ua(dst, src, n, sizeof(float));
+}
+
+// copy n fp32 elements : destination and source are unaligned
 static inline void hvx_copy_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_uu(dst, src, n, sizeof(float));
 }
@@ -170,26 +170,26 @@ static inline void hvx_copy_f32_uu(uint8_t * restrict dst, const uint8_t * restr
         }                                                                           \
     } while(0)
 
-// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is aligned
+// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
 static inline void hvx_copy_f16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     assert((unsigned long) dst % 128 == 0);
     assert((unsigned long) src % 128 == 0);
     hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
 }
 
-// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is aligned
+// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
 static inline void hvx_copy_f16_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     assert((unsigned long) dst % 128 == 0);
     hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
 }
 
-// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is unaligned
+// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
 static inline void hvx_copy_f16_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     assert((unsigned long) src % 128 == 0);
     hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
 }
 
-// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is unaligned
+// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
 static inline void hvx_copy_f16_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
 }
@@ -235,28 +235,98 @@ static inline void hvx_copy_f16_f32_uu(uint8_t * restrict dst, const uint8_t * r
         }                                                                           \
     } while(0)
 
-// copy/convert n fp16 elements into n fp32 elements : source is aligned, destination is aligned
+// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
 static inline void hvx_copy_f32_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     assert((unsigned long) dst % 128 == 0);
     assert((unsigned long) src % 128 == 0);
     hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
 }
 
-// copy/convert n fp16 elements into n fp32 elements : source is unaligned, destination is aligned
+// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
 static inline void hvx_copy_f32_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     assert((unsigned long) dst % 128 == 0);
     hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
 }
 
-// copy/convert n fp16 elements into n fp32 elements : source is aligned, destination is unaligned
+// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
 static inline void hvx_copy_f32_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     assert((unsigned long) src % 128 == 0);
     hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
 }
 
-// copy/convert n fp16 elements into n fp32 elements : source is unaligned, destination is unaligned
+// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
 static inline void hvx_copy_f32_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
     hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
 }
 
+//// fp32 -> int32
+
+#define hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
+    do {                                                          \
+        dst_type * restrict vdst = (dst_type *) dst;              \
+        src_type * restrict vsrc = (src_type *) src;              \
+                                                                  \
+        const uint32_t elem_size = sizeof(int32_t);               \
+        const uint32_t epv  = 128 / elem_size;                    \
+        const uint32_t nvec = n / epv;                            \
+        const uint32_t nloe = n % epv;                            \
+                                                                  \
+        uint32_t i = 0;                                           \
+        _Pragma("unroll(4)")                                      \
+        for (; i < nvec; i++) {                                   \
+            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
+        }                                                         \
+        if (nloe) {                                               \
+            HVX_Vector v = Q6_Vw_equals_Vsf(vsrc[i]);             \
+            vec_store((void *) &vdst[i], nloe * elem_size, v);    \
+        }                                                         \
+    } while(0)
+
+// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
+static inline void hvx_copy_i32_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    assert((unsigned long) dst % 128 == 0);
+    assert((unsigned long) src % 128 == 0);
+    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
+}
+
+// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
+static inline void hvx_copy_i32_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
+}
+
+//// int32 -> fp32
+
+#define hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
+    do {                                                          \
+        dst_type * restrict vdst = (dst_type *) dst;              \
+        src_type * restrict vsrc = (src_type *) src;              \
+                                                                  \
+        const uint32_t elem_size = sizeof(float);                 \
+        const uint32_t epv  = 128 / elem_size;                    \
+        const uint32_t nvec = n / epv;                            \
+        const uint32_t nloe = n % epv;                            \
+                                                                  \
+        uint32_t i = 0;                                           \
+        _Pragma("unroll(4)")                                      \
+        for (; i < nvec; i++) {                                   \
+            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
+        }                                                         \
+        if (nloe) {                                               \
+            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
+            vec_store((void *) &vdst[i], nloe * elem_size, v);    \
+        }                                                         \
+    } while(0)
+
+// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
+static inline void hvx_copy_f32_i32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    assert((unsigned long) dst % 128 == 0);
+    assert((unsigned long) src % 128 == 0);
+    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
+}
+
+// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
+static inline void hvx_copy_f32_i32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
+}
+
 #endif // HVX_COPY_H
diff --git src/ggml-hexagon/htp/hvx-erf.h src/ggml-hexagon/htp/hvx-erf.h
index 6eefc300..b17c657a 100644
--- src/ggml-hexagon/htp/hvx-erf.h
+++ src/ggml-hexagon/htp/hvx-erf.h
@@ -12,7 +12,7 @@ static __attribute__((noinline)) HVX_Vector hvx_vec_erf_f32(HVX_Vector x) {
     HVX_Vector t = hvx_vec_inverse_f32(hvx_vec_add_f32_f32(
         hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));
 
-    HVX_Vector poly = hvx_vec_mul_f32_f32(hvx_vec_splat_f32(1.061405429f), t);
+    HVX_Vector poly = hvx_vec_splat_f32(1.061405429f);
     poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
     poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
     poly = hvx_vec_add_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
@@ -27,6 +27,7 @@ static __attribute__((noinline)) HVX_Vector hvx_vec_erf_f32(HVX_Vector x) {
     return result;
 }
 
+// GELU_ERF uses the erf definition.
 static inline HVX_Vector hvx_vec_gelu_erf_f32(HVX_Vector x) {
     const HVX_Vector scale = hvx_vec_splat_f32(0.7071067811865475f);
     const HVX_Vector half  = hvx_vec_splat_f32(0.5f);
diff --git src/ggml-hexagon/htp/main.c src/ggml-hexagon/htp/main.c
index fcb0a720..2dd1b3fc 100644
--- src/ggml-hexagon/htp/main.c
+++ src/ggml-hexagon/htp/main.c
@@ -851,6 +851,7 @@ static int execute_op(struct htp_ops_context * octx) {
         case HTP_OP_UNARY_SIGMOID:
         case HTP_OP_UNARY_SILU:
         case HTP_OP_UNARY_GELU:
+        case HTP_OP_UNARY_GELU_QUICK:
         case HTP_OP_UNARY_GELU_ERF:
         case HTP_OP_UNARY_NEG:
         case HTP_OP_UNARY_EXP:
@@ -931,6 +932,12 @@ static int execute_op(struct htp_ops_context * octx) {
         case HTP_OP_ROLL:
             return op_roll(octx);
 
+        case HTP_OP_POOL_2D:
+            return op_pool_2d(octx);
+
+        case HTP_OP_POOL_1D:
+            return op_pool_1d(octx);
+
         case HTP_OP_CONCAT:
             return op_concat(octx);
 
diff --git src/ggml-hexagon/htp/matmul-ops.c src/ggml-hexagon/htp/matmul-ops.c
index 9dfd3564..1a513d16 100644
--- src/ggml-hexagon/htp/matmul-ops.c
+++ src/ggml-hexagon/htp/matmul-ops.c
@@ -27,8 +27,8 @@
 
 typedef struct {
     float        *dst;
-    dma_addr_t    src2_addr;
-    size_t        src2_bytes;
+    dma_addr_t    bias_addr;
+    size_t        bias_bytes;
     dma_addr_t    act_dma_addr;
     dma_addr_t    weight;
     dma_queue *   weight_dma;
@@ -36,9 +36,10 @@ typedef struct {
     int           k;
     int           n;
     int           act_stride;
+    uint32_t      act_elem_size;  // 4=F32 src1, 2=F16 src1
     int           weight_stride;
     int           dst_stride;
-    uint32_t      src2_stride;
+    uint32_t      bias_stride;
     int           ne02;
     int           ne03;
     int           ne12;
@@ -47,8 +48,8 @@ typedef struct {
     size_t        src0_nb3;
     size_t        act_nb2;
     size_t        act_nb3;
-    size_t        src2_nb2;
-    size_t        src2_nb3;
+    size_t        bias_nb2;
+    size_t        bias_nb3;
     size_t        dst_nb2;
     size_t        dst_nb3;
     int           r2;
@@ -118,24 +119,19 @@ struct htp_mm_context {
 
     // Dynamic VTCM pointers allocated sequentially
     uint8_t * vtcm_src0;
-    uint8_t * vtcm_src1;
-    uint8_t * vtcm_src2;
-    uint8_t * vtcm_src3;
+    uint8_t * vtcm_act;
+    uint8_t * vtcm_bias;
     uint8_t * vtcm_dst;
     uint8_t * vtcm_act_raw;
 
     // Cached strides
     uint32_t vtcm_src0_stride;
-    uint32_t vtcm_src1_stride;
-    uint32_t vtcm_src2_stride;
-    uint32_t vtcm_src3_stride;
+    uint32_t vtcm_act_stride;
     uint32_t vtcm_act_raw_stride;
 
     // Cached thread offsets/sizes
     uint32_t vtcm_src0_size_per_thread;
-    uint32_t vtcm_src1_size_per_thread;
-    uint32_t vtcm_src2_size_per_thread;
-    uint32_t vtcm_src3_size_per_thread;
+    uint32_t vtcm_act_size_per_thread;
     uint32_t vtcm_dst_size_per_thread;
 };
 
@@ -190,6 +186,7 @@ static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
     const struct htp_tensor * restrict src1 = octx->src[1];         \
     const struct htp_tensor * restrict src2 = octx->src[2];         \
     const struct htp_tensor * restrict  dst = octx->dst;            \
+    const struct htp_tensor * restrict  act = src1;                 \
                                                                     \
     const uint32_t ne00 = src0->ne[0];                              \
     const uint32_t ne01 = src0->ne[1];                              \
@@ -247,7 +244,7 @@ static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     htp_matmul_preamble;                                                                                                                   \
                                                                                                                                            \
     const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;                                                               \
-    const uint32_t src1_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);                                              \
+    const uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);                                               \
     const uint32_t cur_m_start = mmctx->cur_m_start;                                                                                       \
                                                                                                                                            \
     const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;                                                  \
@@ -260,13 +257,13 @@ static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                                  \
                                                                                                                                            \
     const size_t dst_row_size  = nb1;                                                                                                      \
-    const size_t src1_row_size = nb11;                                                                                                     \
-    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                                    \
+    const size_t act_row_size = nb11;                                                                                                      \
+    const size_t act_stride = mmctx->vtcm_act_stride;                                                                                      \
     const size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;                                                          \
                                                                                                                                            \
     uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                          \
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                          \
-    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                       \
+    uint8_t * restrict act_data = mmctx->vtcm_act;                                                                                         \
                                                                                                                                            \
     const dma_addr_t src0_row = src0->data;                                                                                                \
                                                                                                                                            \
@@ -301,9 +298,9 @@ static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
                                                                                                                                            \
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                             \
         uint32_t ir1 = 0;                                                                                                                  \
-        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                           \
-            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                                    \
-            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                                    \
+        for (; ir1 + 1 < act_nrows; ir1 += 2) {                                                                                            \
+            const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);                                       \
+            const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);                                       \
             float * restrict dst_row0 = (float *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));                                    \
             float * restrict dst_row1 = (float *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));                                    \
                                                                                                                                            \
@@ -318,11 +315,11 @@ static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
                 src2_ptr0 = &src2_row0[ct * 32];                                                                                           \
                 src2_ptr1 = &src2_row1[ct * 32];                                                                                           \
             }                                                                                                                              \
-            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, src2_ptr0, src2_ptr1);                             \
+            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);                               \
         }                                                                                                                                  \
                                                                                                                                            \
-        for (; ir1 < src1_nrows; ++ir1) {                                                                                                  \
-            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                         \
+        for (; ir1 < act_nrows; ++ir1) {                                                                                                   \
+            const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);                                            \
             float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));                              \
             float * dst_ptr = &dst_row[ct * 32];                                                                                           \
                                                                                                                                            \
@@ -331,7 +328,7 @@ static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
                 const float * restrict src2_row = (const float *) ((const uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));    \
                 src2_ptr = &src2_row[ct * 32];                                                                                             \
             }                                                                                                                              \
-            DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, src2_ptr);                                                                \
+            DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);                                                                 \
         }                                                                                                                                  \
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                              \
                                                                                                                                            \
@@ -359,18 +356,18 @@ static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                \
                                                                                                                          \
     const size_t dst_row_size  = nb1;                                                                                    \
-    const size_t src1_row_size = nb11;                                                                                   \
-    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                  \
+    const size_t act_row_size = nb11;                                                                                    \
+    const size_t act_stride = mmctx->vtcm_act_stride;                                                                    \
                                                                                                                          \
     uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                   \
     uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
-    uint8_t * src1_data = mmctx->vtcm_src1;                                                                              \
+    uint8_t * act_data = mmctx->vtcm_act;                                                                                \
                                                                                                                          \
     float * tmp = (float *) vtcm_dst_ptr;                                                                                \
                                                                                                                          \
     const dma_addr_t src0_row = src0->data;                                                                              \
                                                                                                                          \
-    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                     \
+    const uint8_t * restrict act_col = (const uint8_t *) act_data;                                                       \
     float * restrict dst_col          = (float *) dst->data;                                                             \
                                                                                                                          \
     const uint32_t tile_size = TILE_SIZE;                                                                                \
@@ -387,11 +384,11 @@ static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     uint32_t push_ct = ct_start;                                                                                         \
     if (src0_start_row < src0_end_row) {                                                                                 \
         if (src2) {                                                                                                      \
-            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;                                         \
+            float * vtcm_bias_ptr = (float *) mmctx->vtcm_bias + src0_start_row;                                         \
             const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);                                    \
             int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                          \
             if (slice_size > 0) {                                                                                        \
-                dma_queue_push(dma_q, dma_make_data(vtcm_src2_ptr, src2_addr),                                           \
+                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),                                           \
                                slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);   \
                 dma_queue_pop_nowait(dma_q);                                                                             \
             }                                                                                                            \
@@ -414,7 +411,7 @@ static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
         valid_rows = MIN(32, MAX(0, valid_rows));                                                                        \
                                                                                                                          \
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                           \
-        DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                      \
+        DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                       \
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                            \
                                                                                                                          \
         if (push_ct < ct_end) {                                                                                          \
@@ -428,9 +425,9 @@ static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     if (copy_cnt > 0) {                                                                                                  \
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                       \
         if (src2) {                                                                                                      \
-            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],                                                        \
+            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],                                                        \
                             (const uint8_t *) tmp,                                                                       \
-                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),                       \
+                            (const uint8_t *) ((const float *) mmctx->vtcm_bias + src0_start_row),                       \
                             copy_cnt);                                                                                   \
         } else {                                                                                                         \
             hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                            \
@@ -448,11 +445,11 @@ static void hvx_mm_nx_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, v
                                                                                                                                   \
     const struct htp_tensor * restrict act = octx->src[n_weights]; /* x */                                                        \
     const uint32_t ne10 = act->ne[0];                                                                                             \
-    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];                                                             \
-    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
+    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];                                                              \
+    const size_t act_stride = mmctx->vtcm_act_stride;                                                                             \
                                                                                                                                   \
     uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                               \
-    uint8_t * restrict src1_data       = mmctx->vtcm_src1;                                                                        \
+    uint8_t * restrict act_data        = mmctx->vtcm_act;                                                                         \
                                                                                                                                   \
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
     const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
@@ -511,23 +508,23 @@ static void hvx_mm_nx_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, v
                                                                                                                                   \
             htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                \
             uint32_t ir1 = 0;                                                                                                     \
-            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
-                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
-                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
+            for (; ir1 + 1 < act_nrows; ir1 += 2) {                                                                               \
+                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);                          \
+                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);                          \
                                                                                                                                   \
                 float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                     \
                 float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                     \
                 float * dst_ptr0 = &dst_row0[ct * 32];                                                                            \
                 float * dst_ptr1 = &dst_row1[ct * 32];                                                                            \
                                                                                                                                   \
-                DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, NULL, NULL);                          \
+                DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL);                            \
             }                                                                                                                     \
                                                                                                                                   \
-            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
-                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
+            for (; ir1 < act_nrows; ++ir1) {                                                                                      \
+                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);                               \
                 float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));                                          \
                 float * dst_ptr = &dst_row[ct * 32];                                                                              \
-                DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                       \
+                DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                        \
             }                                                                                                                     \
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                 \
                                                                                                                                   \
@@ -550,10 +547,10 @@ MATMUL_2D_REPACKED_IMPL(q2_k,       512,  tiled_vec_dot_q2_k_32x2,  tiled_vec_do
 MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
 MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)
 
-static void hvx_mm_transfer_src1_dma(
+static void hvx_mm_transfer_act_dma(
     struct htp_ops_context * octx,
     const struct htp_mm_kernel_params * kparams,
-    const struct htp_tensor * src1,
+    const struct htp_tensor * act,
     uint8_t * dst_base,
     size_t dst_row_size,
     uint32_t m_start,
@@ -564,22 +561,22 @@ static void hvx_mm_transfer_src1_dma(
     }
 
     dma_queue * dma_q = octx->ctx->dma[0];
-    const uint32_t ne0 = src1->ne[0];
-    const size_t elem_size = (src1->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
+    const uint32_t ne0 = act->ne[0];
+    const size_t elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
     const size_t row_bytes = ne0 * elem_size;
-    const size_t src1_nb1 = src1->nb[1];
-    const dma_addr_t src_base = src1->data;
+    const size_t act_nb1 = act->nb[1];
+    const dma_addr_t act_base = act->data;
 
-    const bool is_contiguous = (src1->nb[2] == src1->ne[1] * src1_nb1) &&
-                               (src1->nb[3] == src1->ne[2] * src1->nb[2]);
+    const bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
+                               (act->nb[3] == act->ne[2] * act->nb[2]);
 
     if (is_contiguous) {
-        const dma_addr_t src_addr = src_base + m_start * src1_nb1;
+        const dma_addr_t src_addr = act_base + m_start * act_nb1;
         dma_queue_push(dma_q, dma_make_data(dst_base, src_addr),
-                       dst_row_size, src1_nb1, row_bytes, m_rows);
+                       dst_row_size, act_nb1, row_bytes, m_rows);
         dma_queue_pop(dma_q);
     } else {
-        const uint32_t ne12_ne1 = src1->ne[2] * src1->ne[1];
+        const uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
         const bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
         for (uint32_t ir = 0; ir < m_rows; ++ir) {
             const uint32_t ir1 = m_start + ir;
@@ -588,19 +585,19 @@ static void hvx_mm_transfer_src1_dma(
                 i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                 const uint32_t rem = ir1 - i13 * ne12_ne1;
                 i12 = fastdiv(rem, &kparams->div_ne1);
-                i11 = rem - i12 * src1->ne[1];
+                i11 = rem - i12 * act->ne[1];
             } else {
                 i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                 const uint32_t rem = ir1 - i13 * ne12_ne1;
-                i12 = src1->ne[1] ? rem / src1->ne[1] : 0;
-                i11 = rem - i12 * src1->ne[1];
+                i12 = act->ne[1] ? rem / act->ne[1] : 0;
+                i11 = rem - i12 * act->ne[1];
             }
-            const dma_addr_t row_src = src_base + (i11 * src1->nb[1] +
-                                                   i12 * src1->nb[2] +
-                                                   i13 * src1->nb[3]);
+            const dma_addr_t row_src = act_base + (i11 * act->nb[1] +
+                                                   i12 * act->nb[2] +
+                                                   i13 * act->nb[3]);
             uint8_t * row_dst = dst_base + ir * dst_row_size;
             dma_queue_push(dma_q, dma_make_data(row_dst, row_src),
-                           dst_row_size, src1_nb1, row_bytes, 1);
+                           dst_row_size, act_nb1, row_bytes, 1);
             dma_queue_pop(dma_q);
         }
     }
@@ -640,7 +637,7 @@ static void name(unsigned int nth, unsigned int ith, void * data) {
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
     htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                            \
-    uint8_t * restrict dst = mmctx->vtcm_src1;                                                             \
+    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
     const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
     const size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
     const size_t dst_row_size = (dst_row_size_expr);                                                       \
@@ -655,9 +652,9 @@ static void name(unsigned int nth, unsigned int ith, void * data) {
 QUANTIZE_IMPL(quantize_f32_q8_0_tiled, "quantize-f32-q8_0_tiled", quantize_f32_q8_0_tiled_kernel, htp_mm_q8_0_tiled_row_size(ne0))
 QUANTIZE_IMPL(quantize_f32_q8_1_tiled, "quantize-f32-q8_1_tiled", quantize_f32_q8_1_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
 QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
-QUANTIZE_IMPL(quantize_f32_f32,        "quantize-f32-f32",        quantize_f32_f32_kernel,        mmctx->vtcm_src1_stride)
-QUANTIZE_IMPL(quantize_f32_f16,        "quantize-f32-f16",        quantize_f32_f16_kernel,        mmctx->vtcm_src1_stride)
-QUANTIZE_IMPL(quantize_f16_f16,        "quantize-f16-f16",        quantize_f16_f16_kernel,        mmctx->vtcm_src1_stride)
+QUANTIZE_IMPL(quantize_f32_f32,        "quantize-f32-f32",        quantize_f32_f32_kernel,        mmctx->vtcm_act_stride)
+QUANTIZE_IMPL(quantize_f32_f16,        "quantize-f32-f16",        quantize_f32_f16_kernel,        mmctx->vtcm_act_stride)
+QUANTIZE_IMPL(quantize_f16_f16,        "quantize-f16-f16",        quantize_f16_f16_kernel,        mmctx->vtcm_act_stride)
 
 static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
     (void) nth;
@@ -673,7 +670,7 @@ static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, vo
 
     quantize_f32_q8_0_tiled_block_kernel(
         (const float *) mmctx->vtcm_act_raw,
-        mmctx->vtcm_src1,
+        mmctx->vtcm_act,
         NULL,
         src->ne[0],
         mmctx->quant_ib_first[ith],
@@ -701,7 +698,7 @@ static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, vo
 
     quantize_f32_q8_1_tiled_block_kernel(
         (const float *) mmctx->vtcm_act_raw,
-        mmctx->vtcm_src1,
+        mmctx->vtcm_act,
         NULL,
         src->ne[0],
         mmctx->quant_ib_first[ith],
@@ -729,7 +726,7 @@ static void quantize_f32_q8_1_s16_tiled_block(unsigned int nth, unsigned int ith
 
     quantize_f32_q8_1_s16_tiled_block_kernel(
         (const float *) mmctx->vtcm_act_raw,
-        mmctx->vtcm_src1,
+        mmctx->vtcm_act,
         NULL,
         src->ne[0],
         mmctx->quant_ib_first[ith],
@@ -795,10 +792,10 @@ static void hvx_mm_4d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
     assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                                                   \
                                                                                                                                                             \
     const size_t dst_row_size = nb1;                                                                                                                        \
-    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                                                     \
+    const size_t act_stride = mmctx->vtcm_act_stride;                                                                                                       \
                                                                                                                                                             \
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                                           \
-    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                                                        \
+    uint8_t * restrict act_data = mmctx->vtcm_act;                                                                                                          \
                                                                                                                                                             \
     const uint32_t tile_size = TILE_SIZE;                                                                                                                   \
     const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                                                        \
@@ -874,19 +871,19 @@ static void hvx_mm_4d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void
                                                                                                                                                             \
                 uint32_t ir1 = 0;                                                                                                                           \
                 for (; ir1 + 1 < batch_nrows; ir1 += 2) {                                                                                                   \
-                    const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 0) * src1_stride);                          \
-                    const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 1) * src1_stride);                          \
+                    const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);                             \
+                    const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);                             \
                     float * restrict dst_row0 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);                                       \
                     float * restrict dst_row1 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);                                       \
                     float * dst_ptr0 = &dst_row0[ct * 32];                                                                                                  \
                     float * dst_ptr1 = &dst_row1[ct * 32];                                                                                                  \
-                    DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, NULL, NULL);                                                \
+                    DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, NULL, NULL);                                                  \
                 }                                                                                                                                           \
                 for (; ir1 < batch_nrows; ++ir1) {                                                                                                          \
-                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (chunk_m_offset + ir1) * src1_stride);                               \
+                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);                                  \
                     float * restrict dst_row = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);                                            \
                     float * dst_ptr = &dst_row[ct * 32];                                                                                                    \
-                    DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                                             \
+                    DOT_2X1(ne10, dst_ptr, w_tile, act_col, valid_rows, NULL);                                                                              \
                 }                                                                                                                                           \
             }                                                                                                                                               \
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                                           \
@@ -923,7 +920,7 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t prefetch_mask = n_prefetch - 1;
 
     const uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
-    const uint32_t src1_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                          // src1 rows
+    const uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
     const uint32_t cur_m_start = mmctx->cur_m_start;
 
     const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
@@ -934,15 +931,15 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
 
     const size_t dst_row_size  = nb1;
     const size_t src0_row_size = nb01;
-    const size_t src1_row_size = nb11;
+    const size_t act_row_size = nb11;
 
     const size_t src0_stride = mmctx->vtcm_src0_stride;
-    const size_t src1_stride = mmctx->vtcm_src1_stride;
+    const size_t act_stride = mmctx->vtcm_act_stride;
 
     // Per-thread VTCMs for all tensors
     uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data     = mmctx->vtcm_src1;
+    uint8_t * restrict act_data      = mmctx->vtcm_act;
 
     const dma_addr_t src0_row = src0->data;
 
@@ -968,21 +965,21 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
         const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
 
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
-        // Process src1 columns in pairs (2x2 tiling)
+        // Process act columns in pairs (2x2 tiling)
         uint32_t ir1 = 0;
-        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
-            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
-            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
+        for (; ir1 + 1 < act_nrows; ir1 += 2) {
+            const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);
+            const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);
             float * restrict dst_row0 = (float *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
             float * restrict dst_row1 = (float *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
-            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
+            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
         }
 
-        // Handle remaining src1 rows (fallback to 2x1)
-        for (; ir1 < src1_nrows; ++ir1) {
-            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
+        // Handle remaining act rows (fallback to 2x1)
+        for (; ir1 < act_nrows; ++ir1) {
+            const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
             float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
-            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
+            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
         }
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
 
@@ -1005,15 +1002,15 @@ static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
 
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
         #pragma unroll(2)
-        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
-            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
+        for (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
+            const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
             float * restrict dst_row          = (float *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
-            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
+            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
         }
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
     }
     if (src2) {
-        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + src1_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
+        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
     }
 }
 
@@ -1029,21 +1026,21 @@ static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
 
     const size_t dst_row_size  = nb1;
     const size_t src0_row_size = nb01;
-    const size_t src1_row_size = nb11;
+    const size_t act_row_size  = nb11;
 
     const size_t src0_stride = mmctx->vtcm_src0_stride;
-    const size_t src1_stride = mmctx->vtcm_src1_stride;
+    const size_t act_stride  = mmctx->vtcm_act_stride;
 
     // Per-thread VTCMs for all tensors
     uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
     uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * src1_data     = mmctx->vtcm_src1;
+    uint8_t * act_data      = mmctx->vtcm_act;
 
     float * tmp = (float *) vtcm_dst_ptr;
 
     const dma_addr_t src0_row = src0->data;
-    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
-    float * restrict dst_col          = (float *) dst->data;
+    const uint8_t * restrict act_col = (const uint8_t *) act_data;
+    float * restrict dst_col         = (float *) dst->data;
 
     const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);
 
@@ -1055,11 +1052,11 @@ static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
     // Prefill vtcm with 2x src0 rows
     if (src0_start_row < src0_end_row) {
         if (src2) {
-            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;
+            float * vtcm_bias_ptr = (float *) mmctx->vtcm_bias + src0_start_row;
             const dma_addr_t src2_addr = src2->data + src0_start_row * sizeof(float);
             int slice_size = (int)src0_end_row - (int)src0_start_row;
             if (slice_size > 0) {
-                dma_queue_push(dma_q, dma_make_data(vtcm_src2_ptr, src2_addr),
+                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                                slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);
                 dma_queue_pop_nowait(dma_q);
             }
@@ -1083,7 +1080,7 @@ static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
     for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
         const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
-        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
+        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
 
         // Prefetch next (n + vtcm_nrows) row
@@ -1103,7 +1100,7 @@ static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
                        src0_stride, src0_row_size, src0_row_size, 1);
         const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
-        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
+        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
     }
 
@@ -1111,9 +1108,9 @@ static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
     if (copy_cnt > 0) {
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
         if (src2) {
-            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],
+            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                             (const uint8_t *) tmp,
-                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),
+                            (const uint8_t *) ((const float *) mmctx->vtcm_bias + src0_start_row),
                             copy_cnt);
         } else {
             hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
@@ -1143,10 +1140,10 @@ static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
     const size_t dst_row_size  = nb1;
     const size_t src0_row_size = nb01;
     const size_t src0_stride = mmctx->vtcm_src0_stride;
-    const size_t src1_stride = mmctx->vtcm_src1_stride;
+    const size_t act_stride  = mmctx->vtcm_act_stride;
 
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data     = mmctx->vtcm_src1;
+    uint8_t * restrict act_data      = mmctx->vtcm_act;
 
 
     if (src0_start_row >= src0_end_row || cur_m_rows == 0) {
@@ -1208,16 +1205,16 @@ static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
 
                 uint32_t ir1 = 0;
                 for (; ir1 + 1 < batch_nrows; ir1 += 2) {
-                    const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 0) * src1_stride);
-                    const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (chunk_m_offset + ir1 + 1) * src1_stride);
+                    const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
+                    const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                     float * restrict dst_row0 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                     float * restrict dst_row1 = (float *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
-                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
+                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                 }
                 for (; ir1 < batch_nrows; ++ir1) {
-                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (chunk_m_offset + ir1) * src1_stride);
+                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                     float * restrict dst_row          = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
-                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
+                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                 }
             }
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
@@ -1253,9 +1250,9 @@ static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
                 const uint32_t batch_nrows    = m_last - m_first;
 
                 for (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
-                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (chunk_m_offset + ir1) * src1_stride);
+                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                     float * restrict dst_row          = (float *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
-                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
+                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                 }
             }
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
@@ -1277,7 +1274,7 @@ static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
     const struct htp_tensor * restrict ids = octx->src[2];
 
     const uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
-    const uint32_t src1_nrows      = ne11;
+    const uint32_t act_nrows       = ne11;
     const uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
     const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
 
@@ -1299,13 +1296,13 @@ static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
     const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
 
     const size_t dst_row_size  = nb1;
-    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);
+    const size_t act_row_size  = htp_mm_q8_0_tiled_row_size(ne10);
 
-    const size_t src1_stride = mmctx->vtcm_src1_stride;
+    const size_t act_stride = mmctx->vtcm_act_stride;
 
     // Per-thread VTCMs for all tensors
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data = mmctx->vtcm_src1;
+    uint8_t * restrict act_data = mmctx->vtcm_act;
 
     for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
         const int32_t cne1 = matrix_row_counts[cur_a];
@@ -1343,11 +1340,11 @@ static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
                 const int               rm1         = row_mapping.i1;  // expert idx
                 const int               rm2         = row_mapping.i2;  // token idx
 
-                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // src1 row idx
-                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_stride);
+                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
+                const uint8_t * restrict act_col = (const uint8_t *) (act_data + (ir1 + rm2 * ne11 + 0) * act_stride);
                 float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));
 
-                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
+                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
             }
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);
 
@@ -1383,14 +1380,14 @@ static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
     assert(ne13 % ne03 == 0);
 
     const size_t dst_row_size  = nb1;
-    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);
+    const size_t act_row_size = htp_mm_q8_0_tiled_row_size(ne10);
 
     const uint32_t n_aids = src2->ne[0];  // num activated experts
     const uint32_t n_ids  = ne02;         // num experts
 
     // Per-thread VTCMs for all tensors
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data = mmctx->vtcm_src1;
+    uint8_t * restrict act_data = mmctx->vtcm_act;
 
     for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
         const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
@@ -1400,7 +1397,7 @@ static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
         assert(eid < (int32_t) n_ids);
 
         const dma_addr_t src0_row = src0->data + eid * nb02;
-        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
+        const uint8_t * restrict act_col = (const uint8_t *) act_data;
         float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);
 
         const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
@@ -1426,7 +1423,7 @@ static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
             valid_rows = MIN(32, MAX(0, valid_rows));
 
             htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
-            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
+            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);
 
             if (push_ct < ct_end) {
@@ -1457,7 +1454,7 @@ static void hvx_mv_id_nx(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t n_ids  = src0->ne[2];
 
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data = mmctx->vtcm_src1;
+    uint8_t * restrict act_data = mmctx->vtcm_act;
 
     for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
         const int32_t eid = *(const int32_t *) ((const uint8_t *) ids->data + ie1 * ids->nb[0]);
@@ -1489,7 +1486,7 @@ static void hvx_mv_id_nx(unsigned int nth, unsigned int ith, void * data) {
             if (src0_start_row >= src0_end_row) continue;
 
             const dma_addr_t src0_row = src_w->data + eid * src_w->nb[2];
-            const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
+            const uint8_t * restrict act_col = (const uint8_t *) act_data;
             float * restrict dst_row = (float *) (dst->data + ie1 * dst->nb[1]);
 
             const uint32_t tile_size = htp_mm_get_weight_tile_size(src_w->type);
@@ -1515,7 +1512,7 @@ static void hvx_mv_id_nx(unsigned int nth, unsigned int ith, void * data) {
                 valid_rows = MIN(32, MAX(0, valid_rows));
 
                 htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
-                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
+                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
                 htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);
 
                 if (push_ct < ct_end) {
@@ -1548,10 +1545,10 @@ static void hvx_mm_id_nx(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
     const struct mmid_row_mapping * matrix_rows = mmctx->matrix_rows;
 
-    const size_t src1_stride = mmctx->vtcm_src1_stride;
+    const size_t act_stride = mmctx->vtcm_act_stride;
 
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data = mmctx->vtcm_src1;
+    uint8_t * restrict act_data = mmctx->vtcm_act;
 
     for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
         const int32_t cne1 = matrix_row_counts[cur_a];
@@ -1612,10 +1609,10 @@ static void hvx_mm_id_nx(unsigned int nth, unsigned int ith, void * data) {
                     const int rm2 = row_mapping.i2;
 
                     const uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
-                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * act->ne[1]) * src1_stride);
+                    const uint8_t * restrict act_col = (const uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                     float * restrict dst_row = (float *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));
 
-                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
+                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, NULL);
                 }
                 htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);
 
@@ -1682,13 +1679,13 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
     struct htp_mm_context mmctx_struct = {0};
     struct htp_mm_context * mmctx = &mmctx_struct;
     mmctx->octx = octx;
-    mmctx->act = src1;
+    mmctx->act = act;
 
     const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
 
     const uint32_t src0_nrows = ne01;
-    const uint32_t src1_nrows = ne11 * ne12 * ne13;
-    mmctx->act_nrows = src1_nrows;
+    const uint32_t act_nrows  = ne11 * ne12 * ne13;
+    mmctx->act_nrows = act_nrows;
 
     uint32_t src0_row_start = 0;
     uint32_t src0_row_end   = src0_nrows;
@@ -1724,10 +1721,9 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
     const size_t src0_row_size = nb01;
     const size_t dst_row_size  = nb1;
-    size_t       src1_row_size = nb11;
+    size_t       act_row_size  = nb11;
 
     const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
-    size_t       src1_row_size_padded;
 
     worker_callback_t quant_task_func;
     worker_callback_t matmul_job_func;
@@ -1751,7 +1747,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
         } else {
             matmul_job_func = hvx_mm_4d;
         }
-    } else if (src1_nrows > 1) {
+    } else if (act_nrows > 1) {
         if (is_repacked) {
             switch (src0->type) {
                 case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;
@@ -1793,13 +1789,13 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
     switch (kparams->kernel_type) {
         case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
-            quant_task_func        = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
-            need_quant             = (src1->type == HTP_TYPE_F32);
-            mmctx->type            = (src1->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
+            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
+            need_quant             = (act->type == HTP_TYPE_F32);
+            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
             mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
             mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
             mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
-            src1_row_size          = hex_round_up(ne10 * 2, 128);
+            act_row_size           = kparams->act_row_size;
             break;
 
         case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
@@ -1809,7 +1805,7 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
             mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
             mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
             mmctx->vec_dot_2x2     = vec_dot_f32_f32_aa_2x2;
-            src1_row_size          = hex_round_up(ne10 * 4, 128);
+            act_row_size           = kparams->act_row_size;
             break;
 
         case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
@@ -1821,9 +1817,9 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
             const uint32_t qk = QK_Q8_0_TILED;
             const uint32_t nb = (ne10 + qk - 1) / qk;
-            const uint32_t total_nb = src1_nrows * nb;
+            const uint32_t total_nb = act_nrows * nb;
 
-            if (src1_nrows < octx->n_threads && !is_batched) {
+            if (act_nrows < octx->n_threads && !is_batched) {
                 n_quant_tasks = MIN(total_nb, octx->n_threads);
                 quant_task_func = htp_mm_act_quant_block_func(src0->type);
                 for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
@@ -1835,28 +1831,28 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
                     mmctx->quant_c[ith]        = ib_first % nb;
                 }
             } else {
-                n_quant_tasks = MIN(src1_nrows, octx->n_threads);
+                n_quant_tasks = MIN(act_nrows, octx->n_threads);
                 quant_task_func = htp_mm_act_quant_row_func(src0->type);
             }
-            src1_row_size = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+            act_row_size = kparams->act_row_size;
             break;
     }
 
-    const uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < src1_nrows)
-                           ? (uint32_t) kparams->m_chunk : src1_nrows;
+    const uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
+                           ? (uint32_t) kparams->m_chunk : act_nrows;
     const uint32_t m_layout_rows = m_chunk;
 
     struct htp_mm_hvx_vtcm_layout L;
     htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
-                                 dst_row_size, src0_row_size, src1_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);
+                                 dst_row_size, src0_row_size, act_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);
 
     if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
         kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
         kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
         kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
-        mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
+        mmctx->vtcm_act_size_per_thread = L.act_bytes;
     } else {
-        mmctx->vtcm_src1_size_per_thread = fastdiv(L.src1_bytes, &octx->n_threads_div);
+        mmctx->vtcm_act_size_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
     }
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
@@ -1864,12 +1860,12 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
     const size_t vtcm_size = L.total_bytes;
 
-    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
-         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);
+    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
+         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);
 
     FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
-         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
-         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);
+         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
+         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);
 
     if (octx->ctx->vtcm_size < vtcm_size) {
         FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
@@ -1878,18 +1874,14 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
     }
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
-    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
     mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src2     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
+    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
     mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
     mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
-    octx->src1_spad.src  = NULL;
-    octx->src0_spad.src  = NULL;
-    octx->dst_spad.src   = NULL;
-
     mmctx->vtcm_src0_stride = src0_row_size_padded;
-    mmctx->vtcm_src1_stride = src1_row_size;
+    mmctx->vtcm_act_stride  = act_row_size;
     if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
         mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
     } else if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
@@ -1900,14 +1892,14 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < src1_nrows) {
-        for (uint32_t m_start = 0; m_start < src1_nrows; m_start += m_chunk) {
-            const uint32_t cur_m_rows = MIN(src1_nrows - m_start, m_chunk);
+    if (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
+        for (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
+            const uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
             mmctx->cur_m_start = m_start;
             mmctx->cur_m_rows  = cur_m_rows;
 
             if (need_quant) {
-                hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);
+                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);
 
                 const uint32_t qk = QK_Q8_0_TILED;
                 const uint32_t nb = (ne10 + qk - 1) / qk;
@@ -1933,22 +1925,22 @@ static int hvx_mm_matmul(struct htp_ops_context * octx) {
                 mmctx->n_quant_tasks = quant_tasks;
                 work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
             } else {
-                hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_src1, mmctx->vtcm_src1_stride, m_start, cur_m_rows);
+                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
             }
 
             work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
         }
     } else {
         mmctx->cur_m_start = 0;
-        mmctx->cur_m_rows  = src1_nrows;
+        mmctx->cur_m_rows  = act_nrows;
 
         if (need_quant) {
-            hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, src1_nrows);
-            mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
+            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
             mmctx->n_quant_tasks = n_quant_tasks;
             work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
         } else {
-            hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_src1, mmctx->vtcm_src1_stride, 0, src1_nrows);
+            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
         }
 
         work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
@@ -1964,11 +1956,11 @@ static void hvx_mm_nx_2d(unsigned int nth, unsigned int ith, void * data) {
     const uint32_t n_weights = kparams->n_weights;
 
     const struct htp_tensor * restrict act = octx->src[n_weights];
-    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];
-    const size_t src1_stride = mmctx->vtcm_src1_stride;
+    const uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
+    const size_t act_stride  = mmctx->vtcm_act_stride;
 
     uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
-    uint8_t * restrict src1_data     = mmctx->vtcm_src1;
+    uint8_t * restrict act_data      = mmctx->vtcm_act;
 
     const uint32_t n_prefetch = kparams->n_prefetch;
     assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
@@ -2020,17 +2012,17 @@ static void hvx_mm_nx_2d(unsigned int nth, unsigned int ith, void * data) {
             const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
             htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
             uint32_t ir1 = 0;
-            for (; ir1 + 1 < src1_nrows; ir1 += 2) {
-                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
-                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
+            for (; ir1 + 1 < act_nrows; ir1 += 2) {
+                const uint8_t * restrict act_col0 = (const uint8_t *) (act_data + (ir1+0) * act_stride);
+                const uint8_t * restrict act_col1 = (const uint8_t *) (act_data + (ir1+1) * act_stride);
                 float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
                 float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
-                mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
+                mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
             }
-            for (; ir1 < src1_nrows; ++ir1) {
-                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
+            for (; ir1 < act_nrows; ++ir1) {
+                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
                 float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
-                mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
+                mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
             }
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
 
@@ -2049,10 +2041,10 @@ static void hvx_mm_nx_2d(unsigned int nth, unsigned int ith, void * data) {
                            src0_stride, src0_row_size, src0_row_size, 1);
             const uint8_t * ss0 = (void *) dma_queue_pop(dma_q).dst;
             htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
-            for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
-                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
+            for (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
+                const uint8_t * restrict act_col = (const uint8_t *) (act_data + ir1 * act_stride);
                 float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
-                mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
+                mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
             }
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
         }
@@ -2144,6 +2136,7 @@ typedef struct {
     size_t                          vtcm_f32_act_bytes_per_thread;
     uint32_t                        dma_step_rows;
     uint32_t                        dma_step_rows_shift;
+    uint32_t                        act_elem_size;  // 4=F32 src1, 2=F16 src1
 } activation_transfer_task_state_t;
 
 typedef struct {
@@ -2262,7 +2255,7 @@ static void transfer_activation_chunk_col_chunk_worker_fn(unsigned int n, unsign
     );
 }
 
-static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
+static void transfer_activation_chunk_to_fp16_dma_pipelined(
         dma_queue *dma_q,
         __fp16 *restrict vtcm_dst,
         dma_addr_t act_dma_addr,
@@ -2270,7 +2263,8 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
         uint32_t k_block,
         uint32_t k_stride,
         uint32_t k_valid,
-        float *thread_f32_act,
+        uint8_t *thread_act,
+        uint32_t act_elem_size,
         struct htp_thread_trace *tr,
         uint32_t dma_step_rows,
         uint32_t dma_step_rows_shift) {
@@ -2280,38 +2274,56 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
 
     const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;
 
+    const bool act_is_f16 = (act_elem_size == sizeof(__fp16));
+    const size_t row_bytes = (size_t) k_block * act_elem_size;  // staging row (DMA dst stride)
+    const size_t src_row_bytes = (size_t) k_stride * act_elem_size;  // DDR row (DMA src stride)
+    const size_t width_bytes = (size_t) k_valid * act_elem_size;
+
     // Push step 0
     if (n_steps > 0 && n_rows > 0) {
         uint32_t nrows_to_fetch = hex_smin(n_rows, R);
-        dma_queue_push(dma_q, dma_make_data(thread_f32_act, act_dma_addr),
-                       k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
+        dma_queue_push(dma_q, dma_make_data(thread_act, act_dma_addr),
+                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
     }
     // Push step 1 (if valid)
     if (n_steps > 1) {
         uint32_t next_r = R * 1;
         if (next_r < n_rows) {
             uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
-            float *next_buf = thread_f32_act + 1 * R * k_block;
-            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * k_stride * sizeof(float)),
-                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
+            uint8_t *next_buf = thread_act + 1 * R * row_bytes;
+            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
+                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
         }
     }
     for (uint32_t s = 0; s < n_steps; ++s) {
         uint32_t r = s << dma_step_rows_shift;
-        float *curr_buf = thread_f32_act;
+        uint8_t *curr_buf = thread_act;
 
         if (r < n_rows) {
-            curr_buf = (float *) dma_queue_pop(dma_q).dst;
+            curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;
         }
 
         htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
-        for (uint32_t p = 0; p < (R >> 1); ++p) {
-            uint32_t row_idx = r + (p << 1);
-            float *pair_buf = curr_buf + (p << 1) * k_block;
-            bool r0_valid = ((row_idx + 0) < n_rows);
-            bool r1_valid = ((row_idx + 1) < n_rows);
+        // Two copies of the pair loop so the type is resolved once per step and the
+        // row-pair kernels stay direct (inlinable) calls.
+        if (act_is_f16) {
+            for (uint32_t p = 0; p < (R >> 1); ++p) {
+                uint32_t row_idx = r + (p << 1);
+                const __fp16 *pair_buf = (const __fp16 *) (curr_buf + (p << 1) * row_bytes);
+                bool r0_valid = ((row_idx + 0) < n_rows);
+                bool r1_valid = ((row_idx + 1) < n_rows);
+
+                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
+            }
+        } else {
+            for (uint32_t p = 0; p < (R >> 1); ++p) {
+                uint32_t row_idx = r + (p << 1);
+                const float *pair_buf = (const float *) (curr_buf + (p << 1) * row_bytes);
+                bool r0_valid = ((row_idx + 0) < n_rows);
+                bool r1_valid = ((row_idx + 1) < n_rows);
 
-            transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
+                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
+            }
         }
         htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
 
@@ -2320,8 +2332,8 @@ static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
         uint32_t next_r = next_s << dma_step_rows_shift;
         if (next_r < n_rows) {
             uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
-            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * k_stride * sizeof(float)),
-                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
+            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
+                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
         }
     }
 }
@@ -2336,11 +2348,11 @@ static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i,
         size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);
 
         __fp16      *dst = st->dst + chunk_idx * st->k_block;
-        const dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * sizeof(float);
+        const dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;
 
-        float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread);
-        transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
-            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
+        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
+        transfer_activation_chunk_to_fp16_dma_pipelined(
+            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
         );
     }
 }
@@ -2446,10 +2458,9 @@ static void dequantize_tiled_weight_chunk_to_fp16_tiles(
         int n_k_tiles, struct fastdiv_values n_k_tiles_div,
         worker_callback_t dequant_worker_fn, int n_threads) {
 
-    assert(n_cols  % HTP_MM_HMX_TILE_N_COLS == 0);
     assert(k_block % HTP_MM_HMX_TILE_N_COLS == 0);
 
-    size_t n_col_tiles = n_cols / HTP_MM_HMX_TILE_N_COLS;
+    size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
     size_t n_tot_tiles = n_col_tiles * n_k_tiles;
 
     size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);
@@ -2499,7 +2510,9 @@ static void transfer_output_chunk_col_chunk_worker_fn(unsigned int n, unsigned i
     output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
     struct htp_thread_trace * tr = &st->traces[i];
 
-    uint32_t n_blocks = st->n_cols / 32;
+    // Round up: the last block is partial when N is not 32-aligned. Its pad
+    // columns are dropped by the dst_cols clamp inside the store.
+    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
     uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
     uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
     uint32_t c_first  = b_first * 32;
@@ -2527,11 +2540,9 @@ static void transfer_output_chunk_col_chunk_worker_fn(unsigned int n, unsigned i
 
 static void transfer_output_chunk_threaded(struct htp_context *ctx, float *dst, const float *src2, const __fp16 *vtcm_src,
                                               int n_rows, int n_cols, int dst_stride, uint32_t src2_stride, int dst_cols, int n_threads) {
-    assert(n_cols % HTP_MM_HMX_TILE_N_COLS == 0);
-
     if (n_rows <= 0) return;
 
-    uint32_t n_blocks = (uint32_t)n_cols / 32;
+    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
     if (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
         struct fastdiv_values n_threads_div = (n_threads == (int)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_values(n_threads);
         output_transfer_col_chunk_state_t col_state;
@@ -2590,6 +2601,7 @@ struct activation_transfer_params {
     int                           k_valid;
     float *                       vtcm_f32_act;
     size_t                        vtcm_f32_act_bytes;
+    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
 };
 
 static void transfer_activation_chunk_threaded(const struct activation_transfer_params * params) {
@@ -2605,13 +2617,15 @@ static void transfer_activation_chunk_threaded(const struct activation_transfer_
     int                           k_valid            = params->k_valid;
     float *                       vtcm_f32_act       = params->vtcm_f32_act;
     size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
+    // element size of the activation rows (4 = F32, 2 = F16).
+    const uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) sizeof(float);
 
     if (n_rows <= 0) {
         return;
     }
 
     const size_t n_tasks = (n_rows + 31) >> 5;
-    if (n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
+    if (act_elem_size == sizeof(float) && n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
         // Calculate step rows parameters for column-chunked dma pipelining
         uint32_t dma_step_rows = 2;
         uint32_t dma_step_rows_shift = 1;
@@ -2662,6 +2676,7 @@ static void transfer_activation_chunk_threaded(const struct activation_transfer_
     state.traces             = ctx->trace;
     state.ctx                = ctx;
     state.vtcm_f32_act       = vtcm_f32_act;
+    state.act_elem_size = act_elem_size;
 
     state.vtcm_f32_act_bytes_per_thread = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
 
@@ -2729,6 +2744,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
                                   dma_addr_t weight,
                                   int m, int k, int n,
                                   int act_stride,
+                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                   int weight_stride,
                                   int weight_type,
                                   int k_valid,
@@ -2748,7 +2764,12 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
     struct htp_thread_trace * tr = &ctx->trace[0];
     htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);
 
-    if (k % 32 != 0 || n % 32 != 0) { return -1; }
+    // Quantized weights are repacked and padded to 32, so we it has to be 32-aligned.
+    // Only F16/F32 weights can be non-32-aligned, they will be padded in the following kernel.
+    const bool wtype_is_quant = (weight_type != HTP_TYPE_F16 && weight_type != HTP_TYPE_F32);
+    if (k % 32 != 0 || (wtype_is_quant && n % 32 != 0)) {
+        return -1;
+    }
     if (!hex_is_aligned(dst, VLEN) || (act_dma_addr & (VLEN - 1)) != 0) { return -1; }
 
     size_t row_stride = htp_mm_get_tiled_row_stride(weight_type, k);
@@ -2817,10 +2838,10 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
 
     hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16
 
-    const bool has_src2      = (src2_bytes > 0 && src2_addr != 0);
-    float   *vtcm_src2       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_src2, has_src2);
-    if (has_src2) {
-        dma_queue_push(weight_dma, dma_make_data(vtcm_src2, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
+    const bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
+    float   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_bias, has_bias);
+    if (has_bias) {
+        dma_queue_push(weight_dma, dma_make_data(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
         dma_queue_pop(weight_dma);
     }
 
@@ -2844,7 +2865,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
+                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -2854,18 +2875,19 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
                 .k_valid = k_valid,
                 .vtcm_f32_act = vtcm_f32_act,
                 .vtcm_f32_act_bytes = L.act_f32_bytes,
+                .act_elem_size = act_elem_size,
             };
             transfer_activation_chunk_threaded(&act_params);
 
             // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
             const size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
-            const uint32_t height_A0 = is_quant ? (n_cols_A0 / 32) * n_k_tiles : n_cols_A0;
+            const uint32_t height_A0 = is_quant ? hmx_ceil_div(n_cols_A0, 32) * n_k_tiles : n_cols_A0;
             dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight),
                            dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);
 
             if (1 < n_chunk_cnt) {
                 const size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
-                const uint32_t height_A1 = is_quant ? (n_cols_A1 / 32) * n_k_tiles : n_cols_A1;
+                const uint32_t height_A1 = is_quant ? hmx_ceil_div(n_cols_A1, 32) * n_k_tiles : n_cols_A1;
                 dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride),
                                dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
             }
@@ -2889,7 +2911,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
 
                 // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                 if (i + 2 < n_chunk_cnt) {
-                    const uint32_t height_p2 = is_quant ? (n_cols_p2 / 32) * n_k_tiles : n_cols_p2;
+                    const uint32_t height_p2 = is_quant ? hmx_ceil_div(n_cols_p2, 32) * n_k_tiles : n_cols_p2;
                     dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_p2 * weight_stride),
                                    dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                 }
@@ -2907,10 +2929,10 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
                     const size_t nc_prev = (i - 1) * n_chunk_n_cols;
                     const size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                     float *output_chunk = dst + (mr * dst_stride + nc_prev);
-                    const float *src2_chunk = has_src2 ? (vtcm_src2 + mr * src2_stride + nc_prev) : NULL;
+                    const float *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : NULL;
                     int chunk_dst_cols = dst_cols - (int)nc_prev;
                     if (chunk_dst_cols > 0) {
-                        transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
+                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                     }
                 }
             }
@@ -2920,10 +2942,10 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
             const size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
             const size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
             float *output_chunk = dst + (mr * dst_stride + nc_last);
-            const float *src2_chunk = has_src2 ? (vtcm_src2 + mr * src2_stride + nc_last) : NULL;
+            const float *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : NULL;
             int chunk_dst_cols = dst_cols - (int)nc_last;
             if (chunk_dst_cols > 0) {
-                transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
+                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
             }
         }
     } else {
@@ -2935,7 +2957,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
+                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -2945,13 +2967,14 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
                 .k_valid = k_valid,
                 .vtcm_f32_act = vtcm_f32_act,
                 .vtcm_f32_act_bytes = L.act_f32_bytes,
+                .act_elem_size = act_elem_size,
             };
             transfer_activation_chunk_threaded(&act_params);
 
             // A0: Pre-fetch the first weight chunk (nc = 0)
             if (n > 0) {
                 const size_t n_cols = hex_smin(n, n_chunk_n_cols);
-                const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
+                const uint32_t height = is_quant ? hmx_ceil_div(n_cols, 32) * n_k_tiles : n_cols;
                 dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
             }
 
@@ -2973,7 +2996,7 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
                 const size_t nc_next = nc + n_chunk_n_cols;
                 if (nc_next < n) {
                     const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
-                    const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
+                    const uint32_t height_next = is_quant ? hmx_ceil_div(n_cols_next, 32) * n_k_tiles : n_cols_next;
                     dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                 }
 
@@ -2984,10 +3007,10 @@ static int hmx_mm_2d_f32(struct htp_context *ctx,
 
                 // D: Output Store
                 float *output_chunk = dst + (mr * dst_stride + nc);
-                const float *src2_chunk = has_src2 ? (vtcm_src2 + mr * src2_stride + nc) : NULL;
+                const float *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : NULL;
                 int chunk_dst_cols = dst_cols - (int)nc;
                 if (chunk_dst_cols > 0) {
-                    transfer_output_chunk_threaded(ctx, output_chunk, src2_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
+                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                 }
             }
         }
@@ -3013,7 +3036,8 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
     const int k           = (int) act->ne[0];
     const int k_valid     = (int) act->ne[0];
     const int m           = (int) (act->ne[1] * act->ne[2] * act->ne[3]);
-    const int act_stride  = (int) (act->nb[1] / sizeof(float));
+    const uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
+    const int act_stride  = (int) (act->nb[1] / act_elem_size);
     const dma_addr_t act_dma_addr = act->data;
 
     if (k % 32 != 0) { return HTP_STATUS_NO_SUPPORT; }
@@ -3089,7 +3113,16 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
     int m_start = 0;
     int m_rows  = m;
     if (octx->ctx->mdev.count > 1) {
-        const bool can_split = htp_tensor_can_row_partition(octx->dsts[0], sizeof(float));
+        bool can_split = htp_tensor_can_row_partition(act, act_elem_size);
+        if (can_split) {
+            for (uint32_t p = 0; p < n_weights; ++p) {
+                const struct htp_tensor * restrict dst = octx->dsts[p];
+                if (!htp_tensor_can_row_partition(dst, sizeof(float))) {
+                    can_split = false;
+                    break;
+                }
+            }
+        }
         const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
         m_start = (int) range.start;
         m_rows  = (int) range.count;
@@ -3118,7 +3151,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
+                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -3128,6 +3161,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
                 .k_valid = k_valid,
                 .vtcm_f32_act = vtcm_f32_act,
                 .vtcm_f32_act_bytes = L.act_f32_bytes,
+                .act_elem_size = act_elem_size,
             };
             transfer_activation_chunk_threaded(&act_params);
 
@@ -3149,13 +3183,13 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
                 const uint32_t dma_src_stride = is_quant ? tile_size : weight_stride;
 
                 const size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
-                const uint32_t height_A0 = is_quant ? (n_cols_A0 / 32) * n_k_tiles : n_cols_A0;
+                const uint32_t height_A0 = is_quant ? hmx_ceil_div(n_cols_A0, 32) * n_k_tiles : n_cols_A0;
                 dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight),
                                dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);
 
                 if (1 < n_chunk_cnt) {
                     const size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
-                    const uint32_t height_A1 = is_quant ? (n_cols_A1 / 32) * n_k_tiles : n_cols_A1;
+                    const uint32_t height_A1 = is_quant ? hmx_ceil_div(n_cols_A1, 32) * n_k_tiles : n_cols_A1;
                     dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[1], weight + n_chunk_n_cols * weight_stride),
                                    dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                 }
@@ -3175,7 +3209,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
                         n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);
 
                     if (i + 2 < n_chunk_cnt) {
-                        const uint32_t height_p2 = is_quant ? (n_cols_p2 / 32) * n_k_tiles : n_cols_p2;
+                        const uint32_t height_p2 = is_quant ? hmx_ceil_div(n_cols_p2, 32) * n_k_tiles : n_cols_p2;
                         dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_p2 * weight_stride),
                                        dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                     }
@@ -3216,7 +3250,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
             struct activation_transfer_params act_params = {
                 .ctx = ctx,
                 .dst = vtcm_f16_act,
-                .act_dma_addr = act_dma_addr + mr * act_stride * sizeof(float),
+                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                 .n_rows = (int) n_rows,
                 .k_block = k,
                 .k_stride = act_stride,
@@ -3226,6 +3260,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
                 .k_valid = k_valid,
                 .vtcm_f32_act = vtcm_f32_act,
                 .vtcm_f32_act_bytes = L.act_f32_bytes,
+                .act_elem_size = act_elem_size,
             };
             transfer_activation_chunk_threaded(&act_params);
 
@@ -3247,7 +3282,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
 
                 if (n > 0) {
                     const size_t n_cols = hex_smin(n, n_chunk_n_cols);
-                    const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
+                    const uint32_t height = is_quant ? hmx_ceil_div(n_cols, 32) * n_k_tiles : n_cols;
                     dma_queue_push(weight_dma, dma_make_data(vtcm_weight_raw[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                 }
 
@@ -3266,7 +3301,7 @@ static int hmx_mm_nx_2d_f32(struct htp_ops_context * octx, const struct htp_mm_k
                     const size_t nc_next = nc + n_chunk_n_cols;
                     if (nc_next < n) {
                         const size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
-                        const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
+                        const uint32_t height_next = is_quant ? hmx_ceil_div(n_cols_next, 32) * n_k_tiles : n_cols_next;
                         dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                     }
 
@@ -3314,16 +3349,16 @@ static int hmx_mm_f16_f32_batched_simple(struct htp_context *ctx,
     int ret = 0;
     for (int b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
         for (int b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
-            dma_addr_t cur_src2_addr = params->src2_addr ? (params->src2_addr +
-                                       b2 * params->src2_nb2 +
-                                       b3 * params->src2_nb3) : 0;
+            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
+                                       b2 * params->bias_nb2 +
+                                       b3 * params->bias_nb3) : 0;
             ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
-                                cur_src2_addr, params->src2_bytes,
+                                cur_bias_addr, params->bias_bytes,
                                 hmx_mm_act_batch_addr(params, b2, b3),
                                 hmx_mm_weight_batch_data(params, b2, b3),
                                 params->m, params->k, params->n,
-                                params->act_stride, params->weight_stride * (int)sizeof(__fp16),
-                                HTP_TYPE_F16, params->k, params->dst_stride, params->src2_stride, params->n,
+                                params->act_stride, params->act_elem_size, params->weight_stride * (int)sizeof(__fp16),
+                                HTP_TYPE_F16, params->k, params->dst_stride, params->bias_stride, params->n,
                                 m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                 act_threads_div, k_div, 0, 0, vtcm_size);
         }
@@ -3339,7 +3374,8 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
     if (params->act_stride < params->k || params->weight_stride < params->k || params->dst_stride < params->n) { return -1; }
     if (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { return -1; }
     if (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { return -1; }
-    if (params->k % 32 != 0 || params->n % 32 != 0) { return -1; }
+    // N (the weight row count) does not have to be 32-aligned:
+    if (params->k % 32 != 0) { return -1; }
     if (!hex_is_aligned(params->dst, VLEN) || (params->act_dma_addr & (VLEN - 1)) != 0) { return -1; }
 
     const int group_size = params->r2;
@@ -3363,7 +3399,7 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
     size_t vtcm_used = vtcm_size;
 
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->src2_bytes);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->bias_bytes);
 
     if (L.total_bytes > vtcm_budget) {
         FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
@@ -3380,10 +3416,10 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
     __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
     float   *vtcm_f32_act    = VTCM_LAYOUT_PTR(float, base, L.off_act_f32);
 
-    const bool has_src2      = (params->src2_bytes > 0 && params->src2_addr != 0);
-    float   *vtcm_src2       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_src2, has_src2);
-    if (has_src2) {
-        dma_queue_push(params->weight_dma, dma_make_data(vtcm_src2, params->src2_addr), hex_align_up(params->src2_bytes, 128), 0, params->src2_bytes, 1);
+    const bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
+    float   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_bias, has_bias);
+    if (has_bias) {
+        dma_queue_push(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
         dma_queue_pop(params->weight_dma);
     }
 
@@ -3417,7 +3453,7 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
                 // thrashing from HVX loads at large strides.
                 for (int g = 0; g < group_size; ++g) {
                     const dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
-                                                      mr * params->act_stride * sizeof(float);
+                                                      (size_t) mr * params->act_stride * params->act_elem_size;
                     __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                     struct activation_transfer_params act_params = {
                         .ctx = ctx,
@@ -3432,6 +3468,7 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
                         .k_valid = params->k,
                         .vtcm_f32_act = vtcm_f32_act,
                         .vtcm_f32_act_bytes = L.act_f32_bytes,
+                        .act_elem_size = params->act_elem_size,
                     };
                     transfer_activation_chunk_threaded(&act_params);
                 }
@@ -3444,7 +3481,8 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
                 }
                 if (n_chunk_n_cols < (size_t) params->n) {
                     const size_t n_cols_second = hex_smin((size_t) params->n - n_chunk_n_cols, n_chunk_n_cols);
-                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch1, weight_group + params->weight_stride * sizeof(__fp16)),
+                    const dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * params->weight_stride * sizeof(__fp16);
+                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch1, second_weight_chunk),
                                       fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                 }
 
@@ -3455,7 +3493,9 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
                     {
                         void * curr_raw = (void *) dma_queue_pop(weight_dma).dst;
 
-                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) curr_raw, n_cols, params->k, params->k, 0, n_cols);
+                        const size_t n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);
+
+                        hmx_interleave_rows_to_tiles(vtcm_weight, (const __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);
 
                         const size_t nc_next = nc + n_chunk_n_cols * 2;
                         if (nc_next < (size_t) params->n) {
@@ -3478,11 +3518,11 @@ static int hmx_mm_f16_f32_batched(struct htp_context *ctx, const hmx_mm_f16_f32_
 
                         {
                             float *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
-                            const float *src2_chunk = has_src2 ? (vtcm_src2 + mr * params->src2_stride + nc) : NULL;
+                            const float *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : NULL;
                             int chunk_dst_cols = params->n - (int)nc;
                             if (chunk_dst_cols > 0) {
-                                transfer_output_chunk_threaded(ctx, output, src2_chunk, vtcm_output, (int) n_rows, (int) n_cols,
-                                                               params->dst_stride, params->src2_stride, chunk_dst_cols, n_threads);
+                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (int) n_rows, (int) n_cols,
+                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
                             }
                         }
                     }
@@ -3612,7 +3652,8 @@ static int hmx_mm_id_2d_f32(struct htp_context *ctx,
     htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);
 
     const int cne1 = m;
-    const int m_padded = hex_align_up(m, 32);
+    const int m_core = m_end - m_start;
+    const int m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);
 
     if (k % 32 != 0 || n % 32 != 0) { return -1; }
     if (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { return -1; }
@@ -3666,10 +3707,10 @@ static int hmx_mm_id_2d_f32(struct htp_context *ctx,
     const size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
     size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
     if (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, size_per_n, size_per_m, size_per_mn,
-                           m_padded, n,
+                           m_core_padded, n,
                            /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
-                           /*n_block_cost=*/(size_t) m_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
-        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_padded, k, n, vtcm_budget);
+                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
+        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
         return -1;
     }
 
@@ -3709,7 +3750,7 @@ static int hmx_mm_id_2d_f32(struct htp_context *ctx,
         // A0: Pre-fetch the first weight chunk (nc = 0)
         if (n > 0) {
             const size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
-            const uint32_t height = is_quant ? (n_cols / 32) * n_k_tiles : n_cols;
+            const uint32_t height = is_quant ? hmx_ceil_div(n_cols, 32) * n_k_tiles : n_cols;
             dma_queue_push(weight_dma, dma_make_data(vtcm_weight, weight),
                            dma_dst_stride, dma_src_stride, dma_width_bytes, height);
         }
@@ -3732,7 +3773,7 @@ static int hmx_mm_id_2d_f32(struct htp_context *ctx,
             const size_t nc_next = nc + n_chunk_n_cols;
             if (nc_next < (size_t) n) {
                 const size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
-                const uint32_t height_next = is_quant ? (n_cols_next / 32) * n_k_tiles : n_cols_next;
+                const uint32_t height_next = is_quant ? hmx_ceil_div(n_cols_next, 32) * n_k_tiles : n_cols_next;
                 dma_queue_push(weight_dma, dma_make_data(curr_raw, weight + nc_next * weight_stride),
                                dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
             }
@@ -3759,14 +3800,19 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
 
     int k = (int) src0->ne[0];
     int n = (int) src0->ne[1];
-    const int m_total    = (int) src1->ne[1];
-    const int act_stride = (int)(src1->nb[1] / sizeof(float));
+    const int m_total    = (int) act->ne[1];
+    const uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
+    const int act_stride = (int) (act->nb[1] / act_elem_size);
     const int wgt_stride = (int)(src0->nb[1] / sizeof(__fp16));
 
     int m_start = 0;
     int m_rows  = m_total;
     if (octx->ctx->mdev.count > 1) {
-        const bool can_split = htp_tensor_can_row_partition(dst, sizeof(float));
+        bool can_split = htp_tensor_can_row_partition(dst, sizeof(float)) &&
+                         htp_tensor_can_row_partition(act, act_elem_size);
+        if (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, sizeof(float))) {
+            can_split = false;
+        }
         const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
         m_start = (int) range.start;
         m_rows  = (int) range.count;
@@ -3784,22 +3830,23 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
     if (src2) {
         src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / sizeof(float));
         src2_addr = src2->data + m_start * src2_stride * sizeof(float);
-        src2_bytes = (size_t) kparams->vtcm_src2_size;
+        src2_bytes = (size_t) kparams->vtcm_bias_size;
         src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
         src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
     }
 
     const int dst_stride = (int)(dst->nb[1] / sizeof(float));
     float       * dst_ptr = (float *) dst->data + m_start * dst_stride;
-    const dma_addr_t act_addr = src1->data + m_start * act_stride * sizeof(float);
+    // byte offset, act_stride is in activation elements (F16 or F32)
+    const dma_addr_t act_addr = act->data + (size_t) m_start * act_stride * act_elem_size;
 
     int ret = -1;
     const int n_threads = kparams->n_threads;
     if (kparams->kernel_type == HTP_MM_KERNEL_HMX_F16_BATCHED) {
         hmx_mm_f16_f32_batched_params_t batch_params = {
             .dst             = dst_ptr,
-            .src2_addr       = src2_addr,
-            .src2_bytes      = src2_bytes,
+            .bias_addr       = src2_addr,
+            .bias_bytes      = src2_bytes,
             .act_dma_addr    = act_addr,
             .weight          = src0->data,
             .weight_dma      = octx->ctx->dma[0],
@@ -3807,21 +3854,22 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
             .k               = k,
             .n               = n,
             .act_stride      = act_stride,
+            .act_elem_size = act_elem_size,
             .weight_stride   = wgt_stride,
             .dst_stride      = dst_stride,
-            .src2_stride     = src2_stride,
+            .bias_stride     = src2_stride,
             .ne02            = ne02,
             .ne03            = ne03,
             .ne12            = ne12,
             .ne13            = ne13,
             .src0_nb2        = src0->nb[2],
             .src0_nb3        = src0->nb[3],
-            .act_nb2         = src1->nb[2],
-            .act_nb3         = src1->nb[3],
+            .act_nb2         = act->nb[2],
+            .act_nb3         = act->nb[3],
             .dst_nb2         = dst->nb[2],
             .dst_nb3         = dst->nb[3],
-            .src2_nb2        = src2_nb2,
-            .src2_nb3        = src2_nb3,
+            .bias_nb2        = src2_nb2,
+            .bias_nb3        = src2_nb3,
             .r2              = (ne02 > 0) ? (ne12 / ne02) : 1,
             .r3              = (ne03 > 0) ? (ne13 / ne03) : 1,
             .div_r2          = kparams->div_r2,
@@ -3838,7 +3886,7 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
         ret = hmx_mm_2d_f32(
             octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
             act_addr, src0->data,
-            m_rows, k, n, act_stride, (int) src0->nb[1], (int) src0->type, (int) src1->ne[0],
+            m_rows, k, n, act_stride, act_elem_size, (int) src0->nb[1], (int) src0->type, (int) act->ne[0],
             dst_stride, src2_stride, (int)dst->ne[0],
             kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
             kparams->n_act_threads,
@@ -3855,7 +3903,17 @@ static int hmx_mm_op_matmul(struct htp_ops_context * octx, const struct htp_mm_k
     return HTP_STATUS_OK;
 }
 
-int op_matmul(struct htp_ops_context * octx) {
+static inline void htp_mm_tensor_collapse_rows(struct htp_tensor * c, const struct htp_tensor * t, uint32_t stride) {
+    *c = *t;
+    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
+    c->ne[2] = 1;
+    c->ne[3] = 1;
+    c->nb[1] = stride;
+    c->nb[2] = c->nb[1] * c->ne[1];
+    c->nb[3] = c->nb[2];
+}
+
+static int op_matmul_impl(struct htp_ops_context * octx) {
     const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
 
     const int status = htp_mm_init_context(octx, kparams);
@@ -3870,6 +3928,40 @@ int op_matmul(struct htp_ops_context * octx) {
     return hvx_mm_matmul(octx);
 }
 
+int op_matmul(struct htp_ops_context * octx) {
+    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
+
+    if (kparams->collapse) {
+        const struct htp_tensor * act  = octx->src[1];
+        const struct htp_tensor * dst  = octx->dst;
+        const uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
+        const uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
+        struct htp_tensor act_collapsed, dst_collapsed;
+        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
+        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
+        octx->src[1] = &act_collapsed;
+        octx->dst    = &dst_collapsed;
+
+        struct htp_tensor src2_collapsed;
+        const struct htp_tensor * src2 = octx->src[2];
+        if (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
+            const uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
+            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
+            octx->src[2] = &src2_collapsed;
+        }
+
+        const int status = op_matmul_impl(octx);
+        octx->src[1] = act;
+        octx->dst    = dst;
+        if (src2) {
+            octx->src[2] = src2;
+        }
+        return status;
+    }
+
+    return op_matmul_impl(octx);
+}
+
 static int hmx_mm_op_matmul_id(
     struct htp_ops_context * octx,
     struct htp_mm_context * mmctx
@@ -3881,21 +3973,40 @@ static int hmx_mm_op_matmul_id(
     const int n_ids = octx->src[2]->ne[0];
     const int n_as  = ne02;
 
-    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
+    const bool mdev_split = (octx->ctx->mdev.count > 1) && htp_tensor_can_row_partition(dst, sizeof(float));
+    if (octx->ctx->mdev.count > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
+        return HTP_STATUS_OK;
+    }
+    uint32_t n_active = 0;
+    if (mdev_split) {
+        for (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
+            if (matrix_row_counts[a] > 0) n_active++;
+        }
+    }
+    const bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);
+
+    uint32_t target_dev = 0;
+    for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
         const int32_t cne1 = matrix_row_counts[cur_a];
         if (cne1 == 0) continue;
 
         const int m_padded = hex_align_up(cne1, 32);
         int m_start = 0, m_end = m_padded;
-        if (octx->ctx->mdev.count > 1) {
-            const bool can_split = htp_tensor_mdev_data_aligned(dst) && (uint32_t) cne1 >= octx->ctx->mdev.count;
+        if (expert_split) {
+            const bool my_expert = (target_dev == octx->ctx->mdev.idx);
+            if (++target_dev == octx->ctx->mdev.count) {
+                target_dev = 0;
+            }
+            if (!my_expert) continue;
+        } else if (mdev_split) {
+            const bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
             const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
             m_start = (int) range.start;
             m_end   = (int) (range.start + range.count);
         }
         if (m_start >= m_end) continue;
 
-        int ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (float*) dst->data, (float*) src1->data,
+        int ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (float*) dst->data, (float*) act->data,
                                    src0->data + cur_a * nb02,
                                    cne1, ne00, ne01,
                                    ne10,
@@ -3951,21 +4062,21 @@ static int hvx_mm_matmul_id(
         n_quant_tasks = MIN(act_nrows, octx->n_threads);
         quant_task_func = htp_mm_act_quant_row_func(src0->type);
     }
-    size_t src1_row_size  = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+    size_t act_row_size   = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
 
     struct htp_mm_hvx_vtcm_layout L;
     htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
-                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);
+                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);
 
     const size_t vtcm_size = L.total_bytes;
 
-    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu src1-spad-size %zu src2-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
-         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);
+    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
+         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);
 
     FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
-         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
+         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
          ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
-         src1->data, dst->data);
+         act->data, dst->data);
 
     // Make sure the reserved vtcm size is sufficient
     if (octx->ctx->vtcm_size < vtcm_size) {
@@ -3974,24 +4085,18 @@ static int hvx_mm_matmul_id(
     }
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
-    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
     mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src2     = NULL;
+    mmctx->vtcm_bias     = NULL;
     mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
     mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
-    octx->src1_spad.src  = NULL;
-    octx->src0_spad.src  = NULL;
-    octx->src2_spad.src  = NULL;
-    octx->dst_spad.src   = NULL;
-
     mmctx->vtcm_src0_stride    = src0_row_size_padded;
-    mmctx->vtcm_src1_stride    = src1_row_size;
+    mmctx->vtcm_act_stride     = act_row_size;
     mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
-    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
-    mmctx->vtcm_src2_size_per_thread = 0;
+    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
     mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);
 
     mmctx->cur_m_start = 0;
@@ -3999,7 +4104,7 @@ static int hvx_mm_matmul_id(
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    hvx_mm_transfer_src1_dma(octx, kparams, src1, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
 
     mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
     mmctx->n_quant_tasks = n_quant_tasks;
@@ -4022,18 +4127,37 @@ static int hmx_mm_op_matmul_id_nx(
     const struct htp_tensor * restrict act  = octx->src[n_weights];
     const int n_as = src0->ne[2];
 
+    bool mdev_split = (octx->ctx->mdev.count > 1);
+    for (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
+        const struct htp_tensor * restrict dst = octx->dsts[p];
+        mdev_split = htp_tensor_can_row_partition(dst, sizeof(float));
+    }
+    if (octx->ctx->mdev.count > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
+        return HTP_STATUS_OK;
+    }
+    uint32_t n_active = 0;
+    if (mdev_split) {
+        for (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
+            if (matrix_row_counts[a] > 0) n_active++;
+        }
+    }
+    const bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);
+
+    uint32_t target_dev = 0;
     for (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
         const int32_t cne1 = matrix_row_counts[cur_a];
         if (cne1 == 0) continue;
 
         const int m_padded = hex_align_up(cne1, 32);
         int m_start = 0, m_end = m_padded;
-        if (octx->ctx->mdev.count > 1) {
-            bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
-            for (uint32_t p = 0; p < n_weights && can_split; ++p) {
-                const struct htp_tensor * restrict dst = octx->dsts[p];
-                can_split = !dst || htp_tensor_mdev_data_aligned(dst);
+        if (expert_split) {
+            const bool my_expert = (target_dev == octx->ctx->mdev.idx);
+            if (++target_dev == octx->ctx->mdev.count) {
+                target_dev = 0;
             }
+            if (!my_expert) continue;
+        } else if (mdev_split) {
+            const bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
             const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
             m_start = (int) range.start;
             m_end   = (int) (range.start + range.count);
@@ -4104,11 +4228,11 @@ static int hvx_mm_matmul_id_nx(
         n_quant_tasks = MIN(act_nrows, octx->n_threads);
         quant_task_func = htp_mm_act_quant_row_func(src0->type);
     }
-    size_t src1_row_size = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);
+    size_t act_row_size  = htp_mm_weight_has_offset(src0->type) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);
 
     struct htp_mm_hvx_vtcm_layout L;
     htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
-                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);
+                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);
 
     const size_t vtcm_size = L.total_bytes;
 
@@ -4120,35 +4244,29 @@ static int hvx_mm_matmul_id_nx(
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
     mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
     mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
     mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
-    octx->src0_spad.src = NULL;
-    octx->src1_spad.src = NULL;
-    octx->src2_spad.src = NULL;
-    octx->src3_spad.src = NULL;
-    octx->dst_spad.src  = NULL;
-
     mmctx->vtcm_src0_stride    = 0;
-    mmctx->vtcm_src1_stride    = src1_row_size;
+    mmctx->vtcm_act_stride     = act_row_size;
     mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
-    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
+    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
     mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);
 
     mmctx->cur_m_start = 0;
     mmctx->cur_m_rows  = act_nrows;
 
-    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, src1 %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
+    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
          src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
          act->ne[0], act->ne[1], act->ne[2], act_nrows,
          L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    hvx_mm_transfer_src1_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
 
     mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
     mmctx->n_quant_tasks           = n_quant_tasks;
@@ -4242,10 +4360,10 @@ int op_matmul_id(struct htp_ops_context * octx) {
     htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);
 
     mmctx->octx = octx;
-    mmctx->act = src1;
+    mmctx->act = act;
 
     const struct htp_tensor * restrict ids = octx->src[2];
-    if (htp_tensor_is_extended(ids) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
+    if (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
         return HTP_STATUS_NO_SUPPORT;
     }
 
@@ -4255,7 +4373,7 @@ int op_matmul_id(struct htp_ops_context * octx) {
     const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
 
     const uint32_t src0_nrows = ne01;  // per expert
-    const uint32_t src1_nrows = ne11 * ne12 * ne13;
+    const uint32_t act_nrows  = ne11 * ne12 * ne13;
 
     // row groups
     const int n_ids = ids->ne[0];  // n_expert_used
@@ -4266,7 +4384,7 @@ int op_matmul_id(struct htp_ops_context * octx) {
     uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
     struct mmid_row_mapping * matrix_rows = NULL;
 
-    if (src1_nrows > 1) {
+    if (act_nrows > 1) {
         const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
         assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);
 
@@ -4300,9 +4418,9 @@ int op_matmul_id(struct htp_ops_context * octx) {
     mmctx->mapping_stride       = mapping_stride;
     mmctx->mm_div_ne11          = kparams->div_ne1;
     mmctx->src0_row_size_padded = src0_row_size_padded;
-    mmctx->act_nrows            = src1_nrows;
+    mmctx->act_nrows            = act_nrows;
     mmctx->cur_m_start          = 0;
-    mmctx->cur_m_rows           = src1_nrows;
+    mmctx->cur_m_rows           = act_nrows;
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
@@ -4334,7 +4452,7 @@ int op_matmul_id(struct htp_ops_context * octx) {
         mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
 
         if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
-            s = hvx_mm_matmul_id(octx, mmctx, src1_nrows > 1 ? hvx_mm_id : hvx_mv_id);
+            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id : hvx_mv_id);
         } else {
             s = HTP_STATUS_NO_SUPPORT;
         }
@@ -4369,7 +4487,7 @@ int op_matmul_id_nx(struct htp_ops_context * octx) {
         return HTP_STATUS_NO_SUPPORT;
     }
     for (uint32_t p = 0; p < n_weights; p++) {
-        if (octx->dsts[p] && htp_tensor_is_extended(octx->dsts[p])) {
+        if (htp_tensor_is_extended(octx->dsts[p])) {
             return HTP_STATUS_NO_SUPPORT;
         }
     }
@@ -4446,7 +4564,7 @@ int op_matmul_id_nx(struct htp_ops_context * octx) {
 
     return s;
 }
-int op_matmul_nx(struct htp_ops_context * octx) {
+static int op_matmul_nx_impl(struct htp_ops_context * octx) {
     const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
 
     const int status = htp_mm_init_context(octx, kparams);
@@ -4511,13 +4629,13 @@ int op_matmul_nx(struct htp_ops_context * octx) {
         quant_task_func = htp_mm_act_quant_row_func(src0->type);
     }
 
-    const size_t src1_row_size = htp_mm_weight_has_offset(src0->type)
+    const size_t act_row_size = htp_mm_weight_has_offset(src0->type)
                                ? htp_mm_q8_1_tiled_row_size(act->ne[0])
                                : htp_mm_q8_0_tiled_row_size(act->ne[0]);
 
     struct htp_mm_hvx_vtcm_layout L;
     htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
-                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, false, true);
+                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, false, true);
 
     const size_t vtcm_size = L.total_bytes;
 
@@ -4529,22 +4647,16 @@ int op_matmul_nx(struct htp_ops_context * octx) {
 
     uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
     mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
-    mmctx->vtcm_src1     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
+    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
     mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
     mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);
 
-    octx->src0_spad.src  = NULL;
-    octx->src1_spad.src  = NULL;
-    octx->src2_spad.src  = NULL;
-    octx->src3_spad.src  = NULL;
-    octx->dst_spad.src   = NULL;
-
     mmctx->vtcm_src0_stride    = is_repacked ? 0 : src0_row_size_padded;
-    mmctx->vtcm_src1_stride    = src1_row_size;
+    mmctx->vtcm_act_stride     = act_row_size;
     mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
     mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
-    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
+    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
     mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);
 
     // Run fused matmul
@@ -4571,7 +4683,7 @@ int op_matmul_nx(struct htp_ops_context * octx) {
 
     htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
 
-    hvx_mm_transfer_src1_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
+    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
 
     mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
     mmctx->n_quant_tasks = n_quant_tasks;
@@ -4581,3 +4693,37 @@ int op_matmul_nx(struct htp_ops_context * octx) {
 
     return HTP_STATUS_OK;
 }
+
+int op_matmul_nx(struct htp_ops_context * octx) {
+    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
+
+    if (kparams->collapse) {
+        const uint32_t n_weights = kparams->n_weights;
+        const struct htp_tensor * act = octx->src[n_weights];
+        const uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
+        struct htp_tensor act_collapsed;
+        struct htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
+        const struct htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];
+
+        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
+        octx->src[n_weights] = &act_collapsed;
+
+        for (uint32_t p = 0; p < n_weights; p++) {
+            orig_dsts[p] = octx->dsts[p];
+            const struct htp_tensor * d = octx->dsts[p];
+            const uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
+            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
+            octx->dsts[p] = &dsts_collapsed[p];
+        }
+
+        const int status = op_matmul_nx_impl(octx);
+
+        octx->src[n_weights] = act;
+        for (uint32_t p = 0; p < n_weights; p++) {
+            octx->dsts[p] = orig_dsts[p];
+        }
+        return status;
+    }
+
+    return op_matmul_nx_impl(octx);
+}
diff --git src/ggml-hexagon/htp/matmul-ops.h src/ggml-hexagon/htp/matmul-ops.h
index 386cb304..1d11a4c1 100644
--- src/ggml-hexagon/htp/matmul-ops.h
+++ src/ggml-hexagon/htp/matmul-ops.h
@@ -87,24 +87,26 @@ enum htp_mm_kernel_type {
 
 // Op-specific struct for precomputed matmul params
 struct htp_mm_kernel_params {
-    int32_t  kernel_type;        // enum htp_mm_kernel_type
-    int32_t  pipeline;           // 1 = pipelined execution, 0 = standard
+    uint8_t  kernel_type;        // enum htp_mm_kernel_type
+    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
+    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
+    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX
+
+    uint8_t  n_threads;          // Number of threads to spawn
+    uint8_t  n_act_threads;      // Number of threads for activation preparation
+    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
+    uint8_t  n_weights;          // Number of weights for fused NX
+
     int32_t  m_chunk;            // Row chunk size (M chunk)
     int32_t  n_chunk;            // Col chunk size (N chunk)
-    int32_t  n_threads;          // Number of threads to spawn
-    int32_t  n_act_threads;      // Number of threads for activation preparation
-    int32_t  n_hmx;              // 1 = use HMX, 0 = use HVX
-    int32_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
     int32_t  tile_size;          // Weight tile size
     int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
-    int32_t  src1_row_size;      // Row size for quantized activation
+    int32_t  act_row_size;       // Row size for activation scratchpad
     int32_t  vtcm_size;          // Total required scratchpad size in VTCM
     int32_t  vtcm_src0_size;     // src0 scratchpad size in VTCM
-    int32_t  vtcm_src1_size;     // src1 scratchpad size in VTCM
-    int32_t  vtcm_src2_size;     // src2 scratchpad size in VTCM (fused only)
-    int32_t  vtcm_src3_size;     // src3 scratchpad size in VTCM (fused only)
+    int32_t  vtcm_act_size;      // activation scratchpad size in VTCM
+    int32_t  vtcm_bias_size;     // bias scratchpad size in VTCM (fused only)
     int32_t  vtcm_dst_size;      // dst scratchpad size in VTCM
-    int32_t  n_weights;          // Number of weights for fused NX
 
     // Precomputed division values
     struct fastdiv_values div_ne12_ne1;
@@ -147,6 +149,7 @@ static inline int htp_mm_hmx_compute_chunks(size_t   vtcm_total,
     const size_t usable = vtcm_total - overhead;
 
     size_t best_cost = SIZE_MAX;
+    size_t best_tail_waste = SIZE_MAX;
     size_t best_mn   = 0;
     size_t best_m = 0, best_n = 0;
 
@@ -173,12 +176,17 @@ static inline int htp_mm_hmx_compute_chunks(size_t   vtcm_total,
             size_t mblocks = ((size_t) m + mc - 1) / mc;
             size_t nblocks = ((size_t) n + nc - 1) / nc;
             size_t cost    = mblocks * m_block_cost + nblocks * n_block_cost;
+            size_t rem     = n % nc;
+            size_t tail_waste = (rem == 0) ? 0 : (nc - rem);
             size_t mn      = mc * nc;
-            if (cost < best_cost || (cost == best_cost && mn > best_mn)) {
-                best_cost = cost;
-                best_mn   = mn;
-                best_m    = mc;
-                best_n    = nc;
+            if (cost < best_cost ||
+                (cost == best_cost && tail_waste < best_tail_waste) ||
+                (cost == best_cost && tail_waste == best_tail_waste && mn > best_mn)) {
+                best_cost       = cost;
+                best_tail_waste = tail_waste;
+                best_mn         = mn;
+                best_m          = mc;
+                best_n          = nc;
             }
         }
 
@@ -349,7 +357,7 @@ struct htp_mm_hmx_vtcm_layout {
     size_t off_dst[2];        // [1] is only used when pipelined
     size_t off_scratch[2];    // dequantization scratch pads
     size_t off_scales;        // HMX scales (256 bytes)
-    size_t off_src2;          // src2 bias in VTCM
+    size_t off_bias;          // bias in VTCM
 
     // Cached sizes of regions for HMX kernel use
     size_t weight_area_bytes;
@@ -358,25 +366,23 @@ struct htp_mm_hmx_vtcm_layout {
     size_t output_area_bytes;
     size_t scratch_bytes[2];
     size_t act_head_stride;
-    size_t src2_bytes;
+    size_t bias_bytes;
 
     size_t total_bytes;
 };
 
 struct htp_mm_hvx_vtcm_layout {
     // Byte offsets from vtcm_base for each region
-    size_t off_src1;          // vtcm_src1 (activation)
+    size_t off_act;           // vtcm_act (activation)
     size_t off_src0;          // vtcm_src0 (weight/Wk)
-    size_t off_src2;          // vtcm_src2 (Wq / fused only)
-    size_t off_src3;          // vtcm_src3 (Wv / fused only)
+    size_t off_bias;          // vtcm_bias (bias / fused add only)
     size_t off_dst;           // vtcm_dst (output scratch)
     size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)
 
     // Cached sizes
     size_t src0_bytes;
-    size_t src1_bytes;
-    size_t src2_bytes;
-    size_t src3_bytes;
+    size_t act_bytes;
+    size_t bias_bytes;
     size_t dst_bytes;
     size_t act_raw_bytes;
 
@@ -394,7 +400,7 @@ static inline void htp_mm_hmx_vtcm_layout_build(
     bool pipeline,
     uint32_t act_threads,
     uint32_t aligned_tile_size,
-    size_t src2_size
+    size_t bias_size
 ) {
     size_t off = 0;
 
@@ -411,7 +417,7 @@ static inline void htp_mm_hmx_vtcm_layout_build(
         size_t off_group_a = 0;
         VTCM_LAYOUT_ALLOC(off_group_a, off_act, activation_area_size);
         VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data
-        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_src2, hex_align_up(src2_size, HTP_MM_HMX_TILE_SIZE), src2_size > 0);
+        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE), bias_size > 0);
 
         // Group B: Compute-only buffers (starts at off_group_a)
         size_t off_group_b = off_group_a;
@@ -439,7 +445,7 @@ static inline void htp_mm_hmx_vtcm_layout_build(
         L->scratch_bytes[0]  = scratch_area_size;
         L->scratch_bytes[1]  = scratch_area_size;
         L->act_head_stride   = act_head_stride;
-        L->src2_bytes        = src2_size;
+        L->bias_bytes        = bias_size;
 
         off = off_group_a + hex_smax(group_b_size, group_c_size);
     } else {
@@ -463,7 +469,7 @@ static inline void htp_mm_hmx_vtcm_layout_build(
         size_t off_group_a = 0;
         VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data
         VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_size);
-        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_src2, hex_align_up(src2_size, HTP_MM_HMX_TILE_SIZE), src2_size > 0);
+        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE), bias_size > 0);
 
         // Group B: Compute-only buffers (starts at off_group_a)
         size_t off_group_b = off_group_a;
@@ -491,7 +497,7 @@ static inline void htp_mm_hmx_vtcm_layout_build(
         L->scratch_bytes[0]  = scratch0_size;
         L->scratch_bytes[1]  = scratch1_size;
         L->act_head_stride   = 0;
-        L->src2_bytes        = src2_size;
+        L->bias_bytes        = bias_size;
 
         off = off_group_a + hex_smax(group_b_size, group_c_size);
     }
@@ -504,21 +510,20 @@ static inline void htp_mm_hvx_vtcm_layout_build(
     int kernel_type,
     int wtype,
     uint32_t ne10,       // k
-    uint32_t src1_nrows, // m_total
+    uint32_t act_nrows,  // m_total
     uint32_t n_threads,
     size_t dst_row_size,
     size_t src0_row_size,
-    size_t src1_row_size,
-    size_t src2_row_size,
+    size_t act_row_size,
+    size_t bias_row_size,
     uint32_t n_prefetch,
     bool is_matmul_id,
     bool is_fused_nx
 ) {
-    (void)src1_row_size;
+    (void)act_row_size;
     size_t src0_sz    = 0;
-    size_t src1_sz    = 0;
-    size_t src2_sz    = src2_row_size > 0 ? htp_mm_round_up(src2_row_size, 128) : 0;
-    size_t src3_sz    = 0;
+    size_t act_sz     = 0;
+    size_t bias_sz    = bias_row_size > 0 ? htp_mm_round_up(bias_row_size, 128) : 0;
     size_t dst_sz     = 0;
     size_t act_raw_sz = 0;
 
@@ -544,22 +549,21 @@ static inline void htp_mm_hvx_vtcm_layout_build(
         }
 
         size_t tiled_act_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
-        size_t act_sz = hex_round_up(tiled_act_row_size * src1_nrows, 128);
+        size_t q_act_sz = hex_round_up(tiled_act_row_size * act_nrows, 128);
         size_t raw_row_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
 
         src0_sz    = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
-        src1_sz    = act_sz;                           // quantized activation buffer
-        src2_sz    = 0;
-        src3_sz    = 0;
+        act_sz     = q_act_sz;                         // quantized activation buffer
+        bias_sz    = 0;
         dst_sz     = 0;
-        act_raw_sz = hex_round_up(raw_row_size * src1_nrows, 128);
+        act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
     } else if (is_matmul_id) {
         const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
-        const size_t src1_row_size_tiled = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10)
-                                                                                               : htp_mm_q8_0_tiled_row_size(ne10);
+        const size_t act_row_size_tiled = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10)
+                                                                          : htp_mm_q8_0_tiled_row_size(ne10);
 
         size_t src0_sz_per_thread = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
-        src1_sz                   = htp_mm_round_up(src1_row_size_tiled * src1_nrows, 256);
+        act_sz                    = htp_mm_round_up(act_row_size_tiled * act_nrows, 256);
 
         if (is_repack) {
             const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
@@ -573,25 +577,24 @@ static inline void htp_mm_hvx_vtcm_layout_build(
 
         src0_sz    = src0_sz_per_thread * n_threads;
         dst_sz     = 0;
-        src2_sz    = 0;
-        src3_sz    = 0;
-        act_raw_sz = hex_round_up(raw_row_size * src1_nrows, 128);
+        bias_sz    = 0;
+        act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
     } else {
         const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
-        const size_t dst_nrows = (src1_nrows > 1) ? 0 : 1;
+        const size_t dst_nrows = (act_nrows > 1) ? 0 : 1;
 
         switch (kernel_type) {
             case HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
-                size_t f16_src1_row_size = htp_mm_round_up(ne10 * 2, 128);
-                src1_sz    = htp_mm_round_up(f16_src1_row_size * src1_nrows, 256);
+                size_t f16_act_row_size = htp_mm_round_up(ne10 * 2, 128);
+                act_sz     = htp_mm_round_up(f16_act_row_size * act_nrows, 256);
                 src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                 dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
-                act_raw_sz = hex_round_up(hex_round_up(ne10 * sizeof(float), 128) * src1_nrows, 128);
+                act_raw_sz = hex_round_up(hex_round_up(ne10 * sizeof(float), 128) * act_nrows, 128);
                 break;
             }
             case HTP_MM_KERNEL_HVX_F32_F32_VTCM: {
-                size_t f32_src1_row_size = htp_mm_round_up(ne10 * 4, 128);
-                src1_sz    = htp_mm_round_up(f32_src1_row_size * src1_nrows, 256);
+                size_t f32_act_row_size = htp_mm_round_up(ne10 * 4, 128);
+                act_sz     = htp_mm_round_up(f32_act_row_size * act_nrows, 256);
                 src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                 dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                 act_raw_sz = 0;
@@ -599,10 +602,10 @@ static inline void htp_mm_hvx_vtcm_layout_build(
             }
             case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
             case HTP_MM_KERNEL_HVX_QUANT_ROW: {
-                size_t q_src1_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
+                size_t q_act_row_size = htp_mm_weight_has_offset(wtype) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
 
                 src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
-                src1_sz = htp_mm_round_up(q_src1_row_size * src1_nrows, 256);
+                act_sz  = htp_mm_round_up(q_act_row_size * act_nrows, 256);
 
                 src0_sz = src0_sz * n_threads;
 
@@ -614,10 +617,10 @@ static inline void htp_mm_hvx_vtcm_layout_build(
                     src0_sz = repacked_vtcm_size * n_threads;
                 }
 
-                size_t dst_slice_per_thread = (dst_nrows > 0 && src1_nrows == 1) ? htp_mm_round_up((dst_row_size + n_threads - 1) / n_threads, 128) : 0;
+                size_t dst_slice_per_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_round_up((dst_row_size + n_threads - 1) / n_threads, 128) : 0;
                 dst_sz = dst_slice_per_thread * n_threads;
                 size_t raw_row_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
-                act_raw_sz = hex_round_up(raw_row_size * src1_nrows, 128);
+                act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
                 break;
             }
             default:
@@ -627,9 +630,8 @@ static inline void htp_mm_hvx_vtcm_layout_build(
 
     // Group A: Persistent buffers across chunk compute
     size_t off_group_a = 0;
-    VTCM_LAYOUT_ALLOC(off_group_a, off_src1, src1_sz);
-    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_src2, src2_sz, src2_sz > 0);
-    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_src3, src3_sz, src3_sz > 0);
+    VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_sz);
+    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, bias_sz, bias_sz > 0);
 
     // Group B: Compute-only buffers (starts at off_group_a)
     size_t off_group_b = off_group_a;
@@ -643,9 +645,8 @@ static inline void htp_mm_hvx_vtcm_layout_build(
     const size_t group_c_size = off_group_c - off_group_a;
 
     L->src0_bytes    = src0_sz;
-    L->src1_bytes    = src1_sz;
-    L->src2_bytes    = src2_sz;
-    L->src3_bytes    = src3_sz;
+    L->act_bytes     = act_sz;
+    L->bias_bytes    = bias_sz;
     L->dst_bytes     = dst_sz;
     L->act_raw_bytes = act_raw_sz;
     L->total_bytes   = off_group_a + hex_smax(group_b_size, group_c_size);
@@ -655,12 +656,12 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
     int kernel_type,
     int wtype,
     uint32_t ne10,
-    uint32_t src1_nrows,
+    uint32_t act_nrows,
     uint32_t n_threads,
     size_t dst_row_size,
     size_t src0_row_size,
-    size_t src1_row_size,
-    size_t src2_row_size,
+    size_t act_row_size,
+    size_t bias_row_size,
     uint32_t n_prefetch,
     size_t vtcm_budget,
     struct htp_mm_hvx_vtcm_layout * L_out,
@@ -668,17 +669,17 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
 ) {
     struct htp_mm_hvx_vtcm_layout L;
     htp_mm_hvx_vtcm_layout_build(
-        &L, kernel_type, wtype, ne10, src1_nrows, n_threads,
-        dst_row_size, src0_row_size, src1_row_size, src2_row_size, n_prefetch, false, false
+        &L, kernel_type, wtype, ne10, act_nrows, n_threads,
+        dst_row_size, src0_row_size, act_row_size, bias_row_size, n_prefetch, false, false
     );
 
     if (L.total_bytes <= vtcm_budget) {
         *L_out = L;
-        *m_chunk_out = src1_nrows;
+        *m_chunk_out = act_nrows;
         return true;
     }
 
-    const size_t fixed_bytes = L.src0_bytes + L.src2_bytes + L.dst_bytes;
+    const size_t fixed_bytes = L.src0_bytes + L.bias_bytes + L.dst_bytes;
     if (vtcm_budget <= fixed_bytes) {
         return false;
     }
@@ -707,8 +708,8 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
     if (m_chunk > 1) {
         m_chunk &= ~1U;
     }
-    if (m_chunk > src1_nrows) {
-        m_chunk = src1_nrows;
+    if (m_chunk > act_nrows) {
+        m_chunk = act_nrows;
     }
     if (m_chunk < 1) {
         return false;
@@ -716,14 +717,14 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
 
     htp_mm_hvx_vtcm_layout_build(
         &L, kernel_type, wtype, ne10, m_chunk, n_threads,
-        dst_row_size, src0_row_size, src1_row_size, src2_row_size, n_prefetch, false, false
+        dst_row_size, src0_row_size, act_row_size, bias_row_size, n_prefetch, false, false
     );
 
     while (m_chunk > 2 && L.total_bytes > vtcm_budget) {
         m_chunk -= 2;
         htp_mm_hvx_vtcm_layout_build(
             &L, kernel_type, wtype, ne10, m_chunk, n_threads,
-            dst_row_size, src0_row_size, src1_row_size, src2_row_size, n_prefetch, false, false
+            dst_row_size, src0_row_size, act_row_size, bias_row_size, n_prefetch, false, false
         );
     }
 
@@ -737,18 +738,18 @@ static inline bool htp_mm_hvx_solve_vtcm_params(
 }
 
 static inline size_t htp_mm_hmx_get_2d_vtcm_size(
-    int wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_size, size_t src2_size
+    int wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_size, size_t bias_size
 ) {
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_threads, aligned_tile_size, src2_size);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_threads, aligned_tile_size, bias_size);
     return L.total_bytes;
 }
 
 static inline size_t htp_mm_hmx_get_batched_vtcm_size(
-    int wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool pipeline, uint32_t act_threads, size_t src2_size) {
+    int wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool pipeline, uint32_t act_threads, size_t bias_size) {
     (void)pipeline;
     struct htp_mm_hmx_vtcm_layout L;
-    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, wtype, k, mc, nc, group_size, false, act_threads, 0, src2_size);
+    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_F16_BATCHED, wtype, k, mc, nc, group_size, false, act_threads, 0, bias_size);
     return L.total_bytes;
 }
 
@@ -760,7 +761,7 @@ static inline bool htp_mm_hmx_solve_batched_params(
     uint32_t group_size,
     int n_threads,
     bool pipeline,
-    size_t src2_size,
+    size_t bias_size,
     size_t vtcm_budget,
     size_t * m_chunk_out,
     size_t * n_chunk_out,
@@ -775,7 +776,7 @@ static inline bool htp_mm_hmx_solve_batched_params(
 
     int act_threads = n_threads;
     while (act_threads >= 1) {
-        size_t group_overhead = htp_mm_hmx_get_batched_overhead() + (src2_size > 0 ? hex_align_up(src2_size, HTP_MM_HMX_TILE_SIZE) : 0);
+        size_t group_overhead = htp_mm_hmx_get_batched_overhead() + (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0);
         size_t group_size_per_n, group_size_per_m, group_size_per_mn;
         htp_mm_hmx_get_batched_chunk_costs(k, group_size, &group_size_per_n, &group_size_per_m, &group_size_per_mn);
 
@@ -785,8 +786,8 @@ static inline bool htp_mm_hmx_solve_batched_params(
 
         if (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_size_per_n, group_size_per_m, group_size_per_mn, hex_align_up(ne11, 32), ne01_padded,
                                (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
-                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
-            size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, group_size, pipeline, act_threads, src2_size);
+                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
+            size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, group_size, pipeline, act_threads, bias_size);
             if (exact_size <= vtcm_budget) {
                 size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                 if (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
@@ -826,7 +827,7 @@ static inline bool htp_mm_hmx_solve_2d_params(
     bool pipeline,
     bool is_matmul_id,
     uint32_t aligned_tile_size,
-    size_t src2_size,
+    size_t bias_size,
     size_t vtcm_budget,
     size_t * m_chunk_out,
     size_t * n_chunk_out,
@@ -843,7 +844,7 @@ static inline bool htp_mm_hmx_solve_2d_params(
 
     int act_threads = n_threads;
     while (act_threads >= 1) {
-        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (src2_size > 0 ? hex_align_up(src2_size, HTP_MM_HMX_TILE_SIZE) : 0);
+        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0);
         size_t simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn;
         htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_size, &simple_2d_size_per_n, &simple_2d_size_per_m, &simple_2d_size_per_mn);
 
@@ -854,7 +855,7 @@ static inline bool htp_mm_hmx_solve_2d_params(
         if (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn, m_for_chunks, ne01_padded,
                                (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
-            size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_size, src2_size);
+            size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_size, bias_size);
             if (exact_size <= vtcm_budget) {
                 size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                 if (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
diff --git src/ggml-hexagon/htp/pool-ops.c src/ggml-hexagon/htp/pool-ops.c
new file mode 100644
index 00000000..ea1f1a59
--- /dev/null
+++ src/ggml-hexagon/htp/pool-ops.c
@@ -0,0 +1,399 @@
+#pragma clang diagnostic ignored "-Wunused-variable"
+
+#include <float.h>
+#include <HAP_farf.h>
+
+#include "hex-common.h"
+#include "dma-queue.h"
+#include "hex-profile.h"
+#include "htp-ctx.h"
+#include "htp-ops.h"
+#include "htp-tensor.h"
+#include "hvx-inverse.h"
+#include "hvx-types.h"
+#include "hvx-utils.h"
+#include "pool-ops.h"
+
+#define HTP_POOL_MAX 0
+#define HTP_POOL_AVG 1
+
+// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
+// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
+static void pool_plane_hvx(
+    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
+    const bool is_max = (p->pool_op == HTP_POOL_MAX);
+    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
+    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
+
+    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
+        const uint32_t sy = oy * p->kernel_y;
+        for (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
+            const uint32_t rem = p->dst_x - ox;
+            const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;
+            HVX_Vector acc = seed;
+            for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
+                const float * row = src + (sy + ky) * p->src_x;
+                if (p->kernel_x == 2) {
+                    const HVX_Vector v0 = *(const HVX_UVector *) (row + ox * 2);
+                    const HVX_Vector v1 = *(const HVX_UVector *) (row + ox * 2 + VLEN_FP32);
+                    const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
+                    const HVX_Vector lo = Q6_V_lo_W(deinterleaved);
+                    const HVX_Vector hi = Q6_V_hi_W(deinterleaved);
+                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
+                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
+                } else if (p->kernel_x == 1) {
+                    const HVX_Vector v = *(const HVX_UVector *) (row + ox);
+                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
+                }
+            }
+            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
+        }
+    }
+}
+
+// Narrow exact-tiling path. The input is staged in VTCM with one vector of
+// guard space, so full-width loads are safe even when src_x is below 32.
+static void pool_plane_hvx_narrow(
+    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
+    const bool is_max = (p->pool_op == HTP_POOL_MAX);
+    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
+    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
+
+    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
+        const uint32_t sy = oy * p->kernel_y;
+        HVX_Vector acc = seed;
+        for (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
+            const float * row = src + (sy + ky) * p->src_x;
+            const HVX_Vector v0 = *(const HVX_UVector *) row;
+            HVX_Vector v = v0;
+            if (p->kernel_x == 2) {
+                // The second vector is zero because a narrow row has fewer
+                // than 32 input elements. The low lanes still contain the
+                // complete even/odd pairs needed by the output.
+                const HVX_Vector zero = Q6_V_vsplat_R(0);
+                const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
+                const HVX_Vector even = Q6_V_lo_W(deinterleaved);
+                const HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
+                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
+                           : hvx_vec_add_f32_f32(even, odd);
+            }
+            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
+                         : hvx_vec_add_f32_f32(acc, v);
+        }
+        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * sizeof(float),
+                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
+    }
+}
+
+static void pool_plane_global(
+    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
+    const uint32_t n = p->src_x * p->src_y;
+    float val;
+    if (p->pool_op == HTP_POOL_MAX) {
+        val = hvx_reduce_max_f32((const uint8_t *) src, n);
+    } else {
+        val = hvx_reduce_sum_f32((const uint8_t *) src, n) * p->inv_kernel_area;
+    }
+    hvx_vec_store_u(dst, sizeof(float), hvx_vec_splat_f32(val));
+}
+
+static void pool_plane_block(
+    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
+    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
+        const float * row = src + oy * p->src_x;
+        float * dst_row = dst + oy * p->dst_x;
+        for (uint32_t ox = 0; ox < p->dst_x; ++ox) {
+            const uint8_t * block = (const uint8_t *) (row + ox * p->kernel_x);
+            float val;
+            if (p->pool_op == HTP_POOL_MAX) {
+                val = hvx_reduce_max_f32(block, p->kernel_x);
+            } else {
+                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
+            }
+            hvx_vec_store_u(dst_row + ox, sizeof(float), hvx_vec_splat_f32(val));
+        }
+    }
+}
+
+// General path: arbitrary kernel/stride/padding
+
+// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
+// so it collapses to one valid-ky range per row instead of a per-element check.
+static inline void pool_row_bounds_y(
+    const struct htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
+    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
+    const int32_t lo = -(*iy0);
+    const int32_t hi = (int32_t) p->src_y - *iy0;
+    *ky_lo = (uint32_t) MAX(0, lo);
+    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
+}
+
+static inline void pool_pixel_boundary_vec(
+    const float * src, float * dst_row, const struct htp_pool_2d_kernel_params * p,
+    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
+    const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
+    const int32_t kx_lo = MAX(0, -ix0);
+    const int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);
+
+    if (kx_lo >= kx_hi || ky_lo >= ky_hi) {
+        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
+        hvx_vec_store_u(dst_row + ox, sizeof(float), empty_val);
+        return;
+    }
+
+    const uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
+    const HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
+
+    HVX_Vector acc = mask_identity;
+    for (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
+        const float * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
+        for (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
+            const uint32_t k_rem = valid_kx - k;
+            const uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
+            const HVX_VectorPred q = Q6_Q_vsetq_R(n * sizeof(float));
+            const HVX_Vector raw = *(const HVX_UVector *) (row + ix0 + kx_lo + k);
+            const HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
+            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
+        }
+    }
+
+    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
+    if (!is_max) {
+        HVX_Vector scale_vec;
+        if (p->avg_divide_count) {
+            const uint32_t count = (ky_hi - ky_lo) * valid_kx;
+            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((float) count));
+        } else {
+            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
+        }
+        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
+    }
+    hvx_vec_store_u(dst_row + ox, sizeof(float), reduced);
+}
+
+static inline void pool_row_general_boundary_vec(
+    const float * src, float * dst_row, const struct htp_pool_2d_kernel_params * p,
+    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
+    for (uint32_t ox = ox_start; ox < ox_end; ++ox) {
+        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
+    }
+}
+
+// Vectorized interior loop.
+static inline void pool_row_general_vec(
+    const float * src, float * dst_row, const struct htp_pool_2d_kernel_params * p,
+    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
+    if (ox_start >= ox_end) {
+        return;
+    }
+
+    if (p->stride_x != 1 && p->stride_x != 2) {
+        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
+        return;
+    }
+
+    const HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
+    const HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
+
+    for (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
+        const uint32_t rem = ox_end - ox;
+        const uint32_t nbytes = (rem < VLEN_FP32) ? (rem * sizeof(float)) : VLEN;
+        const int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
+        HVX_Vector acc = seed;
+
+        for (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
+            const float * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
+            if (p->stride_x == 1) {
+                for (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
+                    const HVX_Vector v = *(const HVX_UVector *) (row + ix0 + (int32_t) kx);
+                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
+                }
+            } else {
+                for (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
+                    const HVX_Vector v0 = *(const HVX_UVector *) (row + ix0 + (int32_t) kx);
+                    const HVX_Vector v1 = *(const HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
+                    const HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
+                    const HVX_Vector lo = Q6_V_lo_W(deinterleaved);
+                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
+                    if (kx + 1 < p->kernel_x) {
+                        const HVX_Vector hi = Q6_V_hi_W(deinterleaved);
+                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
+                    }
+                }
+            }
+        }
+        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
+    }
+}
+
+static void pool_plane_general(
+    const float * src, float * dst, const struct htp_pool_2d_kernel_params * p) {
+    const bool is_max = (p->pool_op == HTP_POOL_MAX);
+
+    for (uint32_t oy = 0; oy < p->dst_y; ++oy) {
+        int32_t iy0;
+        uint32_t ky_lo, ky_hi;
+        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
+        float * dst_row = dst + oy * p->dst_x;
+
+        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
+        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
+        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
+    }
+}
+
+typedef void (*pool_plane_fn_t)(const float * src, float * dst, const struct htp_pool_2d_kernel_params * p);
+
+struct pool_2d_context {
+    struct htp_ops_context * octx;
+    const struct htp_pool_2d_kernel_params * kparams;
+    pool_plane_fn_t pool_plane;
+    uint32_t n_threads;
+    uint32_t plane_start;
+    uint32_t plane_count;
+    uint32_t planes_per_thread;
+};
+
+static void pool_2d_thread(unsigned int nth, unsigned int ith, void * data) {
+    struct pool_2d_context * ctx = (struct pool_2d_context *) data;
+    const struct htp_pool_2d_kernel_params * p = ctx->kparams;
+    const struct htp_tensor * src0 = ctx->octx->src[0];
+    const struct htp_tensor * dst = ctx->octx->dst;
+    pool_plane_fn_t pool_plane = ctx->pool_plane;
+    const uint32_t planes_per_thread = ctx->planes_per_thread;
+    const uint32_t first = ctx->plane_start + ith * planes_per_thread;
+    const uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_count);
+
+    if (first >= last) {
+        return;
+    }
+
+    struct htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
+    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];
+
+    const uint32_t src_spad_half = p->src_plane_bytes_aligned;
+    const uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
+    const uint32_t src_bytes_per_thread = 2 * src_spad_half;
+    const uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
+    const size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;
+
+    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
+    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
+    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;
+
+    float * srcb2[2] = { (float *) src_spad, (float *) (src_spad + src_spad_half) };
+    float * dstb2[2] = { (float *) dst_spad, (float *) (dst_spad + dst_spad_half) };
+
+    const uint32_t total = last - first;
+
+    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
+    for (uint32_t i = 0; i < total && i < 2; ++i) {
+        dma_queue_push(dma_queue,
+                       dma_make_data(dst->data, dstb2[i]),
+                       p->dst_plane_bytes, dst_spad_half,
+                       p->dst_plane_bytes, 0);
+
+        const dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
+        dma_queue_push(dma_queue,
+                       dma_make_data(srcb2[i], src_addr),
+                       src_spad_half, p->src_plane_bytes,
+                       p->src_plane_bytes, 1);
+    }
+
+    for (uint32_t i = 0; i < total; ++i) {
+        const uint32_t plane = first + i;
+        const uint32_t buf   = i & 1u;
+        float * srcb = srcb2[buf];
+        float * dstb = dstb2[buf];
+
+        dma_queue_pop(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
+        dma_queue_pop(dma_queue); // input for plane i
+
+        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
+        pool_plane(srcb, dstb, p);
+        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
+
+        const dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
+        dma_queue_push(dma_queue,
+                       dma_make_data(dst_addr, dstb),
+                       p->dst_plane_bytes, dst_spad_half,
+                       p->dst_plane_bytes, 1);
+
+        if (i + 2 < total) {
+            const dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
+            dma_queue_push(dma_queue,
+                           dma_make_data(srcb, next_src_addr),
+                           src_spad_half, p->src_plane_bytes,
+                           p->src_plane_bytes, 1);
+        }
+    }
+
+    dma_queue_flush(dma_queue);
+
+    FARF(HIGH, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
+         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
+         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
+    (void) nth;
+}
+
+int op_pool_2d(struct htp_ops_context * octx) {
+    const struct htp_tensor * src0 = octx->src[0];
+    const struct htp_tensor * dst = octx->dst;
+    const struct htp_pool_2d_kernel_params * p =
+        (const struct htp_pool_2d_kernel_params *) octx->kernel_params;
+
+    if (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
+        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
+        return HTP_STATUS_NO_SUPPORT;
+    }
+
+    uint32_t plane_start = 0;
+    uint32_t plane_count = p->planes;
+    if (octx->ctx->mdev.count > 1) {
+        const uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_LINE_SIZE)) : 1;
+        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
+            plane_count, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
+            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
+        plane_start = range.start;
+        plane_count = range.count;
+    }
+    if (plane_count == 0) {
+        return HTP_STATUS_OK;
+    }
+
+    const uint32_t n_threads = MIN(p->n_threads, plane_count);
+    if (!htp_ops_context_set_n_threads(octx, n_threads)) {
+        return HTP_STATUS_INVAL_PARAMS;
+    }
+
+    const uint32_t planes_per_thread = fastdiv(plane_count + n_threads - 1, &octx->n_threads_div);
+
+    pool_plane_fn_t pool_plane;
+    if (p->global_path) {
+        pool_plane = pool_plane_global;
+    } else if (p->block_path) {
+        pool_plane = pool_plane_block;
+    } else if (p->narrow_path) {
+        pool_plane = pool_plane_hvx_narrow;
+    } else if (p->fast_path) {
+        pool_plane = pool_plane_hvx;
+    } else {
+        pool_plane = pool_plane_general;
+    }
+
+    struct pool_2d_context ctx = {
+        .octx = octx,
+        .kparams = p,
+        .pool_plane = pool_plane,
+        .n_threads = n_threads,
+        .plane_start = plane_start,
+        .plane_count = plane_count,
+        .planes_per_thread = planes_per_thread,
+    };
+    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
+    return HTP_STATUS_OK;
+}
+
+int op_pool_1d(struct htp_ops_context * octx) {
+    return op_pool_2d(octx);
+}
diff --git src/ggml-hexagon/htp/pool-ops.h src/ggml-hexagon/htp/pool-ops.h
new file mode 100644
index 00000000..66b405a8
--- /dev/null
+++ src/ggml-hexagon/htp/pool-ops.h
@@ -0,0 +1,85 @@
+#ifndef HTP_POOL_OPS_H
+#define HTP_POOL_OPS_H
+
+#include <stdint.h>
+#include <stddef.h>
+#include <stdbool.h>
+#include <string.h>
+
+#include "hex-common.h"
+
+struct htp_pool_2d_kernel_params {
+    uint32_t src_x;
+    uint32_t src_y;
+    uint32_t dst_x;
+    uint32_t dst_y;
+    uint32_t kernel_x;
+    uint32_t kernel_y;
+    uint32_t stride_x;
+    uint32_t stride_y;
+    int32_t  pad_x;
+    int32_t  pad_y;
+    uint32_t src_plane_bytes;
+    uint32_t dst_plane_bytes;
+    uint32_t src_plane_bytes_aligned;
+    uint32_t dst_plane_bytes_aligned;
+    uint32_t n_threads;
+    uint32_t planes;
+    uint32_t pool_op;
+    uint32_t fast_path;
+    uint32_t narrow_path;
+    uint32_t global_path;
+    uint32_t block_path;
+    uint32_t avg_divide_count;
+    uint32_t ox_lo;
+    uint32_t ox_hi;
+    float    inv_kernel_area;
+};
+
+#if defined(__cplusplus)
+static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
+#else
+_Static_assert(sizeof(struct htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
+#endif
+
+struct htp_pool_vtcm_layout {
+    size_t   total_bytes;
+    size_t   off_src;
+    size_t   off_dst;
+    size_t   src_bytes_per_thread;
+    size_t   dst_bytes_per_thread;
+    size_t   src_spad_half_size;
+    size_t   dst_spad_half_size;
+};
+
+static inline bool htp_pool_solve_layout(
+    struct htp_pool_vtcm_layout * layout,
+    uint32_t src_x,
+    uint32_t src_y,
+    uint32_t dst_x,
+    uint32_t dst_y,
+    uint32_t n_threads,
+    size_t   vtcm_budget
+) {
+    // Full-plane double buffering (256 bytes guard space for vector loads)
+    const size_t src_plane_bytes   = (size_t) src_x * src_y * sizeof(float);
+    const size_t dst_plane_bytes   = (size_t) dst_x * dst_y * sizeof(float);
+    const size_t src_plane_aligned = hex_round_up((uint32_t) src_plane_bytes + 256, 128);
+    const size_t dst_plane_aligned = hex_round_up((uint32_t) dst_plane_bytes, 128);
+
+    const size_t spad_per_thread = 2 * (src_plane_aligned + dst_plane_aligned);
+    if (spad_per_thread * n_threads > vtcm_budget) {
+        return false;
+    }
+
+    layout->src_spad_half_size   = src_plane_aligned;
+    layout->dst_spad_half_size   = dst_plane_aligned;
+    layout->src_bytes_per_thread = 2 * src_plane_aligned;
+    layout->dst_bytes_per_thread = 2 * dst_plane_aligned;
+    layout->off_src              = 0;
+    layout->off_dst              = layout->src_bytes_per_thread * n_threads;
+    layout->total_bytes          = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
+    return true;
+}
+
+#endif // HTP_POOL_OPS_H
diff --git src/ggml-hexagon/htp/ssm-conv.c src/ggml-hexagon/htp/ssm-conv.c
index 931aa406..0d28c899 100644
--- src/ggml-hexagon/htp/ssm-conv.c
+++ src/ggml-hexagon/htp/ssm-conv.c
@@ -135,48 +135,49 @@ static inline void hvx_ssm_conv_unpack_to_T(const float * raw, float * T, uint32
     }
 }
 
-// HVX 32x32 src0 transpose for prefill: src0 {tile_n, ncs} (VTCM) -> src0_T {ncs, d_inner_tile} (VTCM)
-static inline void transpose_src0_block(const float * src0_block,
-                                        uint32_t      ncs,
-                                        uint32_t      cb_n,
-                                        uint32_t      d_inner_tile,
-                                        float *       src0_T_block_dst,
-                                        uint32_t      cb) {
-    const uint32_t T_TILE = VLEN_FP32;
-
-    HVX_Vector __attribute__((aligned(VLEN))) sub[32];
-
-    for (uint32_t t0 = 0; t0 < ncs; t0 += T_TILE) {
-        const uint32_t t_n = MIN(T_TILE, ncs - t0);
-
-        uint32_t __attribute__((aligned(VLEN))) mask_buf[VLEN_FP32] = { 0 };
-        for (uint32_t k = 0; k < t_n; ++k) {
-            mask_buf[k] = 0xFFFFFFFF;
-        }
-        const HVX_Vector mask = *(const HVX_Vector *) mask_buf;
+// Decode dot product specialization for d_conv == 4: multiply in the raw channel-major layout,
+// then deinterleave the products so each vector holds one tap of 32 channels, and sum.
+// Keeps both operands in DMA layout - no transpose, no scratch.
+static inline void hvx_ssm_conv_decode_4(const float * x, const float * w, float * out, uint32_t n_ch) {
+    for (uint32_t cb = 0; cb < n_ch; cb += VLEN_FP32) {
+        const float * xp = x + cb * 4;
+        const float * wp = w + cb * 4;
 
-        for (uint32_t r = 0; r < cb_n; ++r) {
-            const float * src_row = src0_block + r * ncs + t0;
-            sub[r] = (t_n == T_TILE) ? *(const HVX_UVector *) src_row : Q6_V_vand_VV(*(const HVX_UVector *) src_row, mask);
-        }
-        for (uint32_t r = cb_n; r < T_TILE; ++r) {
-            sub[r] = hvx_vec_splat_f32(0.0f);
-        }
+        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp +  0), *(const HVX_Vector *)(wp +  0));
+        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp + 32), *(const HVX_Vector *)(wp + 32));
+        HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp + 64), *(const HVX_Vector *)(wp + 64));
+        HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(xp + 96), *(const HVX_Vector *)(wp + 96));
 
-        hvx_transpose_32x32_f32(sub);
+        HVX_VectorPair p01 = Q6_W_vdeal_VVR(p1, p0, -4);
+        HVX_VectorPair p23 = Q6_W_vdeal_VVR(p3, p2, -4);
 
-        for (uint32_t r = 0; r < t_n; ++r) {
-            float * dst = src0_T_block_dst + (t0 + r) * d_inner_tile + cb;
-            if (cb_n == T_TILE) {
-                *(HVX_UVector *) dst = sub[r];
-            } else {
-                hvx_vec_store_u(dst, cb_n * sizeof(float), sub[r]);
-            }
-        }
+        HVX_VectorPair q02 = Q6_W_vdeal_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -4);
+        HVX_VectorPair q13 = Q6_W_vdeal_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -4);
+
+        HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(q02), Q6_V_lo_W(q13));
+        HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(q02), Q6_V_hi_W(q13));
+
+        *(HVX_Vector *)(out + cb) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
+    }
+}
+
+// Transpose src0 for prefill: one 32-channel block {32, ncs} (VTCM) -> T {ncs, 32} (VTCM).
+// One VTCM gather per output row: lane r picks channel cb+r, the region bound drops
+// the lanes past the channel tail.
+static inline void hvx_ssm_conv_transpose_block(const float * raw_block,
+                                                float *       T,
+                                                uint32_t      ncs,
+                                                uint32_t      cb_n,
+                                                HVX_Vector    vv) {
+    const size_t   base = (size_t) raw_block;
+    const uint32_t mu   = cb_n * ncs * sizeof(float) - 1;
+
+    for (uint32_t t = 0; t < ncs; ++t) {
+        Q6_vgather_ARMVw((HVX_Vector *) (T + (size_t) t * VLEN_FP32), base + t * sizeof(float), mu, vv);
     }
 }
 
-// Single-row decode worker (n_t == 1)
+// Single-token decode worker (n_t == 1)
 static void ssm_conv_thread_f32_decode(unsigned int nth, unsigned int ith, void * data) {
     struct htp_ssm_conv_context *             scctx   = (struct htp_ssm_conv_context *) data;
     struct htp_ops_context *                  octx    = scctx->octx;
@@ -202,9 +203,11 @@ static void ssm_conv_thread_f32_decode(unsigned int nth, unsigned int ith, void
 
     const uint32_t d_inner_per_thread = ir1 - ir0;
     const uint32_t d_inner_stride     = hex_round_up(d_inner_per_thread, VLEN_FP32);
+    const uint32_t d_inner_tile       = scctx->d_inner_tile;
 
-    const size_t src0_stride_seq_bytes = src0->nb[2];
-    const size_t dst_stride_seq_bytes  = dst->nb[2];
+    const size_t src0_stride_inner_bytes = src0->nb[1];
+    const size_t src0_stride_seq_bytes   = src0->nb[2];
+    const size_t dst_stride_seq_bytes    = dst->nb[2];
 
     uint8_t * src1_spad_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;
     uint8_t * src0_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
@@ -216,57 +219,121 @@ static void ssm_conv_thread_f32_decode(unsigned int nth, unsigned int ith, void
     float * src1_raw = (float *) src1_spad_base;
     float * src1_T   = (float *) (src1_spad_base + weight_raw_size);
 
-    float * src0_raw = (float *) src0_spad_base;
-    float * src0_T   = (float *) (src0_spad_base + weight_raw_size);
+    const size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * d_conv * sizeof(float), 128);
+    const size_t dst_tile_bytes      = hex_round_up(d_inner_tile * sizeof(float), 128);
+
+    float * src0_tile_raw[2] = { (float *) src0_spad_base, (float *) (src0_spad_base + src0_tile_raw_bytes) };
+    float * src0_T           = (float *) (src0_spad_base + 2 * src0_tile_raw_bytes);
 
-    float * dst_spad = (float *) dst_spad_base;
+    float * dst_tile[2] = { (float *) dst_spad_base, (float *) (dst_spad_base + dst_tile_bytes) };
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
-    // 1. Fetch weights src1 from DDR into VTCM via DMA (DMA64-safe)
+    // raw_dot keeps both operands in the DMA layout, so src1 needs no prep pass
+    const bool raw_dot = (d_conv == 4) && (d_inner_per_thread % VLEN_FP32 == 0);
+
+    const uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
+    const uint32_t n_chunks  = n_s * n_tiles;
+    const size_t   row_bytes = d_conv * sizeof(float);
+
+    uint32_t s_fetch        = 0;
+    uint32_t tile_off_fetch = 0;
+
+    #define SSM_CONV_DECODE_PUSH_FETCH(c)                                                         \
+        do {                                                                                      \
+            const uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
+            const dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
+                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
+            dma_queue_push(dma_q,                                                                 \
+                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
+                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
+            tile_off_fetch += d_inner_tile;                                                       \
+            if (tile_off_fetch >= d_inner_per_thread) {                                           \
+                tile_off_fetch = 0;                                                               \
+                s_fetch++;                                                                        \
+            }                                                                                     \
+        } while (0)
+
+    // Queue weights and initial input tiles together so DDR reads overlap
     const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
     dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
-    dma_queue_pop(dma_q);
 
-    // 2. Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
-    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
-    hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
-    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
-
-    const size_t input_bytes  = (size_t) d_inner_per_thread * d_conv * sizeof(float);
-    const size_t output_bytes = (size_t) d_inner_per_thread * sizeof(float);
-
-    // 3. Process each sequence
-    for (uint32_t s = 0; s < n_s; ++s) {
-        const dma_addr_t src0_ddr = src0->data + s * src0_stride_seq_bytes + ir0 * d_conv * sizeof(float);
-        dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_raw, src0_ddr), input_bytes, input_bytes, input_bytes, 1);
-        dma_queue_pop(dma_q);
-
-        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) s);
-        hvx_ssm_conv_unpack_to_T(src0_raw, src0_T, d_inner_per_thread, d_inner_stride, d_conv);
-
-        for (uint32_t cb = 0; cb < d_inner_per_thread; cb += VLEN_FP32) {
-            const uint32_t cb_n = MIN(VLEN_FP32, d_inner_per_thread - cb);
-            HVX_Vector acc = hvx_vec_splat_f32(0.0f);
-            for (uint32_t j = 0; j < d_conv; ++j) {
-                HVX_Vector x = *(const HVX_Vector *)(src0_T + j * d_inner_stride + cb);
-                HVX_Vector w = *(const HVX_Vector *)(src1_T + j * d_inner_stride + cb);
-                acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
-            }
-            HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
-            if (cb_n == VLEN_FP32) {
-                *(HVX_Vector *)(dst_spad + cb) = y;
-            } else {
-                hvx_vec_store_u(dst_spad + cb, cb_n * sizeof(float), y);
+    SSM_CONV_DECODE_PUSH_FETCH(0);
+    if (n_chunks > 1) {
+        SSM_CONV_DECODE_PUSH_FETCH(1);
+    }
+
+    dma_queue_pop(dma_q);  // weights
+
+    if (!raw_dot) {
+        // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
+        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
+        hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
+        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
+    }
+
+    uint32_t i3       = 0;
+    uint32_t tile_off = 0;
+
+    for (uint32_t c = 0; c < n_chunks; ++c) {
+        const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
+
+        if (c >= 2) {
+            dma_queue_pop(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
+        }
+        dma_queue_pop(dma_q);      // fetch chunk c
+
+        float * restrict out = dst_tile[c & 1];
+
+        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
+        if (raw_dot) {
+            const float * xp = src0_tile_raw[c & 1];
+            const float * wp = src1_raw + tile_off * 4;
+            hvx_ssm_conv_decode_4(xp, wp, out, tile_n);
+        } else {
+            const uint32_t tile_stride = hex_round_up(tile_n, VLEN_FP32);
+            hvx_ssm_conv_unpack_to_T(src0_tile_raw[c & 1], src0_T, tile_n, tile_stride, d_conv);
+
+            for (uint32_t cb = 0; cb < tile_n; cb += VLEN_FP32) {
+                const uint32_t cb_n = MIN(VLEN_FP32, tile_n - cb);
+                HVX_Vector acc = hvx_vec_splat_f32(0.0f);
+                for (uint32_t j = 0; j < d_conv; ++j) {
+                    HVX_Vector x = *(const HVX_Vector *)(src0_T + j * tile_stride + cb);
+                    HVX_Vector w = *(const HVX_Vector *)(src1_T + j * d_inner_stride + tile_off + cb);
+                    acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
+                }
+                HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
+                if (cb_n == VLEN_FP32) {
+                    *(HVX_Vector *)(out + cb) = y;
+                } else {
+                    hvx_vec_store_u(out + cb, cb_n * sizeof(float), y);
+                }
             }
         }
-        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) s);
+        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i3);
+
+        const dma_addr_t dst_ddr        = dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * sizeof(float);
+        const size_t     tile_out_bytes = (size_t) tile_n * sizeof(float);
+        dma_queue_push(dma_q, dma_make_data(dst_ddr, (uint8_t *) out),
+                       tile_out_bytes, tile_out_bytes, tile_out_bytes, 1);
 
-        const dma_addr_t dst_ddr = dst->data + s * dst_stride_seq_bytes + ir0 * sizeof(float);
-        dma_queue_push(dma_q, dma_make_data(dst_ddr, (uint8_t *) dst_spad), output_bytes, output_bytes, output_bytes, 1);
-        dma_queue_pop(dma_q);
+        if (c + 2 < n_chunks) {
+            SSM_CONV_DECODE_PUSH_FETCH(c + 2);
+        }
+
+        tile_off += d_inner_tile;
+        if (tile_off >= d_inner_per_thread) {
+            tile_off = 0;
+            i3++;
+        }
+    }
+
+    for (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
+        dma_queue_pop(dma_q);  // drain the last writebacks
     }
 
+    #undef SSM_CONV_DECODE_PUSH_FETCH
+
     FARF(HIGH, "ssm-conv-f32-decode %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
          ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
@@ -318,82 +385,157 @@ static void ssm_conv_thread_f32_prefill(unsigned int nth, unsigned int ith, void
     float * src1_raw = (float *) src1_spad_base;
     float * src1_T   = (float *) (src1_spad_base + weight_raw_size);
 
+    // src0 spad holds two raw tiles (fetch of tile n+1 overlaps compute of tile n) plus
+    // the transposed block. dst spad holds two tiles so a writeback can stay in flight.
     const size_t src0_tile_raw_bytes = hex_round_up(d_inner_tile * ncs * sizeof(float), 128);
-    float * src0_tile_raw = (float *) src0_spad_base;
-    float * src0_T        = (float *) (src0_spad_base + src0_tile_raw_bytes);
+    const size_t dst_tile_bytes      = hex_round_up(d_inner_tile * n_t * sizeof(float), 128);
 
-    float * dst_tile = (float *) dst_spad_base;
+    float * src0_tile_raw[2] = { (float *) src0_spad_base, (float *) (src0_spad_base + src0_tile_raw_bytes) };
+    float * src0_T           = (float *) (src0_spad_base + 2 * src0_tile_raw_bytes);
+
+    float * dst_tile[2] = { (float *) dst_spad_base, (float *) (dst_spad_base + dst_tile_bytes) };
 
     struct htp_thread_trace * tr = &octx->ctx->trace[ith];
 
-    // 1. Fetch weights src1 from DDR into VTCM via DMA (DMA64-safe)
+    const uint32_t n_tiles   = (d_inner_per_thread + d_inner_tile - 1) / d_inner_tile;
+    const uint32_t n_chunks  = n_s * n_tiles;
+    const size_t   row_bytes = ncs * sizeof(float);
+
+    uint32_t s_fetch        = 0;
+    uint32_t tile_off_fetch = 0;
+
+    // Chunk c fetches into src0_tile_raw[c & 1] and writes back from dst_tile[c & 1].
+    // Two fetches run ahead, so the queue order is F0 F1 W0 F2 W1 ... and pops follow it.
+    #define SSM_CONV_PUSH_FETCH(c)                                                                \
+        do {                                                                                      \
+            const uint32_t   cur_tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off_fetch); \
+            const dma_addr_t fetch_ddr  = src0->data + s_fetch * src0_stride_seq_bytes +          \
+                                          (ir0 + tile_off_fetch) * src0_stride_inner_bytes;       \
+            dma_queue_push(dma_q,                                                                 \
+                           dma_make_data((uint8_t *) src0_tile_raw[(c) & 1], fetch_ddr),          \
+                           row_bytes, src0_stride_inner_bytes, row_bytes, cur_tile_n);            \
+            tile_off_fetch += d_inner_tile;                                                       \
+            if (tile_off_fetch >= d_inner_per_thread) {                                           \
+                tile_off_fetch = 0;                                                               \
+                s_fetch++;                                                                        \
+            }                                                                                     \
+        } while (0)
+
+    // Queue weights and initial input tiles together so DDR reads overlap
     const dma_addr_t src1_ddr = src1->data + ir0 * d_conv * sizeof(float);
     dma_queue_push(dma_q, dma_make_data((uint8_t *) src1_raw, src1_ddr), weight_bytes, weight_bytes, weight_bytes, 1);
-    dma_queue_pop(dma_q);
 
-    // 2. Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
-    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
+    SSM_CONV_PUSH_FETCH(0);
+    if (n_chunks > 1) {
+        SSM_CONV_PUSH_FETCH(1);
+    }
+
+    dma_queue_pop(dma_q);  // weights
+
+    // Unpack/transpose src1_raw into src1_T {d_conv, d_inner_stride}
+    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
     hvx_ssm_conv_unpack_to_T(src1_raw, src1_T, d_inner_per_thread, d_inner_stride, d_conv);
-    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);
+    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ir0);
 
     const uint32_t C_TILE = VLEN_FP32;
 
-    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
-        for (uint32_t tile_off = 0; tile_off < d_inner_per_thread; tile_off += d_inner_tile) {
-            const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
+    // gather offsets: lane r reads channel r of the raw tile block
+    uint32_t __attribute__((aligned(VLEN))) gather_off[VLEN_FP32];
+    for (uint32_t r = 0; r < VLEN_FP32; ++r) {
+        gather_off[r] = r * ncs * sizeof(float);
+    }
+    const HVX_Vector vv = *(const HVX_Vector *) gather_off;
+
+    uint32_t i3       = 0;
+    uint32_t tile_off = 0;
+
+    for (uint32_t c = 0; c < n_chunks; ++c) {
+        const uint32_t tile_n = MIN(d_inner_tile, d_inner_per_thread - tile_off);
+
+        const float * restrict raw = src0_tile_raw[c & 1];
+        float * restrict       out = dst_tile[c & 1];
+
+        if (c >= 2) {
+            dma_queue_pop(dma_q);  // writeback of chunk c-2, frees dst_tile[c & 1]
+        }
+        dma_queue_pop(dma_q);      // fetch chunk c
+
+        // Channel block outer, token inner: the taps and the sliding window stay in
+        // registers, so a new output row costs one src0_T load.
+        const uint32_t dst_tile_stride = hex_round_up(tile_n, C_TILE);
+
+        for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
+            const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
 
-            // Fetch src0 chunk from DDR to VTCM via 2D DMA
-            const dma_addr_t src0_tile_ddr = src0->data +
-                i3 * src0_stride_seq_bytes +
-                (ir0 + tile_off) * src0_stride_inner_bytes;
-            const size_t row_bytes = ncs * sizeof(float);
+            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
+            hvx_ssm_conv_transpose_block(raw + (size_t) cb * ncs, src0_T, ncs, cb_n, vv);
+            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) tile_off);
 
-            dma_queue_push(dma_q, dma_make_data((uint8_t *) src0_tile_raw, src0_tile_ddr),
-                           row_bytes, src0_stride_inner_bytes, row_bytes, tile_n);
-            dma_queue_pop(dma_q);
+            const float * restrict wp = src1_T + tile_off + cb;
+            float * restrict       op = out + cb;
 
-            // Transpose src0 chunk in VTCM into {d_inner_tile, ncs} layout
             htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
-            for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
-                const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
-                transpose_src0_block(src0_tile_raw + cb * ncs, ncs, cb_n, d_inner_tile, src0_T, cb);
-            }
+            if (d_conv == 4) {
+                const HVX_Vector w0 = *(const HVX_Vector *) (wp);
+                const HVX_Vector w1 = *(const HVX_Vector *) (wp + d_inner_stride);
+                const HVX_Vector w2 = *(const HVX_Vector *) (wp + 2 * d_inner_stride);
+                const HVX_Vector w3 = *(const HVX_Vector *) (wp + 3 * d_inner_stride);
+
+                HVX_Vector x0 = *(const HVX_Vector *) (src0_T);
+                HVX_Vector x1 = *(const HVX_Vector *) (src0_T + C_TILE);
+                HVX_Vector x2 = *(const HVX_Vector *) (src0_T + 2 * C_TILE);
 
-            // Compute convolution
-            for (uint32_t t = 0; t < n_t; ++t) {
-                for (uint32_t cb = 0; cb < tile_n; cb += C_TILE) {
-                    const uint32_t cb_n = MIN(C_TILE, tile_n - cb);
+                for (uint32_t t = 0; t < n_t; ++t) {
+                    const HVX_Vector x3 = *(const HVX_Vector *) (src0_T + (t + 3) * C_TILE);
 
+                    HVX_Vector a = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x0, w0),
+                                                            Q6_Vqf32_vmpy_VsfVsf(x1, w1));
+                    HVX_Vector b = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, w2),
+                                                            Q6_Vqf32_vmpy_VsfVsf(x3, w3));
+
+                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a, b));
+
+                    x0 = x1;
+                    x1 = x2;
+                    x2 = x3;
+                }
+            } else {
+                for (uint32_t t = 0; t < n_t; ++t) {
                     HVX_Vector acc = hvx_vec_splat_f32(0.0f);
                     for (uint32_t j = 0; j < d_conv; ++j) {
-                        HVX_Vector x = *(const HVX_Vector *) (src0_T + (t + j) * d_inner_tile + cb);
-                        HVX_Vector w = *(const HVX_Vector *) (src1_T + j * d_inner_stride + tile_off + cb);
+                        HVX_Vector x = *(const HVX_Vector *) (src0_T + (t + j) * C_TILE);
+                        HVX_Vector w = *(const HVX_Vector *) (wp + j * d_inner_stride);
                         acc          = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                     }
-
-                    HVX_Vector y = Q6_Vsf_equals_Vqf32(acc);
-                    float * dst_tile_ptr = dst_tile + t * tile_n + cb;
-                    if (cb_n == C_TILE) {
-                        *(HVX_Vector *) dst_tile_ptr = y;
-                    } else {
-                        hvx_vec_store_u(dst_tile_ptr, cb_n * sizeof(float), y);
-                    }
+                    *(HVX_Vector *) (op + t * dst_tile_stride) = Q6_Vsf_equals_Vqf32(acc);
                 }
             }
             htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) tile_off);
+        }
+
+        // Writeback dst_tile VTCM -> DDR via 2D DMA
+        dma_queue_push(dma_q,
+                       dma_make_data(dst->data + i3 * dst_stride_seq_bytes + (ir0 + tile_off) * sizeof(float),
+                                     (uint8_t *) out),
+                       dst_stride_token_bytes, dst_tile_stride * sizeof(float), tile_n * sizeof(float), n_t);
 
-            // Writeback dst_tile from VTCM to DDR via 2D DMA
-            const dma_addr_t dst_tile_ddr = dst->data +
-                i3 * dst_stride_seq_bytes +
-                (ir0 + tile_off) * sizeof(float);
-            const size_t dst_row_bytes = tile_n * sizeof(float);
+        if (c + 2 < n_chunks) {
+            SSM_CONV_PUSH_FETCH(c + 2);
+        }
 
-            dma_queue_push(dma_q, dma_make_data(dst_tile_ddr, (uint8_t *) dst_tile),
-                           dst_stride_token_bytes, dst_row_bytes, dst_row_bytes, n_t);
-            dma_queue_pop(dma_q);
+        tile_off += d_inner_tile;
+        if (tile_off >= d_inner_per_thread) {
+            tile_off = 0;
+            i3++;
         }
     }
 
+    for (uint32_t k = MIN(n_chunks, 2); k > 0; --k) {
+        dma_queue_pop(dma_q);  // drain the last writebacks
+    }
+
+    #undef SSM_CONV_PUSH_FETCH
+
     FARF(HIGH, "ssm-conv-f32-prefill %d/%d: %ux%ux%ux%u (%u:%u) * %ux%ux%ux%u -> %ux%ux%ux%u\n",
          ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0], dst->ne[1],
diff --git src/ggml-hexagon/htp/ssm-conv.h src/ggml-hexagon/htp/ssm-conv.h
index be62d7bf..a071d74b 100644
--- src/ggml-hexagon/htp/ssm-conv.h
+++ src/ggml-hexagon/htp/ssm-conv.h
@@ -12,13 +12,8 @@ struct htp_ssm_conv_kernel_params {
     uint32_t d_inner;
     uint32_t n_t;
     uint32_t n_s;
-    uint32_t d_inner_per_thread;
     uint32_t d_inner_tile;
 
-    uint32_t src0_row_size_aligned;
-    uint32_t src1_row_size_aligned;
-    uint32_t dst_row_size_aligned;
-
     uint32_t vtcm_src0_size_per_thread;
     uint32_t vtcm_src1_size_per_thread;
     uint32_t vtcm_dst_size_per_thread;
@@ -27,8 +22,6 @@ struct htp_ssm_conv_kernel_params {
     uint32_t vtcm_src1_size;
     uint32_t vtcm_dst_size;
     uint32_t vtcm_size;
-
-    struct fastdiv_values div_n_threads;
 };
 
 #if defined(__cplusplus)
diff --git src/ggml-hexagon/htp/unary-ops.c src/ggml-hexagon/htp/unary-ops.c
index b63fd4c2..c557b090 100644
--- src/ggml-hexagon/htp/unary-ops.c
+++ src/ggml-hexagon/htp/unary-ops.c
@@ -497,7 +497,72 @@ static void silu_f32(const void * restrict src,
     }
 }
 
-// gelu(x) = x * sigmoid(1.702 * x)  (quick/sigmoid approximation, matches CPU GELU_QUICK reference)
+// GELU uses the tanh approximation.
+static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
+    const HVX_Vector half = hvx_vec_splat_f32(0.5f);
+    const HVX_Vector one  = hvx_vec_splat_f32(1.0f);
+
+    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
+    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
+    inner = hvx_vec_add_f32_f32(inner, one);
+    inner = hvx_vec_mul_f32_f32(inner, x);
+    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));
+
+    return hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
+                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
+}
+
+static inline void hvx_gelu_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    assert((unsigned long) dst % 128 == 0);
+    assert((unsigned long) src % 128 == 0);
+
+    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
+    HVX_Vector * restrict vsrc = (HVX_Vector *) src;
+
+    const uint32_t nvec = n / VLEN_FP32;
+    const uint32_t nloe = n % VLEN_FP32;
+
+    uint32_t i = 0;
+    _Pragma("unroll(4)")
+    for (; i < nvec; i++) {
+        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
+    }
+    if (nloe) {
+        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_f32(vsrc[i]));
+    }
+}
+
+// GELU_QUICK uses x * sigmoid(1.702 * x).
+static __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
+    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
+    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
+    const HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
+    const HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
+    const HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);
+
+    return hvx_vec_mul_f32_f32(x, sigmoid);
+}
+
+static inline void hvx_gelu_quick_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
+    assert((unsigned long) dst % 128 == 0);
+    assert((unsigned long) src % 128 == 0);
+
+    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
+    HVX_Vector * restrict vsrc = (HVX_Vector *) src;
+
+    const uint32_t nvec = n / VLEN_FP32;
+    const uint32_t nloe = n % VLEN_FP32;
+
+    uint32_t i = 0;
+    _Pragma("unroll(4)")
+    for (; i < nvec; i++) {
+        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
+    }
+    if (nloe) {
+        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_quick_f32(vsrc[i]));
+    }
+}
+
 static void gelu_f32(const void * restrict src,
                      void * restrict dst,
                      const uint32_t num_rows,
@@ -508,9 +573,21 @@ static void gelu_f32(const void * restrict src,
         const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
         uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);
 
-        hvx_mul_scalar_f32(dst_local, src_local, 1.702f, ne0);
-        hvx_sigmoid_f32_aa(dst_local, dst_local, ne0);
-        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
+        hvx_gelu_f32_aa(dst_local, src_local, ne0);
+    }
+}
+
+static void gelu_quick_f32(const void * restrict src,
+                           void * restrict dst,
+                           const uint32_t num_rows,
+                           const struct htp_unary_context * uctx) {
+    htp_unary_op_preamble;
+
+    for (uint32_t ir = 0; ir < num_rows; ir++) {
+        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * src0_row_size_aligned);
+        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);
+
+        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
     }
 }
 
@@ -774,9 +851,12 @@ static void tile_silu_f32(void * restrict dst, const void * restrict src, uint32
 
 static void tile_gelu_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
     (void) uctx;
-    hvx_mul_scalar_f32((uint8_t *) dst, (const uint8_t *) src, 1.702f, tw);
-    hvx_sigmoid_f32_aa((uint8_t *) dst, (uint8_t *) dst, tw);
-    hvx_mul_f32_aaa((uint8_t *) dst, (const uint8_t *) src, (uint8_t *) dst, tw);
+    hvx_gelu_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
+}
+
+static void tile_gelu_quick_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
+    (void) uctx;
+    hvx_gelu_quick_f32_aa((uint8_t *) dst, (const uint8_t *) src, tw);
 }
 
 static void tile_gelu_erf_f32(void * restrict dst, const void * restrict src, uint32_t tw, const struct htp_unary_context * uctx) {
@@ -1533,6 +1613,7 @@ static int execute_op_unary(struct htp_ops_context * octx) {
         case HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                break;
         case HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   break;
         case HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   break;
+        case HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             break;
         case HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               break;
         case HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               break;
         case HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   break;
@@ -1684,6 +1765,7 @@ static int execute_op_unary(struct htp_ops_context * octx) {
             case HTP_OP_UNARY_SIGMOID:   compute_func = (void *) tile_sigmoid_f32;        break;
             case HTP_OP_UNARY_SILU:      compute_func = (void *) tile_silu_f32;           break;
             case HTP_OP_UNARY_GELU:      compute_func = (void *) tile_gelu_f32;           break;
+            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) tile_gelu_quick_f32;     break;
             case HTP_OP_UNARY_GELU_ERF:  compute_func = (void *) tile_gelu_erf_f32;       break;
             case HTP_OP_UNARY_SOFTPLUS:  compute_func = (void *) tile_softplus_f32;       break;
             case HTP_OP_UNARY_TANH:      compute_func = (void *) tile_tanh_f32;           break;
@@ -1731,6 +1813,7 @@ static int execute_op_unary(struct htp_ops_context * octx) {
             case HTP_OP_UNARY_SIGMOID:   compute_func = (void *) sigmoid_f32;             break;
             case HTP_OP_UNARY_SILU:      compute_func = (void *) silu_f32;                break;
             case HTP_OP_UNARY_GELU:      compute_func = (void *) gelu_f32;                break;
+            case HTP_OP_UNARY_GELU_QUICK: compute_func = (void *) gelu_quick_f32;          break;
             case HTP_OP_UNARY_GELU_ERF:  compute_func = (void *) gelu_erf_f32;            break;
             case HTP_OP_UNARY_SOFTPLUS:  compute_func = (void *) softplus_f32;            break;
             case HTP_OP_UNARY_TANH:      compute_func = (void *) tanh_f32;                break;
diff --git src/ggml-hexagon/htp/unary-ops.h src/ggml-hexagon/htp/unary-ops.h
index 7b73cf10..7e5779e5 100644
--- src/ggml-hexagon/htp/unary-ops.h
+++ src/ggml-hexagon/htp/unary-ops.h
@@ -54,6 +54,7 @@ static inline bool htp_op_is_unary(uint32_t opcode) {
         case HTP_OP_UNARY_SIGMOID:
         case HTP_OP_UNARY_SILU:
         case HTP_OP_UNARY_GELU:
+        case HTP_OP_UNARY_GELU_QUICK:
         case HTP_OP_UNARY_GELU_ERF:
         case HTP_OP_UNARY_SOFTPLUS:
         case HTP_OP_UNARY_TANH:
diff --git src/ggml-hip/CMakeLists.txt src/ggml-hip/CMakeLists.txt
index a6a6b727..75d4d60e 100644
--- src/ggml-hip/CMakeLists.txt
+++ src/ggml-hip/CMakeLists.txt
@@ -12,7 +12,7 @@ list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
 list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")
 
 if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
-    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
+    set(CMAKE_HIP_FLAGS_DEBUG "-g -Xarch_device -O2 -Xarch_host -O0")
 endif()
 
 # CMake on Windows doesn't support the HIP language yet
diff --git src/ggml-metal/ggml-metal-common.cpp src/ggml-metal/ggml-metal-common.cpp
index 8b065f54..a00f87c9 100644
--- src/ggml-metal/ggml-metal-common.cpp
+++ src/ggml-metal/ggml-metal-common.cpp
@@ -80,6 +80,22 @@ static bool ggml_metal_mul_mv_mma_type_supported(enum ggml_type type) {
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_K:
         case GGML_TYPE_Q6_K:
+        case GGML_TYPE_BF16:
+        case GGML_TYPE_Q1_0:
+        case GGML_TYPE_Q2_0:
+        case GGML_TYPE_MXFP4:
+        case GGML_TYPE_Q2_K:
+        case GGML_TYPE_Q3_K:
+        case GGML_TYPE_IQ2_XXS:
+        case GGML_TYPE_IQ2_XS:
+        case GGML_TYPE_IQ2_S:
+        case GGML_TYPE_IQ3_XXS:
+        case GGML_TYPE_IQ3_S:
+        case GGML_TYPE_IQ1_S:
+        case GGML_TYPE_IQ1_M:
+        case GGML_TYPE_IQ4_NL:
+        case GGML_TYPE_IQ4_XS:
+        case GGML_TYPE_TQ2_0:
             return true;
         default:
             return false;
@@ -105,10 +121,18 @@ static int64_t ggml_metal_mul_mv_mma_rows_min(enum ggml_type type) {
     switch (type) {
         case GGML_TYPE_F32:
             return 6;
+        case GGML_TYPE_TQ2_0:
+            return 5;
+        case GGML_TYPE_BF16:
+            return 4;
         case GGML_TYPE_F16:
+        case GGML_TYPE_Q2_0:
         case GGML_TYPE_Q4_K:
         case GGML_TYPE_Q5_0:
         case GGML_TYPE_Q5_1:
+        case GGML_TYPE_Q2_K:
+        case GGML_TYPE_IQ4_NL:
+        case GGML_TYPE_MXFP4:
             return 3;
         default:
             return 2;
diff --git src/ggml-metal/ggml-metal-ops.cpp src/ggml-metal/ggml-metal-ops.cpp
index dfe46bac..597d5300 100644
--- src/ggml-metal/ggml-metal-ops.cpp
+++ src/ggml-metal/ggml-metal-ops.cpp
@@ -2476,7 +2476,8 @@ static int ggml_metal_op_mul_mat_mma(ggml_metal_op_t ctx, int idx) {
 
     if (fuse_add) {
         dst = ctx->node(idx + n_fuse - 1);
-        res = dst->src[0]->op == GGML_OP_MUL_MAT ? dst->src[1] : dst->src[0];
+        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
+        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
     }
 
     auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_add);
@@ -3088,7 +3089,9 @@ static bool ggml_metal_op_flash_attn_ext_use_kv_f16(const ggml_tensor * op) {
     // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
     // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
     // TODO: tune per device
-    if (op->src[0]->ne[1] < 32) {
+    // large heads need the upfront dequant to fit the non-vec threadgroup memory
+    if (op->src[0]->ne[1] < 32 &&
+        (op->src[0]->ne[0] < 512 || ggml_metal_op_flash_attn_ext_use_vec(op))) {
         return false;
     }
 
@@ -3827,6 +3830,8 @@ int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
         ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
         ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);
 
+        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);
+
         ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
 
         ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
diff --git src/ggml-metal/kernels/mul_mv_mma.metal src/ggml-metal/kernels/mul_mv_mma.metal
index fcf981a9..5cf245d5 100644
--- src/ggml-metal/kernels/mul_mv_mma.metal
+++ src/ggml-metal/kernels/mul_mv_mma.metal
@@ -605,4 +605,23 @@ MUL_MV_MMA_GEN("q5_1", block_q5_1, 2,     dequantize_q5_1)
 MUL_MV_MMA_GEN("q4_K", block_q4_K, QK_NL, dequantize_q4_K)
 MUL_MV_MMA_GEN("q6_K", block_q6_K, QK_NL, dequantize_q6_K)
 
+#if defined(GGML_METAL_HAS_BF16)
+MUL_MV_MMA_GEN("bf16",    bfloat4x4,     1,     dequantize_bf16)
+#endif
+MUL_MV_MMA_GEN("q1_0",    block_q1_0,    8,     dequantize_q1_0)
+MUL_MV_MMA_GEN("q2_0",    block_q2_0,    4,     dequantize_q2_0)
+MUL_MV_MMA_GEN("mxfp4",   block_mxfp4,   2,     dequantize_mxfp4)
+MUL_MV_MMA_GEN("q2_K",    block_q2_K,    QK_NL, dequantize_q2_K)
+MUL_MV_MMA_GEN("q3_K",    block_q3_K,    QK_NL, dequantize_q3_K)
+MUL_MV_MMA_GEN("iq2_xxs", block_iq2_xxs, QK_NL, dequantize_iq2_xxs)
+MUL_MV_MMA_GEN("iq2_xs",  block_iq2_xs,  QK_NL, dequantize_iq2_xs)
+MUL_MV_MMA_GEN("iq2_s",   block_iq2_s,   QK_NL, dequantize_iq2_s)
+MUL_MV_MMA_GEN("iq3_xxs", block_iq3_xxs, QK_NL, dequantize_iq3_xxs)
+MUL_MV_MMA_GEN("iq3_s",   block_iq3_s,   QK_NL, dequantize_iq3_s)
+MUL_MV_MMA_GEN("iq1_s",   block_iq1_s,   QK_NL, dequantize_iq1_s)
+MUL_MV_MMA_GEN("iq1_m",   block_iq1_m,   QK_NL, dequantize_iq1_m)
+MUL_MV_MMA_GEN("iq4_nl",  block_iq4_nl,  2,     dequantize_iq4_nl)
+MUL_MV_MMA_GEN("iq4_xs",  block_iq4_xs,  QK_NL, dequantize_iq4_xs)
+MUL_MV_MMA_GEN("tq2_0",   block_tq2_0,   QK_NL, dequantize_tq2_0)
+
 #undef MUL_MV_MMA_GEN
diff --git src/ggml-opencl/ggml-opencl.cpp src/ggml-opencl/ggml-opencl.cpp
index 4f1f937b..347c7693 100644
--- src/ggml-opencl/ggml-opencl.cpp
+++ src/ggml-opencl/ggml-opencl.cpp
@@ -18864,9 +18864,11 @@ static void ggml_cl_mul_mat_f16_f32_adreno_xmem(
     const int kpack = K / 4;
     const int npack = CEIL_DIV(M, 4);
     const int os = 8;
+    // Pad weights to the 32-row tiles read by the xmem kernel.
+    const int npack_padded = CEIL_DIV(npack, os)*os;
 
     const size_t xmem_bytes = 6144;
-    const size_t weight_bytes = static_cast<size_t>(kpack) * static_cast<size_t>(npack) * 4u * sizeof(cl_half4);
+    const size_t weight_bytes = static_cast<size_t>(kpack) * static_cast<size_t>(npack_padded) * 4u * sizeof(cl_half4);
 
     backend_ctx->prealloc_adreno_xmem_const.allocate(backend_ctx->context, xmem_bytes);
 
@@ -18899,14 +18901,14 @@ static void ggml_cl_mul_mat_f16_f32_adreno_xmem(
     CL_CHECK(clSetKernelArg(prepack, 3, sizeof(int),      &K));
     CL_CHECK(clSetKernelArg(prepack, 4, sizeof(int),      &M));
     CL_CHECK(clSetKernelArg(prepack, 5, sizeof(int),      &kpack));
-    CL_CHECK(clSetKernelArg(prepack, 6, sizeof(int),      &npack));
+    CL_CHECK(clSetKernelArg(prepack, 6, sizeof(int),      &npack_padded));
     CL_CHECK(clSetKernelArg(prepack, 7, sizeof(int),      &os));
     size_t lws = 256;
     size_t max_wg = backend_ctx->get_kernel_workgroup_size(prepack);
     if (lws > max_wg) {
         lws = max_wg;
     }
-    size_t gws = CEIL_DIV(static_cast<size_t>(kpack) * static_cast<size_t>(npack), lws) * lws;
+    size_t gws = CEIL_DIV(static_cast<size_t>(kpack) * static_cast<size_t>(npack_padded), lws) * lws;
     backend_ctx->enqueue_ndrange_kernel(prepack, 1, &gws, &lws, dst);
 
     cl_kernel pack_src = backend_ctx->kernel_adreno_xmem_pack_src_f32;
diff --git src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl
index df9d9aed..0f17df47 100644
--- src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl
+++ src/ggml-opencl/kernels/gemm_xmem_f16_f32_os8.cl
@@ -101,7 +101,7 @@ __kernel void kernel_gemm_xmem_f16_f32_os8(
     const int X = get_group_id(1)*get_local_size(0) + get_local_id(0);
     const int Z = get_group_id(0)*get_local_size(2) + get_local_id(2);
 
-    if (X >= N || Z*8 >= npack) {
+    if (Z*8 >= npack) {
         return;
     }
 
@@ -198,6 +198,11 @@ __kernel void kernel_gemm_xmem_f16_f32_os8(
         r7 += src1.w * weights_cache[15].scdef;
     } while (coord_s < kpack);
 
+    // Keep all lanes active until the subgroup loads and syncs are done.
+    if (X >= N) {
+        return;
+    }
+
     int coord_s_out = Z*8;
     if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r0); coord_s_out++; }
     if (coord_s_out < npack) { write_imageh(dst_img, (int2)(X, coord_s_out), r1); coord_s_out++; }
diff --git src/ggml-openvino/ggml-decoder.cpp src/ggml-openvino/ggml-decoder.cpp
index ae36a095..10e82f3e 100644
--- src/ggml-openvino/ggml-decoder.cpp
+++ src/ggml-openvino/ggml-decoder.cpp
@@ -336,7 +336,9 @@ int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
         }
         if (op_case == 1 && m_is_stateful) {
             // Recurrent convolution and GDN gates retain their rank-4 layout.
-            bool recurrent = src->op == GGML_OP_GET_ROWS && is_recurrent_cache(src->src[0]);
+            // The gathered states may come through a view of the one gather of all states (see build_rs).
+            const auto * gather = src->op == GGML_OP_VIEW ? src->src[0] : src;
+            bool recurrent = gather->op == GGML_OP_GET_ROWS && is_recurrent_cache(gather->src[0]);
             for (int i = 0; i < m_cgraph->n_nodes && !recurrent; ++i) {
                 const auto * consumer = m_cgraph->nodes[i];
                 if (consumer->op == GGML_OP_GATED_DELTA_NET) {
@@ -411,15 +413,19 @@ int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
         break;
     }
     case GGML_OP_GET_ROWS: {
-        if (node->src[1]->op == GGML_OP_VIEW) {
-            // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
-            // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf.
-            // op_case 1/2: active/extra rows of a multi-slot cache
-            // op_case 3/4: active/extra rows of a single-slot cache
-            if (node->src[0]->op == GGML_OP_RESHAPE && node->src[0]->src[0] != nullptr &&
-                is_recurrent_cache(node->src[0]->src[0])) {
-                const bool single_slot = node->src[0]->src[0]->ne[1] == 1;
+        // GET_ROWS gathering recurrent state cache rows via the inp->s_copy index list:
+        // src[0] is a reshape of cache_r/cache_s, src[1] is a view of the s_copy leaf, or the leaf itself
+        // when one gather covers all states (see build_rs).
+        // op_case 1/2: active/extra rows of a multi-slot cache
+        // op_case 3/4: active/extra rows of a single-slot cache
+        if (node->src[0]->op == GGML_OP_RESHAPE && node->src[0]->src[0] != nullptr &&
+            is_recurrent_cache(node->src[0]->src[0])) {
+            const bool single_slot = node->src[0]->src[0]->ne[1] == 1;
+            if (node->src[1]->op == GGML_OP_VIEW) {
                 op_case = (node->src[1]->view_offs == 0 ? 1 : 2) + (single_slot ? 2 : 0);
+            } else if (single_slot) {
+                // a single-slot cache holds exactly the one state the gather selects
+                op_case = 3;
             }
         }
         break;
@@ -579,6 +585,12 @@ int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
                        node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src == node->view_src) {
                 op_case = 4;
                 break;
+            } else if (node->src[0]->src[0]->op == GGML_OP_GET_ROWS && node->src[1] != nullptr &&
+                       node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src != nullptr &&
+                       is_recurrent_cache(node->src[1]->view_src) && node->src[1]->view_src->ne[1] == 1) {
+                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
+                // all states (see build_rs)
+                op_case = 9;
             }
         } else if (node->src[0]->op == GGML_OP_GET_ROWS && node->src[1] != nullptr &&
                    node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src != nullptr &&
@@ -1097,6 +1109,9 @@ ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor * op,
         input_shape = m_is_static ? ov::PartialShape{1, 1, input->ne[1], m_prefill_chunk_size} :
                                     ov::PartialShape{1, 1, -1, -1};
 
+    } else if (is_inp_scale_rows(input, op)) {
+        input_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};
+
     } else if (is_inp_mask(input, op)) {
         // mask
         if (m_is_static) {
@@ -2101,6 +2116,10 @@ void GgmlOvDecoder::compute_node_dynamic_dims() {
                     m_node_dynamic_dims[src] = 0;
                     continue;
                 }
+                if (is_inp_scale_rows(src, node)) {
+                    m_node_dynamic_dims[src] = 1;
+                    continue;
+                }
                 if (node->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !is_stateful() && !m_model_is_splitted) {
                     m_node_dynamic_dims[src] = 1;
                     continue;
@@ -2181,8 +2200,11 @@ void GgmlOvDecoder::compute_node_dynamic_dims() {
                 }
                 if (m_node_dynamic_dims[node] != -1 && dynamic_dim_value != node->ne[m_node_dynamic_dims[node]]) {
                     m_node_dynamic_dims[node] = -1;
-                    GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
-                                  node->name, node->src[0]->name);
+                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
+                    if (ggml_nelements(node) > 0) {
+                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
+                                      node->name, node->src[0]->name);
+                    }
                 }
             }
             break;
@@ -2307,6 +2329,7 @@ void GgmlOvDecoder::compute_node_dynamic_dims() {
         case GGML_OP_DIAG:
         case GGML_OP_TRI:
         case GGML_OP_REPEAT:
+        case GGML_OP_DUP:
         // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
         // DIV/CLAMP are used in the MoE routing-weight normalization
         // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
diff --git src/ggml-openvino/ggml-decoder.h src/ggml-openvino/ggml-decoder.h
index 33b95340..a614d291 100644
--- src/ggml-openvino/ggml-decoder.h
+++ src/ggml-openvino/ggml-decoder.h
@@ -393,6 +393,11 @@ public:
                op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
     }
 
+    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
+    static bool is_inp_scale_rows(const ggml_tensor * tensor, const ggml_tensor * op) {
+        return op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
+    }
+
     static bool is_rope_freqs_weight(const ggml_tensor * tensor, const ggml_tensor * op) {
         return op->op == GGML_OP_ROPE && tensor == op->src[2];
     }
diff --git src/ggml-openvino/ggml-openvino-extra.cpp src/ggml-openvino/ggml-openvino-extra.cpp
index 0257e23d..14a0b038 100644
--- src/ggml-openvino/ggml-openvino-extra.cpp
+++ src/ggml-openvino/ggml-openvino-extra.cpp
@@ -131,6 +131,7 @@ void ggml_openvino_device_config::init() {
         "GGML_OPENVINO_DISABLE_KV_SLICE",
         "GGML_OPENVINO_ENABLE_FALLBACK",
         "GGML_OPENVINO_MANUAL_GQA_ATTN",
+        "GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN",
         "GGML_OPENVINO_MOE_OP",
         "GGML_OPENVINO_MEMORY_OPTIMIZE",
         "GGML_OPENVINO_RELEASE_WEIGHTS",
diff --git src/ggml-openvino/ggml-openvino.cpp src/ggml-openvino/ggml-openvino.cpp
index 87ee096f..22b2f45e 100644
--- src/ggml-openvino/ggml-openvino.cpp
+++ src/ggml-openvino/ggml-openvino.cpp
@@ -1369,6 +1369,10 @@ static ggml_openvino_op_support is_op_supported_case(const ggml_tensor * op) {
         if (op->type == GGML_TYPE_I64) {
             return {false, "CONCAT with I64 type is not supported"};
         }
+        // quantized inputs are dequantized, so the output cannot be written in the quantized type
+        if (ggml_is_quantized(op->type)) {
+            return {false, "CONCAT with quantized type is not supported"};
+        }
         if (ggml_openvino_is_gpu() && op->type == GGML_TYPE_BF16 && has_view_op_input(op)) {
             return {false, "CONCAT with BF16 type and VIEW input is not supported on GPU"};
         }
@@ -1529,6 +1533,13 @@ static ggml_openvino_op_support is_op_supported_case(const ggml_tensor * op) {
         }
         break;
     }
+    case GGML_OP_DUP: {
+        // translated as CONT, so only a plain copy
+        if (op->type != op->src[0]->type || !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op->src[0])) {
+            return {false, "DUP with type conversion or non-contiguous src is not supported"};
+        }
+        break;
+    }
     case GGML_OP_CPY: {
         if (op->src[0]->type != GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_BF16) {
             return {false, "CPY with BF16 src[1] type is not supported"};
@@ -1568,6 +1579,14 @@ static ggml_openvino_op_support is_op_supported_case(const ggml_tensor * op) {
             (op->src[0]->buffer == nullptr || op->src[0]->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
             return {false, "MUL_MAT scalar dot product with non-weight src[0] on GPU is not supported"};
         }
+        // The GPU plugin fails to compile u4 weights with an f16 zero point for some row counts
+        // (clFinish CL_OUT_OF_RESOURCES). Op tests build Q4_1/Q4_K weights in that form; model weights use a
+        // u4 zero point. Op tests check support before allocating, while model loading checks with a dummy
+        // buffer, so only unbound weights are excluded. Remove once the GPU plugin is fixed.
+        if (ggml_openvino_is_gpu() && (op->src[0]->type == GGML_TYPE_Q4_1 || op->src[0]->type == GGML_TYPE_Q4_K) &&
+            op->src[0]->buffer == nullptr) {
+            return {false, "MUL_MAT with unbound Q4_1/Q4_K src[0] on GPU is not supported"};
+        }
         if (op->src[0]->ne[3] != op->src[1]->ne[3] && op->src[0]->ne[3] != 1 && op->src[1]->ne[3] != 1) {
             return {false, "MUL_MAT with incompatible broadcast on ne[3]: src0->ne[3]=" + std::to_string(op->src[0]->ne[3]) +
                            ", src1->ne[3]=" + std::to_string(op->src[1]->ne[3])};
diff --git src/ggml-openvino/openvino/op/fill.cpp src/ggml-openvino/openvino/op/fill.cpp
index db2fecb5..87358e12 100644
--- src/ggml-openvino/openvino/op/fill.cpp
+++ src/ggml-openvino/openvino/op/fill.cpp
@@ -20,7 +20,7 @@ OutputVector translate_fill(const NodeContext & context) {
 
     auto shape = context.get_input_shape(0).to_shape();
 
-    auto val = ov::op::v0::Constant::create(ov::element::f32, {}, {c});
+    auto val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
     auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
         std::vector<int64_t>(shape.begin(), shape.end()));
     auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);
diff --git src/ggml-openvino/openvino/op_table.cpp src/ggml-openvino/openvino/op_table.cpp
index 12a0953e..23b3a2cf 100644
--- src/ggml-openvino/openvino/op_table.cpp
+++ src/ggml-openvino/openvino/op_table.cpp
@@ -37,6 +37,7 @@ std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
         {"GGML_OP_ADD_ID",          op::translate_add_id                           },
         {"GGML_OP_CONCAT",          op::translate_concat                           },
         {"GGML_OP_CONT",            op::translate_cont                             },
+        {"GGML_OP_DUP",             op::translate_cont                             },
         {"GGML_OP_DIV",             op::translate_div                              },
         {"GGML_OP_FILL",            op::translate_fill                             },
         {"GGML_OP_GET_ROWS",        op::translate_get_rows                         },
diff --git src/ggml-openvino/openvino/pass/align_eltwise_ranks.cpp src/ggml-openvino/openvino/pass/align_eltwise_ranks.cpp
new file mode 100644
index 00000000..eae5b2c4
--- /dev/null
+++ src/ggml-openvino/openvino/pass/align_eltwise_ranks.cpp
@@ -0,0 +1,96 @@
+#include "align_eltwise_ranks.h"
+
+#include <numeric>
+#include <openvino/op/add.hpp>
+#include <openvino/op/constant.hpp>
+#include <openvino/op/divide.hpp>
+#include <openvino/op/multiply.hpp>
+#include <openvino/op/sqrt.hpp>
+#include <openvino/op/subtract.hpp>
+#include <openvino/op/unsqueeze.hpp>
+#include <openvino/pass/pattern/op/wrap_type.hpp>
+
+namespace ov {
+namespace frontend {
+namespace ggml {
+namespace pass {
+
+namespace {
+
+// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
+// scaled by the norm weight.
+bool is_rms_norm_output(const ov::Output<ov::Node> & value, int depth = 1) {
+    const auto * node = value.get_node();
+    if (!ov::is_type<ov::op::v1::Multiply>(node)) {
+        return false;
+    }
+    for (const auto & input : node->input_values()) {
+        const auto * src = input.get_node();
+        if (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_input_node_ptr(1))) {
+            return true;
+        }
+        if (depth > 0 && is_rms_norm_output(input, depth - 1)) {
+            return true;
+        }
+    }
+    return false;
+}
+
+}  // namespace
+
+AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
+    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();
+
+    const auto callback = [this](ov::pass::pattern::Matcher & m) {
+        auto node = m.get_match_root();
+        if (node->get_input_size() != 2) {
+            return false;
+        }
+
+        auto lhs = node->input_value(0);
+        auto rhs = node->input_value(1);
+
+        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
+        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
+        // so it is not affected and is left alone.
+        if (ov::is_type<ov::op::v0::Constant>(lhs.get_node()) ||
+            ov::is_type<ov::op::v0::Constant>(rhs.get_node())) {
+            return false;
+        }
+
+        const auto lhs_rank = lhs.get_partial_shape().rank();
+        const auto rhs_rank = rhs.get_partial_shape().rank();
+        if (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
+            return false;
+        }
+
+        const size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
+        const auto & shorter = shorter_idx == 0 ? lhs : rhs;
+
+        // The defect is with the norm output as the higher-rank operand. When the norm output is the
+        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
+        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
+        if (is_rms_norm_output(shorter)) {
+            return false;
+        }
+        const int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());
+
+        std::vector<int64_t> axes(static_cast<size_t>(diff));
+        std::iota(axes.begin(), axes.end(), 0);
+        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
+            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));
+
+        node->input(shorter_idx).replace_source_output(unsqueeze->output(0));
+        register_new_node(unsqueeze);
+        return true;
+    };
+
+    register_matcher(
+        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
+        callback);
+}
+
+}  // namespace pass
+}  // namespace ggml
+}  // namespace frontend
+}  // namespace ov
diff --git src/ggml-openvino/openvino/pass/align_eltwise_ranks.h src/ggml-openvino/openvino/pass/align_eltwise_ranks.h
new file mode 100644
index 00000000..04f59cce
--- /dev/null
+++ src/ggml-openvino/openvino/pass/align_eltwise_ranks.h
@@ -0,0 +1,39 @@
+#pragma once
+
+#include <openvino/pass/matcher_pass.hpp>
+
+namespace ov {
+namespace frontend {
+namespace ggml {
+namespace pass {
+
+// Give a binary eltwise op's two operands the same rank, by unsqueezing leading axes onto the
+// shorter one.
+//
+// This is semantically a no-op -- NUMPY broadcasting already left-pads the lower-rank operand
+// with 1s, and the result rank is max(rank_a, rank_b) either way. It exists purely to work
+// around an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in rank is computed
+// incorrectly once the plugin fuses it as a post-op into an `rms` primitive. gemma-4 dense hits
+// this under stateful execution, where the layer tail adds a rank-3 residual to the rank-4
+// RMS-norm output; the model then degenerates into repeated tokens on GPU while CPU is correct.
+// Equalising the ranks keeps the fusion and makes it compute the right answer.
+//
+// Deliberately a graph pass rather than something the op translators do: rank is load-bearing
+// during translation (several translators and later passes read operand ranks), and rewriting
+// operands mid-translate breaks the attention path. Running after the graph is complete avoids
+// that entirely.
+//
+// Only applies when both operands are real tensors -- a tensor-vs-Constant mismatch is the
+// RMS norm's own eps/rsqrt arithmetic, which folds into the `rms` primitive rather than
+// becoming a fused post-op, and is not affected by the defect. The norm output itself is never
+// unsqueezed: when it is the lower-rank operand (gemma-3), that breaks the fused path instead.
+class AlignEltwiseOperandRanks : public ov::pass::MatcherPass {
+public:
+    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::AlignEltwiseOperandRanks")
+    AlignEltwiseOperandRanks();
+};
+
+}  // namespace pass
+}  // namespace ggml
+}  // namespace frontend
+}  // namespace ov
diff --git src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp
index db8fc613..3041a6ca 100644
--- src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp
+++ src/ggml-openvino/openvino/pass/fuse_moe_compressed.cpp
@@ -321,22 +321,39 @@ FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
     auto gate_up_w_m = any_input();
     auto ids_gate_up_m = any_input();
     auto bgm_fused_m = wrap_type<ov::op::internal::GatherMatmul>({ a_m, gate_up_w_m, ids_gate_up_m, any_input() });
-    auto gu_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
-        { wrap_type<ov::op::v1::Transpose>({ bgm_fused_m, any_input() }), any_input() }) });
+    // Two graph shapes reach here. The rank-4 (stateless) graph restores the batch dim with an
+    // Unsqueeze after the Transpose and may Convert afterwards:
+    //     GatherMatmul -> Transpose -> Unsqueeze -> [Convert] -> Slice
+    // The rank-3 (stateful) graph never drops to a batch dim at all, so there is no Unsqueeze
+    // here - it appears later, between the GEGLU and the down Reshape - and the Convert sits on
+    // the other side of the Transpose:
+    //     GatherMatmul -> Convert -> Transpose -> Slice
+    // optional<T>({a, b}) matches T(a, b) or bare a, so one pattern covers both.
+    auto gu_t_m =
+        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_fused_m }), any_input() });
+    auto gu_u_m =
+        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ gu_t_m, any_input() }) });
 
     auto gate_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_input(), any_input(), any_input(), any_input() });
     auto up_slice_m = wrap_type<ov::op::v8::Slice>({ gu_u_m, any_input(), any_input(), any_input(), any_input() });
     auto gelu_m = wrap_type<ov::op::v7::Gelu>({ gate_slice_m });
     auto geglu_m = wrap_type<ov::op::v1::Multiply>({ gelu_m, up_slice_m });
 
+    // The rank-3 graph inserts the Unsqueeze the gate/up branch lacked right here, before the
+    // Reshape that feeds the down projection; the rank-4 graph goes straight from the GEGLU
+    // into the Reshape.
     auto d_t_m = wrap_type<ov::op::v1::Transpose>(
-        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>({ geglu_m, any_input() }) }),
+        { optional<ov::op::v0::Convert>({ wrap_type<ov::op::v1::Reshape>(
+              { optional<ov::op::v0::Unsqueeze>({ geglu_m, any_input() }), any_input() }) }),
           any_input() });
     auto down_w_m = any_input();
     auto ids_down_m = any_input();
     auto bgm_down_m = wrap_type<ov::op::internal::GatherMatmul>({ d_t_m, down_w_m, ids_down_m, any_input() });
-    auto down_u_m = optional<ov::op::v0::Convert>({ wrap_type<ov::op::v0::Unsqueeze>(
-        { wrap_type<ov::op::v1::Transpose>({ bgm_down_m, any_input() }), any_input() }) });
+    // Same two shapes as the gate/up branch above.
+    auto down_t_m =
+        wrap_type<ov::op::v1::Transpose>({ optional<ov::op::v0::Convert>({ bgm_down_m }), any_input() });
+    auto down_u_m =
+        optional<ov::op::v0::Convert>({ optional<ov::op::v0::Unsqueeze>({ down_t_m, any_input() }) });
 
     // gemma-4 applies an extra per-expert output scale to the down projection before the
     // router-weight multiply (llama-graph.cpp's ffn_down_exps.scale); FuseMoeCompressed's
@@ -421,13 +438,20 @@ FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
         }
         const size_t top_k = ids_pshape[ids_pshape.rank().get_length() - 1].get_length();
 
-        // routing weights arrive as [1, n_tokens, top_k, 1]; the op wants [..., top_k]
-        auto routing = pm.at(routing_m);
-        const auto routing_pshape = routing.get_partial_shape();
-        if (routing_pshape.rank().is_dynamic() || routing_pshape.rank().get_length() != 4 ||
-            routing_pshape[3] != 1) {
+        // Routing weights arrive as [1, n_tokens, top_k, 1] on the rank-4 (stateless) graph and
+        // as [n_tokens, top_k, 1] on the rank-3 (stateful) one, which is the same thing without
+        // the leading batch dim. Normalise the rank-3 form up to the rank-4 one so everything
+        // below - and the op's own config - stays in the shape that is already validated on the
+        // stateless path; the batch dim is taken back off the result at the end.
+        const auto routing_in = pm.at(routing_m);
+        const auto routing_pshape = routing_in.get_partial_shape();
+        const auto routing_rank = routing_pshape.rank();
+        if (routing_rank.is_dynamic() || (routing_rank.get_length() != 4 && routing_rank.get_length() != 3) ||
+            routing_pshape[routing_rank.get_length() - 1] != 1) {
             return false;
         }
+        const bool batchless = routing_rank.get_length() == 3;
+
         // Fold gemma-4's per-expert output scale into the routing weights: the reduction is
         // sum_e(routing[e] * scale[e] * down_out[e]), and MOECompressed only takes one
         // per-expert weight, so pre-multiply it into routing here (same [.., top_k, 1] shape).
@@ -435,7 +459,11 @@ FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
         if (down_scale.get_partial_shape() != routing_pshape) {
             return false;
         }
-        routing = std::make_shared<ov::op::v1::Multiply>(routing, down_scale);
+        ov::Output<ov::Node> routing = std::make_shared<ov::op::v1::Multiply>(routing_in, down_scale);
+        if (batchless) {
+            routing = std::make_shared<ov::op::v0::Unsqueeze>(
+                routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
+        }
         routing = std::make_shared<ov::op::v0::Squeeze>(
             routing, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 3 }));
         if (ids_pshape.rank().get_length() == 2) {
@@ -494,10 +522,19 @@ FuseMoeCompressedFusedGateUp::FuseMoeCompressedFusedGateUp() {
         auto moe = std::make_shared<ov::op::internal::MOECompressed>(args, config);
 
         ov::Output<ov::Node> result = moe->output(0);
+        // The op was fed the batched form, so it produces [1, n_tokens, hidden]. On the rank-3
+        // graph the ReduceSum being replaced is [n_tokens, hidden], so drop the batch dim again.
+        if (batchless) {
+            result = std::make_shared<ov::op::v0::Squeeze>(
+                result, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { 0 }));
+        }
         const auto root_type = m.get_match_root()->get_output_element_type(0);
         if (result.get_element_type() != root_type) {
             result = std::make_shared<ov::op::v0::Convert>(result, root_type);
         }
+        if (result.get_partial_shape() != m.get_match_root()->get_output_partial_shape(0)) {
+            return false;
+        }
 
         result.get_node_shared_ptr()->set_friendly_name(m.get_match_root()->get_friendly_name());
         ov::copy_runtime_info(m.get_matched_nodes(), result.get_node_shared_ptr());
diff --git src/ggml-openvino/openvino/translate_session.cpp src/ggml-openvino/openvino/translate_session.cpp
index 5a3d11f2..ec3596de 100644
--- src/ggml-openvino/openvino/translate_session.cpp
+++ src/ggml-openvino/openvino/translate_session.cpp
@@ -7,6 +7,7 @@
 #include "input_model.h"
 #include "pass/fuse_argsort_topk.h"
 #include "pass/fuse_moe_router.h"
+#include "pass/align_eltwise_ranks.h"
 #include "pass/fuse_moe_compressed.h"
 #include "pass/fuse_to_conv.h"
 #include "pass/kv_state_seq_axis.h"
@@ -498,6 +499,17 @@ std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<M
             manager.register_pass<pass::FuseMoeCompressedFusedGateUp>();
         }
 
+        // Workaround for an OpenVINO GPU-plugin defect: an eltwise op whose operands differ in
+        // rank is computed wrongly once the plugin fuses it as a post-op into `rms`. gemma-4
+        // dense under stateful execution adds a rank-3 residual to the rank-4 norm output and
+        // decodes garbage on GPU while CPU is correct. Equalising the ranks is a no-op for
+        // NUMPY broadcasting and makes the fused path correct.
+        // Remove once the plugin guards that fusion. Opt out with
+        // GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN=1.
+        if (ggml_openvino_is_gpu() && !ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN")) {
+            manager.register_pass<pass::AlignEltwiseOperandRanks>();
+        }
+
         if (ggml_model_decoder->is_stateful()) {
             const auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
             const auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
diff --git src/ggml-openvino/utils.cpp src/ggml-openvino/utils.cpp
index 67a4123c..f07e180a 100644
--- src/ggml-openvino/utils.cpp
+++ src/ggml-openvino/utils.cpp
@@ -676,6 +676,15 @@ ov::Tensor get_ov_input_tensor_static_prefill(const std::shared_ptr<GgmlOvDecode
         return input_tensor;
     }
 
+    if (GgmlOvDecoder::is_inp_scale_rows(ggml_tensor, op)) {
+        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, 1});
+        auto * dst = input_tensor.data<float>();
+        const auto * src = static_cast<const float *>(ggml_tensor->data) + chunk_index * chunk_size;
+        std::copy(src, src + chunk_valid_size, dst);
+        std::fill(dst + chunk_valid_size, dst + chunk_size, 1.0f);
+        return input_tensor;
+    }
+
     if (GgmlOvDecoder::is_inp_mean(ggml_tensor, op)) {
         const size_t n_seqs = ggml_tensor->ne[1];
         const size_t src_stride = ggml_tensor->ne[0];
@@ -1726,6 +1735,25 @@ enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, const std::shared
 }
 }  // namespace
 
+// Nodes on the unselected branches of ggml_build_forward_select() stay in the graph but must not be
+// computed. Keep them out of the OV model, or their inputs become parameters with fixed shapes.
+static ggml_cgraph * get_compute_graph(ggml_cgraph * cgraph, ov_runtime_context & r_ctx) {
+    auto is_skipped = [](const ggml_tensor * node) {
+        return node->op != GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_COMPUTE);
+    };
+    if (std::none_of(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, is_skipped)) {
+        return cgraph;
+    }
+    auto & compute = r_ctx.compute_graphs[cgraph];
+    compute.nodes.clear();
+    std::copy_if(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, std::back_inserter(compute.nodes),
+                 [&](const ggml_tensor * node) { return !is_skipped(node); });
+    compute.graph = *cgraph;
+    compute.graph.nodes = compute.nodes.data();
+    compute.graph.n_nodes = (int) compute.nodes.size();
+    return &compute.graph;
+}
+
 // Both execution paths use two cache levels:
 // 1. Reuse this backend's decoder/request via graph_key and compatibility checks.
 // 2. On a local miss, look up compiled_graph_key in the shared compilation cache,
@@ -1744,6 +1772,7 @@ enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend)
         GGML_ASSERT(ctx->runtime_context != nullptr);
         std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
         std::lock_guard<std::mutex> execution_lock(r_ctx->execution_mutex);
+        cgraph = get_compute_graph(cgraph, *r_ctx);
 
         return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
     } catch (const ov::Exception & e) {
diff --git src/ggml-openvino/utils.h src/ggml-openvino/utils.h
index 491dbadc..c34441f5 100644
--- src/ggml-openvino/utils.h
+++ src/ggml-openvino/utils.h
@@ -119,6 +119,12 @@ struct ov_runtime_context {
     std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;
     size_t stateful_kv_size;
     std::map<std::string, std::string> kv_state_input_name_map;
+    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
+    struct compute_graph {
+        ggml_cgraph graph;
+        std::vector<ggml_tensor *> nodes;
+    };
+    std::unordered_map<const ggml_cgraph *, compute_graph> compute_graphs;
 
     ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}
 
diff --git src/ggml-rpc/ggml-rpc.cpp src/ggml-rpc/ggml-rpc.cpp
index 158a15bb..e9af7122 100644
--- src/ggml-rpc/ggml-rpc.cpp
+++ src/ggml-rpc/ggml-rpc.cpp
@@ -5,9 +5,11 @@
 #include "transport.h"
 
 #include <array>
+#include <chrono>
 #include <cinttypes>
 #include <optional>
 #include <string>
+#include <thread>
 #include <vector>
 #include <queue>
 #include <condition_variable>
@@ -21,7 +23,6 @@
 #include <filesystem>
 #include <algorithm>
 #include <atomic>
-#include <thread>
 
 static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");
 
@@ -77,6 +78,11 @@ enum rpc_cmd {
     RPC_CMD_DEVICE_COUNT,
     RPC_CMD_GRAPH_RECOMPUTE,
     RPC_CMD_MEMSET_TENSOR,
+    RPC_CMD_SET_TENSOR_2D,
+    RPC_CMD_GET_TENSOR_2D,
+    RPC_CMD_COMM_INIT,
+    RPC_CMD_COMM_ALLREDUCE,
+    RPC_CMD_COMM_FREE,
     RPC_CMD_NONE,
     RPC_CMD_COUNT,
 };
@@ -86,6 +92,10 @@ static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");
 // Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
 const size_t HASH_THRESHOLD = 10 * 1024 * 1024;
 
+// Maximum number of graphs cached per device; client and server must use the same value
+// so that both sides clear their caches at the same point in the message stream
+const size_t GRAPH_CACHE_MAX = 1024;
+
 struct rpc_msg_hello_req {
     uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
 };
@@ -202,6 +212,36 @@ struct rpc_msg_get_device_memory_rsp {
 
 struct rpc_msg_graph_recompute_req {
     uint32_t device;
+    uint64_t uid;
+};
+
+struct rpc_msg_get_tensor_2d_req {
+    rpc_tensor tensor;
+    uint64_t offset;
+    uint64_t size;
+    uint64_t n_copies;
+    uint64_t stride;
+};
+
+struct rpc_msg_comm_init_req {
+    uint32_t device;
+    uint32_t rank;
+    uint32_t world;
+    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
+    char     host[64];  // rank > 0: rank 0's host
+};
+
+struct rpc_msg_comm_init_rsp {
+    uint8_t ok;
+};
+
+struct rpc_msg_comm_allreduce_req {
+    uint32_t   device;
+    rpc_tensor tensor;
+};
+
+struct rpc_msg_comm_free_req {
+    uint32_t device;
 };
 
 #pragma pack(pop)
@@ -218,7 +258,6 @@ struct ggml_backend_rpc_device_context {
     uint32_t    device;
     std::string name;
     std::string description;
-    uint64_t    last_graph_uid;
 };
 
 struct ggml_backend_rpc_buffer_type_context {
@@ -232,6 +271,7 @@ struct ggml_backend_rpc_buffer_type_context {
 class rpc_dispatcher;
 struct ggml_backend_rpc_context {
     std::shared_ptr<rpc_dispatcher> dispatcher;
+    std::string                     endpoint;
     uint32_t                        device;
     std::string                     name;
 };
@@ -341,6 +381,17 @@ static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input,
 
 // RPC client-side implementation
 
+// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
+static constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);
+
+static inline void rpc_cpu_relax() {
+#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
+    __asm__ volatile("yield" ::: "memory");
+#else
+    std::this_thread::yield();
+#endif
+}
+
 // Performs HELLO handshake with transport auto-negotiation.
 // Advertises local capabilities via conn_caps; if the server responds with
 // matching capabilities, the socket is upgraded transparently.
@@ -389,6 +440,16 @@ public:
         return true;
     }
 
+    bool try_pop(T* out) {
+        std::unique_lock<std::mutex> lock(mutex);
+        if (interrupted || queue.empty()) {
+            return false;
+        }
+        *out = queue.front();
+        queue.pop();
+        return true;
+    }
+
     void interrupt() {
         std::unique_lock<std::mutex> lock(mutex);
         interrupted = true;
@@ -418,6 +479,9 @@ public:
     void event_synchronize(ggml_backend_event_t event);
     void event_record(ggml_backend_event_t event);
     void synchronize();
+    void busy_spin_acquire();
+    void busy_spin_release();
+    void graph_compute(uint32_t device, const ggml_cgraph * cgraph);
 
     void start(const std::string & endpoint);
     void work();
@@ -439,8 +503,11 @@ private:
         rpc_msg_ptr              msg;
         std::shared_future<void> sf;
     };
+    std::mutex graph_mutex;
+    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
     rpc_msg_queue    queue;
     socket_ptr       sock;
+    std::atomic_uint busy_spin_users = 0;
     std::atomic_bool running;
     std::thread      thread;
 };
@@ -532,6 +599,15 @@ void rpc_dispatcher::synchronize() {
     msg->completion.get_future().wait();
 }
 
+void rpc_dispatcher::busy_spin_acquire() {
+    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
+}
+
+void rpc_dispatcher::busy_spin_release() {
+    const unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
+    GGML_ASSERT(previous > 0);
+}
+
 void rpc_dispatcher::start(const std::string & endpoint) {
     std::string host;
     int port;
@@ -555,9 +631,18 @@ void rpc_dispatcher::start(const std::string & endpoint) {
 }
 
 void rpc_dispatcher::work() {
+    auto last_cmd = std::chrono::steady_clock::now();
     while (running) {
         rpc_msg_ptr msg_ptr;
-        if (!queue.pop(&msg_ptr)) {
+        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
+        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
+                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
+        if (spin) {
+            if (!queue.try_pop(&msg_ptr)) {
+                rpc_cpu_relax();
+                continue;
+            }
+        } else if (!queue.pop(&msg_ptr)) {
             break;
         }
         if (msg_ptr->cmd != RPC_CMD_NONE) {
@@ -570,6 +655,7 @@ void rpc_dispatcher::work() {
             }
         }
         msg_ptr->completion.set_value();
+        last_cmd = std::chrono::steady_clock::now();
     }
 }
 
@@ -625,7 +711,7 @@ static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
     return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
 }
 
-static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const std::shared_ptr<rpc_dispatcher> & dispatcher = nullptr) {
+static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const rpc_dispatcher * dispatcher = nullptr) {
     rpc_tensor result;
     if (!tensor) {
         memset(&result, 0, sizeof(result));
@@ -638,7 +724,7 @@ static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const std::shared
         ggml_backend_buffer_t buffer = tensor->buffer;
         ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
         // ref: https://github.com/ggml-org/llama.cpp/pull/26500
-        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher == dispatcher)) {
+        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
             result.buffer = ctx->remote_ptr;
             result.data = reinterpret_cast<uint64_t>(tensor->data);
         } else {
@@ -740,6 +826,46 @@ static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggm
     ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input, input_size);
 }
 
+static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
+        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
+    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
+    rpc_tensor rpc_tensor = serialize_tensor(tensor);
+    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
+    size_t input_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t) + size*n_copies;
+    uint8_t * input = new uint8_t[input_size]();
+    uint8_t * dest = input;
+    memcpy(dest, &rpc_tensor, sizeof(rpc_tensor));
+    dest += sizeof(rpc_tensor);
+    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
+    memcpy(dest, header, sizeof(header));
+    dest += sizeof(header);
+    for (size_t i = 0; i < n_copies; i++) {
+        memcpy(dest + i*size, (const char *)data + i*stride_data, size);
+    }
+    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
+    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, input_ptr, input_size);
+}
+
+static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
+        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
+    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
+    auto request = std::make_shared<rpc_msg_get_tensor_2d_req>();
+    request->tensor   = serialize_tensor(tensor);
+    request->offset   = offset;
+    request->size     = size;
+    request->n_copies = n_copies;
+    request->stride   = stride_tensor;
+    if (stride_data == size) {
+        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), data, size*n_copies);
+    } else {
+        std::vector<uint8_t> packed(size*n_copies);
+        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), packed.data(), packed.size());
+        for (size_t i = 0; i < n_copies; i++) {
+            memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
+        }
+    }
+}
+
 static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
     ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
     auto request = std::make_shared<rpc_msg_get_tensor_req>();
@@ -785,8 +911,8 @@ static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
     /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
     /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
     /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
-    /* .set_tensor_2d   = */ NULL,
-    /* .get_tensor_2d   = */ NULL,
+    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
+    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
     /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
     /* .clear           = */ ggml_backend_rpc_buffer_clear,
     /* .reset           = */ NULL,
@@ -989,7 +1115,7 @@ static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
     rpc_ctx->dispatcher->synchronize();
 }
 
-static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const std::shared_ptr<rpc_dispatcher> & dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
+static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
     if (tensor == nullptr) {
         return;
     }
@@ -1009,7 +1135,7 @@ static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const s
     tensors.push_back(result);
 }
 
-static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const std::shared_ptr<rpc_dispatcher> & dispatcher, size_t * output_size) {
+static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, size_t * output_size) {
     uint32_t n_nodes = cgraph->n_nodes;
     std::vector<rpc_tensor> tensors;
     std::unordered_set<ggml_tensor*> visited;
@@ -1017,13 +1143,15 @@ static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, co
         add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
     }
     // serialization format:
-    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
+    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
     uint32_t n_tensors = tensors.size();
-    *output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
+    *output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
     uint8_t * output = new uint8_t[*output_size]();
     uint8_t * dest = output;
     memcpy(dest, &device, sizeof(device));
     dest += sizeof(device);
+    memcpy(dest, &cgraph->uid, sizeof(cgraph->uid));
+    dest += sizeof(cgraph->uid);
     memcpy(dest, &n_nodes, sizeof(n_nodes));
     dest += sizeof(n_nodes);
     for (uint32_t i = 0; i < n_nodes; i++) {
@@ -1037,24 +1165,33 @@ static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, co
     return output;
 }
 
-static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
-    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
-    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
-    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;
-
+void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
+    std::lock_guard<std::mutex> lock(graph_mutex);
     GGML_ASSERT(cgraph->n_nodes > 0);
-    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
+    auto & device_graph_uids = graph_uids[device];
+    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
     if (reuse) {
         auto request = std::make_shared<rpc_msg_graph_recompute_req>();
-        request->device = rpc_ctx->device;
-        rpc_ctx->dispatcher->send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
+        request->device = device;
+        request->uid    = cgraph->uid;
+        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
     } else {
-        rpc_dev_ctx->last_graph_uid = cgraph->uid;
+        if (cgraph->uid != 0) {
+            if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
+                device_graph_uids.clear();
+            }
+            device_graph_uids.insert(cgraph->uid);
+        }
         size_t input_size = 0;
-        uint8_t * input = serialize_graph(rpc_ctx->device, cgraph, rpc_ctx->dispatcher, &input_size);
+        uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
         std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
-        rpc_ctx->dispatcher->send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
+        send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
     }
+}
+
+static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
+    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
+    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
     return GGML_STATUS_SUCCESS;
 }
 
@@ -1069,13 +1206,25 @@ static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_eve
     GGML_UNUSED(event);
 }
 
+static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
+        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
+    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
+    GGML_UNUSED(backend);
+}
+
+static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
+        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
+    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
+    GGML_UNUSED(backend);
+}
+
 static ggml_backend_i ggml_backend_rpc_interface = {
     /* .get_name                = */ ggml_backend_rpc_name,
     /* .free                    = */ ggml_backend_rpc_free,
     /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
     /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
-    /* .set_tensor_2d_async     = */ NULL,
-    /* .get_tensor_2d_async     = */ NULL,
+    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
+    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
     /* .cpy_tensor_async        = */ NULL,
     /* .synchronize             = */ ggml_backend_rpc_synchronize,
     /* .graph_plan_create       = */ NULL,
@@ -1123,6 +1272,7 @@ ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
     auto dispatcher = get_dispatcher(endpoint);
     ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
         /* .dispatcher = */ dispatcher,
+        /* .endpoint   = */ endpoint,
         /* .device     = */ device,
         /* .name       = */ dev_name,
     };
@@ -1157,6 +1307,7 @@ public:
     rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
         : backends(std::move(all_backends)), cache_dir(cache_dir) {
         stored_graphs.resize(backends.size());
+        comm_states.resize(backends.size());
     }
     ~rpc_server();
 
@@ -1169,11 +1320,16 @@ public:
     bool buffer_clear(const rpc_msg_buffer_clear_req & request);
     bool memset_tensor(const rpc_msg_memset_tensor_req & request);
     bool set_tensor(const std::vector<uint8_t> & input);
+    bool set_tensor_2d(const std::vector<uint8_t> & input);
     bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
     bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
+    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response);
     bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
     bool graph_compute(const std::vector<uint8_t> & input);
     bool graph_recompute(const rpc_msg_graph_recompute_req & request);
+    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
+    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
+    bool comm_free(const rpc_msg_comm_free_req & request);
     bool init_tensor(const rpc_msg_init_tensor_req & request);
     bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
     bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
@@ -1184,6 +1340,7 @@ public:
     };
 
 private:
+    void sync_all_backends();
     bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
     ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
     ggml_tensor * create_node(uint64_t id,
@@ -1192,11 +1349,23 @@ private:
                               std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);
 
 
+    // pairwise allreduce over a direct connection to the peer server
+    struct comm_state {
+        socket_ptr              peer;
+        uint32_t                rank = 0;
+        uint32_t                world = 0;
+        ggml_backend_buffer_ptr scratch;
+        size_t                  scratch_size = 0;
+        std::vector<uint8_t>    send_buf;
+        std::vector<uint8_t>    recv_buf;
+    };
+
     std::vector<ggml_backend_t> backends;
     const char * cache_dir;
     std::unordered_set<ggml_backend_buffer_t> buffers;
-    // store the last computed graph for each backend
-    std::vector<stored_graph> stored_graphs;
+    // computed graphs cached per backend, keyed by uid
+    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
+    std::vector<comm_state> comm_states;
 };
 
 void rpc_server::hello(rpc_msg_hello_rsp & response) {
@@ -1304,6 +1473,7 @@ bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rp
 }
 
 bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
+    sync_all_backends();
     LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
     ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
     if (buffers.find(buffer) == buffers.end()) {
@@ -1312,8 +1482,10 @@ bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
     }
     // Discard all cached graphs to avoid use-after-free in graph_recompute,
     // since their nodes may hold pointers to the buffer being freed.
-    for (auto & sg : stored_graphs) {
-        sg.graph = nullptr;
+    for (auto & sgs : stored_graphs) {
+        for (auto & sg : sgs) {
+            sg.second.graph = nullptr;
+        }
     }
     ggml_backend_buffer_free(buffer);
     buffers.erase(buffer);
@@ -1321,6 +1493,7 @@ bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
 }
 
 bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
+    sync_all_backends();
     LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
     ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
     if (buffers.find(buffer) == buffers.end()) {
@@ -1332,6 +1505,7 @@ bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
 }
 
 bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
+    sync_all_backends();
     struct ggml_init_params params {
         /*.mem_size   =*/ ggml_tensor_overhead(),
         /*.mem_buffer =*/ NULL,
@@ -1407,13 +1581,20 @@ ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rp
         result->buffer = nullptr;
     }
 
-    if (result->buffer) {
+    if (result->buffer && ggml_nelements(result) > 0) {
         // require that the tensor data does not go beyond the buffer end
         uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
         uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
         uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
-        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
-        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
+        if (tensor->data + tensor_size < tensor->data ||
+            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
+            GGML_LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
+                           "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
+                           __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
+                           result->ne[0], result->ne[1], result->ne[2], result->ne[3],
+                           tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
+            return nullptr;
+        }
     }
 
     result->op = (ggml_op) tensor->op;
@@ -1428,6 +1609,7 @@ ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rp
 
 
 bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
+    sync_all_backends();
     // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
     uint8_t  cache_flag;
     uint64_t offset;
@@ -1482,6 +1664,67 @@ bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
     return true;
 }
 
+bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
+    sync_all_backends();
+    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
+    if (input.size() < sizeof(rpc_tensor) + 4*sizeof(uint64_t)) {
+        return false;
+    }
+    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
+    uint64_t header[4];
+    memcpy(header, input.data() + sizeof(rpc_tensor), sizeof(header));
+    const uint64_t offset   = header[0];
+    const uint64_t size     = header[1];
+    const uint64_t n_copies = header[2];
+    const uint64_t stride   = header[3];
+
+    const uint64_t data_size = input.size() - sizeof(rpc_tensor) - 4*sizeof(uint64_t);
+    if (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
+        return false;
+    }
+
+    struct ggml_init_params params {
+        /*.mem_size   =*/ ggml_tensor_overhead(),
+        /*.mem_buffer =*/ NULL,
+        /*.no_alloc   =*/ true,
+    };
+    ggml_context_ptr ctx_ptr { ggml_init(params) };
+    GGML_ASSERT(ctx_ptr != nullptr);
+    ggml_context * ctx = ctx_ptr.get();
+    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
+    if (tensor == nullptr || tensor->buffer == nullptr) {
+        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
+        return false;
+    }
+    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
+            __func__, (void*)tensor->buffer, tensor->data, offset, size, n_copies, stride);
+
+    // sanitize tensor->data
+    {
+        if (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
+            return false;
+        }
+        const uint64_t span = (n_copies - 1)*stride + size;
+        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
+        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
+
+        if (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
+            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
+                           __func__, in_tensor->data, offset, span, p0, p1);
+            return false;
+        }
+        if (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
+            GGML_LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
+                           __func__, offset, span, ggml_nbytes(tensor));
+            return false;
+        }
+    }
+
+    const void * data = input.data() + sizeof(rpc_tensor) + 4*sizeof(uint64_t);
+    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
+    return true;
+}
+
 bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
     if (!cache_dir) {
         return false;
@@ -1504,6 +1747,7 @@ bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
 
 bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
 {
+    sync_all_backends();
     std::vector<uint8_t> cached_file;
     if (!get_cached_file(request.hash, cached_file)) {
         response.result = 0;
@@ -1545,6 +1789,7 @@ bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rp
 }
 
 bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
+    sync_all_backends();
     struct ggml_init_params params {
         /*.mem_size   =*/ ggml_tensor_overhead(),
         /*.mem_buffer =*/ NULL,
@@ -1580,6 +1825,7 @@ bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
 }
 
 bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
+    sync_all_backends();
     struct ggml_init_params params {
         /*.mem_size   =*/ ggml_tensor_overhead(),
         /*.mem_buffer =*/ NULL,
@@ -1614,7 +1860,56 @@ bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<
     return true;
 }
 
+bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
+    sync_all_backends();
+    struct ggml_init_params params {
+        /*.mem_size   =*/ ggml_tensor_overhead(),
+        /*.mem_buffer =*/ NULL,
+        /*.no_alloc   =*/ true,
+    };
+    ggml_context_ptr ctx_ptr { ggml_init(params) };
+    GGML_ASSERT(ctx_ptr != nullptr);
+    ggml_context * ctx = ctx_ptr.get();
+    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
+    if (tensor == nullptr || tensor->buffer == nullptr) {
+        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
+        return false;
+    }
+    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
+            __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size, request.n_copies, request.stride);
+
+    // sanitize tensor->data
+    {
+        if (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
+            return false;
+        }
+        if (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
+            return false;
+        }
+        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
+        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
+        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
+
+        if (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
+                span > p1 - request.tensor.data - request.offset) {
+            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
+                           __func__, request.tensor.data, request.offset, span, p0, p1);
+            return false;
+        }
+        if (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
+            GGML_LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
+                           __func__, request.offset, span, ggml_nbytes(tensor));
+            return false;
+        }
+    }
+
+    response.resize(request.size * request.n_copies, 0);
+    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
+    return true;
+}
+
 bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
+    sync_all_backends();
     struct ggml_init_params params {
         /*.mem_size   =*/ 2*ggml_tensor_overhead(),
         /*.mem_buffer =*/ NULL,
@@ -1713,8 +2008,8 @@ ggml_tensor * rpc_server::create_node(uint64_t id,
 
 bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
     // serialization format:
-    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
-    if (input.size() < 2*sizeof(uint32_t)) {
+    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
+    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
         return false;
     }
     const uint8_t * src = input.data();
@@ -1724,10 +2019,13 @@ bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
     if (device >= backends.size()) {
         return false;
     }
+    uint64_t uid;
+    memcpy(&uid, src, sizeof(uid));
+    src += sizeof(uid);
     uint32_t n_nodes;
     memcpy(&n_nodes, src, sizeof(n_nodes));
     src += sizeof(n_nodes);
-    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
+    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
         return false;
     }
     const uint64_t * nodes = (const uint64_t *)src;
@@ -1735,19 +2033,26 @@ bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
     uint32_t n_tensors;
     memcpy(&n_tensors, src, sizeof(n_tensors));
     src += sizeof(n_tensors);
-    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
+    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
         return false;
     }
     const rpc_tensor * tensors = (const rpc_tensor *)src;
-    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);
+    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);
+
+    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
+    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
+        stored_graphs[device].clear();
+    }
+    stored_graph sg_tmp;
+    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;
 
     size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
-    if (stored_graphs[device].buffer.size() < buf_size) {
-        stored_graphs[device].buffer.resize(buf_size);
+    if (sg.buffer.size() < buf_size) {
+        sg.buffer.resize(buf_size);
     }
     struct ggml_init_params params = {
         /*.mem_size   =*/ buf_size,
-        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
+        /*.mem_buffer =*/ sg.buffer.data(),
         /*.no_alloc   =*/ true,
     };
     ggml_context_ptr ctx_ptr { ggml_init(params) };
@@ -1778,9 +2083,9 @@ bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
             graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
         }
     }
-    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
+    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
     GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
-    stored_graphs[device].graph = graph;
+    sg.graph = graph;
     return true;
 }
 
@@ -1789,16 +2094,216 @@ bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
     if (device >= backends.size()) {
         return false;
     }
-    if (stored_graphs[device].graph == nullptr) {
+    auto it = stored_graphs[device].find(request.uid);
+    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
+        GGML_LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
         return false;
     }
-    ggml_cgraph * graph = stored_graphs[device].graph;
-    LOG_DBG("[%s] device: %u\n", __func__, device);
-    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
+    ggml_cgraph * graph = it->second.graph;
+    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
+    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
     GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
     return true;
 }
 
+// graph compute is asynchronous; commands that read or write buffer data synchronize first
+void rpc_server::sync_all_backends() {
+    for (ggml_backend_t backend : backends) {
+        ggml_backend_synchronize(backend);
+    }
+}
+
+// The comm link between two servers uses the same caps negotiation as the client HELLO,
+// so it gets the same transport upgrades (e.g. RDMA).
+bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
+    response.ok = 0;
+    if (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
+        return true;
+    }
+    comm_state & state = comm_states[request.device];
+    if (state.peer != nullptr) {
+        response.ok = 1;
+        return true;
+    }
+    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
+    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
+    if (request.rank == 0) {
+        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
+        if (srv == nullptr) {
+            GGML_LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
+            return true;
+        }
+        state.peer = srv->accept();
+        if (state.peer == nullptr) {
+            return true;
+        }
+        if (!state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
+            state.peer = nullptr;
+            return true;
+        }
+        state.peer->get_caps(local_caps);
+        if (!state.peer->send_data(local_caps, sizeof(local_caps))) {
+            state.peer = nullptr;
+            return true;
+        }
+        state.peer->update_caps(remote_caps);
+    } else {
+        const std::string host(request.host, strnlen(request.host, sizeof(request.host)));
+        // rank 0 may not be listening yet, retry for a few seconds
+        for (int i = 0; i < 100 && state.peer == nullptr; i++) {
+            state.peer = socket_t::connect(host.c_str(), request.port);
+            if (state.peer == nullptr) {
+                std::this_thread::sleep_for(std::chrono::milliseconds(50));
+            }
+        }
+        if (state.peer == nullptr) {
+            GGML_LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
+            return true;
+        }
+        state.peer->get_caps(local_caps);
+        if (!state.peer->send_data(local_caps, sizeof(local_caps)) ||
+            !state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
+            state.peer = nullptr;
+            return true;
+        }
+        state.peer->update_caps(remote_caps);
+    }
+    state.rank  = request.rank;
+    state.world = request.world;
+    GGML_LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
+    response.ok = 1;
+    return true;
+}
+
+bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
+    if (request.device >= backends.size()) {
+        return false;
+    }
+    comm_state & state = comm_states[request.device];
+    if (state.peer == nullptr) {
+        GGML_LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
+        return false;
+    }
+    ggml_backend_t backend = backends[request.device];
+
+    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
+    struct ggml_init_params params = {
+        /*.mem_size   =*/ ctx_size,
+        /*.mem_buffer =*/ NULL,
+        /*.no_alloc   =*/ true,
+    };
+    ggml_context_ptr ctx_ptr { ggml_init(params) };
+    GGML_ASSERT(ctx_ptr != nullptr);
+    ggml_context * ctx = ctx_ptr.get();
+    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
+    if (t_dst == nullptr || t_dst->buffer == nullptr) {
+        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
+        return false;
+    }
+    const size_t  nbytes = ggml_nbytes(t_dst);
+    const int64_t ne     = ggml_nelements(t_dst);
+    if (nbytes == 0) {
+        return true;
+    }
+    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
+    // stay f32 since the extra casts and sync cost more than the bytes saved
+    const bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
+    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
+    const size_t need       = wire_bf16 ? 2*nbytes : nbytes;
+    if (state.scratch_size < need) {
+        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
+        state.scratch_size = need;
+    }
+    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
+    state.send_buf.resize(wire_bytes);
+    state.recv_buf.resize(wire_bytes);
+
+    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
+        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
+        t->buffer = state.scratch.get();
+        t->data   = scratch_base + offset;
+        return t;
+    };
+    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
+        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
+        t->op     = GGML_OP_CPY;
+        t->src[0] = src;
+        t->src[1] = dst;
+        t->buffer = dst->buffer;
+        t->data   = dst->data;
+        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
+        return t;
+    };
+    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
+        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
+        graph->nodes[0] = n0;
+        graph->nodes[1] = n1;
+        graph->n_nodes  = n1 != nullptr ? 2 : 1;
+        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
+        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
+    };
+
+    // wait for the pending subgraph that produced this partial
+    ggml_backend_synchronize(backend);
+
+    ggml_tensor * t_wire_send = nullptr;
+    ggml_tensor * t_wire_recv = nullptr;
+    if (wire_bf16) {
+        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
+        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
+        compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
+        ggml_backend_synchronize(backend);
+        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
+    } else {
+        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
+    }
+
+    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
+    if (state.rank == 0) {
+        if (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
+            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
+            return false;
+        }
+    } else {
+        if (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
+            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
+            return false;
+        }
+    }
+
+    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
+    ggml_tensor * t_cast = nullptr;
+    if (wire_bf16) {
+        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
+        t_cast = new_cpy_node(t_wire_recv, t_peer);
+    } else {
+        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
+    }
+
+    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
+    t_red->op     = GGML_OP_ADD;
+    t_red->src[0] = t_dst;
+    t_red->src[1] = t_peer;
+    t_red->buffer = t_dst->buffer;
+    t_red->data   = t_dst->data;
+    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
+
+    if (t_cast != nullptr) {
+        compute_nodes(t_cast, t_red);
+    } else {
+        compute_nodes(t_red, nullptr);
+    }
+    return true;
+}
+
+bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
+    if (request.device >= backends.size()) {
+        return false;
+    }
+    comm_states[request.device] = comm_state();
+    return true;
+}
+
 bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
     uint32_t dev_id = request.device;
     if (dev_id >= backends.size()) {
@@ -1993,6 +2498,30 @@ static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const
                 }
                 break;
             }
+            case RPC_CMD_SET_TENSOR_2D: {
+                std::vector<uint8_t> input;
+                if (!recv_msg(sock, input)) {
+                    return;
+                }
+                if (!server.set_tensor_2d(input)) {
+                    return;
+                }
+                break;
+            }
+            case RPC_CMD_GET_TENSOR_2D: {
+                rpc_msg_get_tensor_2d_req request;
+                if (!recv_msg(sock, &request, sizeof(request))) {
+                    return;
+                }
+                std::vector<uint8_t> response;
+                if (!server.get_tensor_2d(request, response)) {
+                    return;
+                }
+                if (!send_msg(sock, response.data(), response.size())) {
+                    return;
+                }
+                break;
+            }
             case RPC_CMD_SET_TENSOR_HASH: {
                 rpc_msg_set_tensor_hash_req request;
                 if (!recv_msg(sock, &request, sizeof(request))) {
@@ -2065,6 +2594,40 @@ static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const
                 }
                 break;
             }
+            case RPC_CMD_COMM_INIT: {
+                rpc_msg_comm_init_req request;
+                if (!recv_msg(sock, &request, sizeof(request))) {
+                    return;
+                }
+                rpc_msg_comm_init_rsp response;
+                if (!server.comm_init(request, response)) {
+                    return;
+                }
+                if (!send_msg(sock, &response, sizeof(response))) {
+                    return;
+                }
+                break;
+            }
+            case RPC_CMD_COMM_ALLREDUCE: {
+                rpc_msg_comm_allreduce_req request;
+                if (!recv_msg(sock, &request, sizeof(request))) {
+                    return;
+                }
+                if (!server.comm_allreduce(request)) {
+                    return;
+                }
+                break;
+            }
+            case RPC_CMD_COMM_FREE: {
+                rpc_msg_comm_free_req request;
+                if (!recv_msg(sock, &request, sizeof(request))) {
+                    return;
+                }
+                if (!server.comm_free(request)) {
+                    return;
+                }
+                break;
+            }
             case RPC_CMD_GET_DEVICE_MEMORY: {
                 rpc_msg_get_device_memory_req request;
                 if (!recv_msg(sock, &request, sizeof(request))) {
@@ -2294,6 +2857,137 @@ static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg
     }
 }
 
+// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
+// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
+// exchanged between the servers and never passes through the client.
+struct ggml_backend_rpc_comm_context {
+    struct rank_info {
+        std::string                     endpoint;
+        uint32_t                        device;
+        std::shared_ptr<rpc_dispatcher> dispatcher;
+    };
+    std::vector<rank_info> ranks;
+};
+
+static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
+    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
+    if (comm_ctx == nullptr) {
+        return;
+    }
+    for (const auto & rank : comm_ctx->ranks) {
+        auto request = std::make_shared<rpc_msg_comm_free_req>();
+        request->device = rank.device;
+        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, sizeof(*request));
+        rank.dispatcher->busy_spin_release();
+    }
+    delete comm_ctx;
+}
+
+static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
+    if (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
+        return nullptr;
+    }
+    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
+    ranks.reserve(n_backends);
+    for (size_t i = 0; i < n_backends; i++) {
+        if (!ggml_backend_is_rpc(backends[i])) {
+            return nullptr;
+        }
+        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
+        // one rank per endpoint: a server processes its socket sequentially, so a second
+        // COMM_INIT on the same connection would deadlock behind the first
+        for (const auto & rank : ranks) {
+            if (rank.endpoint == rpc_ctx->endpoint) {
+                GGML_LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
+                return nullptr;
+            }
+        }
+        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
+    }
+
+    // TODO: simplify this logic
+    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
+    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
+    std::string host0;
+    int port0;
+    if (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
+        return nullptr;
+    }
+    const uint32_t comm_port = (uint32_t) port0 + 1000;
+    if (host0.size() >= 64) {
+        return nullptr;
+    }
+
+    for (const auto & rank : ranks) {
+        rank.dispatcher->busy_spin_acquire();
+    }
+
+    // Send all init requests before reading any response: rank 0 blocks in accept
+    // until rank 1 has connected.
+    std::vector<rpc_msg_comm_init_rsp> responses(n_backends);
+    for (size_t i = 0; i < n_backends; i++) {
+        auto request = std::make_shared<rpc_msg_comm_init_req>();
+        request->device = ranks[i].device;
+        request->rank   = (uint32_t) i;
+        request->world  = (uint32_t) n_backends;
+        request->port   = comm_port;
+        if (i > 0) {
+            memcpy(request->host, host0.c_str(), host0.size());
+        }
+        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, sizeof(*request), &responses[i], sizeof(responses[i]));
+    }
+    for (size_t i = 0; i < n_backends; i++) {
+        ranks[i].dispatcher->synchronize();
+    }
+    bool ok = true;
+    for (size_t i = 0; i < n_backends; i++) {
+        if (!responses[i].ok) {
+            GGML_LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
+            ok = false;
+        }
+    }
+    if (!ok) {
+        for (const auto & rank : ranks) {
+            rank.dispatcher->busy_spin_release();
+        }
+        return nullptr;
+    }
+    GGML_LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
+                  ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
+    return new ggml_backend_rpc_comm_context{std::move(ranks)};
+}
+
+static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
+    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
+    if (comm_ctx == nullptr) {
+        return false;
+    }
+    const size_t n_ranks = comm_ctx->ranks.size();
+    const int64_t ne = ggml_nelements(tensors[0]);
+    if (ne == 0) {
+        return true;
+    }
+    for (size_t i = 0; i < n_ranks; i++) {
+        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
+                !ggml_is_contiguously_allocated(tensors[i]) ||
+                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
+            return false;
+        }
+        // a rank with a disabled node has garbage in its partial and must contribute zeros,
+        // which only the fallback path handles
+        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
+            return false;
+        }
+    }
+    for (size_t i = 0; i < n_ranks; i++) {
+        auto request = std::make_shared<rpc_msg_comm_allreduce_req>();
+        request->device = comm_ctx->ranks[i].device;
+        request->tensor = serialize_tensor(tensors[i]);
+        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, sizeof(*request));
+    }
+    return true;
+}
+
 static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
     if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
         return (void *)ggml_backend_rpc_add_server;
@@ -2301,6 +2995,15 @@ static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const ch
     if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
         return (void *)ggml_backend_rpc_start_server;
     }
+    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
+        return (void *)ggml_backend_rpc_comm_init;
+    }
+    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
+        return (void *)ggml_backend_rpc_comm_free;
+    }
+    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
+        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
+    }
     return NULL;
 
     GGML_UNUSED(reg);
@@ -2359,7 +3062,6 @@ ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
             /* .device      = */    ind,
             /* .name        = */    dev_name,
             /* .description = */    dev_desc,
-            /* .last_graph_uid = */ 0,
         };
 
         ggml_backend_dev_t dev = new ggml_backend_device {
diff --git src/ggml-sycl/CMakeLists.txt src/ggml-sycl/CMakeLists.txt
index d2196f74..1697ecd9 100644
--- src/ggml-sycl/CMakeLists.txt
+++ src/ggml-sycl/CMakeLists.txt
@@ -221,4 +221,27 @@ if (GGML_SYCL_DEVICE_ARCH)
         "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
         -fsycl-max-parallel-link-jobs=${GGML_SYCL_MAX_PARALLEL_LINK_JOBS}
     )
+
+    # The XMX dequant-GEMM tiles need the sub-group size of the target: 8 on Xe-HPG (DG2, ARL-H),
+    # 16 on Xe-HPC and Xe2 or newer. ocloc fails on the other size, so build only the one that fits.
+    # 0 (unknown name, mixed list, or no XMX) builds no XMX tile and the path stays off.
+    set(_ggml_sycl_xmx_sg "")
+    string(TOLOWER "${GGML_SYCL_DEVICE_ARCH}" _ggml_sycl_archs)
+    string(REPLACE "," ";" _ggml_sycl_archs "${_ggml_sycl_archs}")
+    foreach(_arch IN LISTS _ggml_sycl_archs)
+        if (_arch MATCHES "^(dg2|acm|ats-m|arl-h|xe-hpg|12\\.5[567]\\.|12\\.74\\.)")
+            set(_sg 8)
+        elseif (_arch MATCHES "^(pvc|bmg|lnl|ptl|wcl|nvl|cri|xe2|xe3|xe-hpc|12\\.60\\.|20\\.|30\\.)")
+            set(_sg 16)
+        else()
+            set(_sg 0)
+        endif()
+        if (_ggml_sycl_xmx_sg STREQUAL "" OR _ggml_sycl_xmx_sg EQUAL _sg)
+            set(_ggml_sycl_xmx_sg ${_sg})
+        else()
+            set(_ggml_sycl_xmx_sg 0)
+        endif()
+    endforeach()
+    message(STATUS "GGML_SYCL_DEVICE_ARCH: XMX dequant-GEMM sub-group size ${_ggml_sycl_xmx_sg} (0 = off)")
+    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_XMX_AOT_SG=${_ggml_sycl_xmx_sg})
 endif()
diff --git src/ggml-sycl/common.hpp src/ggml-sycl/common.hpp
index 11fe86cc..47550937 100644
--- src/ggml-sycl/common.hpp
+++ src/ggml-sycl/common.hpp
@@ -26,7 +26,6 @@
 #include "presets.hpp"
 #include "type.hpp"
 #include "sycl_hw.hpp"
-#include "fattn-buffers.hpp"
 #include "memtrace.hpp"
 
 namespace syclexp = sycl::ext::oneapi::experimental;
@@ -66,6 +65,61 @@ extern int g_ggml_sycl_enable_fusion;
 extern int g_ggml_sycl_enable_esimd;
 extern int g_ggml_sycl_mmvq_wide;
 extern int g_ggml_sycl_prioritize_dmmv;
+
+// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
+// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
+enum ggml_sycl_xmx_gather_type {
+    GGML_SYCL_XMX_GATHER_IQ4_NL   = 1 << 0,
+    GGML_SYCL_XMX_GATHER_IQ3_S    = 1 << 1,
+    GGML_SYCL_XMX_GATHER_IQ4_XS   = 1 << 2,
+    GGML_SYCL_XMX_GATHER_IQ3_XXS  = 1 << 3,
+    GGML_SYCL_XMX_GATHER_IQ2_XXS  = 1 << 4,
+    GGML_SYCL_XMX_GATHER_IQ2_XS   = 1 << 5,
+    GGML_SYCL_XMX_GATHER_IQ2_S    = 1 << 6,
+    GGML_SYCL_XMX_GATHER_IQ1_S    = 1 << 7,
+    GGML_SYCL_XMX_GATHER_IQ1_M    = 1 << 8,
+    GGML_SYCL_XMX_GATHER_Q8_0     = 1 << 9,
+    GGML_SYCL_XMX_GATHER_Q4_K     = 1 << 10,
+    GGML_SYCL_XMX_GATHER_Q5_K     = 1 << 11,
+    GGML_SYCL_XMX_GATHER_Q6_K     = 1 << 12,
+};
+static constexpr int GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
+extern int g_ggml_sycl_xmx_gather_types;
+// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
+// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
+static constexpr int GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
+extern int g_ggml_sycl_xmx_gather_shapes;
+
+// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
+// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
+enum ggml_sycl_dynamic_precision {
+    GGML_SYCL_DYNAMIC_PRECISION_F16,
+    GGML_SYCL_DYNAMIC_PRECISION_BF16,
+    GGML_SYCL_DYNAMIC_PRECISION_TF32,
+    GGML_SYCL_DYNAMIC_PRECISION_F32,
+};
+#ifdef GGML_SYCL_F16
+static constexpr int GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
+#else
+static constexpr int GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
+#endif
+extern int g_ggml_sycl_dynamic_precision;
+// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
+// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
+extern int g_ggml_sycl_dynamic_required_precision;
+
+// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
+static inline int32_t ggml_sycl_src1_prec(const ggml_tensor * dst) {
+    return g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
+                                                                                       dst->op_params[3];
+}
+
+// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
+static inline bool ggml_sycl_src1_f16_ok(const ggml_tensor * dst) {
+    const int32_t src1_prec = ggml_sycl_src1_prec(dst);
+    return g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
+           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
+}
 extern int g_ggml_sycl_enable_flash_attention;
 extern int g_ggml_sycl_dev2dev_memcpy;
 extern int g_ggml_sycl_fa_onednn;
@@ -334,6 +388,12 @@ struct mmid_row_mapping {
     int32_t i2;
 };
 
+struct ggml_sycl_gg_tile {
+    int32_t expert;
+    int32_t n0;
+    int32_t n1;
+};
+
 namespace sycl_ex = sycl::ext::oneapi::experimental;
 struct ggml_backend_sycl_context {
     int device;
@@ -408,18 +468,15 @@ struct ggml_backend_sycl_context {
     // pool
     std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];
 
-    std::unique_ptr<ggml_sycl_fattn_kv_buffers> fattn_bufs[GGML_SYCL_MAX_DEVICES];
-
     std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];
 
     std::vector<mmid_row_mapping> mmid_row_mapping_host;
+    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;
 
     static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);
 
     static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);
 
-    static std::unique_ptr<ggml_sycl_fattn_kv_buffers> new_fattn_kv_buffers(queue_ptr qptr, int device);
-
     ggml_sycl_pool & pool(int device) {
         if (pools[device] == nullptr) {
             pools[device] = new_pool_for_device(stream(device,0), device);
@@ -431,17 +488,6 @@ struct ggml_backend_sycl_context {
         return pool(device);
     }
 
-    ggml_sycl_fattn_kv_buffers & fattn_buffers(int device) {
-        if (fattn_bufs[device] == nullptr) {
-            fattn_bufs[device] = new_fattn_kv_buffers(stream(device, 0), device);
-        }
-        return *fattn_bufs[device];
-    }
-
-    ggml_sycl_fattn_kv_buffers & fattn_buffers() {
-        return fattn_buffers(device);
-    }
-
 #ifdef GGML_SYCL_GRAPH
     std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
 #endif
diff --git src/ggml-sycl/element_wise.cpp src/ggml-sycl/element_wise.cpp
index 2e926abe..0d263aa7 100644
--- src/ggml-sycl/element_wise.cpp
+++ src/ggml-sycl/element_wise.cpp
@@ -452,6 +452,46 @@ static void unary_mul_sycl(const T * x, const T * g, T * dst, const int64_t k, c
     });
 }
 
+// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
+// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
+template<typename F>
+static void add_unary_mul_flat_kernel(const float * a, const float * bias, const float * scale, float * dst,
+                                      const int64_t k, const sycl::nd_item<1> &item_ct1, F op) {
+    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
+        dst[i] = op(a[i] + bias[i]) * scale[i];
+    }
+}
+
+template<typename F>
+static void add_unary_mul_bcast_kernel(const float * a, const float * bias, const float * scale, float * dst,
+                                       const int64_t k, const sycl::uint3 ne0_fd, const sycl::nd_item<1> &item_ct1, F op) {
+    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
+        const uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
+        dst[i] = op(a[i] + bias[h]) * scale[h];
+    }
+}
+
+template<typename F>
+static void add_unary_mul_sycl(const float * a, const float * bias, const float * scale, float * dst,
+                               const int64_t k, const int64_t ne0, queue_ptr main_stream, F op) {
+    const size_t            num_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
+    const sycl::nd_range<1> range(num_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));
+
+    if (k == ne0) {
+        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
+            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
+        });
+        return;
+    }
+
+    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
+    GGML_ASSERT(k < ((int64_t) 1 << 31));
+    const sycl::uint3 ne0_fd = init_fastdiv_values((uint32_t) ne0);
+    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
+        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
+    });
+}
+
 namespace ggml_sycl_detail {
 static void acc_f32_sycl(const char *x, const char *y, float *dst,
                          const int64_t n_elements,
@@ -995,6 +1035,19 @@ static inline void ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_ten
     });
 }
 
+// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
+// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
+template<typename F>
+static void dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
+    switch (uop) {
+        case GGML_UNARY_OP_SILU:     launch([](float v) { return op_silu(v); });     break;
+        case GGML_UNARY_OP_SIGMOID:  launch([](float v) { return op_sigmoid(v); });  break;
+        case GGML_UNARY_OP_SOFTPLUS: launch([](float v) { return op_softplus(v); }); break;
+        default:
+            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
+    }
+}
+
 // dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
 // standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
 void ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
@@ -1032,13 +1085,41 @@ void ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor *
         }
     };
 
-    switch (ggml_get_unary_op(unary_node)) {
-        case GGML_UNARY_OP_SILU:     dispatch_type([](float v) { return op_silu(v); });     break;
-        case GGML_UNARY_OP_SIGMOID:  dispatch_type([](float v) { return op_sigmoid(v); });  break;
-        case GGML_UNARY_OP_SOFTPLUS: dispatch_type([](float v) { return op_softplus(v); }); break;
-        default:
-            GGML_ABORT("fused unary+mul: unsupported unary op %s", ggml_unary_op_name(ggml_get_unary_op(unary_node)));
-    }
+    dispatch_fused_unary_op(ggml_get_unary_op(unary_node), dispatch_type);
+}
+
+// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
+// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
+void ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_node,
+                                      ggml_tensor * unary_node, ggml_tensor * mul_node) {
+    // the dst-arity convention the other fusions follow; a and bias live on add_node
+    scope_op_debug_print scope_dbg_print(__func__, mul_node, /*num_src=*/2);
+
+    const ggml_tensor * a     = add_node->src[0];
+    const ggml_tensor * bias  = add_node->src[1];
+    const ggml_tensor * scale = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];
+
+    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
+    GGML_ASSERT(scale != unary_node);
+    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
+    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_node->type == GGML_TYPE_F32);
+    GGML_ASSERT(ggml_are_same_shape(a, mul_node));
+    // a and dst are indexed flat
+    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_node));
+    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
+    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
+    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
+    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));
+
+    queue_ptr main_stream = ctx.stream();
+    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
+
+    const auto dispatch_op = [&](auto op) {
+        add_unary_mul_sycl((const float *) a->data, (const float *) bias->data, (const float *) scale->data,
+                           (float *) mul_node->data, ggml_nelements(mul_node), mul_node->ne[0], main_stream, op);
+    };
+
+    dispatch_fused_unary_op(ggml_get_unary_op(unary_node), dispatch_op);
 }
 
 __dpct_inline__ float ggml_sycl_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
diff --git src/ggml-sycl/element_wise.hpp src/ggml-sycl/element_wise.hpp
index d280066e..0a5bfa95 100644
--- src/ggml-sycl/element_wise.hpp
+++ src/ggml-sycl/element_wise.hpp
@@ -132,4 +132,9 @@ void ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
 // fused UNARY(silu|sigmoid|softplus) + MUL; see ggml_sycl_can_fuse() for the accepted shapes
 void ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);
 
+// fused f32 ADD + UNARY(silu|sigmoid|softplus) + MUL with the bias and the scale broadcast
+// over dim 0; see ggml_sycl_can_fuse() for the accepted shapes
+void ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_node,
+                                      ggml_tensor * unary_node, ggml_tensor * mul_node);
+
 #endif // GGML_SYCL_ELEMENTWISE_HPP
diff --git src/ggml-sycl/fattn-common.hpp src/ggml-sycl/fattn-common.hpp
index 3c2d1a77..2e8ac6af 100644
--- src/ggml-sycl/fattn-common.hpp
+++ src/ggml-sycl/fattn-common.hpp
@@ -6,7 +6,6 @@
 #include "common.hpp"
 #include "convert.hpp"
 #include "vecdotq.hpp"
-#include "fattn-buffers.hpp"
 #include "fattn.hpp"
 
 #include "ggml.h"
@@ -933,13 +932,10 @@ void launch_fattn(
     GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
 
     ggml_sycl_pool & pool = ctx.pool();
-    ggml_sycl_fattn_kv_buffers & fbuf = ctx.fattn_buffers();
     dpct::queue_ptr  main_stream = ctx.stream();
     const int id  = ggml_sycl_get_device();
     const int nsm = ggml_sycl_info().devices[id].nsm;
 
-    ggml_sycl_fattn_alloc        K_f16(fbuf.K);
-    ggml_sycl_fattn_alloc        V_f16(fbuf.V);
     const ggml_sycl_fattn_extra  extra = ggml_sycl_fattn_get_extra(dst);
     ggml_sycl_pool_alloc<int>    KV_max(pool);
     ggml_sycl_pool_alloc<float>  dst_tmp(pool);
@@ -959,8 +955,8 @@ void launch_fattn(
         const size_t bs = ggml_blck_size(K->type);
         const size_t ts = ggml_type_size(K->type);
 
-        sycl::half * K_f16_ptr = extra.K_buffer_ptr ? (sycl::half *) extra.K_buffer_ptr
-                                                    : K_f16.alloc(ggml_nelements(K));
+        GGML_ASSERT(extra.K_buffer_ptr);
+        sycl::half * K_f16_ptr = (sycl::half *) extra.K_buffer_ptr;
         if (ggml_is_contiguously_allocated(K)) {
             to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
             to_fp16(K_data, K_f16_ptr, ggml_nelements(K), main_stream);
@@ -993,8 +989,8 @@ void launch_fattn(
             const size_t bs = ggml_blck_size(V->type);
             const size_t ts = ggml_type_size(V->type);
 
-            sycl::half * V_f16_ptr = extra.V_buffer_ptr ? (sycl::half *) extra.V_buffer_ptr
-                                                        : V_f16.alloc(ggml_nelements(V));
+            GGML_ASSERT(extra.V_buffer_ptr);
+            sycl::half * V_f16_ptr = (sycl::half *) extra.V_buffer_ptr;
             if (ggml_is_contiguously_allocated(V)) {
                 to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                 to_fp16(V_data, V_f16_ptr, ggml_nelements(V), main_stream);
diff --git src/ggml-sycl/fattn-mkl.cpp src/ggml-sycl/fattn-mkl.cpp
index 30947b17..5a5cf69f 100644
--- src/ggml-sycl/fattn-mkl.cpp
+++ src/ggml-sycl/fattn-mkl.cpp
@@ -8,7 +8,6 @@
 
 #include "common.hpp"
 #include "fattn-common.hpp"
-#include "fattn-buffers.hpp"
 #include "convert.hpp"
 #include "fattn.hpp"
 
@@ -283,8 +282,10 @@ static mkl_fa_kv_desc mkl_fa_make_desc(const ggml_tensor * T, bool interleaved,
     d.ts   = (int64_t)ggml_type_size(T->type);
 
     if (T->type == GGML_TYPE_F16) {
-        d.mode = interleaved ? MKL_FA_KV_MODE_F16_INTERLEAVED
-                             : MKL_FA_KV_MODE_F16_DENSE;
+        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
+        // padded row stride as strided even when there is only one KV head.
+        d.mode = interleaved || d.nb1 != d.D * (int64_t)sizeof(sycl::half)
+            ? MKL_FA_KV_MODE_F16_INTERLEAVED : MKL_FA_KV_MODE_F16_DENSE;
     } else if (ggml_is_contiguously_allocated(T) && !interleaved) {
         d.mode = MKL_FA_KV_MODE_QUANT_CONTIG;
     } else {
@@ -413,7 +414,9 @@ void ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor *
     const int64_t q_row_stride  = Q->nb[1] / sizeof(float);
     const int64_t q_head_stride = Q->nb[2] / sizeof(float);
 
-    const bool V_is_K_view = V->view_src
+    // Alias the dequantized buffers only when K and V expose the same values.
+    // MLA V is a narrower view of K and needs its own strided dequantization.
+    const bool V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
         && (V->view_src == K || (V->view_src == K->view_src
             && V->view_offs == K->view_offs));
 
diff --git src/ggml-sycl/fattn-vec.hpp src/ggml-sycl/fattn-vec.hpp
index 9ec88c28..5e40c88f 100644
--- src/ggml-sycl/fattn-vec.hpp
+++ src/ggml-sycl/fattn-vec.hpp
@@ -589,10 +589,10 @@ void ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggm
     const bool need_f16_V = type_V == GGML_TYPE_F16;
     constexpr size_t nbytes_shared = 0;
 
+    const auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
     // D=512 does not fit the default register file; it spills up to 343 bytes per thread, against at most 57 for D <= 256. This kernel is decode only, so thread occupancy is not the limit. It is 1.9x faster at every KV depth on Battlemage.
     constexpr bool use_large_grf = D >= 512;
 
-    const auto arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
     const int nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
     if constexpr (D <= 256) {
         if (nthreads == 256) {
diff --git src/ggml-sycl/fattn.cpp src/ggml-sycl/fattn.cpp
index 541ae8a8..8bd946e0 100644
--- src/ggml-sycl/fattn.cpp
+++ src/ggml-sycl/fattn.cpp
@@ -146,15 +146,17 @@ static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const
     // Set GGML_SYCL_ENABLE_MKL_FA=0 to force TILE/VEC path for A/B testing.
     // Example: GGML_SYCL_ENABLE_MKL_FA=0 llama-cli -m model.gguf -fa -ngl 99 ...
     // Note: MKL GEMM calls are incompatible with SYCL graph capture replay.
-    // MKL is validated for the mainstream GQA envelope: grouped-query
-    // (gqa_ratio >= 2), head_dim a multiple of 64 in [64,512] with matching
-    // K/V head size, mask, no sinks/ALiBi/softcap. Gemma's global layers use
-    // head_dim 512, so the cap must include it. Head sizes not a multiple of
-    // 64 (72/80/96), MHA (gqa_ratio == 1), and MLA (DKQ != DV, e.g. 576/512)
-    // fall through to TILE/VEC; see follow-up work.
+    const bool standard_shape = Q->ne[0] >= 64 && Q->ne[0] <= 512 &&
+        Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0];
+    // GLM-4.7 Flash's MLA shape is already expressible by the MKL pipeline:
+    // KQ GEMM uses DKQ=576 while VKQ and output use DV=512. Keep this narrow
+    // until other mismatched K/V shapes have independent correctness data.
+    const bool glm_mla_shape = Q->ne[0] == 576 && K->ne[0] == 576 &&
+        V->ne[0] == 512 && gqa_ratio == 20 &&
+        K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
+
     if (g_ggml_sycl_enable_mkl_fa == 1 && mask && !sinks && gqa_ratio >= 2 &&
-        Q->ne[0] >= 64 && Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 &&
-        Q->ne[0] == V->ne[0] &&
+        (standard_shape || glm_mla_shape) &&
         Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
         max_bias == 0.0f && logit_softcap == 0.0f &&
         (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
@@ -164,7 +166,10 @@ static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const
         // nb1=75 for ne0=40 fall through to TILE.
         bool kv_strides_ok = true;
         for (const ggml_tensor * t : {K, V}) {
-            if (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
+            const bool glm_v_stride = glm_mla_shape && t == V &&
+                V->view_src && V->nb[1] == K->nb[1];
+            if (!glm_v_stride && t->type == GGML_TYPE_F16 &&
+                    t->nb[1] % (t->ne[0] * 2) != 0) {
                 kv_strides_ok = false;
                 break;
             }
diff --git src/ggml-sycl/fused-gemm.cpp src/ggml-sycl/fused-gemm.cpp
new file mode 100644
index 00000000..38c69bc6
--- /dev/null
+++ src/ggml-sycl/fused-gemm.cpp
@@ -0,0 +1,1093 @@
+#include "fused-gemm.hpp"
+
+#include <sycl/ext/oneapi/matrix/matrix.hpp>
+
+#include <algorithm>
+#include <string>
+#include <tuple>
+#include <mutex>
+#include <set>
+#include <unordered_map>
+
+namespace mx = sycl::ext::oneapi::experimental::matrix;
+
+// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.
+
+// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
+// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
+// and are summed at the end.
+static constexpr int FG_BK     = QK4_NL;
+static constexpr int FG_KSPLIT = 4;
+
+// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
+// convert once, in registers, when they write the element, so any type costs the same one pass.
+//   store: storage in SLM (A) and in the packed B buffer
+//   mtype: matrix_type in matrix_combinations
+//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
+//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
+//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
+//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
+template <typename T> struct fg_elem;
+
+template <> struct fg_elem<sycl::half> {
+    using store = sycl::half;
+    using pair  = sycl::half2;
+    static constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
+    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
+    static constexpr ggml_type       src   = GGML_TYPE_F16;
+    static constexpr int             mant  = 10;
+    static constexpr int             slow  = 0;
+    static store cvt(float x) { return (store) x; }
+    static pair make(float x, float y) { return pair((store) x, (store) y); }
+};
+
+// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
+// DG2 AOT target rejects
+static inline uint32_t fg_round_bits(float x, int drop) {
+    const uint32_t u = sycl::bit_cast<uint32_t>(x);
+    if ((u & 0x7f800000u) == 0x7f800000u) {
+        return (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
+    }
+    return u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
+}
+
+struct alignas(4) fg_bf16x2 {
+    sycl::ext::oneapi::bfloat16 x, y;
+};
+
+template <> struct fg_elem<sycl::ext::oneapi::bfloat16> {
+    using store = sycl::ext::oneapi::bfloat16;
+    using pair  = fg_bf16x2;
+    static constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
+    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
+    static constexpr ggml_type       src   = GGML_TYPE_BF16;
+    static constexpr int             mant  = 7;
+    static constexpr int             slow  = 0;
+    static store cvt(float x) { return store(x); }
+    static pair make(float x, float y) { return { cvt(x), cvt(y) }; }
+};
+
+// tf32 keeps f32 range and f16 mantissa, in f32 storage
+template <> struct fg_elem<mx::precision::tf32> {
+    using store = float;
+    using pair  = sycl::float2;
+    static constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
+    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
+    static constexpr ggml_type       src   = GGML_TYPE_COUNT;
+    static constexpr int             mant  = 10;
+    static constexpr int             slow  = 1;
+    static store cvt(float x) { return sycl::bit_cast<float>(fg_round_bits(x, 13) & ~0x1fffu); }
+    static pair make(float x, float y) { return pair(cvt(x), cvt(y)); }
+};
+
+// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
+// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
+// columns of B. A and B may differ: the device lists the pairs it supports.
+template <typename TA, typename TB, int TM_, int TN_, int TK_, int SG_> struct fg_combo {
+    using ta  = TA;
+    using tb  = TB;
+    using EA  = fg_elem<TA>;
+    using EB  = fg_elem<TB>;
+    using tsa = typename EA::store;
+    using tsb = typename EB::store;
+    static constexpr int TM = TM_;
+    static constexpr int TN = TN_;
+    static constexpr int TK = TK_;
+    static constexpr int SG = SG_;
+    static constexpr int VNNI    = 4 / sizeof(tsb);  // K rows of B packed in one 32-bit word
+    static constexpr int SG_ROWS = TM > 16 ? TM : 16;
+    static constexpr int RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
+    static constexpr int MT      = SG_ROWS / TM;
+    static constexpr int BN      = TN > 32 ? TN : 32;
+    static constexpr int NT      = BN / TN;
+    static constexpr int WG_SIZE = FG_KSPLIT * SG;
+    static constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
+    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
+    // so it spills: 13x slower on B60
+    static constexpr bool efficient = TN <= 32;
+    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
+    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
+};
+
+using fg_half = sycl::half;
+using fg_bf16 = sycl::ext::oneapi::bfloat16;
+using fg_tf32 = mx::precision::tf32;
+
+// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
+// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
+template <typename F> static void fg_visit_combo(int idx, F && f);
+static constexpr int FG_N_COMBOS = 8;
+
+// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
+// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
+template <int SG> static constexpr bool fg_listed() {
+#if defined(GGML_SYCL_XMX_AOT_SG)
+    return SG == GGML_SYCL_XMX_AOT_SG;
+#else
+    return true;
+#endif
+}
+
+template <typename S, typename F> static void fg_call_combo(F && f) {
+    if constexpr (fg_listed<S::SG>()) {
+        f(S{});
+    }
+}
+
+template <typename F> static void fg_visit_combo(int idx, F && f) {
+    switch (idx) {
+        case 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
+        case 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
+        case 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
+        case 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); break; // Xe2, Xe3, Xe-HPC
+        case 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
+        case 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   break; // Xe2, Xe3, Xe-HPC
+        case 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
+        case 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
+        default: GGML_ABORT("bad XMX combination %d", idx);
+    }
+}
+
+// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
+// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
+// its own device image (joint_matrix is an optional kernel feature) and only a combination the
+// device reports is launched, so the runtime never asks IGC for the others.
+#if defined(__SYCL_DEVICE_ONLY__)
+#    if __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
+        __SYCL_TARGET_INTEL_GPU_ARL_H__
+#        define FG_AOT_SG 8
+#    elif __SYCL_TARGET_INTEL_GPU_PVC__ || __SYCL_TARGET_INTEL_GPU_PVC_VG__ || __SYCL_TARGET_INTEL_GPU_BMG_G21__ || \
+        __SYCL_TARGET_INTEL_GPU_BMG_G31__ || __SYCL_TARGET_INTEL_GPU_LNL_M__ || __SYCL_TARGET_INTEL_GPU_PTL_H__ ||   \
+        __SYCL_TARGET_INTEL_GPU_PTL_U__ || __SYCL_TARGET_INTEL_GPU_WCL__ || __SYCL_TARGET_INTEL_GPU_NVL_S__ ||       \
+        __SYCL_TARGET_INTEL_GPU_NVL_U__ || __SYCL_TARGET_INTEL_GPU_NVL_P__
+#        define FG_AOT_SG 16
+#    elif __SYCL_TARGET_INTEL_GPU_TGLLP__ || __SYCL_TARGET_INTEL_GPU_RKL__ || __SYCL_TARGET_INTEL_GPU_ADL_S__ || \
+        __SYCL_TARGET_INTEL_GPU_ADL_P__ || __SYCL_TARGET_INTEL_GPU_ADL_N__ || __SYCL_TARGET_INTEL_GPU_DG1__ ||   \
+        __SYCL_TARGET_INTEL_GPU_MTL_U__ || __SYCL_TARGET_INTEL_GPU_MTL_H__
+#        define FG_AOT_SG 0 // no XMX
+#    endif
+#endif
+
+template <int SG> static constexpr bool fg_built() {
+#if defined(FG_AOT_SG)
+    return SG == FG_AOT_SG;
+#else
+    return true;
+#endif
+}
+
+// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
+// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
+// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
+static constexpr int64_t grouped_gemm_max_tiles(int64_t total_rows, int64_t n_as, int64_t BN) {
+    return total_rows <= n_as ? total_rows : n_as + (total_rows - n_as) / BN;
+}
+// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
+// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
+static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);
+
+// the device lists S with an f32 accumulator and output
+template <typename S> static bool fg_device_has_combo(const std::vector<mx::combination> & combinations) {
+    for (const auto & c : combinations) {
+        if (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
+            c.dtype == mx::matrix_type::fp32 &&
+            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
+            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
+            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
+            return true;
+        }
+    }
+    return false;
+}
+
+template <typename T> static const char * fg_type_name() {
+    return std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
+}
+
+static std::string fg_combo_name(int idx) {
+    std::string name;
+    fg_visit_combo(idx, [&](auto s) {
+        using S = decltype(s);
+        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
+               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
+               std::to_string(S::SG);
+    });
+    return name;
+}
+
+// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
+// on a mixed box the first caller's verdict is not the others'.
+static int fg_device_combos(const sycl::device & dev) {
+    static std::mutex                            mtx;
+    static std::unordered_map<sycl::device, int> known;
+    std::lock_guard<std::mutex>                  lock(mtx);
+    const auto                                   it = known.find(dev);
+    if (it != known.end()) {
+        return it->second;
+    }
+    int available = 0;
+    try {
+        const auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
+        const auto sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
+        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
+            fg_visit_combo(idx, [&](auto s) {
+                using S = decltype(s);
+                const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
+                if (sg_ok && fg_device_has_combo<S>(combinations)) {
+                    available |= 1 << idx;
+                }
+            });
+        }
+    } catch (const sycl::exception &) {
+        available = 0;
+    }
+    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
+                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
+    known.emplace(dev, available);
+    return available;
+}
+
+// Rank of combination S for a src1 of type src1_type, lower is better. Order:
+//  1. throughput: a tile that does not spill, then the fastest type class of A and B
+//  2. B type equal to the src1 type, so the pack is a plain copy
+//  3. B at least as precise as f16
+//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
+// A costs nothing to convert: the A stage emits any type at the same cost.
+template <typename S> static int64_t fg_rank(ggml_type src1_type) {
+    const int64_t spills  = !S::efficient;
+    const int64_t slow    = std::max(S::EA::slow, S::EB::slow);
+    const int64_t convert = S::EB::src != src1_type;
+    const int64_t lossy   = S::EB::mant < 10;
+    const int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
+    const int64_t mk      = 1024 - S::TM * S::TK;
+    return ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
+}
+
+// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
+// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
+// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
+static bool fg_mode_meets(int mode, int32_t src1_prec) {
+    if (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
+        return true;
+    }
+    if (src1_prec >= GGML_PREC_BF16) {
+        return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
+    }
+    switch (g_ggml_sycl_dynamic_required_precision) {
+        case GGML_SYCL_DYNAMIC_PRECISION_TF32: return mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
+        case GGML_SYCL_DYNAMIC_PRECISION_BF16: return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
+        default:                               return false;
+    }
+}
+
+// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
+// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
+// for an F32 request (most mantissa first).
+static int fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
+    const sycl::device dev     = stream->get_device();
+    const int          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
+    const bool         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
+    const int          modes[] = {
+        g_ggml_sycl_dynamic_precision,
+        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
+        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
+    };
+    int     best      = -1;
+    int64_t best_rank = 0;
+    for (int i = 0; i < 3 && best < 0; ++i) {
+        const int mode = modes[i];
+        if (!fg_mode_meets(mode, src1_prec)) {
+            continue;
+        }
+        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
+            if (!(allowed & (1 << idx))) {
+                continue;
+            }
+            fg_visit_combo(idx, [&](auto s) {
+                using S = decltype(s);
+                if (S::EA::mode != mode || S::EB::mode != mode) {
+                    return;
+                }
+                const int64_t rank = fg_rank<S>(src1_type);
+                if (best < 0 || rank < best_rank) {
+                    best      = idx;
+                    best_rank = rank;
+                }
+            });
+        }
+    }
+    // log each distinct decision once
+    static std::mutex                                      mtx;
+    static std::set<std::tuple<size_t, int, int32_t, int>> seen;
+    std::lock_guard<std::mutex>                            lock(mtx);
+    if (seen.emplace(std::hash<sycl::device>{}(dev), (int) src1_type, src1_prec, best).second) {
+        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
+                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
+    }
+    return best;
+}
+
+// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
+template <typename E, typename T_src>
+static void fused_gemm_pack_b(const T_src * y, typename E::store * packed, int N, int Npad, int K,
+                              dpct::queue_ptr stream) {
+    constexpr int V   = 4 / sizeof(typename E::store);
+    const int     kqs = K / V;
+    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
+        const int idx = id[0];
+        const int n   = idx / kqs;
+        const int kq  = idx - n * kqs;
+        typename E::store vals[V] = {};
+        if (n < N) {
+            const T_src * src = y + (size_t) n * K + V * kq;
+#pragma unroll
+            for (int v = 0; v < V; ++v) {
+                vals[v] = E::cvt((float) src[v]);
+            }
+        }
+        typename E::store * out = packed + ((size_t) kq * Npad + n) * V;
+#pragma unroll
+        for (int v = 0; v < V; ++v) {
+            out[v] = vals[v];
+        }
+    });
+}
+
+// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
+// folded into the value so the mad below sees plain A elements. One overload per weight format.
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq4_nl * __restrict__ xrow, const int kb, typename E::pair * a) {
+    const block_iq4_nl blk = xrow[kb];
+    const float        d   = (float) blk.d;
+#pragma unroll
+    for (int j = 0; j < QK4_NL / 2; j += 2) {
+        const uint8_t q0 = blk.qs[j];
+        const uint8_t q1 = blk.qs[j + 1];
+        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
+        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
+    }
+}
+
+// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
+// only the fields of that sub-block instead of copying the block. Same decode as
+// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq3_s * __restrict__ xrow, const int kb, typename E::pair * a) {
+    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
+    const block_iq3_s * blk = xrow + kb / (QK_K / 32);
+    const int           ib8 = kb % (QK_K / 32);
+    const uint8_t *     qs  = blk->qs + 8 * ib8;
+    const int           qh  = blk->qh[ib8];
+    const float         d   = (float) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
+        const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
+        const int      signs = blk->signs[4 * ib8 + il];
+#pragma unroll
+        for (int j = 0; j < 2; ++j) {
+            const float g1a = (float) ((grid1 >> (16 * j + 0)) & 0xff);
+            const float g1b = (float) ((grid1 >> (16 * j + 8)) & 0xff);
+            const float g2a = (float) ((grid2 >> (16 * j + 0)) & 0xff);
+            const float g2b = (float) ((grid2 >> (16 * j + 8)) & 0xff);
+            const int   s   = 2 * j;
+            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
+                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
+            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
+                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
+        }
+    }
+}
+
+// values per stored block, so a row of K values is K/qk blocks
+template <typename block_q_t> struct fg_block_traits;
+template <> struct fg_block_traits<block_iq4_nl> { static constexpr int qk = QK4_NL; };
+template <> struct fg_block_traits<block_iq3_s>  { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq3_xxs> { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq4_xs>  { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq2_xxs> { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq2_xs>  { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq2_s>   { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq1_s>   { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_iq1_m>   { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_q8_0>    { static constexpr int qk = QK8_0; };
+template <> struct fg_block_traits<block_q4_K>    { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_q5_K>    { static constexpr int qk = QK_K; };
+template <> struct fg_block_traits<block_q6_K>    { static constexpr int qk = QK_K; };
+
+// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
+// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
+// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
+// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
+#define FG_SUPERBLOCK(T)                                                         \
+    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
+    const T * blk = xrow + kb / (QK_K / 32);                                     \
+    const int ib  = kb % (QK_K / 32)
+
+template <typename E>
+static __dpct_inline__ void fg_pack_quarter(const float * __restrict__ t, typename E::pair * a, int il) {
+#pragma unroll
+    for (int j = 0; j < 4; ++j) {
+        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq4_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq4_xs);
+    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
+    // land at a[0..7] and a[8..15] and no quarter loop is needed
+    const float d = (float) blk->d *
+        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
+    const uint8_t * q4 = blk->qs + 16 * ib;
+#pragma unroll
+    for (int j = 0; j < 8; ++j) {
+        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
+        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq3_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq3_xxs);
+    const uint8_t *  q3    = blk->qs + 8 * ib;
+    const uint16_t * gas   = (const uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
+    const uint32_t   aux32 = gas[0] | (gas[1] << 16);
+    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
+        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
+        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
+        float t[8];
+#pragma unroll
+        for (int j = 0; j < 4; ++j) {
+            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
+            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
+        }
+        fg_pack_quarter<E>(t, a, il);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq2_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq2_xxs);
+    const uint16_t * q2    = blk->qs + 4 * ib;
+    const uint8_t *  aux8  = (const uint8_t *) q2;
+    const uint32_t   aux32 = q2[2] | (q2[3] << 16);
+    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
+        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
+        float t[8];
+#pragma unroll
+        for (int j = 0; j < 8; ++j) {
+            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
+        }
+        fg_pack_quarter<E>(t, a, il);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq2_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq2_xs);
+    const uint16_t * q2 = blk->qs + 4 * ib;
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
+        const float     d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
+        const uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
+        float t[8];
+#pragma unroll
+        for (int j = 0; j < 8; ++j) {
+            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
+        }
+        fg_pack_quarter<E>(t, a, il);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq2_s * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq2_s);
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        const uint8_t * grid =
+            (const uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
+        const float   d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
+        const uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
+        float t[8];
+#pragma unroll
+        for (int j = 0; j < 8; ++j) {
+            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
+        }
+        fg_pack_quarter<E>(t, a, il);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq1_s * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq1_s);
+    const float delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
+    const float d     = (float) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        uint32_t       grid32[2];
+        const int8_t * q = (const int8_t *) grid32;
+        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
+        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
+        grid32[0] &= 0x0f0f0f0f;
+        float t[8];
+#pragma unroll
+        for (int j = 0; j < 8; ++j) {
+            t[j] = d * (q[j] + delta);
+        }
+        fg_pack_quarter<E>(t, a, il);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_iq1_m * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_iq1_m);
+    const uint16_t * sc = (const uint16_t *) blk->scales;
+    iq1m_scale_t     scale;
+    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
+#pragma unroll
+    for (int il = 0; il < 4; ++il) {
+        const int   ib16  = 2 * ib + il / 2;
+        const float d     = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
+        const float delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
+        uint32_t       grid32[2];
+        const int8_t * q = (const int8_t *) grid32;
+        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
+                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
+        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
+        grid32[0] &= 0x0f0f0f0f;
+        float t[8];
+#pragma unroll
+        for (int j = 0; j < 8; ++j) {
+            t[j] = d * (q[j] + delta);
+        }
+        fg_pack_quarter<E>(t, a, il);
+    }
+}
+
+
+// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
+// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
+// live, as in dequantize.hpp.
+template <typename E>
+static __dpct_inline__ void fg_decode_q8_0(const int8_t * __restrict__ qs, const float d, typename E::pair * a) {
+#pragma unroll
+    for (int j = 0; j < QK8_0 / 2; ++j) {
+        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
+    }
+}
+
+// same unpack as get_scale_min_k4() in dequantize.hpp
+static __dpct_inline__ void fg_scale_min_k4(const int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
+    if (j < 4) {
+        d = q[j] & 63;
+        m = q[j + 4] & 63;
+    } else {
+        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
+        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
+    }
+}
+
+// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
+// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
+template <typename E>
+static __dpct_inline__ void fg_decode_q4_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
+                                           const sycl::half2 dm, const int ib, typename E::pair * a) {
+    uint8_t sc, mb;
+    fg_scale_min_k4(ib, scales, sc, mb);
+    const float     d     = (float) dm[0] * sc;
+    const float     m     = (float) dm[1] * mb;
+    const uint8_t * q     = qs + 32 * (ib / 2);
+    const int       shift = 4 * (ib % 2);
+#pragma unroll
+    for (int j = 0; j < 16; ++j) {
+        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
+    }
+}
+
+// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
+template <typename E>
+static __dpct_inline__ void fg_decode_q5_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
+                                           const uint8_t * __restrict__ scales, const sycl::half2 dm, const int ib,
+                                           typename E::pair * a) {
+    uint8_t sc, mb;
+    fg_scale_min_k4(ib, scales, sc, mb);
+    const float     d     = (float) dm[0] * sc;
+    const float     m     = (float) dm[1] * mb;
+    const uint8_t * q     = qs + 32 * (ib / 2);
+    const int       shift = 4 * (ib % 2);
+#pragma unroll
+    for (int j = 0; j < 16; ++j) {
+        const int l = 2 * j;
+        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
+                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
+    }
+}
+
+// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
+// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
+// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
+template <typename E>
+static __dpct_inline__ void fg_decode_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
+                                           const int8_t * __restrict__ scales, const float d, const int ib,
+                                           typename E::pair * a) {
+    const int       h  = ib / 4;
+    const int       r  = ib % 4;
+    const uint8_t * q  = ql + 64 * h + 32 * (r & 1);
+    const uint8_t * hb = qh + 32 * h;
+    const int8_t *  sc = scales + 8 * h + 2 * r;
+#pragma unroll
+    for (int j = 0; j < 16; ++j) {
+        const int   l  = 2 * j;
+        const float dl = d * sc[j / 8];
+        const int   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
+        const int   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
+        a[j] = E::make(dl * q0, dl * q1);
+    }
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_q8_0 * __restrict__ xrow, const int kb, typename E::pair * a) {
+    fg_decode_q8_0<E>(xrow[kb].qs, (float) xrow[kb].d, a);
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_q4_K * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_q4_K);
+    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_q5_K * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_q5_K);
+    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
+}
+
+template <typename E>
+static __dpct_inline__ void fg_stage_a(const block_q6_K * __restrict__ xrow, const int kb, typename E::pair * a) {
+    FG_SUPERBLOCK(block_q6_K);
+    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (float) blk->d, ib, a);
+}
+
+// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
+// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
+template <typename block_q_t> struct fg_soa {
+    static constexpr bool supported = false;
+};
+
+template <> struct fg_soa<block_q8_0> {
+    static constexpr bool supported = true;
+    // [qs][d]
+    template <typename E>
+    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int,
+                                      typename E::pair * a) {
+        const float d = (float) ((const sycl::half *) (x + nblocks * QK8_0))[ib];
+        fg_decode_q8_0<E>((const int8_t *) x + ib * QK8_0, d, a);
+    }
+};
+
+template <> struct fg_soa<block_q4_K> {
+    static constexpr bool supported = true;
+    // [qs][scales][dm]
+    template <typename E>
+    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
+                                      typename E::pair * a) {
+        const uint8_t *   scales = x + nblocks * (QK_K / 2);
+        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
+        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_SIZE, dm, kb % (QK_K / 32), a);
+    }
+};
+
+template <> struct fg_soa<block_q5_K> {
+    static constexpr bool supported = true;
+    // [qs][qh][scales][dm]
+    template <typename E>
+    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
+                                      typename E::pair * a) {
+        const uint8_t *   qh     = x + nblocks * (QK_K / 2);
+        const uint8_t *   scales = qh + nblocks * (QK_K / 8);
+        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
+        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_SIZE, dm,
+                          kb % (QK_K / 32), a);
+    }
+};
+
+template <> struct fg_soa<block_q6_K> {
+    static constexpr bool supported = true;
+    // [ql][qh][scales][d]
+    template <typename E>
+    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
+                                      typename E::pair * a) {
+        const uint8_t * qh     = x + nblocks * (QK_K / 2);
+        const uint8_t * scales = qh + nblocks * (QK_K / 4);
+        const float     d      = (float) ((const sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
+        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (const int8_t *) scales + ib * (QK_K / 16), d,
+                          kb % (QK_K / 32), a);
+    }
+};
+
+
+// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
+// n1 - n0 <= BN
+template <typename S, typename block_q_t, bool reordered>
+static void fused_dequant_gemm_tile(
+    const block_q_t * __restrict__ x,
+    const typename S::tsb * __restrict__ packed_b,
+    float * __restrict__ dst,
+    const int M, const int Npad, const int K, const int ldd,
+    const int b0, const int n0, const int n1,
+    sycl::local_accessor<typename S::tsa, 1> tile_a,
+    sycl::local_accessor<float, 1> tile_c,
+    const sycl::nd_item<2> & item) {
+    if constexpr (fg_built<S::SG>()) {
+        using EA = typename S::EA;
+        using TA = typename S::ta;
+        using TB = typename S::tb;
+        const auto sg     = item.get_sub_group();
+        const int  sg_id  = sg.get_group_id()[0];
+        const int  lane   = sg.get_local_id()[0];
+        const int  m0     = item.get_group(1) * S::SG_ROWS;
+        const int  nstep  = K / FG_BK;
+        const int  a_base = sg_id * S::SG_ROWS * FG_BK;
+        const int  c_base = sg_id * S::SG_ROWS * S::BN;
+
+        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
+#pragma unroll
+        for (int mt = 0; mt < S::MT; ++mt) {
+#pragma unroll
+            for (int nt = 0; nt < S::NT; ++nt) {
+                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
+            }
+        }
+
+        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
+        constexpr int         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
+        const size_t          bpr     = K / fg_block_traits<block_q_t>::qk;
+        const size_t          nblocks = (size_t) M * bpr;
+        size_t                row_blk[S::RPL];
+        bool                  row_ok[S::RPL];
+        typename EA::pair *   a[S::RPL];
+#pragma unroll
+        for (int r = 0; r < S::RPL; ++r) {
+            const int row = m0 + r * S::SG + lane;
+            row_ok[r]  = row < M;
+            row_blk[r] = (size_t) (row_ok[r] ? row : 0) * bpr;
+            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
+        }
+
+        const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
+                                                    sycl::access::decorated::no>(packed_b);
+        const int b_stride = Npad * S::VNNI;
+
+        const int kb_begin = (sg_id * nstep) / FG_KSPLIT;
+        const int kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
+        for (int kb = kb_begin; kb < kb_end; ++kb) {
+#pragma unroll
+            for (int r = 0; r < S::RPL; ++r) {
+                if (row_ok[r]) {
+                    if constexpr (reordered) {
+                        fg_soa<block_q_t>::template stage<EA>((const uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
+                                                              a[r]);
+                    } else {
+                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
+                    }
+                } else {
+#pragma unroll
+                    for (int j = 0; j < FG_BK / 2; ++j) {
+                        a[r][j] = EA::make(0.0f, 0.0f);
+                    }
+                }
+            }
+            sycl::group_barrier(sg);
+
+#pragma unroll
+            for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
+                const int kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
+                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
+#pragma unroll
+                for (int nt = 0; nt < S::NT; ++nt) {
+                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
+                }
+#pragma unroll
+                for (int mt = 0; mt < S::MT; ++mt) {
+                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
+                    mx::joint_matrix_load(sg, sub_a,
+                        tile_a.template get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
+                        FG_BK);
+#pragma unroll
+                    for (int nt = 0; nt < S::NT; ++nt) {
+                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
+                    }
+                }
+            }
+            // the next step overwrites tile_a
+            sycl::group_barrier(sg);
+        }
+
+#pragma unroll
+        for (int mt = 0; mt < S::MT; ++mt) {
+#pragma unroll
+            for (int nt = 0; nt < S::NT; ++nt) {
+                mx::joint_matrix_store(sg, acc[mt][nt],
+                    tile_c.template get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
+                    S::BN, mx::layout::row_major);
+            }
+        }
+        sycl::group_barrier(item.get_group());
+
+        // sum the K splits; consecutive lanes write consecutive rows of one dst column
+        for (int idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_SIZE) {
+            const int r = idx % S::SG_ROWS;
+            const int c = idx / S::SG_ROWS;
+            const int m = m0 + r;
+            const int n = n0 + c;
+            if (m < M && n < n1) {
+                float sum = 0.0f;
+#pragma unroll
+                for (int s = 0; s < FG_KSPLIT; ++s) {
+                    sum += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
+                }
+                dst[(size_t) n * ldd + m] = sum;
+            }
+        }
+    }
+}
+
+template <typename S, typename block_q_t>
+static void fused_dequant_gemm_launch(const void * src0, const typename S::tsb * packed, float * dst, const int M,
+                                      const int N, const int Npad, const int K, const int ldd,
+                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
+    stream->submit([&](sycl::handler & cgh) {
+        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
+        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
+        cgh.parallel_for(
+            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
+            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
+                const int n0 = item.get_group(0) * S::BN;
+                fused_dequant_gemm_tile<S, block_q_t, false>((const block_q_t *) src0, packed, dst, M, Npad, K, ldd,
+                                                      n0, n0, N, tile_a, tile_c, item);
+            });
+    });
+}
+
+// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
+template <typename S, typename block_q_t, bool reordered>
+static void grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
+                                        const ggml_sycl_gg_tile * tiles_ptr, const typename S::tsb * packed, float * dst,
+                                        const int M, const int Npad, const int K, const int64_t n_tiles,
+                                        const int64_t groups_m, dpct::queue_ptr stream) {
+    stream->submit([&](sycl::handler & cgh) {
+        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
+        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
+        cgh.parallel_for(
+            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
+            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
+                const int               t    = item.get_group(0);
+                const ggml_sycl_gg_tile tile = tiles_ptr[t];
+                const block_q_t *       x    = (const block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
+                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
+                                                                 tile.n1, tile_a, tile_c, item);
+            });
+    });
+}
+
+// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
+// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
+template <typename S>
+static void grouped_gemm_pack_b(const float * y, typename S::tsb * packed, const ggml_sycl_gg_tile * tiles, int Npad,
+                                int K, dpct::queue_ptr stream) {
+    using E        = typename S::EB;
+    constexpr int V = S::VNNI;
+    const int kqs   = K / V;
+    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
+        const size_t idx = id[0];
+        const int    kq  = idx / Npad;
+        const int    n   = idx - (size_t) kq * Npad;
+        const ggml_sycl_gg_tile tile = tiles[n / S::BN];
+        const int    row = tile.n0 + n % S::BN;
+        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
+        typename S::tsb vals[V] = {};
+        if (row < tile.n1) {
+            const float * src = y + (size_t) row * K + V * kq;
+#pragma unroll
+            for (int v = 0; v < V; ++v) {
+                vals[v] = E::cvt(src[v]);
+            }
+        }
+        typename S::tsb * out = packed + ((size_t) kq * Npad + n) * V;
+#pragma unroll
+        for (int v = 0; v < V; ++v) {
+            out[v] = vals[v];
+        }
+    });
+}
+
+// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
+// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
+template <typename T> static constexpr bool fg_plain_ok() {
+    return !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
+           !std::is_same_v<T, block_q6_K>;
+}
+
+template <typename T, bool R> struct fg_tag {
+    using type                     = T;
+    static constexpr bool reordered = R;
+};
+
+// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
+template <typename T, typename F> static bool fg_visit_layout(bool reordered, F && f) {
+    if (!reordered) {
+        f(fg_tag<T, false>{});
+        return true;
+    }
+    if constexpr (fg_soa<T>::supported) {
+        f(fg_tag<T, true>{});
+        return true;
+    }
+    return false;
+}
+
+template <typename F> static bool fg_visit_type(ggml_type type, bool reordered, F && f) {
+    switch (type) {
+        case GGML_TYPE_IQ4_NL:  return fg_visit_layout<block_iq4_nl>(reordered, f);
+        case GGML_TYPE_IQ3_S:   return fg_visit_layout<block_iq3_s>(reordered, f);
+        case GGML_TYPE_IQ4_XS:  return fg_visit_layout<block_iq4_xs>(reordered, f);
+        case GGML_TYPE_IQ3_XXS: return fg_visit_layout<block_iq3_xxs>(reordered, f);
+        case GGML_TYPE_IQ2_XXS: return fg_visit_layout<block_iq2_xxs>(reordered, f);
+        case GGML_TYPE_IQ2_XS:  return fg_visit_layout<block_iq2_xs>(reordered, f);
+        case GGML_TYPE_IQ2_S:   return fg_visit_layout<block_iq2_s>(reordered, f);
+        case GGML_TYPE_IQ1_S:   return fg_visit_layout<block_iq1_s>(reordered, f);
+        case GGML_TYPE_IQ1_M:   return fg_visit_layout<block_iq1_m>(reordered, f);
+        case GGML_TYPE_Q8_0:    return fg_visit_layout<block_q8_0>(reordered, f);
+        case GGML_TYPE_Q4_K:    return fg_visit_layout<block_q4_K>(reordered, f);
+        case GGML_TYPE_Q5_K:    return fg_visit_layout<block_q5_K>(reordered, f);
+        case GGML_TYPE_Q6_K:    return fg_visit_layout<block_q6_K>(reordered, f);
+        default:                return false;
+    }
+}
+
+template <typename S>
+static bool fg_fused_run(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
+                         float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
+                         dpct::queue_ptr stream) {
+    const int64_t groups_n = (N + S::BN - 1) / S::BN;
+    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
+    const int     Npad     = (int) (groups_n * S::BN);
+
+    // src1 is read in its own type: one pass, converted in registers only if B differs
+    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * Npad);
+    if (src1_type == GGML_TYPE_F16) {
+        fused_gemm_pack_b<typename S::EB>((const sycl::half *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
+    } else if (src1_type == GGML_TYPE_BF16) {
+        fused_gemm_pack_b<typename S::EB>((const fg_bf16 *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
+    } else {
+        fused_gemm_pack_b<typename S::EB>((const float *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
+    }
+
+    const typename S::tsb * packed = packed_b.get();
+    return fg_visit_type(src0_type, false, [&](auto tag) {
+        using block_q_t = typename decltype(tag)::type;
+        if constexpr (fg_plain_ok<block_q_t>()) {
+            fused_dequant_gemm_launch<S, block_q_t>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
+                                                    groups_n, groups_m, stream);
+        }
+    });
+}
+
+bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
+                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
+                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
+    // every BN columns dequantize A again, so wide N is left to the library GEMM
+    bool plain_ok = false;
+    fg_visit_type(src0_type, false, [&](auto tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
+    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
+        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
+        return false;
+    }
+    if (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
+        return false;
+    }
+    if (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
+        return false;
+    }
+    const int combo = fg_pick_combo(stream, src1_type, src1_prec);
+    if (combo < 0) {
+        return false;
+    }
+    bool launched = false;
+    fg_visit_combo(combo, [&](auto s) {
+        launched = fg_fused_run<decltype(s)>(src0_type, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
+    });
+    return launched;
+}
+
+template <typename S>
+static bool fg_grouped_run(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
+                           const float * src1, float * dst, const int64_t * expert_row_offsets, int64_t n_as, int64_t M,
+                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
+                           dpct::queue_ptr stream) {
+    // the host knows every slice, so it lays out the work-groups: no search on the device
+    tiles.clear();
+    for (int64_t e = 0; e < n_as; ++e) {
+        const int64_t end = expert_row_offsets[e + 1];
+        for (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
+            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
+        }
+    }
+    const int64_t n_tiles  = tiles.size();
+    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
+    const int     Npad     = (int) (n_tiles * S::BN);
+
+    const int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
+    GGML_ASSERT(n_tiles <= max_tiles);
+    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
+    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_gg_tile))));
+
+    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
+    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (int) K, stream);
+
+    const typename S::tsb *   packed    = packed_b.get();
+    const ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
+    const char *              src0_dd   = (const char *) src0_base;
+    return fg_visit_type(src0_type, reordered, [&](auto tag) {
+        using T = decltype(tag);
+        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
+                                                                       (int) M, Npad, (int) K, n_tiles, groups_m,
+                                                                       stream);
+    });
+}
+
+bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
+                                    const float * src1, int32_t src1_prec, float * dst,
+                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
+                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
+                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
+    int64_t n_active = 0;
+    for (int64_t e = 0; e < n_as; ++e) {
+        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
+    }
+    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
+        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {})) {
+        return false;
+    }
+    if (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows, n_active)) {
+        return false;
+    }
+    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
+    if (combo < 0) {
+        return false;
+    }
+    bool launched = false;
+    fg_visit_combo(combo, [&](auto s) {
+        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
+                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
+    });
+    return launched;
+}
diff --git src/ggml-sycl/fused-gemm.hpp src/ggml-sycl/fused-gemm.hpp
new file mode 100644
index 00000000..29898920
--- /dev/null
+++ src/ggml-sycl/fused-gemm.hpp
@@ -0,0 +1,88 @@
+#ifndef GGML_SYCL_FUSED_GEMM_HPP
+#define GGML_SYCL_FUSED_GEMM_HPP
+
+#include "common.hpp"
+
+
+// Shape and type gates for the kernels below. Device capability is separate: it needs a queue to ask.
+static constexpr int GGML_SYCL_FG_MAX_N = 64; // widest N taken; each shape covers it in BN-wide tiles
+
+// weight formats the fused A stage decodes; K must cover whole stored blocks
+constexpr bool ggml_sycl_fused_dequant_gemm_type_ok(ggml_type src0_type, int64_t K) {
+    // iq4_nl and q8_0 store 32 values per block; every other format here is a 256-value superblock
+    // that the A stage walks in steps of 32, so K must cover whole superblocks.
+    if (src0_type == GGML_TYPE_IQ4_NL || src0_type == GGML_TYPE_Q8_0) {
+        return K % 32 == 0;
+    }
+    const bool superblock =
+           src0_type == GGML_TYPE_Q4_K ||
+           src0_type == GGML_TYPE_Q5_K ||
+           src0_type == GGML_TYPE_Q6_K ||
+           src0_type == GGML_TYPE_IQ3_S ||
+           src0_type == GGML_TYPE_IQ4_XS ||
+           src0_type == GGML_TYPE_IQ3_XXS ||
+           src0_type == GGML_TYPE_IQ2_XXS ||
+           src0_type == GGML_TYPE_IQ2_XS ||
+           src0_type == GGML_TYPE_IQ2_S ||
+           src0_type == GGML_TYPE_IQ1_S ||
+           src0_type == GGML_TYPE_IQ1_M;
+    return superblock && QK_K == 256 && K % QK_K == 0;
+}
+
+constexpr bool ggml_sycl_fused_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t N, int64_t K,
+                                                     int64_t ldd) {
+    return ggml_sycl_fused_dequant_gemm_type_ok(src0_type, K) && M > 0 && N > 0 && K > 0 &&
+           N <= GGML_SYCL_FG_MAX_N &&
+           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && ldd <= INT32_MAX;
+}
+
+// grouped variant: the per-expert fused kernel is only worth it while each expert is narrow,
+// so wider average slices are left to the per-expert library GEMM loop
+constexpr bool ggml_sycl_grouped_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t K,
+                                                       int64_t total_rows, int64_t n_active) {
+    return ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, 1, K, M) && total_rows > 0 &&
+           total_rows <= INT32_MAX && total_rows <= n_active * GGML_SYCL_FG_MAX_N;
+}
+
+// Runtime type gate, kept out of the constexpr predicates above so those stay pure.
+inline bool ggml_sycl_xmx_gather_type_enabled(ggml_type src0_type) {
+    switch (src0_type) {
+        case GGML_TYPE_IQ4_NL:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_NL  ) != 0;
+        case GGML_TYPE_IQ3_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_S   ) != 0;
+        case GGML_TYPE_IQ4_XS:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_XS  ) != 0;
+        case GGML_TYPE_IQ3_XXS: return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_XXS ) != 0;
+        case GGML_TYPE_IQ2_XXS: return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XXS ) != 0;
+        case GGML_TYPE_IQ2_XS:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XS  ) != 0;
+        case GGML_TYPE_IQ2_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_S   ) != 0;
+        case GGML_TYPE_IQ1_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_S   ) != 0;
+        case GGML_TYPE_IQ1_M:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_M   ) != 0;
+        case GGML_TYPE_Q8_0:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q8_0    ) != 0;
+        case GGML_TYPE_Q4_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q4_K    ) != 0;
+        case GGML_TYPE_Q5_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q5_K    ) != 0;
+        case GGML_TYPE_Q6_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q6_K    ) != 0;
+        default:          return false;
+    }
+}
+
+// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1[n*K + k], src1 is F32, F16 or BF16.
+// The XMX combination is picked per call from the src1 type and its precision request src1_prec
+// (op_params[3], [TAG_GGML_PREC]); the accumulator is f32, which meets any request.
+// q8_0 and the k-quants are not handled here, only in the grouped path below.
+// Returns false when the case is not handled (type, device, precision, or shape).
+bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
+                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
+                                  ggml_sycl_pool & pool, dpct::queue_ptr stream);
+
+// One launch for every expert of a MUL_MAT_ID: rows of src1/dst are grouped by expert, expert e
+// owns rows [expert_row_offsets[e], expert_row_offsets[e+1]) and reads its weights at
+// src0_base + e*expert_stride. tiles is host scratch that must stay alive until the queue drains.
+// reordered: each expert slice is in the reorder (SoA) layout of reorder_qw().
+// dst[n*M + m] = sum_k dequant(src0_e)[m*K + k] * src1[n*K + k]
+// Returns false when the case is not handled (type, layout, device, precision, or shape).
+bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
+                                    const float * src1, int32_t src1_prec, float * dst,
+                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
+                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
+                                    ggml_sycl_pool & pool, dpct::queue_ptr stream);
+
+#endif // GGML_SYCL_FUSED_GEMM_HPP
diff --git src/ggml-sycl/fusion.cpp src/ggml-sycl/fusion.cpp
index d3e99523..79fe13a1 100644
--- src/ggml-sycl/fusion.cpp
+++ src/ggml-sycl/fusion.cpp
@@ -64,6 +64,12 @@ static bool ggml_sycl_should_fuse_mul_mat_glu(const ggml_tensor * gate, const gg
     return true;
 }
 
+// the unary ops the fused unary chains in element_wise.cpp have a functor for
+static bool ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
+    return unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
+           unary_op == GGML_UNARY_OP_SOFTPLUS;
+}
+
 bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                         std::initializer_list<enum ggml_unary_op> unary_ops) {
 #ifndef NDEBUG
@@ -184,9 +190,7 @@ bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializ
             return false;
         }
 
-        // the ops ggml_sycl_op_unary_mul_fused() has a kernel for
-        if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_SIGMOID &&
-            unary_op != GGML_UNARY_OP_SOFTPLUS) {
+        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
             return false;
         }
 
@@ -233,6 +237,55 @@ bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializ
         return true;
     }
 
+    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
+    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
+    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
+        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
+        const ggml_tensor * add   = cgraph->nodes[node_idx];
+        const ggml_tensor * unary = cgraph->nodes[node_idx + 1];
+        const ggml_tensor * mul   = cgraph->nodes[node_idx + 2];
+
+        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
+        if (unary_op != unary_ops.begin()[0]) {
+            return false;
+        }
+
+        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
+            return false;
+        }
+
+        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
+        // unary, add and unary have one use each, and all three have the same shape
+        const ggml_tensor * a     = add->src[0];
+        const ggml_tensor * bias  = add->src[1];
+        const ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
+
+        if (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
+            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
+            return false;
+        }
+
+        // the activation and the destination are indexed flat
+        if (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
+            return false;
+        }
+
+        // the kernel reads the bias and the scale as v[col], so each must be a single
+        // contiguous row spanning ne0
+        if (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
+            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
+            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
+            return false;
+        }
+
+        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
+        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
+            return false;
+        }
+
+        return true;
+    }
+
     if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
         ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
         const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
diff --git src/ggml-sycl/fwht.cpp src/ggml-sycl/fwht.cpp
index fb48d7fe..5af549c2 100644
--- src/ggml-sycl/fwht.cpp
+++ src/ggml-sycl/fwht.cpp
@@ -46,8 +46,8 @@ static constexpr float H20[20][20] = {
 #undef P
 #undef N
 
-template <int N>
-static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
+template <int N, typename T>
+static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                         const float scale, const sycl::nd_item<2> & item) {
     const sycl::sub_group sg = item.get_sub_group();
 
@@ -67,7 +67,7 @@ static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst
 
 #pragma unroll
     for (int i = 0; i < el_w; ++i) {
-        reg[i] = src[i * WARP_SIZE + lane] * scale;
+        reg[i] = static_cast<float>(src[i * WARP_SIZE + lane]) * scale;
     }
 
     // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
@@ -107,8 +107,8 @@ static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst
     }
 }
 
-template <int N>
-static void launch_fwht(const float * src, float * dst, const int64_t n_rows, const float scale,
+template <int N, typename T>
+static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale,
                         dpct::queue_ptr stream) {
     constexpr int rows_per_block = 4;
 
@@ -120,7 +120,7 @@ static void launch_fwht(const float * src, float * dst, const int64_t n_rows, co
 
     stream->parallel_for(sycl::nd_range<2>(global, local),
                          [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
-                             fwht_kernel<N>(src, dst, n_rows, scale, item);
+                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                          });
 }
 
@@ -128,8 +128,8 @@ static void launch_fwht(const float * src, float * dst, const int64_t n_rows, co
 // keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
 // still shuffle; those up to NT go through work-group local memory; the rest stay
 // in registers.
-template <int N, int NT>
-static void fwht_kernel_wide(const float * __restrict__ src,
+template <int N, int NT, typename T>
+static void fwht_kernel_wide(const T * __restrict__ src,
                              float * __restrict__ dst,
                              const int64_t            n_rows,
                              const float              scale,
@@ -151,7 +151,7 @@ static void fwht_kernel_wide(const float * __restrict__ src,
     float reg[el_w];
 #pragma unroll
     for (int i = 0; i < el_w; ++i) {
-        reg[i] = src[i * NT + tid] * scale;
+        reg[i] = static_cast<float>(src[i * NT + tid]) * scale;
     }
 
     const sycl::sub_group sg   = item.get_sub_group();
@@ -207,8 +207,8 @@ static void fwht_kernel_wide(const float * __restrict__ src,
     }
 }
 
-template <int N, int NT>
-static void launch_fwht_wide(const float *   src,
+template <int N, int NT, typename T>
+static void launch_fwht_wide(const T *       src,
                              float *         dst,
                              const int64_t   n_rows,
                              const float     scale,
@@ -220,13 +220,13 @@ static void launch_fwht_wide(const float *   src,
         sycl::local_accessor<float, 1> smem(sycl::range<1>(N), cgh);
         cgh.parallel_for(sycl::nd_range<2>(global, local),
                          [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
-                             fwht_kernel_wide<N, NT>(src, dst, n_rows, scale, item, get_pointer(smem));
+                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                          });
     });
 }
 
-template <int N, int m>
-static void kronecker_kernel(const float * __restrict__ src,
+template <int N, int m, typename T>
+static void kronecker_kernel(const T * __restrict__ src,
                              float * __restrict__ dst,
                              const int64_t            n_rows,
                              const float              scale,
@@ -255,7 +255,7 @@ static void kronecker_kernel(const float * __restrict__ src,
 
 #pragma unroll
         for (int j = 0; j < m; ++j) {
-            reg[i * m + j] = src[b_idx * m + j] * scale;
+            reg[i * m + j] = static_cast<float>(src[b_idx * m + j]) * scale;
         }
     }
 
@@ -321,8 +321,8 @@ static void kronecker_kernel(const float * __restrict__ src,
     }
 }
 
-template <int N, int m>
-static void launch_kronecker(const float *   src,
+template <int N, int m, typename T>
+static void launch_kronecker(const T *       src,
                              float *         dst,
                              const int64_t   n_rows,
                              const float     scale,
@@ -337,25 +337,16 @@ static void launch_kronecker(const float *   src,
 
     stream->parallel_for(sycl::nd_range<2>(global, local),
                          [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
-                             kronecker_kernel<N, m>(src, dst, n_rows, scale, item);
+                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                          });
 }
 
-bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
-    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
-        return false;
-    }
-    if (!ggml_are_same_shape(src, dst)) {
-        return false;
-    }
-    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
-        return false;
-    }
-
+template <typename T>
+static bool ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
     const int     n    = (int) src->ne[0];
     const int64_t rows = ggml_nrows(src);
 
-    const float *   src_d  = (const float *) src->data;
+    const T *       src_d  = (const T *) src->data;
     float *         dst_d  = (float *) dst->data;
     dpct::queue_ptr stream = ctx.stream();
 
@@ -402,3 +393,24 @@ bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src,
             return false;
     }
 }
+
+bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
+    if (dst->type != GGML_TYPE_F32) {
+        return false;
+    }
+    if (!ggml_are_same_shape(src, dst)) {
+        return false;
+    }
+    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
+        return false;
+    }
+
+    switch (src->type) {
+        case GGML_TYPE_F32:
+            return ggml_sycl_op_fwht_impl<float>(ctx, src, dst);
+        case GGML_TYPE_F16:
+            return ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
+        default:
+            return false;
+    }
+}
diff --git src/ggml-sycl/ggml-sycl.cpp src/ggml-sycl/ggml-sycl.cpp
index 49148d7c..2a5184e0 100644
--- src/ggml-sycl/ggml-sycl.cpp
+++ src/ggml-sycl/ggml-sycl.cpp
@@ -14,6 +14,7 @@
 #include <array>
 #include <assert.h>
 #include <atomic>
+#include <cctype>
 #include <cinttypes>
 #include <cstddef>
 #include <cstdint>
@@ -60,6 +61,7 @@
 #include "ggml-sycl/common.hpp"
 #include "ggml-sycl/element_wise.hpp"
 #include "ggml-sycl/fwht.hpp"
+#include "ggml-sycl/fused-gemm.hpp"
 #include "ggml-sycl/gemm.hpp"
 #include "ggml-sycl/getrows.hpp"
 #include "ggml-sycl/mem.hpp"
@@ -105,6 +107,30 @@ int g_ggml_sycl_enable_fusion = 1;
 int g_ggml_sycl_enable_esimd = 1;
 int g_ggml_sycl_mmvq_wide = 1;
 int g_ggml_sycl_prioritize_dmmv = 0;
+int g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
+int g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
+int g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
+int g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
+static const char * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };
+
+// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
+static int ggml_sycl_get_env_precision(const char * name, int def) {
+    const char * env = getenv(name);
+    if (!env) {
+        return def;
+    }
+    std::string mode(env);
+    for (char & c : mode) {
+        c = (char) std::toupper((unsigned char) c);
+    }
+    for (int i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
+        if (mode == ggml_sycl_dynamic_precision_names[i]) {
+            return i;
+        }
+    }
+    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
+    return def;
+}
 int g_ggml_sycl_use_async_mem_op = 0;
 int g_ggml_sycl_use_async_mem_op_requested = 1;
 int g_ggml_sycl_use_level_zero_api = 0;
@@ -113,6 +139,7 @@ int g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
 int g_ggml_sycl_usm_system = 0;
 int g_ggml_sycl_enable_host_pinned_mem = 1;
 int g_ggml_sycl_host_pinned_mem_2g = 0;
+int g_ggml_sycl_upload_staging_slots = 4;
 int g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
 int g_ggml_sycl_enable_sparse_fa = 0;
 int g_ggml_sycl_debug_sparse_fa = 0;
@@ -401,6 +428,12 @@ static void ggml_check_sycl() try {
         g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
         g_ggml_sycl_mmvq_wide = ggml_sycl_get_env("GGML_SYCL_MMVQ_WIDE", 1);
         g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
+        g_ggml_sycl_xmx_gather_types = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_TYPES", GGML_SYCL_XMX_GATHER_TYPES_DEFAULT);
+        g_ggml_sycl_xmx_gather_shapes = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_SHAPES", GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT);
+        g_ggml_sycl_dynamic_precision =
+            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_DEFAULT);
+        g_ggml_sycl_dynamic_required_precision =
+            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_REQUIRED_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_F32);
 
 #ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
         g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
@@ -426,6 +459,7 @@ static void ggml_check_sycl() try {
 
         g_ggml_sycl_host_pinned_mem_2g =
             ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;
+        g_ggml_sycl_upload_staging_slots = std::max(0, ggml_sycl_get_env("GGML_SYCL_UPLOAD_STAGING_SLOTS", 4));
 
         g_ggml_sycl_enable_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA", 0);
         g_ggml_sycl_debug_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_DEBUG", 0);
@@ -509,6 +543,12 @@ static void ggml_check_sycl() try {
 #endif
 
         GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
+        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
+        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
+        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
+                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
+        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
+                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);
 
 #if defined(GGML_SYCL_SUPPORT_VMM)
         GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
@@ -517,6 +557,7 @@ static void ggml_check_sycl() try {
 #endif
 
         GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
+        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);
 
 #if defined(__INTEL_LLVM_COMPILER)
         GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
@@ -629,12 +670,23 @@ inline void free_aligned_mem_host(void * memblock) {
 // sycl buffer
 
 struct ggml_backend_sycl_buffer_context {
+    // pinned staging for uploads; the host fills one slot while the previous one transfers
+    static constexpr size_t staging_slot_size = 8*1024*1024;
+
+    struct host_staging {
+        void * data = nullptr;
+        std::vector<sycl::event> events;
+        std::vector<bool> submitted;
+        int next = 0;
+    };
+
     int device;
     void * dev_ptr = nullptr;
     queue_ptr stream;
     std::string name;
     optimize_feature opt_feature;
     std::vector<ggml_tensor_extra_gpu *> tensor_extras;
+    host_staging staging;
     bool is_usm_system;
 
     ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream, bool is_usm_system) :
@@ -644,7 +696,22 @@ struct ggml_backend_sycl_buffer_context {
             opt_feature = ggml_sycl_info().devices[device].opt_feature;
         }
 
+    // waits for every queued upload, then releases the pinned block
+    void drop_host_staging() {
+        for (size_t i = 0; i < staging.submitted.size(); ++i) {
+            if (staging.submitted[i]) {
+                staging.events[i].wait_and_throw();
+                staging.submitted[i] = false;
+            }
+        }
+        if (staging.data != nullptr) {
+            sycl::free(staging.data, *stream);
+            staging.data = nullptr;
+        }
+    }
+
     ~ggml_backend_sycl_buffer_context() {
+        drop_host_staging();
         if (dev_ptr != nullptr) {
             ggml_sycl_set_device(device);
             if (is_usm_system)
@@ -745,6 +812,40 @@ static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
     GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
     ggml_sycl_set_device(ctx->device);
+
+    // copy through pinned memory so the device never reads mmap()ed pages directly
+    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
+    const int n_slots = g_ggml_sycl_upload_staging_slots;
+    if (n_slots > 0 && ctx->staging.data == nullptr) {
+        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
+        if (ctx->staging.data != nullptr) {
+            ctx->staging.events.resize(n_slots);
+            ctx->staging.submitted.assign(n_slots, false);
+        }
+    }
+    if (ctx->staging.data != nullptr) {
+        queue_ptr    stream    = ctx->stream;
+        char *       dst       = (char *) tensor->data + offset;
+        const char * src       = (const char *) data;
+        size_t       remaining = size;
+        while (remaining > 0) {
+            const size_t chunk = std::min(remaining, ctx->staging_slot_size);
+            const int    slot  = ctx->staging.next;
+            ctx->staging.next = (ctx->staging.next + 1) % (int) ctx->staging.submitted.size();
+            if (ctx->staging.submitted[slot]) {
+                ctx->staging.events[slot].wait_and_throw();
+            }
+            void * stage = (char *) ctx->staging.data + slot * ctx->staging_slot_size;
+            memcpy(stage, src, chunk);
+            ctx->staging.events[slot] = stream->memcpy(dst, stage, chunk);
+            ctx->staging.submitted[slot] = true;
+            src       += chunk;
+            dst       += chunk;
+            remaining -= chunk;
+        }
+        return;
+    }
+
     auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
     SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
 #ifndef _WIN32
@@ -792,6 +893,14 @@ static bool ggml_sycl_is_l0_discrete_gpu(int device) {
 }
 #endif
 
+static void memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst,
+                         const void *ptr_src, size_t size) {
+    char *host_buf = (char *)malloc(size);
+    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
+    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
+    free(host_buf);
+}
+
 static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, sycl::queue &q_src, void *ptr_dst,
                     const void *ptr_src, size_t size) {
 
@@ -835,10 +944,7 @@ static void dev2dev_memcpy(int device_dst, sycl::queue &q_dst, int device_src, s
     } else {
         GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
     }
-    char *host_buf = (char *)malloc(size);
-    q_src.memcpy(host_buf, (const char *)ptr_src, size).wait();
-    q_dst.memcpy((char *)ptr_dst, host_buf, size).wait();
-    free(host_buf);
+    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
 }
 
 static bool
@@ -2064,11 +2170,6 @@ std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(q
     return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
 }
 
-
-std::unique_ptr<ggml_sycl_fattn_kv_buffers> ggml_backend_sycl_context::new_fattn_kv_buffers(queue_ptr qptr, int device) {
-    return std::unique_ptr<ggml_sycl_fattn_kv_buffers>(new ggml_sycl_fattn_kv_buffers(qptr, device));
-}
-
 /// kernels
 typedef void (*ggml_sycl_op_mul_mat_t)(
     ggml_backend_sycl_context & ctx,
@@ -3027,22 +3128,18 @@ inline void ggml_sycl_op_mul_mat_sycl(
     }
 #endif
 
+    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
+    // goes in its own type, so there is no separate conversion pass
+    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
+        ggml_sycl_fused_dequant_gemm(src0->type, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
+                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
+        return;
+    }
+
+    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
+    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
     if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
         row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
-        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
-        if (src0->type != GGML_TYPE_F16) {
-            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
-                                                 " : converting src0 to fp16");
-            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
-            GGML_ASSERT(to_fp16_sycl != nullptr);
-            size_t ne = row_diff*ne00;
-            src0_as_f16.alloc(ne);
-            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
-        }
-        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
-                                         ? (const sycl::half *)src0_dd_i
-                                         : src0_as_f16.get();
-
         ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
         if (src1->type != GGML_TYPE_F16) {
             scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
@@ -3057,6 +3154,20 @@ inline void ggml_sycl_op_mul_mat_sycl(
                 ? (const sycl::half *)src1->data + src1_padded_row_size
                                          : src1_as_f16.get();
 
+        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
+        if (src0->type != GGML_TYPE_F16) {
+            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
+                                                 " : converting src0 to fp16");
+            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
+            GGML_ASSERT(to_fp16_sycl != nullptr);
+            size_t ne = row_diff*ne00;
+            src0_as_f16.alloc(ne);
+            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
+        }
+        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
+                                         ? (const sycl::half *)src0_dd_i
+                                         : src0_as_f16.get();
+
 #if GGML_SYCL_DNNL
         if (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                 DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
@@ -4859,6 +4970,10 @@ static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor
 
     // check data types and tensor shapes for custom matrix multiplication kernels:
     bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
+#ifdef GGML_SYCL_F16
+    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
+    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
+#endif
 
     bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);
 
@@ -5272,7 +5387,9 @@ static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
     SYCL_CHECK(CHECK_TRY_ERROR(
         stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
 
-    // also ensures ctx.mmid_row_mapping_host is drained before we use it again
+    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
+    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
+    // this wait would let the next node overwrite a buffer the device is still reading
     SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
 
     ggml_tensor src0_row = *src0;
@@ -5363,7 +5480,25 @@ static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
             });
         }
 
-        for (int64_t i02 = 0; i02 < n_as; i02++) {
+        bool grouped = false;
+        if (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
+            dst->type == GGML_TYPE_F32 && nb11 == sizeof(float)*ne10 && nb1 == sizeof(float)*ne0) {
+            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
+            // anyway: install it here already, so prefill does not depend on whether a decode ran before
+            if (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
+                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
+                opt_for_reorder_id(&ctx, src0);
+            }
+            const bool src0_reordered =
+                src0->extra && ((const ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
+            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
+                                                     (const float *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
+                                                     (float *) dst_contiguous.get(),
+                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
+                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
+        }
+
+        for (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
             const int64_t num_src1_rows = expert_row_counts[i02];
 
             if (num_src1_rows == 0) {
@@ -6177,6 +6312,16 @@ static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * syc
             i++;
             continue;
         }
+        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
+        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
+        if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
+            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
+            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
+                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
+            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
+            i += 2;
+            continue;
+        }
 
         // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
         // norms) into one launch; sources are strided views of the fused qkv buffer, so
diff --git src/ggml-sycl/im2col.hpp src/ggml-sycl/im2col.hpp
index 976d1094..c9eba49e 100644
--- src/ggml-sycl/im2col.hpp
+++ src/ggml-sycl/im2col.hpp
@@ -15,8 +15,6 @@
 
 #include "common.hpp"
 
-#define SYCL_IM2COL_BLOCK_SIZE 256
-
 void ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
 void ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
 
diff --git src/ggml-sycl/memtrace.cpp src/ggml-sycl/memtrace.cpp
index 9c4f8853..ecd82ff6 100644
--- src/ggml-sycl/memtrace.cpp
+++ src/ggml-sycl/memtrace.cpp
@@ -15,7 +15,6 @@ static const char * mem_type_name(ggml_sycl_mem_type type) {
         case GGML_SYCL_MEM_POOL_LEG: return "pool_leg";
         case GGML_SYCL_MEM_POOL_VMM: return "pool_vmm";
         case GGML_SYCL_MEM_ASYNC:    return "async";
-        case GGML_SYCL_MEM_FATTN_KV: return "fattn_kv";
         case GGML_SYCL_MEM_DIRECT:   return "direct";
         default:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (int) type);
     }
diff --git src/ggml-sycl/memtrace.hpp src/ggml-sycl/memtrace.hpp
index 426d9096..c7da47f3 100644
--- src/ggml-sycl/memtrace.hpp
+++ src/ggml-sycl/memtrace.hpp
@@ -10,7 +10,6 @@ enum ggml_sycl_mem_type {
     GGML_SYCL_MEM_POOL_LEG,
     GGML_SYCL_MEM_POOL_VMM,
     GGML_SYCL_MEM_ASYNC,
-    GGML_SYCL_MEM_FATTN_KV,
     GGML_SYCL_MEM_DIRECT,
 
     GGML_SYCL_MEM_TYPE_COUNT,
diff --git src/ggml-sycl/mmvq.cpp src/ggml-sycl/mmvq.cpp
index d0f09070..ceaacb59 100644
--- src/ggml-sycl/mmvq.cpp
+++ src/ggml-sycl/mmvq.cpp
@@ -6,6 +6,39 @@
 #include "quants.hpp"
 #include "vecdotq.hpp"
 
+// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
+// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
+// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
+static __dpct_inline__ float vec_dot_iq2_xxs_q8_1_moe(const void * __restrict__ vbq,
+                                                      const block_q8_1 * __restrict__ bq8_1,
+                                                      const int & iqs) {
+    return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
+}
+
+static __dpct_inline__ float vec_dot_iq2_xs_q8_1_moe(const void * __restrict__ vbq,
+                                                     const block_q8_1 * __restrict__ bq8_1,
+                                                     const int & iqs) {
+    return vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid, ksigns64);
+}
+
+static __dpct_inline__ float vec_dot_iq3_xxs_q8_1_moe(const void * __restrict__ vbq,
+                                                      const block_q8_1 * __restrict__ bq8_1,
+                                                      const int & iqs) {
+    return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid, ksigns64);
+}
+
+static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(const void * __restrict__ vbq,
+                                                    const block_q8_1 * __restrict__ bq8_1,
+                                                    const int & iqs) {
+    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
+}
+
+static __dpct_inline__ float vec_dot_iq1_s_q8_1_moe(const void * __restrict__ vbq,
+                                                    const block_q8_1 * __restrict__ bq8_1,
+                                                    const int & iqs) {
+    return vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
+}
+
 // Minimum weight-row count at which the Q4_K multi-column MMVQ kernel handles two output rows per
 // subgroup (rows_per_sg == 2) instead of one, when ncols_dst == 2.
 //
@@ -2190,6 +2223,67 @@ static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
     }
 }
 
+template <int ncols_dst>
+static void mul_mat_vec_iq3_s_q8_1_sycl_ncols(const void *    vx,
+                                              const void *    vy,
+                                              float *         dst,
+                                              const int       ncols,
+                                              const int       nrows,
+                                              const int       stride_col_y,
+                                              const int       stride_col_dst,
+                                              dpct::queue_ptr stream) {
+    GGML_ASSERT(ncols % QK_K == 0);
+    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
+    const sycl::range<3> block_nums(1, 1, block_num_y);
+    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
+    stream->submit([&](sycl::handler & cgh) {
+        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
+                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
+                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, block_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
+                                 vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
+                         });
+    });
+}
+
+static void mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(const void *    vx,
+                                                     const void *    vy,
+                                                     float *         dst,
+                                                     const int       ncols,
+                                                     const int       nrows,
+                                                     const int       ncols_dst,
+                                                     const int       stride_col_y,
+                                                     const int       stride_col_dst,
+                                                     dpct::queue_ptr stream) {
+    switch (ncols_dst) {
+        case 1:
+            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
+            break;
+        case 2:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        case 3:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        case 4:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        case 5:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        case 6:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        case 7:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        case 8:
+            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
+            break;
+        default:
+            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
+    }
+}
+
 static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                           float *dst, const int ncols,
                                           const int nrows,
@@ -2627,7 +2721,16 @@ void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tens
                 mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                 break;
             case GGML_TYPE_IQ3_S:
-                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
+                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
+                    const int stride_col_y   = src1_padded_col_size / QK8_1;
+                    const int stride_col_dst = dst->ne[0];
+                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (int) src1_ncols);
+                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
+                                                             src1_ncols, stride_col_y, stride_col_dst, stream);
+                    return;
+                } else if (i == 0 || src1_ncols == 1) {
+                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
+                }
                 break;
             case GGML_TYPE_IQ4_NL:
                 mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
@@ -2681,34 +2784,6 @@ void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tens
     GGML_UNUSED(ctx);
 }
 
-// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
-// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
-// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
-static __dpct_inline__ float vec_dot_iq2_xxs_q8_1_moe(const void * __restrict__ vbq,
-                                                      const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
-    return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
-}
-
-static __dpct_inline__ float vec_dot_iq2_xs_q8_1_moe(const void * __restrict__ vbq,
-                                                     const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
-    return vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid, ksigns64);
-}
-
-static __dpct_inline__ float vec_dot_iq3_xxs_q8_1_moe(const void * __restrict__ vbq,
-                                                      const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
-    return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid, ksigns64);
-}
-
-static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(const void * __restrict__ vbq,
-                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
-    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
-}
-
-static __dpct_inline__ float vec_dot_iq1_s_q8_1_moe(const void * __restrict__ vbq,
-                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
-    return vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
-}
-
 // src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
 template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
 static void mul_mat_vec_q_moe(
diff --git src/ggml-sycl/pad.hpp src/ggml-sycl/pad.hpp
index b099e9b7..4bc5a2cf 100644
--- src/ggml-sycl/pad.hpp
+++ src/ggml-sycl/pad.hpp
@@ -15,8 +15,6 @@
 
 #include "common.hpp"
 
-#define SYCL_PAD_BLOCK_SIZE 256
-
 void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
 
 void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
diff --git src/ggml-sycl/rope.hpp src/ggml-sycl/rope.hpp
index b95a5858..94a76989 100644
--- src/ggml-sycl/rope.hpp
+++ src/ggml-sycl/rope.hpp
@@ -15,8 +15,6 @@
 
 #include "common.hpp"
 
-#define SYCL_ROPE_BLOCK_SIZE 256
-
 void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);
 
 void ggml_sycl_rope_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
diff --git src/ggml-sycl/upscale.hpp src/ggml-sycl/upscale.hpp
index c36c1bdc..2b72ea1b 100644
--- src/ggml-sycl/upscale.hpp
+++ src/ggml-sycl/upscale.hpp
@@ -4,6 +4,4 @@
 #include "dpct/helper.hpp"
 #include "common.hpp"
 
-#define SYCL_UPSCALE_BLOCK_SIZE 256
-
 void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
diff --git src/ggml-vulkan/ggml-vulkan-buffers.cpp src/ggml-vulkan/ggml-vulkan-buffers.cpp
index f7a21dd2..35195163 100644
--- src/ggml-vulkan/ggml-vulkan-buffers.cpp
+++ src/ggml-vulkan/ggml-vulkan-buffers.cpp
@@ -623,7 +623,10 @@ void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t sp
     // If the device is not an UMA device the memory is host-accessible through rebar. While writing
     // through PCIe is sufficient fast reading back data from PCIe is slower than going through
     // the HW device to host copy path.
-    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
+    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
+    const bool slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
+                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
+    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
         GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
 
         std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
diff --git src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan.cpp
index 5587dfc2..998cc693 100644
--- src/ggml-vulkan/ggml-vulkan.cpp
+++ src/ggml-vulkan/ggml-vulkan.cpp
@@ -5189,10 +5189,16 @@ void ggml_vk_instance_init() {
     // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
     ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);
 
+    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
+    if (ggml_vk_default_dispatcher_instance.vkEnumerateInstanceVersion == nullptr) {
+        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
+        throw vk::SystemError(vk::Result::eErrorFeatureNotPresent, "Vulkan 1.2 required");
+    }
+
     uint32_t api_version = vk::enumerateInstanceVersion();
 
     if (api_version < VK_API_VERSION_1_2) {
-        std::cerr << "ggml_vulkan: Error: Vulkan 1.2 required." << std::endl;
+        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
         throw vk::SystemError(vk::Result::eErrorFeatureNotPresent, "Vulkan 1.2 required");
     }
 
@@ -8148,10 +8154,14 @@ void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const
     // cm2 dense is fast, so it needs a larger reduction to win.
     // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
     const int64_t min_ratio = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
+    // coopmat2 vector decode requires 8B strides.
+    auto sparse_gather_aligned = [](const ggml_tensor * t) {
+        return (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
+               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * sizeof(ggml_fp16_t)) == 0;
+    };
     const bool use_sparse = !disable_sparse && n_kv_max > 0 && mask &&
                             max_bias == 0.0f && logit_softcap == 0.0f &&
-                            // the cm2 sparse gather only reads f16
-                            (kv_f16 || tuning_params.path != FA_COOPMAT2) &&
+                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                             nem0 == KV &&
                             (int64_t)KV >= std::max<int64_t>(4096, min_ratio * (int64_t)n_kv_max) &&
                             (gqa_ratio > 1 || (tuning_params.path == FA_SCALAR && N == 1));
diff --git src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp
index c6ed63dd..c7253784 100644
--- src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp
+++ src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp
@@ -18,7 +18,8 @@
 #ifdef GL_NV_cooperative_matrix_decode_vector
 #extension GL_NV_cooperative_matrix_decode_vector : enable
 #endif
-#extension GL_EXT_buffer_reference : enable
+#extension GL_EXT_buffer_reference2 : enable
+#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
 #extension GL_KHR_shader_subgroup_ballot : enable
 #extension GL_KHR_shader_subgroup_vote : enable
 #extension GL_EXT_null_initializer : enable
@@ -35,6 +36,10 @@
 #define FA_GATHER_BS 1u
 #endif
 
+layout(buffer_reference, std430, buffer_reference_align = 1) buffer decodeBufFA_Byte {
+    uint8_t raw;
+};
+
 // buffer_reference stride = sizeof(struct) = FaBlockBytesK/V.
 layout(buffer_reference, std430, buffer_reference_align = 1) buffer decodeBufFA_K {
     uint8_t raw[FaBlockBytesK];
@@ -113,48 +118,71 @@ layout (binding = 1) readonly buffer K {uint8_t data_k[];};
 layout (binding = 2) readonly buffer V {uint8_t data_v[];};
 layout (binding = 3) readonly buffer M {uint8_t data_m[];};
 
-// f16 aliases for the sparse gather callbacks.
-layout (binding = 1) readonly buffer KF16 {float16_t data_kf16[];};
-layout (binding = 2) readonly buffer VF16 {float16_t data_vf16[];};
+// Native 16-bit aliases for the sparse gather callbacks.
+layout (binding = 1) readonly buffer K16 {FLOAT_TYPE data_k16[];};
+layout (binding = 2) readonly buffer V16 {FLOAT_TYPE data_v16[];};
 layout (binding = 3) readonly buffer MF16 {float16_t data_mf16[];};
 #ifdef GL_NV_cooperative_matrix_decode_vector
-layout (binding = 1) readonly buffer KF16V4 {f16vec4 data_kf16v4[];};
-layout (binding = 2) readonly buffer VF16V4 {f16vec4 data_vf16v4[];};
+layout (binding = 1) readonly buffer K16V4 {FLOAT_TYPEV4 data_k16v4[];};
+layout (binding = 2) readonly buffer V16V4 {FLOAT_TYPEV4 data_v16v4[];};
 #endif
 
-// K/V/mask f16-element offsets for the current head/batch, set in main().
+// K/V/mask offsets in 16-bit elements for the current head/batch, set in main().
 uint32_t g_k_off_elem, g_v_off_elem, g_m_off_elem;
 
-#if !defined(BFLOAT16)
-// blockCoords are in block units: KV slot = blockCoords[0],
-// head dim = blockCoords[1]*FA_GATHER_BS + coordInBlock[1].
-float16_t faGatherK(const decodeBufFA_K unused, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
-    if (blockCoords[0] >= p.split_kv) { return float16_t(0); }
+FLOAT_TYPE faGatherK(const decodeBufFA_K bl_in, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
+    if (blockCoords[0] >= p.split_kv) { return FLOAT_TYPE(0.0); }
     const int r = data_sparse[sparse_base + blockCoords[0]];
-    return r < 0 ? float16_t(0) : data_kf16[g_k_off_elem + uint(r) * k_stride + blockCoords[1] * FA_GATHER_BS + coordInBlock[1]];
+    if (r < 0) { return FLOAT_TYPE(0.0); }
+#if !defined(BFLOAT16)
+    if (USE_DECODE_K) {
+        decodeBufFA_K block = decodeBufFA_K(decodeBufFA_Byte(bl_in) + uint64_t(uint(r) - blockCoords[0]) * k_stride * FaBlockBytesK);
+        return faDecodeK(block, blockCoords, coordInBlock);
+    }
+#endif
+    return data_k16[g_k_off_elem + uint(r) * k_stride + blockCoords[1] * FA_GATHER_BS + coordInBlock[1]];
 }
 
-float16_t faGatherV(const decodeBufFA_V unused, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
-    if (blockCoords[0] >= p.split_kv) { return float16_t(0); }
+FLOAT_TYPE faGatherV(const decodeBufFA_V bl_in, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
+    if (blockCoords[0] >= p.split_kv) { return FLOAT_TYPE(0.0); }
     const int r = data_sparse[sparse_base + blockCoords[0]];
-    return r < 0 ? float16_t(0) : data_vf16[g_v_off_elem + uint(r) * v_stride + blockCoords[1] * FA_GATHER_BS + coordInBlock[1]];
+    if (r < 0) { return FLOAT_TYPE(0.0); }
+#if !defined(BFLOAT16)
+    if (USE_DECODE_V) {
+        decodeBufFA_V block = decodeBufFA_V(decodeBufFA_Byte(bl_in) + uint64_t(uint(r) - blockCoords[0]) * v_stride * FaBlockBytesV);
+        return faDecodeV(block, blockCoords, coordInBlock);
+    }
+#endif
+    return data_v16[g_v_off_elem + uint(r) * v_stride + blockCoords[1] * FA_GATHER_BS + coordInBlock[1]];
 }
 
 #ifdef GL_NV_cooperative_matrix_decode_vector
-f16vec4 faGatherKVector(const decodeBufFA_K unused, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
-    if (blockCoords[0] >= p.split_kv) { return f16vec4(0); }
+FLOAT_TYPEV4 faGatherKVector(const decodeBufFA_K bl_in, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
+    if (blockCoords[0] >= p.split_kv) { return FLOAT_TYPEV4(0.0); }
     const int r = data_sparse[sparse_base + blockCoords[0]];
-    if (r < 0) { return f16vec4(0); }
+    if (r < 0) { return FLOAT_TYPEV4(0.0); }
+#if !defined(BFLOAT16)
+    if (USE_DECODE_K) {
+        decodeBufFA_K block = decodeBufFA_K(decodeBufFA_Byte(bl_in) + uint64_t(uint(r) - blockCoords[0]) * k_stride * FaBlockBytesK);
+        return faDecodeKVector(block, blockCoords, coordInBlock);
+    }
+#endif
     const uint32_t o = g_k_off_elem + uint(r) * k_stride + blockCoords[1] * FA_GATHER_BS + coordInBlock[1];
-    return data_kf16v4[o / 4];
+    return data_k16v4[o / 4];
 }
 
-f16vec4 faGatherVVector(const decodeBufFA_V unused, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
-    if (blockCoords[0] >= p.split_kv) { return f16vec4(0); }
+FLOAT_TYPEV4 faGatherVVector(const decodeBufFA_V bl_in, const uint32_t blockCoords[2], const uint32_t coordInBlock[2]) {
+    if (blockCoords[0] >= p.split_kv) { return FLOAT_TYPEV4(0.0); }
     const int r = data_sparse[sparse_base + blockCoords[0]];
-    if (r < 0) { return f16vec4(0); }
+    if (r < 0) { return FLOAT_TYPEV4(0.0); }
+#if !defined(BFLOAT16)
+    if (USE_DECODE_V) {
+        decodeBufFA_V block = decodeBufFA_V(decodeBufFA_Byte(bl_in) + uint64_t(uint(r) - blockCoords[0]) * v_stride * FaBlockBytesV);
+        return faDecodeVVector(block, blockCoords, coordInBlock);
+    }
+#endif
     const uint32_t o = g_v_off_elem + uint(r) * v_stride + blockCoords[1] * FA_GATHER_BS + coordInBlock[1];
-    return data_vf16v4[o / 4];
+    return data_v16v4[o / 4];
 }
 
 #define FAGATHERK , faGatherK, faGatherKVector
@@ -163,7 +191,6 @@ f16vec4 faGatherVVector(const decodeBufFA_V unused, const uint32_t blockCoords[2
 #define FAGATHERK , faGatherK
 #define FAGATHERV , faGatherV
 #endif
-#endif
 
 // Add gathered mask to S (slope==1 since sparse requires max_bias==0). col = slot in block jblk.
 ACC_TYPE faAddSparseMask(const uint32_t row, const uint32_t col, const ACC_TYPE elem, const uint32_t jblk) {
@@ -252,8 +279,8 @@ void main() {
 
     tensorViewNV<2, false, 1, 0> tensorViewTranspose = createTensorViewNV(2, false, 1, 0);
 
-    const uint bs_k = USE_SPARSE ? FA_GATHER_BS : fa_block_elems(FaTypeK);
-    const uint bs_v = USE_SPARSE ? FA_GATHER_BS : fa_block_elems(FaTypeV);
+    const uint bs_k = USE_SPARSE ? max(FA_GATHER_BS, BLOCK_SIZE_K) : BLOCK_SIZE_K;
+    const uint bs_v = USE_SPARSE ? max(FA_GATHER_BS, BLOCK_SIZE_V) : BLOCK_SIZE_V;
     tensorLayoutK = setTensorLayoutBlockSizeNV(tensorLayoutK, 1, bs_k);
     tensorLayoutV = setTensorLayoutBlockSizeNV(tensorLayoutV, 1, bs_v);
 
@@ -384,18 +411,15 @@ void main() {
 
         uint32_t k_offset = ik2*p.nb12 + ik3*p.nb13;
         // F16: bs_k==1 (direct load). F32: bs_k==4 (vec4 / dequantFuncF32). Quantized types: bs_k==32.
-#if defined(BFLOAT16)
-        coopMatLoadTensorNV(K_T, data_k, k_offset, sliceTensorLayoutNV(tensorLayoutK, j * Bc, Bc, 0, HSK_pad), tensorViewTranspose);
-#else
-        const bool k_use_decode = (bs_k > 1u);
         if (USE_SPARSE) {
             coopMatLoadTensorNV(K_T, data_k, k_offset, sliceTensorLayoutNV(tensorLayoutK, j * Bc, Bc, 0, HSK_pad), tensorViewTranspose FAGATHERK);
-        } else if (k_use_decode) {
+#if !defined(BFLOAT16)
+        } else if (USE_DECODE_K) {
             coopMatLoadTensorNV(K_T, data_k, k_offset, sliceTensorLayoutNV(tensorLayoutK, j * Bc, Bc, 0, HSK_pad), tensorViewTranspose FADECODEK);
+#endif
         } else {
             coopMatLoadTensorNV(K_T, data_k, k_offset, sliceTensorLayoutNV(tensorLayoutK, j * Bc, Bc, 0, HSK_pad), tensorViewTranspose);
         }
-#endif
         S = coopMatMulAdd(Qf16, K_T, S);
 
         if (LOGIT_SOFTCAP) {
@@ -458,18 +482,15 @@ void main() {
 
         coopmat<FLOAT_TYPE, gl_ScopeWorkgroup, Bc, HSV_pad, gl_MatrixUseB> V;
         uint32_t v_offset = iv2*p.nb22 + iv3*p.nb23;
-#if defined(BFLOAT16)
-        coopMatLoadTensorNV(V, data_v, v_offset, sliceTensorLayoutNV(tensorLayoutV, j * Bc, Bc, 0, HSV_pad));
-#else
-        const bool v_use_decode = (bs_v > 1u);
         if (USE_SPARSE) {
             coopMatLoadTensorNV(V, data_v, v_offset, sliceTensorLayoutNV(tensorLayoutV, j * Bc, Bc, 0, HSV_pad) FAGATHERV);
-        } else if (v_use_decode) {
+#if !defined(BFLOAT16)
+        } else if (USE_DECODE_V) {
             coopMatLoadTensorNV(V, data_v, v_offset, sliceTensorLayoutNV(tensorLayoutV, j * Bc, Bc, 0, HSV_pad) FADECODEV);
+#endif
         } else {
             coopMatLoadTensorNV(V, data_v, v_offset, sliceTensorLayoutNV(tensorLayoutV, j * Bc, Bc, 0, HSV_pad));
         }
-#endif
 
         L = eM*L + rowsum;
 
diff --git src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp
index 0b757f38..3f6296bd 100644
--- src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp
+++ src/ggml-vulkan/vulkan-shaders/topk_nary_search.comp
@@ -60,7 +60,13 @@ void topk(const uint row) {
     if (gl_GlobalInvocationID.x < p.ncols_input) {
         if (p.first_pass != 0) {
             const uint row_offset = row * p.ncols_input;
-            dst_row[tid] = ivec2(gl_GlobalInvocationID.x, floatBitsToInt(data_a[row_offset + gl_GlobalInvocationID.x]));
+            // NaN ranks lowest, like -inf, so that every value has a place in
+            // the ordering the search below counts
+            float a = float(data_a[row_offset + gl_GlobalInvocationID.x]);
+            if (isnan(a)) {
+                a = uintBitsToFloat(0xFF800000);
+            }
+            dst_row[tid] = ivec2(gl_GlobalInvocationID.x, floatBitsToInt(a));
         } else {
             const uint row_offset = row * p.ncols_input;
             dst_row[tid] = data_s[row_offset + gl_GlobalInvocationID.x];
@@ -76,8 +82,10 @@ void topk(const uint row) {
             if (tid < s) {
                 ivec2 a = dst_row[tid];
                 ivec2 b = dst_row[tid + s];
+                // compare as floats: the bit patterns of negative values
+                // order the other way as integers
                 if (a.x >= p.orig_ncols ||
-                    b.x < p.orig_ncols && b.y > a.y) {
+                    b.x < p.orig_ncols && intBitsToFloat(b.y) > intBitsToFloat(a.y)) {
                     dst_row[tid] = b;
                 }
             }
@@ -95,9 +103,11 @@ void topk(const uint row) {
         int shift = 32 - SUBGROUP_SIZE_LOG2;
         uint mask = ((1 << SUBGROUP_SIZE_LOG2) - 1) << shift;
 
-        // The current range.
+        // The current range, [range_min, range_max). It starts as every value
+        // (+inf maps to 0xFF800000 and NaN was replaced by -inf), so the
+        // buckets always hold at least limit values.
         uint range_min = 0;
-        uint range_max = 0xFF800000;
+        uint range_max = 0xFFFFFFFF;
         // How many are above the current range, and how many we need to find.
         uint total = 0;
         uint limit = min(p.k, p.ncols_input - gl_WorkGroupID.x * BLOCK_SIZE);
@@ -138,8 +148,12 @@ void topk(const uint row) {
             total = sh_total;
 
             // Update the range, and break if we've found the K-th largest.
-            range_max = range_min + ((min_idx + 1) << shift);
-            range_min = range_min + (min_idx << shift);
+            // The end of the top bucket wraps past 2^32, clamp it instead.
+            range_min = range_min + (uint(min_idx) << shift);
+            range_max = range_min + (1u << shift);
+            if (range_max < range_min) {
+                range_max = 0xFFFFFFFF;
+            }
 
             if (total == p.k) {
                 break;
diff --git src/ggml-webgpu/ggml-webgpu.cpp src/ggml-webgpu/ggml-webgpu.cpp
index a7986b9c..11a4fc46 100644
--- src/ggml-webgpu/ggml-webgpu.cpp
+++ src/ggml-webgpu/ggml-webgpu.cpp
@@ -4104,7 +4104,7 @@ static void ggml_webgpu_init_memset_pipeline(webgpu_global_context & ctx) {
 static void ggml_backend_webgpu_request_adapter(wgpu::Instance & instance, wgpu::Adapter & adapter) {
     wgpu::RequestAdapterOptions options = {};
 
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
     // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215
     const char * const          adapterEnabledToggles[] = { "vulkan_enable_f16_on_nvidia", "use_vulkan_memory_model" };
     wgpu::DawnTogglesDescriptor adapterTogglesDesc;
@@ -4133,7 +4133,7 @@ static void create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
     ctx->webgpu_global_ctx->adapter.GetLimits(&ctx->webgpu_global_ctx->capabilities.limits);
 
     wgpu::AdapterInfo info{};
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
     wgpu::AdapterPropertiesSubgroupMatrixConfigs subgroup_matrix_configs{};
     if (ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
         info.nextInChain = &subgroup_matrix_configs;
@@ -4150,7 +4150,7 @@ static void create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
         wgpu::WGSLLanguageFeatureName::Packed4x8IntegerDotProduct);
 
     bool valid_subgroup_matrix_config = false;
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
     // Accept f16 subgroup matrix configurations (square or non-square).
     // NVIDIA GPUs typically report square configs (e.g. 16x16x16),
     // while Intel Xe2 GPUs report non-square configs (e.g. 8x16x16).
@@ -4178,7 +4178,7 @@ static void create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
     // Initialize device
     std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };
 
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
     required_features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
     if (ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix) {
         required_features.push_back(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
@@ -4214,7 +4214,7 @@ static void create_webgpu_device(ggml_backend_webgpu_reg_context * ctx) {
                        std::string(message).c_str());
         });
 
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
     // Enable Dawn-specific toggles to increase native performance
     // TODO: Maybe WebGPU needs a "fast" mode where you can request compilers skip adding checks like these,
     //       only for native performance?
@@ -4544,7 +4544,7 @@ static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const
                     break;
                 }
                 if (ggml_webgpu_tensor_binding_overlap(ctx->webgpu_global_ctx, src1, src2) &&
-                    src1->type != src2->type && !ggml_is_quantized(src1->type) && !ggml_is_quantized(src2->type)) {
+                    src1->type != src2->type && !(ggml_is_quantized(src1->type) && ggml_is_quantized(src2->type))) {
                     supports_op = false;
                     break;
                 }
@@ -4865,7 +4865,7 @@ ggml_backend_reg_t ggml_backend_webgpu_reg() {
     instance_descriptor.requiredFeatures                     = instance_features.data();
     instance_descriptor.requiredFeatureCount                 = instance_features.size();
 
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
     const char * const          instanceEnabledToggles[] = { "allow_unsafe_apis" };
     wgpu::DawnTogglesDescriptor instanceTogglesDesc;
     instanceTogglesDesc.enabledToggles     = instanceEnabledToggles;
@@ -4885,7 +4885,7 @@ ggml_backend_reg_t ggml_backend_webgpu_reg() {
 
     // WebGPU backend requires f16 support and, on native, implicit device synchronization.
     if (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)
-#ifndef __EMSCRIPTEN__
+#if !defined(__EMSCRIPTEN__) && !defined(__wasi__)
         && adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)
 #endif
     ) {
