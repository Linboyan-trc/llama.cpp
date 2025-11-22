#pragma once // 1. 这个pragma once用来代替"#ifndef MY_HEADER_H" + "#define MY_HEADER_H" + "#endif", 更简洁的确保本头文件被引用的时候只被引用一次

#include "ggml.h" // 2. 引用"ggml.h"

#ifdef  __cplusplus // 3. 使用C++编译器的时候编译器内置这个宏，使得函数编译后不被重命名，其实就是关闭函数重载功能
extern "C" {
#endif

// 4. 给结构体指针别名
// 4.1 语法是typedef <value_type> <alias>
// 4.2 这里<value_type>是struct ggml_backend_buffer_type *，结构体指针
// 4.3 <alias>是ggml_backend_buffer_type_t
// 4.3 而struct ggml_backend_buffer_type在其他地方
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct      ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct             ggml_backend * ggml_backend_t;

// Tensor allocator
// 5. 声明一个结构体
// 5.1 有4个成员变量，其中size_t在64位操作系统上是unsigned long，8 byte
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

// 6. 声明一个函数
// 6.1 语法是<macro> <return_value_type> <func_name>(<arguments>)
// 6.2 这里<macro>是GGML_API
// 6.2 <return_value_type>是struct ggml_tallocr
// 6.2 <func_name>是ggml_tallocr_new
// 6.2 <arguments>是ggml_backend_buffer_t buffer
GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten

typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);

// Utils
// Create a buffer and allocate all the tensors in a ggml_context
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
