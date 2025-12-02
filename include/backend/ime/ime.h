#ifndef INCLUDE_SHL_IME_H_
#define INCLUDE_SHL_IME_H_

#include "rvv/rvv.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************** initialization ******************************/
// int shl_ime_matmul_init_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
//                              struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_ime_fullyconnected_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);

/*********************************** matmul *********************************/

// void shl_ime_matmul_reorder_weight_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1);

// void shl_ime_matmul_reorder_mat0_4xK_int8(const int8_t *src, int8_t *dst, 
//                                           int m, int k, int lda, 
//                                           int32_t *sum_row);

// void shl_ime_matmul_reorder_mat1_Kx4_int8(const int8_t *src, int8_t *dst, 
//                                           int k, int n, int ldb, 
//                                           int32_t *sum_col);

// void shl_ime_matmul_4x4_int8(int8_t *dst, 
//                              const int8_t *sa, const int8_t *sb, 
//                              int m, int k, int n, int ldc, 
//                              int32_t z1, int32_t z2, int32_t z3, 
//                              int32_t mult, int32_t shift, 
//                              const int32_t *sum_row, const int32_t *sum_col);

// int shl_ime_matmul_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
//                         struct csinn_tensor *output, struct csinn_matmul_params *params);

// int shl_ime_matmul_common_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
//                                struct csinn_tensor *output, struct csinn_matmul_params *params,
//                                void (*reorder_mat0)(const int8_t *src, int8_t *dst, int m, int k, int lda, int32_t *sum_row),
//                                void (*reorder_mat1)(const int8_t *src, int8_t *dst, int k, int n, int ldb, int32_t *sum_col),
//                                void (*matmul)(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k, int n,
//                                               int ldc, int32_t z1, int32_t z2, int32_t z3, int32_t mult,
//                                               int32_t shift, const int32_t *sum_row, const int32_t *sum_col));

/******************************** fully connected ************************************/

void shl_ime_fc_reorder_weight_n4_int4(struct csinn_tensor *weights);
void shl_ime_fc_reorder_weight_n4_int8(struct csinn_tensor *weights);

void shl_ime_reorder_input_z4_int8(int8_t *dst, int8_t *src, int m, int k, int k_stride);

int shl_ime_fullyconnected_gemm_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);
int shl_ime_fullyconnected_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);

/*************************************************************************************/

struct shl_ime_option {
    bool use_packn_layout;
    bool binary_model_op_init;
};

struct shl_ime_option *shl_ime_get_graph_option(struct csinn_session *sess);
bool shl_ime_get_binary_model_op_init(struct csinn_session *sess);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_IME_H_
