#ifndef INCLUDE_SHL_IME_H_
#define INCLUDE_SHL_IME_H_

#include "rvv/rvv.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************** initialization ******************************/
int shl_ime_fullyconnected_init_i8a_i4w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params);

int shl_ime_fullyconnected_init_i8a_i8w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params);

int shl_ime_fullyconnected_init_i8a_i8w_f32o(struct csinn_tensor *input, 
                                         struct csinn_tensor *output,
                                         struct csinn_tensor *weights, 
                                         struct csinn_tensor *bias,
                                         struct csinn_fc_params *params);

/******************************** fully connected ************************************/

void shl_ime_fc_reorder_weight_n4_int4(struct csinn_tensor *weights);
void shl_ime_fc_reorder_weight_n4_int8(struct csinn_tensor *weights);

void shl_ime_reorder_input_z4_int8(int8_t *dst, int8_t *src, int m, int k, int k_stride);

int shl_ime_fullyconnected_exec_i8a_i4w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params);

int shl_ime_fullyconnected_exec_i8a_i8w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params);

int shl_ime_fullyconnected_exec_i8a_i8w_f32o(struct csinn_tensor *input, 
                                             struct csinn_tensor *output,
                                             struct csinn_tensor *weights, 
                                             struct csinn_tensor *bias,
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
