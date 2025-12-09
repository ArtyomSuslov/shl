#include "ime/ime.h"

int shl_ime_fullyconnected_init_i8a_i4w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params)
{
    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_INT4_SYM) {
        params->base.cb->exec = shl_ref_fullyconnected_quant;
        return CSINN_TRUE;
    }

    // Bias Fusion
    if (!params->fc_extra.fuse_zp2bias) {
        params->fc_extra.fuse_zp2bias = true;
        
        if (bias->data == NULL) {
            int out_nodes = weights->dim[0];
            bias->data = shl_mem_alloc(out_nodes * sizeof(int32_t));
            memset(bias->data, 0, out_nodes * sizeof(int32_t));
        }
    }

    // Расчет множителей
    for (int i = 0; i < weights->quant_channel; i++) {
        double real_scale = (double)input->qinfo->scale * 
                            (double)weights->qinfo[i].scale / 
                            (double)output->qinfo->scale;
                            
        shl_quantize_multiplier(real_scale, 
                                &(weights->qinfo[i].multiplier),
                                &(weights->qinfo[i].shift));
    }

    shl_ime_fc_reorder_weight_n4_int4(weights);
    params->base.cb->exec = shl_ime_fullyconnected_exec_i8a_i4w_i8o;

    return CSINN_TRUE;
}

int shl_ime_fullyconnected_init_i8a_i8w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params)
{
    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM) {
        params->base.cb->exec = shl_ref_fullyconnected_quant;
        return CSINN_TRUE;
    }

    // Bias Fusion
    if (!params->fc_extra.fuse_zp2bias) {
        params->fc_extra.fuse_zp2bias = true;
        
        if (bias->data == NULL) {
            int out_nodes = weights->dim[0];
            bias->data = shl_mem_alloc(out_nodes * sizeof(int32_t));
            memset(bias->data, 0, out_nodes * sizeof(int32_t));
        }
    }

    // Расчет множителей
    for (int i = 0; i < weights->quant_channel; i++) {
        double real_scale = (double)input->qinfo->scale * 
                            (double)weights->qinfo[i].scale / 
                            (double)output->qinfo->scale;
                            
        shl_quantize_multiplier(real_scale, 
                                &(weights->qinfo[i].multiplier),
                                &(weights->qinfo[i].shift));
    }

    shl_ime_fc_reorder_weight_n4_int8(weights);
    params->base.cb->exec = shl_ime_fullyconnected_exec_i8a_i8w_i8o;

    return CSINN_TRUE;
}

int shl_ime_fullyconnected_init_i8a_i8w_f32o(struct csinn_tensor *input, 
                                             struct csinn_tensor *output,
                                             struct csinn_tensor *weights, 
                                             struct csinn_tensor *bias,
                                             struct csinn_fc_params *params)
{
    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM_TO_F32) {
        params->base.cb->exec = shl_ref_fullyconnected_quant;
        return CSINN_TRUE;
    }

    // Bias Fusion
    if (!params->fc_extra.fuse_zp2bias) {
        params->fc_extra.fuse_zp2bias = true;
        
        if (bias->data == NULL) {
            int out_nodes = weights->dim[0];
            bias->data = shl_mem_alloc(out_nodes * sizeof(int32_t));
            memset(bias->data, 0, out_nodes * sizeof(int32_t));
        }
    }

    shl_ime_fc_reorder_weight_n4_int8(weights);
    params->base.cb->exec = shl_ime_fullyconnected_exec_i8a_i8w_f32o;

    return CSINN_TRUE;
}