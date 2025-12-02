#include "ime/ime.h"

int shl_ime_fullyconnected_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_INT4_SYM &&
        params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM) {
        params->base.cb->exec = shl_ref_fullyconnected_quant;
        return CSINN_TRUE;
    }

    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);

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

    // Переупорядочивание весов
    if (!binary_model_op_init) {
        switch (params->base.quant_type) {
            case CSINN_QUANT_INT8_ASYM_W_INT4_SYM:
                shl_ime_fc_reorder_weight_n4_int4(weights);
                break;
            case CSINN_QUANT_INT8_ASYM_W_SYM:
                shl_ime_fc_reorder_weight_n4_int8(weights);
                break;
            default:
                break;
        }
    }

    switch (params->base.quant_type) {
        case CSINN_QUANT_INT8_ASYM_W_INT4_SYM:
            params->base.cb->exec = shl_ime_fullyconnected_gemm_int4;
            break;
        case CSINN_QUANT_INT8_ASYM_W_SYM:
            params->base.cb->exec = shl_ime_fullyconnected_gemm_int8;
            break;
        default:
            break;
    }

    return CSINN_TRUE;
}