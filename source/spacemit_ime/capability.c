#include "ime/ime.h"

static int common_all_support(struct csinn_tensor *input, struct csinn_params_base *base)
{
    if ((input->dtype != CSINN_DTYPE_FLOAT16) && (input->dtype != CSINN_DTYPE_FLOAT32) &&
        (input->dtype != CSINN_DTYPE_INT8)) {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_INTRINSIC;
}

static int float_all_support(struct csinn_tensor *input, struct csinn_params_base *base)
{
    if ((input->dtype != CSINN_DTYPE_FLOAT16) && (input->dtype != CSINN_DTYPE_FLOAT32)) {
        return CSINN_OPT_UNSUPPORTED;
    }

    return CSINN_OPT_INTRINSIC;
}

int shl_ime_matmul_cap(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                       struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < mat0->dim_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    if (mat0->dtype == CSINN_DTYPE_INT8) {
        if (batches_a == batches_b) {
            return CSINN_OPT_INTRINSIC;
        } else if (batches_a > 1 && batches_b == 1) {
            if (!params->trans_a && !params->trans_b) {
                return CSINN_OPT_INTRINSIC;
            }
        }
    }

    return CSINN_OPT_UNSUPPORTED;
}
