#include "ime/perf.h"
#include "ime/ime.h"

static struct shl_function_map shl_ime_kernel_map[] = {
    // {shl_ime_matmul_init_int8, "shl_ime_matmul_init_int8"},
    // {shl_ime_matmul_int8, "shl_ime_matmul_int8"},

    // {shl_ime_matmul_common_int8, "shl_ime_matmul_common_int8"},

    // {shl_ime_matmul_reorder_mat0_4xK_int8, "shl_ime_matmul_reorder_mat0_4xK_int8"},
    // {shl_ime_matmul_reorder_mat1_Kx4_int8, "shl_ime_matmul_reorder_mat1_Kx4_int8"},
    // {shl_ime_matmul_4x4_int8, "shl_ime_matmul_4x4_int8"},

    {shl_ime_fullyconnected_init_int8, "shl_ime_fullyconnected_init_int8"},

    {shl_ime_fullyconnected_gemm_int4, "shl_ime_fullyconnected_gemm_int4"},
    {shl_ime_fullyconnected_gemm_int8, "shl_ime_fullyconnected_gemm_int8"},
    
    {NULL, NULL}};

char *shl_ref_get_kernel_name(void *exec);

char *shl_ime_get_kernel_name(void *exec)
{
    char *name = shl_find_function_name(shl_ime_kernel_map, exec);
    if (name == NULL) {
        name = shl_ref_get_kernel_name(exec);
    }
    return name;
}

int shl_ime_matmul_perf(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params,
                        struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_ime_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}