#include "ime/ime.h"

#define IME_OP_PATTERN_MAX 10
static struct shl_cb_table shl_ime_cb_table[IME_OP_PATTERN_MAX];

void shl_ime_reg_op(enum csinn_dtype_enum dtype, 
                    enum csinn_op_enum op_name, 
                    void *init, 
                    void *exec,
                    void *est, 
                    void *cap, 
                    void *perf)
{
    static int i = 0;
    if (i >= IME_OP_PATTERN_MAX) {
        shl_debug_error("IME callback table is full!\n");
    }
    shl_ime_cb_table[i].shl_cb_key = op_name * CSINN_DTYPE_SIZE + dtype;
    shl_ime_cb_table[i].shl_cb_value.init = init;
    shl_ime_cb_table[i].shl_cb_value.exec = exec;
    shl_ime_cb_table[i].shl_cb_value.est = est;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_ime(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < IME_OP_PATTERN_MAX; i++) {
        if (shl_ime_cb_table[i].shl_cb_key == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &(shl_ime_cb_table[i].shl_cb_value);
            break;
        }
    }
    if ((cb == NULL) || (cb->init == NULL)) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

struct shl_ime_option *shl_ime_get_graph_option(struct csinn_session *sess)
{
    struct shl_gref_target_data *gref_td = sess->td;
    if (gref_td) {
        return (struct shl_ime_option *)(gref_td->cpu_option);
    } else {
        return NULL;
    }
}

void __attribute__((weak)) shl_target_init_ime()
{
    shl_ime_reg_op(CSINN_DTYPE_A_INT8_W_INT4_O_INT8, 
                   CSINN_OP_FULLYCONNECTED, 
                   shl_ime_fullyconnected_init_i8a_i4w_i8o, 
                   NULL,
                   shl_gref_fullyconnected, 
                   NULL, 
                   NULL);
    
    shl_ime_reg_op(CSINN_DTYPE_A_INT8_W_INT8_O_INT8, 
                   CSINN_OP_FULLYCONNECTED, 
                   shl_ime_fullyconnected_init_i8a_i8w_i8o, 
                   NULL,
                   shl_gref_fullyconnected, 
                   NULL, 
                   NULL);
    
    shl_ime_reg_op(CSINN_DTYPE_A_INT8_W_INT8_O_FLOAT32, 
                   CSINN_OP_FULLYCONNECTED, 
                   shl_ime_fullyconnected_init_i8a_i8w_f32o, 
                   NULL,
                   shl_gref_fullyconnected, 
                   NULL, 
                   NULL);
    
    shl_register_runtime_callback(CSINN_IME, NULL);
    shl_register_op_callback(CSINN_IME, shl_cb_map_ime);
    shl_register_runtime_callback(CSINN_IME, shl_gref_runtime_callback);
}