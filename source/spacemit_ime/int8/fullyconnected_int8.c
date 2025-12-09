#include "ime/ime.h"

/*
 * Функция-утилита для Init.
 * Берет тензор с INT8 весами (linear layout).
 * 1. Вычисляет размер с паддингом
 * 2. Упаковывает в формат Nn (Block 8x4) для vmadot
 * 3. Подменяет память в тензоре
 */
void shl_ime_fc_reorder_weight_n4_int4(struct csinn_tensor *weights)
{
    int8_t *src_data = (int8_t *)weights->data;
    int32_t N = weights->dim[0];
    int32_t K = weights->dim[1];

    // Расчет размеров с паддингом (Align N to 4, K to 8)
    // K выравниваем до 8 (т.к. в блоке 8 строк)
    // N выравниваем до 4 (т.к. в блоке 4 столбца)
    int32_t K_pad = (K + 7) & ~7;
    int32_t N_pad = (N + 3) & ~3;
    
    // Размер в байтах (2 веса в байте)
    size_t packed_size = (size_t)(N_pad * K_pad) / 2;

    // Аллокация новой памяти
    uint8_t *dest_data = (uint8_t *)shl_mem_alloc(packed_size);
    int dst_idx = 0;

    // Упаковка (Global N -> Global K -> Block 8x4 Column-Major)
    // Внешний цикл по панелям N (ширина 4)
    for (int n_blk = 0; n_blk < N_pad; n_blk += 4) {
        // Внутренний цикл по полосам K (высота 8)
        for (int k_blk = 0; k_blk < K_pad; k_blk += 8) {
            
            // Внутри блока 8x4 идем по столбцам (Column-Major для vmadot)
            for (int n_inner = 0; n_inner < 4; n_inner++) {
                int curr_n = n_blk + n_inner;
                
                // Идем по столбцу K парами (для упаковки в байт)
                for (int k_inner = 0; k_inner < 8; k_inner += 2) {
                    int curr_k = k_blk + k_inner;
                    
                    int8_t w0 = 0;
                    int8_t w1 = 0;

                    // Читаем W0 (Low Nibble) с проверкой границ
                    if (curr_n < N && curr_k < K) {
                        w0 = src_data[curr_n * K + curr_k];
                    }
                    // Читаем W1 (High Nibble)
                    if (curr_n < N && (curr_k + 1) < K) {
                        w1 = src_data[curr_n * K + (curr_k + 1)];
                    }
                    
                    // Pack: [High | Low]
                    uint8_t packed = (uint8_t)((w0 & 0x0F) | ((w1 & 0x0F) << 4));
                    dest_data[dst_idx++] = packed;
                }
            }
        }
    }

    // Подмена памяти
    // Освобождаем старые INT8 данные
    shl_mem_free(weights->data); 
    
    // Назначаем новые упакованные данные
    weights->data = dest_data;
    weights->dtype = CSINN_DTYPE_UINT8; 
}

/*
 * Функция-утилита для Init.
 * Берет тензор с INT8 весами (linear layout).
 * 1. Вычисляет размер с паддингом
 * 2. Упаковывает в формат Nn (Block 8x4) для vmadot
 * 3. Подменяет память в тензоре
 */
void shl_ime_fc_reorder_weight_n4_int8(struct csinn_tensor *weights)
{
    int8_t *src_data = (int8_t *)weights->data;
    int32_t N = weights->dim[0];
    int32_t K = weights->dim[1];

    // Расчет размеров с паддингом (Align N to 4, K to 8)
    // K выравниваем до 8 (т.к. в блоке 8 строк)
    // N выравниваем до 4 (т.к. в блоке 4 столбца)
    int32_t K_pad = (K + 7) & ~7;
    int32_t N_pad = (N + 3) & ~3;
    
    // Размер в байтах
    size_t packed_size = (size_t)(N_pad * K_pad * sizeof(int8_t));

    // Аллокация новой памяти
    int8_t *dest_data = (int8_t *)shl_mem_alloc(packed_size);
    int dst_idx = 0;

    // Упаковка (Global N -> Global K -> Block 8x4 Column-Major)
    // Внешний цикл по панелям N (ширина 4)
    for (int n_blk = 0; n_blk < N_pad; n_blk += 4) {
        // Внутренний цикл по полосам K (высота 8)
        for (int k_blk = 0; k_blk < K_pad; k_blk += 8) {
            
            // Внутри блока 8x4 идем по столбцам (Column-Major для vmadot)
            for (int n_inner = 0; n_inner < 4; n_inner++) {
                int curr_n = n_blk + n_inner;
                
                // Идем по столбцу K
                for (int k_inner = 0; k_inner < 8; k_inner++) {
                    int curr_k = k_blk + k_inner;

                    int8_t w = 0;
                    if (curr_n < N && curr_k < K) {
                        w = src_data[curr_n * K + curr_k];
                    }

                    dest_data[dst_idx++] = (int8_t)w;
                }
            }
        }
    }

    // Подмена памяти
    // Освобождаем старые INT8 данные
    shl_mem_free(weights->data); 
    
    // Назначаем новые упакованные данные
    weights->data = dest_data;
    weights->dtype = CSINN_DTYPE_INT8;
}

/*
 * Переупорядочивание ВХОДНОЙ матрицы (Activations) для IME.
 * Формат: Zz (Block Row-Major).
 * Блок: 4x8 (4 строки M, 8 столбцов K).
 * Внутри блока: Row-Major (элементы K идут подряд).
 * 
 * dst: Буфер назначения (должен быть заранее выделен с учетом паддинга!)
 * src: Исходный тензор input (INT8)
 */
void shl_ime_reorder_input_z4_int8(int8_t *dst, int8_t *src, int m, int k, int k_stride)
{
    // k_stride обычно равен k, но может быть больше, если это часть большой матрицы
    
    // Выравниваем K до 8 (ширина блока)
    // M выравниваем до 4 (высота блока)
    // ВАЖНО: dst должен иметь размер aligned_m * aligned_k
    
    int m_pad = (m + 3) & ~3;
    int k_pad = (k + 7) & ~7;
    
    int dst_idx = 0;

    // 1. Внешний цикл: Полосы по M (высота 4)
    for (int m_blk = 0; m_blk < m_pad; m_blk += 4) {
        
        // 2. Внутренний цикл: Панели по K (ширина 8)
        for (int k_blk = 0; k_blk < k_pad; k_blk += 8) {
            
            // Внутри блока 4x8
            // Мы хотим Row-Major: Строка 0, Строка 1...
            
            for (int m_inner = 0; m_inner < 4; m_inner++) {
                int curr_m = m_blk + m_inner;
                
                // Копируем 8 элементов строки (K)
                for (int k_inner = 0; k_inner < 8; k_inner++) {
                    int curr_k = k_blk + k_inner;
                    
                    if (curr_m < m && curr_k < k) {
                        dst[dst_idx++] = src[curr_m * k_stride + curr_k];
                    } else {
                        dst[dst_idx++] = 0; // Padding
                    }
                }
            }
        }
    }
}

// Вспомогательная функция для сатурации (int32 -> int8)
static inline int8_t sat_s32_to_s8(int32_t v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Вспомогательная функция фиксированной точки (из SHL)
static inline int32_t shl_quant_mult(int32_t val, int32_t mult, int32_t shift) {
    int64_t temp = (int64_t)val * mult;
    
    // mult - это число в формате Q31 (нормализованное 0.5..1.0 * 2^31).
    // Значит, результат умножения (temp) имеет лишний фактор 2^31.
    
    if (shift > 0) {
        // Если shift положительный (редкий случай, scale > 1), 
        // нам нужно сдвинуть вправо на 31 (убрать Q31), но влево на shift.
        // Итого: вправо на (31 - shift).
        int32_t right_shift = 31 - shift;
        if (right_shift < 0) {
            // Экстремальный случай, если scale очень большой.
            return (int32_t)(temp << (-right_shift));
        }
        int64_t rounding = 1LL << (right_shift - 1);
        return (int32_t)((temp + rounding) >> right_shift);
        
    } else {
        // Самый частый случай (shift <= 0).
        // Нам нужно сдвинуть вправо на 31 (убрать Q31) И ЕЩЕ вправо на abs(shift).
        // Итого: вправо на (31 - shift), так как shift отрицательный.
        // Пример: shift = -6. Total shift = 31 - (-6) = 37.
        
        int32_t right_shift = 31 - shift;
        
        // Добавляем округление (Rounding)
        int64_t rounding = 1LL << (right_shift - 1);
        
        return (int32_t)((temp + rounding) >> right_shift);
    }
}

// Прототипы ASM кернелей
// dst: временный буфер int32[16]
// src: блок активаций 4xK (packed Zz)
// weight: блок весов NxK (packed Nn)
// k_steps: количество шагов (K с паддингом)
void shl_ime_gemm_4x4_int8_int4(int32_t *dst, const int8_t *src, const uint8_t *weight, int32_t k_steps);
void shl_ime_gemm_4x4_int8_int8(int32_t *dst, const int8_t *src, const int8_t *weight, int32_t k_steps);

int shl_ime_fullyconnected_exec_i8a_i4w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data; // Результат пишем сюда (INT8)
    
    uint8_t *weights_data = (uint8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t M = input->dim[0];
    int32_t K = input->dim[1];
    int32_t N = weights->dim[0];

    // Подготовка размеров с паддингом
    int32_t M_pad = (M + 3) & ~3;
    int32_t K_pad = (K + 7) & ~7;
    int32_t N_pad = (N + 3) & ~3;
    
    // Reorder Input (Zz format)
    // Используем workspace или временный буфер
    int8_t *input_reordered = shl_mem_alloc(M_pad * K_pad * sizeof(int8_t));
    shl_ime_reorder_input_z4_int8(input_reordered, input_data, M, K, K);

    // Временный буфер для результатов одного тайла 4x4 (int32)
    int32_t acc_buffer[16]; 

    // Цикл по тайлам (M x N)
    for (int m_blk = 0; m_blk < M_pad; m_blk += 4) {
        for (int n_blk = 0; n_blk < N_pad; n_blk += 4) {
            
            // Вызов ASM Кернела
            // input: смещение по полосам M
            int8_t *ptr_in = input_reordered + (m_blk * K_pad);
            
            // weights: смещение по панелям N (размер панели = 4 * K_pad / 2 байт)
            uint8_t *ptr_w = weights_data + (n_blk * (K_pad / 2));

            shl_ime_gemm_4x4_int8_int4(acc_buffer, ptr_in, ptr_w, K_pad);

            for (int i = 0; i < 4; i++) {      // Строки (m)
                for (int j = 0; j < 4; j++) {  // Столбцы (n)
                    int cur_m = m_blk + i;
                    int cur_n = n_blk + j;

                    // Проверка границ (из-за паддинга M/N)
                    if (cur_m >= M || cur_n >= N) continue;

                    // Читаем int32 из аккумулятора
                    int32_t acc = acc_buffer[i * 4 + j]; 

                    // Bias Add (Per-Channel, зависит от n)
                    acc += bias_data[cur_n];

                    // Scale & Shift (Per-Channel)
                    int32_t mult = weights->qinfo[cur_n].multiplier;
                    int32_t shift = weights->qinfo[cur_n].shift;
                    
                    acc = shl_quant_mult(acc, mult, shift);
                    acc += output->qinfo->zero_point; 

                    // Saturation to int8 (доработать?)
                    output_data[cur_m * N + cur_n] = sat_s32_to_s8(acc);
                }
            }
        }
    }

    shl_mem_free(input_reordered);
    return CSINN_TRUE;
}

int shl_ime_fullyconnected_exec_i8a_i8w_i8o(struct csinn_tensor *input, 
                                            struct csinn_tensor *output,
                                            struct csinn_tensor *weights, 
                                            struct csinn_tensor *bias,
                                            struct csinn_fc_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data; // Результат пишем сюда (INT8)
    
    int8_t *weights_data = (int8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t M = input->dim[0];
    int32_t K = input->dim[1];
    int32_t N = weights->dim[0];

    // Подготовка размеров с паддингом
    int32_t M_pad = (M + 3) & ~3;
    int32_t K_pad = (K + 7) & ~7;
    int32_t N_pad = (N + 3) & ~3;
    
    // Reorder Input (Zz format)
    // Используем workspace или временный буфер
    int8_t *input_reordered = shl_mem_alloc(M_pad * K_pad * sizeof(int8_t));
    shl_ime_reorder_input_z4_int8(input_reordered, input_data, M, K, K);

    // Временный буфер для результатов одного тайла 4x4 (int32)
    int32_t acc_buffer[16]; 

    // Цикл по тайлам (M x N)
    for (int m_blk = 0; m_blk < M_pad; m_blk += 4) {
        for (int n_blk = 0; n_blk < N_pad; n_blk += 4) {
            
            // Вызов ASM Кернела
            // input: смещение по полосам M
            int8_t *ptr_in = input_reordered + (m_blk * K_pad);
            
            // weights: смещение по панелям N
            int8_t *ptr_w = weights_data + (n_blk * K_pad);

            shl_ime_gemm_4x4_int8_int8(acc_buffer, ptr_in, ptr_w, K_pad);

            for (int i = 0; i < 4; i++) {      // Строки (m)
                for (int j = 0; j < 4; j++) {  // Столбцы (n)
                    int cur_m = m_blk + i;
                    int cur_n = n_blk + j;

                    // Проверка границ (из-за паддинга M/N)
                    if (cur_m >= M || cur_n >= N) continue;

                    // Читаем int32 из аккумулятора
                    int32_t acc = acc_buffer[i * 4 + j]; 

                    // Bias Add (Per-Channel, зависит от n)
                    acc += bias_data[cur_n];

                    // Scale & Shift (Per-Channel)
                    int32_t mult = weights->qinfo[cur_n].multiplier;
                    int32_t shift = weights->qinfo[cur_n].shift;
                    
                    acc = shl_quant_mult(acc, mult, shift);
                    acc += output->qinfo->zero_point;

                    // Saturation to int8
                    output_data[cur_m * N + cur_n] = sat_s32_to_s8(acc);
                }
            }
        }
    }

    shl_mem_free(input_reordered);
    return CSINN_TRUE;
}

/* 
 * Специализированная функция для OpenVINO.
 * Вход: 
 *   - input: INT8 (Activations, Asymmetric)
 *   - weights: INT8 (Weights, Symmetric, Packed Nn)
 *   - bias: INT32 (содержит: RealBias - InputZP * Sigma(W))
 * Выход:
 *   - output: FLOAT32 (Dequantized result)
 */
int shl_ime_fullyconnected_exec_i8a_i8w_f32o(struct csinn_tensor *input, 
                                             struct csinn_tensor *output,
                                             struct csinn_tensor *weights, 
                                             struct csinn_tensor *bias,
                                             struct csinn_fc_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    float *output_data = (float *)output->data;
    
    int8_t *weights_data = (int8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t M = input->dim[0];
    int32_t K = input->dim[1];
    int32_t N = weights->dim[0];

    int32_t M_pad = (M + 3) & ~3;
    int32_t K_pad = (K + 7) & ~7;
    int32_t N_pad = (N + 3) & ~3;
    
    // Получаем Scale входа (он один на весь тензор)
    float in_scale = input->qinfo->scale;

    // 1. Reorder Input
    int8_t *input_reordered = (int8_t *)shl_mem_alloc(M_pad * K_pad * sizeof(int8_t));
    if (!input_reordered) return CSINN_FALSE;
    shl_ime_reorder_input_z4_int8(input_reordered, input_data, M, K, K);

    int32_t acc_buffer[16]; 

    // 2. Main Loop
    for (int m_blk = 0; m_blk < M_pad; m_blk += 4) {
        for (int n_blk = 0; n_blk < N_pad; n_blk += 4) {
            
            int8_t *ptr_in = input_reordered + (m_blk * K_pad);
            int8_t *ptr_w = weights_data + (n_blk * K_pad); 

            // ASM: Accumulation
            shl_ime_gemm_4x4_int8_int8(acc_buffer, ptr_in, ptr_w, K_pad);

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int cur_m = m_blk + i;
                    int cur_n = n_blk + j;

                    if (cur_m >= M || cur_n >= N) continue;

                    int32_t acc = acc_buffer[i * 4 + j]; 

                    // Add Fused Bias (RealBias_q - Zp*SumW)
                    if (bias_data) {
                        acc += bias_data[cur_n];
                    }
                    
                    float w_scale = weights->qinfo[cur_n].scale; // Per-channel scale
                    float combined_scale = in_scale * w_scale;

                    output_data[cur_m * N + cur_n] = (float)acc * combined_scale;
                }
            }
        }
    }

    shl_mem_free(input_reordered);
    return CSINN_TRUE;
}