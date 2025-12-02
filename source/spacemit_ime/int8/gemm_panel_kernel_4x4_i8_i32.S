    .file "gemm_panel_kernel_4x4x8_i8_i32.S"

/*
 * Function: shl_ime_gemm_4x4_int8_int4
 * 
 * Description:
 *   Computes 4x4 int32 block using SpacemiT IME extension.
 *   Accumulates over K dimension.
 *   Inputs are:
 *     - Activations: int8, Zz format (4x8 blocks, row-major inner)
 *     - Weights: packed int4, Nn format (8x4 blocks, column-major inner)
 *
 * C Signature:
 *   void shl_ime_gemm_4x4_int8_int4(
 *       int32_t *dst,          // a0
 *       const int8_t *src,     // a1 (Activations)
 *       const uint8_t *weight, // a2 (Packed Weights)
 *       int32_t k_steps        // a3
 *   );
 */

    .section .text
    .align 2
    .global shl_ime_gemm_4x4_int8_int4
    .type shl_ime_gemm_4x4_int8_int4, @function

shl_ime_gemm_4x4_int8_int4:
    # ---------------------------------------------------------

    # Инициализация констант для распаковки (Shuffle Constants)
    # Нам нужны индексы для vrgather: 0, 0, 1, 1, 2, 2 ... 15, 15
    # Нам нужна маска для vmerge: 0, 1, 0, 1 ...
    
    # VLEN=256, SEW=8. Мы работаем с 32 элементами (bytes) в векторе.
    li t0, 32
    vsetvli zero, t0, e8, m1, ta, ma

    # Генерируем вектор индексов v30: [0, 0, 1, 1, ..., 15, 15]
    vid.v v30           # v30 = [0, 1, 2, 3 ...]
    vsrl.vi v30, v30, 1 # v30 = [0, 0, 1, 1 ...] (целочисленное деление на 2)

    # Генерируем маску v0: [0, 1, 0, 1 ...] для vmerge
    vid.v v10
    vand.vi v10, v10, 1
    vmseq.vi v0, v10, 1

    # ---------------------------------------------------------

    # Инициализация Аккумуляторов
    # v24, v25 используются для результата (int32). Очищаем их.
    vsetvli zero, zero, e8, m1, ta, ma 
    vmv.v.i v24, 0
    vmv.v.i v25, 0

    # ---------------------------------------------------------

    # Основной цикл (Loop over K)
    # a3 = k_steps (общее количество K, кратное 8)
    
gemm_loop_1:
    beqz a3, gemm_end_1
    
    # --- Загрузка Activations (A) ---
    # Формат Zz: 4 строки, 8 столбцов. Линейно в памяти. 32 байта.
    # v8 = [Row0(8) | Row1(8) | Row2(8) | Row3(8)]
    vle8.v v8, (a1)
    addi a1, a1, 32    # Pointer bump A (32 bytes)

    # --- Загрузка Weights (B) ---
    # Формат Nn: 8 строк (K), 4 столбца (N). Упакованы.
    # Всего 32 веса. Упакованы в 16 байт.
    # v10 = [Packed Bytes ...]
    vle8.v v10, (a2)
    addi a2, a2, 16    # Pointer bump B (16 bytes)

    # --- Распаковка Weights (Packed -> Int8) ---
    # v10 (Packed) -> v16 (Unpacked Int8)
    # Нам нужно расширить 16 байт до 32 байт с дублированием и распаковкой.
    
    # 1. Duplicate bytes: v12 = [B0, B0, B1, B1 ...]
    vrgather.vv v12, v10, v30

    # 2. Extract High Nibbles (Odds): v14 = SignExtend(v12 >> 4)
    vsra.vi v14, v12, 4

    # 3. Extract Low Nibbles (Evens): v12 = SignExtend(v12 << 4 >> 4)
    vsll.vi v12, v12, 4
    vsra.vi v12, v12, 4
    
    # 4. Merge: v16 = (Mask==1 ? v14 : v12)
    # Mask v0 у нас 0101... (0=Even/Low, 1=Odd/High)
    vmerge.vvm v16, v12, v14, v0

    # Теперь v16 содержит честные int8 веса, линейно по K, потом по N.
    # Layout в регистре v16:
    # [ Col0_K0..K7 | Col1_K0..K7 | Col2... | Col3... ]
    # Это именно то, что ожидает vmadot (copies * K * N).

    # --- Matrix Multiply Accumulate ---
    # v24 += v8 (A) * v16 (B)
    vmadot v24, v8, v16

    # Loop counters
    addi a3, a3, -8
    j gemm_loop_1

gemm_end_1:

    # ---------------------------------------------------------
    
    # Сохранение результата
    # Результат лежит в v24, v25 (блок 4x4, int32).
    # Нам нужно сбросить его в память dst (a0).
    # Размер: 16 элементов * 4 байта = 64 байта.
    
    # Переключаем VL на 16 элементов int32 (e32)
    li t0, 16
    vsetvli zero, t0, e32, m2, ta, ma 
    
    # Сохраняем
    vse32.v v24, (a0)

    ret

/*
 * Function: shl_ime_gemm_4x4_int8_int8
 * 
 * Description:
 *   Computes 4x4 int32 block using SpacemiT IME extension.
 *   Accumulates over K dimension.
 *   Inputs are:
 *     - Activations: int8, Zz format (4x8 blocks, row-major inner)
 *     - Weights: int8, Nn format (8x4 blocks, column-major inner)
 *
 * C Signature:
 *   void shl_ime_gemm_4x4_int8_int8(
 *       int32_t *dst,          // a0
 *       const int8_t *src,     // a1 (Activations)
 *       const int8_t *weight,  // a2 (Packed Weights)
 *       int32_t k_steps        // a3
 *   );
 */

    .section .text
    .align 2
    .global shl_ime_gemm_4x4_int8_int8
    .type shl_ime_gemm_4x4_int8_int8, @function

shl_ime_gemm_4x4_int8_int8:

    # Инициализация Аккумуляторов
    # v24, v25 используются для результата (int32). Очищаем их.
    li t0, 32
    vsetvli zero, t0, e8, m1, ta, ma
    vmv.v.i v24, 0
    vmv.v.i v25, 0

    # ---------------------------------------------------------

    # Основной цикл (Loop over K)
    # a3 = k_steps (общее количество K, кратное 8)
    
gemm_loop_2:
    beqz a3, gemm_end_2
    
    # --- Загрузка Activations (A) ---
    # Формат Zz: 4 строки (M), 8 столбцов (K). Линейно в памяти. 32 байта.
    # v8 = [Row0(8) | Row1(8) | Row2(8) | Row3(8)]
    vle8.v v8, (a1)
    addi a1, a1, 32    # Pointer bump A (32 bytes)

    # --- Загрузка Weights (B) ---
    # Формат Nn: 8 строк (K), 4 столбца (N). Линейно в памяти. 32 байта.
    # v10 = [Col0(8) | Col1(8) | Col2(8) | Col3(8)]
    vle8.v v10, (a2)
    addi a2, a2, 32    # Pointer bump B (32 bytes)

    # --- Matrix Multiply Accumulate ---
    # v24 += v8 (A) * v10 (B)
    vmadot v24, v8, v10

    # Loop counters
    addi a3, a3, -8
    j gemm_loop_2

gemm_end_2:

    # ---------------------------------------------------------
    
    # Сохранение результата
    # Результат лежит в v24, v25 (блок 4x4, int32).
    # Нам нужно сбросить его в память dst (a0).
    # Размер: 16 элементов * 4 байта = 64 байта.
    
    # Переключаем VL на 16 элементов int32 (e32)
    li t0, 16
    vsetvli zero, t0, e32, m2, ta, ma 
    
    # Сохраняем
    vse32.v v24, (a0)

    ret