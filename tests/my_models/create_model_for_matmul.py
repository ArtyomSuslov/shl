import numpy as np
import struct
import argparse

def generate_and_write_data(m, k, n, trans_a, trans_b, output_path):
    """
    Генерирует тестовые данные для matmul и записывает их в бинарный файл,
    совместимый с тестовым фреймворком CSINN.
    """
    print(f"Генерация данных для M={m}, K={k}, N={n}")
    print(f"Транспонирование: A={'Да' if trans_a else 'Нет'}, B={'Да' if trans_b else 'Нет'}")

    # 1. Определяем реальные формы матриц на основе флагов транспонирования
    shape0 = (k, m) if trans_a else (m, k)
    shape1 = (n, k) if trans_b else (k, n)
    shape_out = (m, n)

    # 2. Генерируем случайные float32 данные
    # Используем небольшой диапазон, чтобы избежать проблем с точностью
    input0_data = np.random.uniform(-5.0, 5.0, size=shape0).astype(np.float32)
    input1_data = np.random.uniform(-5.0, 5.0, size=shape1).astype(np.float32)

    # 3. Вычисляем эталонный результат с помощью NumPy
    # NumPy @ оператор правильно обрабатывает умножение матриц
    A = input0_data.T if trans_a else input0_data
    B = input1_data.T if trans_b else input1_data
    reference_data = A @ B

    # Проверка корректности размеров
    assert A.shape == (m, k), "Ошибка в логике формы A"
    assert B.shape == (k, n), "Ошибка в логике формы B"
    assert reference_data.shape == shape_out, "Ошибка в логике формы вывода"

    # --- Сборка бинарного файла ---
    # Структура файла, как ее читает C-код:
    # 1. total_size (int32) - общее количество 4-байтных слов далее
    # 2. trans_a (int32)
    # 3. trans_b (int32)
    # 4. dim_count (int32) - для 2D matmul всегда 2
    # 5. input0->dim (2 * int32)
    # 6. input1->dim (2 * int32)
    # 7. output->dim (2 * int32)
    # 8. input0->data (m*k * float32)
    # 9. input1->data (k*n * float32)
    # 10. reference->data (m*n * float32)

    dim_count = 2
    metadata = [
        int(trans_a),
        int(trans_b),
        dim_count,
        *shape0,      # Размеры input0
        *shape1,      # Размеры input1
        *shape_out,   # Размеры output
    ]

    # Выравниваем (flatten) массивы данных
    input0_flat = input0_data.flatten()
    input1_flat = input1_data.flatten()
    reference_flat = reference_data.flatten()

    # Собираем все данные в один список
    all_data = metadata + list(input0_flat) + list(input1_flat) + list(reference_flat)
    total_size = len(all_data)

    # 4. Записываем все в бинарный файл
    # '<' означает little-endian (наиболее распространенный порядок байт)
    # 'i' - 4-байтный integer
    # 'f' - 4-байтный float
    try:
        with open(output_path, 'wb') as f:
            # Сначала записываем общий размер
            f.write(struct.pack('<i', total_size))

            # Создаем строку формата для struct.pack
            num_meta = len(metadata)
            num_in0 = len(input0_flat)
            num_in1 = len(input1_flat)
            num_ref = len(reference_flat)
            
            format_string = f'<{num_meta}i{num_in0}f{num_in1}f{num_ref}f'
            
            # Упаковываем и записываем все остальные данные
            packed_data = struct.pack(format_string, *all_data)
            f.write(packed_data)

        print(f"Файл успешно сгенерирован: {output_path}")
        print(f"Общий размер: {total_size * 4 + 4} байт")

    except Exception as e:
        print(f"Ошибка при записи файла: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор тестовых данных для CSINN matmul.")
    parser.add_argument("M", type=int, help="Количество строк в матрице A и выходной матрице")
    parser.add_argument("K", type=int, help="Количество столбцов в A / строк в B")
    parser.add_argument("N", type=int, help="Количество столбцов в матрице B и выходной матрице")
    parser.add_argument("-o", "--output", default="matmul_case.bin",
                        help="Путь к выходному бинарному файлу (default: matmul_case.bin)")
    parser.add_argument("--trans_a", action="store_true", help="Транспонировать матрицу A")
    parser.add_argument("--trans_b", action="store_true", help="Транспонировать матрицу B")

    args = parser.parse_args()

    generate_and_write_data(args.M, args.K, args.N, args.trans_a, args.trans_b, args.output)