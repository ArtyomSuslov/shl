#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import struct

def create_fullyconnected_file():
    # --- Ввод размеров через консоль ---
    batch = int(input("Введите размер батча (batch): "))
    in_size = int(input("Введите размер входного вектора (in_size): "))
    out_size = int(input("Введите размер выходного вектора (out_size): "))

    # --- Генерация случайных данных ---
    input_data = np.random.rand(batch, in_size).astype(np.float32)
    weight_data = np.random.rand(out_size, in_size).astype(np.float32)  # OI layout
    bias_data = np.random.rand(out_size).astype(np.float32)
    reference_data = input_data @ weight_data.T + bias_data  # матричное умножение + bias

    # --- Создание буфера ---
    in_size0 = batch * in_size
    in_size1 = out_size * in_size
    buffer = [batch, in_size, out_size]
    buffer += list(input_data.flatten().view(np.int32))
    buffer += list(weight_data.flatten().view(np.int32))
    buffer += list(bias_data.view(np.int32))
    buffer += list(reference_data.flatten().view(np.int32))

    # --- Запись в бинарный файл ---
    filename = "fullyconnected_data.bin"
    with open(filename, "wb") as f:
        f.write(struct.pack('i', len(buffer)))  # размер буфера
        f.write(struct.pack(f'{len(buffer)}i', *buffer))

    print(f"Файл '{filename}' сгенерирован успешно!")

if __name__ == "__main__":
    create_fullyconnected_file()
