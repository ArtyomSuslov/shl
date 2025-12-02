#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import struct

def create_fullyconnected_file():
    print("=== Генератор данных для SHL QC4 (Symmetric Weights) ===")
    # --- Ввод размеров через консоль ---
    batch = int(input("Введите размер батча (batch): "))
    in_size = int(input("Введите размер входного вектора (in_size): "))
    out_size = int(input("Введите размер выходного вектора (out_size): "))

    # --- Генерация случайных данных ---
    # Мы вычитаем 0.5, чтобы получить диапазон [-0.5, 0.5].
    # Это критически важно для Symmetric INT4, чтобы использовать
    # и положительные, и отрицательные уровни квантования.
    
    # 1. Input: Можно оставить [0, 1) или сделать [-0.5, 0.5). 
    # Для Asymmetric Input (наш случай) [0, 1) тоже нормально, но [-0.5, 0.5] сложнее для теста.
    # Давай сделаем тоже центрированным, это "честнее" для нейросетей.
    input_data = np.random.rand(batch, in_size).astype(np.float32) # Range [-1.0, 1.0]

    # 2. Weights: ОБЯЗАТЕЛЬНО центрируем для QC4 Symmetric!
    # Делаем диапазон чуть меньше, например [-0.5, 0.5], чтобы эмулировать реальные веса.
    weight_data = (np.random.rand(out_size, in_size).astype(np.float32) - 0.5) # Range [-0.5, 0.5]

    # 3. Bias: Тоже центрируем
    bias_data = (np.random.rand(out_size).astype(np.float32) - 0.5) # Range [-0.5, 0.5]
    
    # --- Вычисление эталона ---
    reference_data = input_data @ weight_data.T + bias_data

    # --- Создание буфера ---
    buffer = [batch, in_size, out_size]
    
    # .view(np.int32) берет битовое представление float без изменений. Это правильно.
    buffer += list(input_data.flatten().view(np.int32))
    buffer += list(weight_data.flatten().view(np.int32))
    buffer += list(bias_data.view(np.int32))
    buffer += list(reference_data.flatten().view(np.int32))

    # --- Запись в бинарный файл ---
    filename = "fullyconnected_data.bin"
    with open(filename, "wb") as f:
        # Пишем длину буфера (в элементах int32)
        f.write(struct.pack('i', len(buffer)))
        # Пишем сами данные
        f.write(struct.pack(f'{len(buffer)}i', *buffer))

    print(f"Файл '{filename}' сгенерирован успешно!")
    print(f"Диапазон весов: [{weight_data.min():.4f}, {weight_data.max():.4f}] (Оптимально для Symmetric INT4)")

if __name__ == "__main__":
    create_fullyconnected_file()