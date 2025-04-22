import numpy as np
import pandas as pd


def mean_most_likely_interval(column: pd.Series):
    n = len(column)
    min_val = column.min()
    max_val = column.max()

    # 1. Определение количества интервалов по Стерджесу (округление рввех)
    l = int(np.ceil(1 + 3.32 * np.log10(n)))

    # 2. Вычисление длины интервала
    h = (max_val - min_val) / l

    # 3. Создание интервалов и подсчёт частот
    bins = np.linspace(min_val, max_val, l)
    freq, bin_edges = np.histogram(column, bins=bins)

    # 4. Нахождение наиболее вероятного интервала
    most_probable_idx = np.argmax(freq)
    min_i = bin_edges[most_probable_idx]

    # 5. Расчёт среднего значения интервала по формуле
    s_int = (min_i + most_probable_idx * h) + h / 2

    return s_int
