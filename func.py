import numpy as np
import pandas as pd


def minmax_normalize(df):
    # Нормализация данных методом Min-Max
    normalized_df = df.copy()
    for column in df.select_dtypes(include=['number']).columns:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val - min_val == 0:
            # Если все значения одинаковые
            normalized_df[column] = 0.0
        else:
            normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
    return normalized_df


def matrix_correlation(df):
    df_new = df.copy()
    cols = df_new.columns
    n = len(cols)

    # Инициализируем матрицу
    corr_matrix = pd.DataFrame(np.eye(n), index=cols, columns=cols)

    # Вычисляем попарные корреляции
    for i in range(n):
        for j in range(i + 1, n):
            x = df_new.iloc[:, i]
            y = df_new.iloc[:, j]

            # Расчет по формуле Пирсона
            mean_x = x.mean()
            mean_y = y.mean()

            covariance = ((x - mean_x) * (y - mean_y)).sum()
            std_x = np.sqrt(((x - mean_x) ** 2).sum())
            std_y = np.sqrt(((y - mean_y) ** 2).sum())

            # Защита от деления на ноль
            if std_x == 0 or std_y == 0:
                corr = 0.0
            else:
                corr = covariance / (std_x * std_y)

            # Заполняем симметричную матрицу
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr

    return corr_matrix


def entropy(target_col):
    counts = target_col.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))


def information_gain(data, feature, target):

    # Энтропия до разбиения
    total_entropy = entropy(data[target])

    # Вычисляем взвешенную энтропию после разбиения
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0

    for value, count in zip(values, counts):
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (count / len(data)) * subset_entropy

    return total_entropy - weighted_entropy


def split_info(data, feature):
    values, counts = np.unique(data[feature], return_counts=True)
    weight = 0

    for value, count in zip(values, counts):
        weight -= (count / len(data)) * np.log2(count / len(data))
    return weight


def gain_ratio(data, target):

    ratios = {}
    features = data.columns.drop(target)

    for feature in features:
        ig = information_gain(data, feature, target)
        si = split_info(data, feature)

        # Избегаем деления на ноль
        ratios[feature] = ig / si if si != 0 else 0

    return pd.Series(ratios).sort_values(ascending=False)