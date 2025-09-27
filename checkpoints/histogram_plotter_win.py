# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('../data/train.csv', encoding='gbk')

# 识别数值型变量列
numerical_columns = df.select_dtypes(include=['number']).columns
print(f'数值型变量列: {list(numerical_columns)}')

# 为每个数值型变量绘制直方图
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=30, edgecolor='black')  # 删除缺失值后绘图
    plt.title(f'{column} 的直方图')
    plt.xlabel(column)
    plt.ylabel('频率')
    plt.grid(True)
    plt.savefig(f'{column}_histogram.png')
    plt.close()
    print(f'已保存 {column} 的直方图至 {column}_histogram.png')