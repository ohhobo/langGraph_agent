# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Attempt to use a Chinese font if available
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
available_fonts = fm.get_font_names()
selected_font = None

for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font]
else:
    # Fallback to default font if no Chinese font is found
    plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['axes.unicode_minus'] = False

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