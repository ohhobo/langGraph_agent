# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 读取数据
df = pd.read_csv('../data/train.csv', encoding='gbk')

# 获取数值型变量列
numerical_columns = df.select_dtypes(include=['number']).columns
print('数值型变量列:', list(numerical_columns))

# 设置中文字体
font_path = 'SimSun.ttf'
font_prop = fm.FontProperties(fname=font_path, size=12)
plt.rcParams['axes.unicode_minus'] = False

# 为每个数值型变量绘制箱线图
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    df.boxplot(column=col)
    plt.title(f'{col} 箱线图', fontproperties=font_prop)
    plt.ylabel(col, fontproperties=font_prop)
    plt.xlabel('数值', fontproperties=font_prop)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{col}_boxplot.png')
    plt.close()
    print(f'已保存 {col} 的箱线图至 {col}_boxplot.png')