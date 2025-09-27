# -*- coding: utf-8 -*-
import pandas as pd

# 读取CSV文件
file_path = '../data/train.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 显示数据框的基本信息
print("数据框的前5行:")
print(df.head())
print("\n数据框的基本信息:")
print(df.info())
print("\n数据框的统计摘要:")
print(df.describe())