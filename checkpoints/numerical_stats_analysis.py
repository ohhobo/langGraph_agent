# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取数据
df = pd.read_csv('../data/train.csv', encoding='gbk')

# 选择数值型变量
numerical_columns = df.select_dtypes(include=['number']).columns
print(f"数值型变量列: {list(numerical_columns)}")

# 生成基本统计信息
stats_summary = df[numerical_columns].describe().T
stats_summary['median'] = df[numerical_columns].median()
stats_summary = stats_summary[['mean', '50%', 'std', 'min', 'max']]
stats_summary.columns = ['均值', '中位数', '标准差', '最小值', '最大值']

# 保存结果
stats_summary.to_csv('numerical_statistics.csv', encoding='utf-8-sig')
print("数值型变量基本统计信息已保存至 'numerical_statistics.csv'")