# -*- coding: utf-8 -*-
import pandas as pd

# 读取数据
df = pd.read_csv('../data/train.csv', encoding='gbk')

# 计算每列的缺失值数量和比例
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# 创建缺失值统计表
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

# 按缺失值数量降序排列
missing_data = missing_data.sort_values(by='Missing Values', ascending=False)

# 打印结果
print(missing_data)

# 保存结果到文件
missing_data.to_csv('missing_values_report.csv')