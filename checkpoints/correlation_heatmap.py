# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 读取数据
df = pd.read_csv('../data/train.csv', encoding='gbk')

# 选择数值型变量
numerical_columns = df.select_dtypes(include=['number']).columns
print('数值型变量列:', list(numerical_columns))

# 计算相关系数矩阵
correlation_matrix = df[numerical_columns].corr()

# 保存相关系数矩阵到CSV文件
correlation_matrix.to_csv('correlation_matrix.csv', encoding='utf-8-sig')
print('相关系数矩阵已保存至 correlation_matrix.csv')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('相关系数热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

print('相关系数热力图已保存至 correlation_heatmap.png')