# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_path = 'SimSun.ttf'  # 请确保字体文件在当前目录或指定正确路径
font = FontProperties(fname=font_path, size=12)

# 读取数据
df = pd.read_csv('../data/train.csv', encoding='gbk')

# 识别分类变量
categorical_columns = df.select_dtypes(include=['object']).columns
print(u'分类变量列:', list(categorical_columns))

# 分析每个分类变量的分布
for col in categorical_columns:
    print(f'\n分析分类变量: {col}')
    # 统计频数和比例
    value_counts = df[col].value_counts()
    proportions = df[col].value_counts(normalize=True) * 100
    
    # 创建频数和比例的DataFrame
    distribution_df = pd.DataFrame({
        u'频数': value_counts,
        u'比例(%)': proportions
    })
    
    # 保存分布情况到CSV文件
    distribution_df.to_csv(f'{col}_distribution.csv', encoding='utf-8-sig')
    print(u'{col} 的分布情况已保存至 {col}_distribution.csv'.format(col=col))
    
    # 绘制条形图（TOP10）
    top10_values = value_counts.head(10)
    plt.figure(figsize=(10, 6))
    top10_values.plot(kind='bar')
    plt.title(u'{col} 分布 (TOP10)'.format(col=col), fontproperties=font)
    plt.xlabel(u'类别', fontproperties=font)
    plt.ylabel(u'频数', fontproperties=font)
    plt.xticks(rotation=45, fontproperties=font)
    plt.tight_layout()
    plt.savefig(f'{col}_distribution_top10.png')
    plt.close()
    print(u'{col} 的分布图已保存至 {col}_distribution_top10.png'.format(col=col))