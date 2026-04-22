import pandas as pd
import os
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')


data = pd.read_csv(os.path.join(data_path, 'data_kmeans.csv'))
groups = [data[data['kmeans'] == i]['user_value_score'].dropna() for i in range(5)]

# 1.方差齐性检验和假设检验
_, levene_p = stats.levene(*groups)
print(f'Levene检验 p-value: {levene_p:.3f}')

# 根据方差齐性选择检验方法
if levene_p < 0.05:
    print('方差不齐，使用Kruskal-Wallis检验')
    stat, p_value = stats.kruskal(*groups)
    print(f'Kruskal-Wallis结果: H={stat:.3f}, p={p_value:.3f}')
else:
    print('方差齐性，使用ANOVA检验')
    stat, p_value = stats.f_oneway(*groups)
    print(f'ANOVA检验结果: F={stat:.3f}, p={p_value:.3f}')

if p_value < 0.05:
    print('差异显著，聚类有效')
else:
    print('差异不显著，需要调整')

print('')

# 2.相关性检验
# 用户价值与忠诚度的相关性
corr1, p1 = pearsonr(data['user_value_score'], data['loyalty_score'])
print(f'Pearson相关系数(用户价值与忠诚度): {corr1:.3f}, p={p1:.3f}')
if abs(corr1) > 0.5:
    print('用户价值与忠诚度显著相关')
else:
    print('用户价值与忠诚度不显著相关')

corr2, p2 = pearsonr(data['risk_score'],data['activity'])
print(f'Pearson相关系数(风险与活跃度): {corr2:.3f}, p={p2:.3f}')
if abs(corr2) > 0.5:
    print('用户价值与忠诚度显著相关')
else:
    print('用户价值与忠诚度不显著相关')
print('')


# 3.卡方检验
# 检验聚类结果与用户价值标签是否相关
contingency_table = pd.crosstab(data['kmeans'], data['label'].str.split('--').str[0])
chi2, p, dof, expected = chi2_contingency(contingency_table)
if p < 0.05:
    print('聚类结果与用户价值标签显著相关')
else:
    print('聚类结果与用户价值标签不相关')
print('')