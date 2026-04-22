import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')

data = pd.read_csv(os.path.join(data_path, 'data_kmeans.csv'))

# 1.绘制雷达图呈现聚类后个类别的特征
dimensions = ['activity', 'avg_sentiment', 'user_value_score',
              'loyalty_score', 'risk_score', 'avg_rating']

cluster_data = data.groupby('kmeans')[dimensions].mean()
transfer = MinMaxScaler()
Std_data = pd.DataFrame(
    transfer.fit_transform(cluster_data),
    columns=dimensions,
    index=cluster_data.index
)

# 圆周分成6份，endpoint不包含终点
angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
# 追加值到末尾
angles += angles[:1]

# 极坐标坐标系polar=True
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for idx in Std_data.index:
    values = Std_data.loc[idx].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}')  # x,y对应角度和标准化的值
    ax.fill(angles, values, alpha=0.15)  # 自动填充颜色

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dimensions, fontsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.10, 1.15))

ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(0, 1.1)

plt.title('Kmeans features')
plt.savefig(os.path.join(data_path, 'Kmeans features'), dpi=300, bbox_inches='tight')

# 2.统计聚类人数分布
plt.figure(figsize=(8, 6))
cluster_counts = data['kmeans'].value_counts().sort_index()
bars = plt.bar(cluster_counts.index, cluster_counts.values)
for bar, count in zip(bars, cluster_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        str(count),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='darkblue'
    )
plt.title('Kmeans count')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.savefig(os.path.join(data_path, 'Kmeans count'), dpi=300, bbox_inches='tight')

# 3.忠诚度与风险热力图
data['risk_level'] = pd.cut(
    data['risk_score'],
    bins=[-1, 0.5, 1.5, 3],
    labels=['Low', 'Medium', 'High']
)
data['loyalty_level'] = pd.cut(
    data['loyalty_score'],
    bins=3,  # 3是忠诚度分箱，表示自动分成三组
    labels=['Low', 'Medium', 'High']
)

cross_tab = pd.crosstab(data['risk_level'], data['loyalty_level'])

plt.figure(figsize=(8, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd',
            cbar_kws={'label': 'User Count'})
plt.title('Risk vs Loyalty Distribution', fontsize=14)
plt.xlabel('Loyalty Level', fontsize=12)
plt.ylabel('Risk Level', fontsize=12)
plt.savefig(os.path.join(data_path, 'risk_loyalty_heatmap.png'), dpi=300, bbox_inches='tight')

# 4.用户价值人数统计的3D图像显示
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(111, projection='3d')

user_value = data[['user_value_score', 'kmeans']]
for idx in user_value.index:
    if 0 <= user_value.loc[idx, 'user_value_score'] < 2:
        user_value.loc[idx, 'user_value_score'] = 1
    elif 2 <= user_value.loc[idx, 'user_value_score'] < 4:
        user_value.loc[idx, 'user_value_score'] = 3
    elif 4 <= user_value.loc[idx, 'user_value_score'] < 6:
        user_value.loc[idx, 'user_value_score'] = 5
    elif 6 <= user_value.loc[idx, 'user_value_score'] < 8:
        user_value.loc[idx, 'user_value_score'] = 7
    else:
        user_value.loc[idx, 'user_value_score'] = 9

user_value_count = user_value.groupby('kmeans')['user_value_score'].value_counts()

user_count = user_value_count.reset_index(name='count')

count = {}
for i in range(5):
    t = 1
    count_data = []
    while t <= 9:
        result = user_count.loc[(user_count['kmeans'] == i) & (user_count['user_value_score'] == t), 'count']
        if not result.empty:
            count_data.append(result.values[0])
        else:
            count_data.append(0)
        t += 2
    count[i] = count_data
print(count)

ys = [1, 3, 5, 7, 9]
colors = ['r', 'g', 'b', 'y', 'orange']
for i in range(5):
    data_z = count[i]
    xs = np.full(len(ys), i)
    ax.bar(xs, data_z, zs=ys, zdir='y', color=colors[i], alpha=0.5, width=0.6)

ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(1, 10, 2))
ax.set_xlabel('Cluster')
ax.set_ylabel('User Value Score')
ax.set_zlabel('Count')
plt.title('User Value Distribution by Cluster')
plt.savefig(os.path.join(data_path, 'user_value_3d.png'), dpi=300, bbox_inches='tight')
# plt.show()
