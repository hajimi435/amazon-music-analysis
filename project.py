"""
数据集列的含义是：
整体评分0
其他用户认为评论有用数1
验证是否购买过2
评论时间3
ID4
产品ID5
产品款式6
评论者名称7
正文8
摘要9
时间戳10

基本要求：
1. 通过评论的内容，识别可能存在的用户群体，并且为用户打上标签（如活跃用
户，高价值用户等，注意融合经济学的角度，包括用户偏好，比如质量，价
格，物流等。分析角度可以自行决定），需要提供全方位的用户画像
2. 为提高用户留存度，通过某些维度的数据分析预警可能的潜在流失客户
3. 找出导致客户流失的潜在因素，并且根据分析提出改进方案

"""

import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import save_npz, load_npz
import matplotlib.pyplot as plt
import numpy as np
import gzip

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', None)  # 不限制列宽

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')


def parse(path):
    g = gzip.open(path, 'rb')  # 打开gzip格式的压缩文件
    for l in g:  # 从压缩文件中逐行读取数据
        yield json.loads(l)  # 将json数据解析成python对象(列表，字典等)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):  # 循环时获取数据后遇到yield暂停，形成逐行提取
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def preprocess(data):
    # 评分是否存在异常值问题
    score_max = max(data.iloc[:, 0])
    score_min = min(data.iloc[:, 0])
    # print(score_max,score_min) 合理区间内

    # 若正文和摘要都不存在，整行删除
    index = data[data.iloc[:, 8].isna() & data.iloc[:, 9].isna()].index
    data = data.drop(index, axis=0)

    # print(data.loc[:,'image'].unique())，偶然发现Image是图像数据，对分析无助，删除该列
    data = data.drop('image', axis=1)

    # 统一转换数据类型
    for i in range(len(data.columns)):
        if i in (0, 1, 2, 10):
            data.iloc[:, i] = pd.to_numeric(data.iloc[:, i], errors='coerce')
        elif i == 3:
            data.iloc[:, i] = pd.to_datetime(data.iloc[:, i], errors='coerce')
        else:
            data.iloc[:, i] = data.iloc[:, i].astype('string')
    # print(data.dtypes)，可见是成功的

    # 处理字典和列表数据
    for col in data.columns:
        if data[col].apply(lambda x: isinstance(x, dict)).any():
            # print(f'{col}有字典数据')
            data[col] = data[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        elif data[col].apply(lambda x: isinstance(x, list)).any():
            # print(f'{col}有列表数据')
            data[col] = data[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    # 重复行
    if data.duplicated().any():
        index = data.duplicated().index
        data.drop_duplicates(inplace=True)

    # 完成上诉转换后，进行数据清洗
    # 空值问题，主要针对数值型数据进行0值填充，对字符串类型进行''空白字符填充
    # print(data.isnull())，大概发现vote是含空值的

    null_list = {}
    for i, j in enumerate(data.columns):
        if data.loc[:, j].isnull().any():
            null_list.setdefault(i, j)
    # print(null_list)
    # --{1: 'vote', 6: 'style', 7: 'reviewerName', 8: 'reviewText', 9: 'summary', 11: 'image'}

    # 对投票数，作0值处理；对除图像外的作空白字符填充
    data.iloc[:, 1].fillna(0, inplace=True)
    for i in range(len(data.columns)):
        if i in (6, 7, 8, 9):
            data.iloc[:, i].fillna(' ', inplace=True)

    # print(data.head(5))
    # print(data.isnull().sum())，成功了

    # 小写化
    data.loc[:, 'reviewText'] = data.loc[:, 'reviewText'].apply(lambda x: x.lower())
    data.loc[:, 'summary'] = data.loc[:, 'summary'].apply(lambda x: x.lower())

    # 原始数据的保存
    data.to_csv(os.path.join(data_path, 'data_clean.csv'), index=False)
    print(data.isnull().sum())

    return None


def describe():
    data = pd.read_csv(os.path.join(data_path, 'data_clean.csv'))

    print(data.loc[:, 'overall':'vote'].describe())  # 评分的mean在4.7，vote最大值有461

    # 1.统计各个星数
    data_star = pd.DataFrame(data['overall'].value_counts()).reset_index()
    print(data_star)  # 评高星的人数比较多
    color = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    plt.figure(figsize=(8, 6))
    plt.xlim(0, 6)
    for i in range(data_star.shape[0]):
        plt.bar(data_star.iloc[i, 0],
                data_star.iloc[i, 1],
                color=color[i],
                )
    plt.xlabel('Score')
    plt.ylabel('Count')

    # 2.星数与产品种类的关系，用来判断主要用户群体和用户偏好
    data_style = data[['overall', 'style']]

    data_information = pd.DataFrame(
        data_style.groupby('style')['overall'].agg([np.mean, np.sum])).reset_index().sort_values('mean', ascending=True,
                                                                                                 ignore_index=True)
    print(data_information)  # 产品均值相差比较大，但有样本量差距大的原因

    mean_score = []
    sum_rating = []
    label = []
    plt.figure(figsize=(8, 6))
    plt.xlim(2, 5.5)
    for i in range(data_information.shape[0]):
        mean_score.append(data_information.iloc[i, 1])
        sum_rating.append(data_information.iloc[i, 2])
        label.append(data_information.iloc[i, 0])
    plt.plot(
        mean_score,
        sum_rating,
        color="lightblue",
        marker='o',
        linestyle='-',
        linewidth=1,
    )

    for i in range(len(sum_rating)):
        if i == 9:
            va = 'center'
            offset_y = 0
            offset_x = 22
        elif i % 2 == 1:
            va = 'bottom'
            offset_y = 8
            offset_x = 0
        elif i % 2 == 0:
            va = 'top'
            offset_y = -8
            offset_x = 0
        plt.annotate(label[i][13:-2],
                     (mean_score[i], sum_rating[i]),
                     xytext=(offset_x, offset_y),
                     ha='center',
                     textcoords='offset points',
                     va=va,
                     fontsize=5,
                     color='lightblue')
    plt.xlabel('Mean Overall Score')
    plt.ylabel('Sum of Ratings')

    # 3.时间序列统计
    data['year'] = data['reviewTime'].str[:4]
    data_year = pd.DataFrame(data['year'].value_counts()).reset_index()
    data_year.columns = ['year', 'count']
    data_year = data_year.sort_values('year', ascending=True, ignore_index=True)
    print(data_year)  # 可以看到评论人数在近几年下降，说明有用户流失的风险

    year = []
    count = []
    plt.figure(figsize=(8, 6))
    for i in range(data_year.shape[0]):
        year.append(data_year.iloc[i, 0][2:])
        count.append(data_year.iloc[i, 1])

    plt.plot(year,
             count,
             color='lightblue',
             linestyle='-',
             marker='o',
             linewidth=1)
    plt.xlabel('year')
    plt.ylabel('count of review')

    # 4.购买用户与未购买用户的评级差异
    data_verified = data[['verified', 'overall']].groupby('verified')['overall'].agg(np.mean).reset_index()
    plt.figure(figsize=(8, 6))
    color = ['red', 'blue']
    plt.xlim(0, 3)
    x = np.arange(1, 3)
    for i in range(data_verified.shape[0]):
        plt.bar(x[i],
                data_verified.iloc[i, 1],
                color=color[i],
                )
        plt.annotate(data_verified.iloc[i, 0],
                     (x[i], data_verified.iloc[i, 1]),
                     xytext=(0, 8),
                     ha='center',
                     textcoords='offset points',
                     va='top',
                     fontsize=10,
                     color='black')
    plt.xlabel('type')
    plt.ylabel('star')
    plt.show()
    # print(data_verified)  # 差距不大

    # 5.用户评论的频率
    frequency = data['reviewerID'].value_counts()
    # print(frequency)

    # 观点输出
    print('')
    print('首先从评分和评论数上看，评分数总体比较高，评论数比较多，用户活跃；\n'
          '从星数统计上看，4，5星占比多，用户对产品和服务大体上比较满意\n'
          '从产品种类上看，用户偏好MP3，Audio CD等产品，可以着重分析这两个产品吸引的用户群体的特点\n'
          '从时间序列上看，评论数大致反映出产品的销售情况由快速上升到下降的特点，说明存在用户流失\n'
          '未购买和购买用户在评级上没有明显区别，后续分析时可以不作详细区分\n'
          '存在用户频繁评论的情况，说明有比较高的用户留存度\n')

    return None


# 获取特征
def feature():
    data = pd.read_csv(os.path.join(data_path, 'data_clean.csv'))

    # 添加时间序列的影响
    data['sequence'] = data['reviewTime'].str[:4].astype(int)

    # 防止空值问题
    data.iloc[:, 1].fillna(0, inplace=True)
    for i in range(len(data.columns)):
        if i in (6, 7, 8, 9):
            data.iloc[:, i].fillna(' ', inplace=True)

    # 1.文本特征抽取，通过TF-idf获取文本质量特征
    stop_word = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'that', 'with', 'on', 'very', 'songs', 'really',
                 'by', 'this', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'song', 'music', 'can',
                 'as', 'your', 'have', 'all', 'her', 'my', 'was', 'his', 'had', 'has', 'been', 'me', 'so', 'one']
    transfer_text = TfidfVectorizer(max_features=250,
                                    max_df=0.9,
                                    min_df=2,
                                    stop_words=stop_word)
    # 稀疏矩阵，形状(数据行数 × 250)
    feature_text = transfer_text.fit_transform(data['reviewText'])
    # print(feature_text.toarray())

    transfer_summary = TfidfVectorizer(max_features=100,
                                       max_df=0.9,
                                       min_df=2,
                                       stop_words=stop_word)
    # 稀疏矩阵
    feature_summary = transfer_summary.fit_transform(data['summary'])

    # 保存
    save_npz(os.path.join(data_path, 'tfidf_text.npz'), feature_text)
    save_npz(os.path.join(data_path, 'tfidf_summary.npz'), feature_summary)

    # 特征名映射，TF-IDF 矩阵的列索引与词汇表索引一一对应，映射关系：矩阵第 j 列 ↔ 词汇表第 j 个词汇
    np.save(os.path.join(data_path, 'names_text.npy'), transfer_text.get_feature_names_out())
    np.save(os.path.join(data_path, 'names_summary.npy'), transfer_summary.get_feature_names_out())

    # 2.活跃度特征
    activity = calculate_activity_time(data)
    data['activity'] = data.merge(activity[['temporal_activity_score']], left_on='reviewerID', right_index=True)[
        'temporal_activity_score']

    # 3.文本情感分析
    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        if pd.isna(text) or text == ' ':
            return 0.0
        score = analyzer.polarity_scores(text)
        return score['compound']

    data['text_sentiment'] = data['reviewText'].apply(get_vader_sentiment)
    data['summary_sentiment'] = data['summary'].apply(get_vader_sentiment)

    # 长度加权
    def weight_sentiment(row):
        text_len = len(row['reviewText'])
        summary_len = len(row['summary'])
        total_len = text_len + summary_len
        if total_len == 0:
            return 0.0

        weight_score = (text_len / total_len) * row['text_sentiment'] + (summary_len / total_len) * row[
            'summary_sentiment']
        return weight_score

    data['sentiment'] = data.apply(weight_sentiment, axis=1)
    # print(data)

    # 4.获取复合特征(已合并的特征)
    data_eliminate = create_business_feature(data)
    # print(data_eliminate.head(10))

    # 构建特征矩阵
    feature_columns = [
        'user_value_score',
        'loyalty_score',
        'risk_score',
        'sentiment_std',
        'avg_sentiment',
        'activity'
    ]

    feature_matrix = data_eliminate[feature_columns].values

    # 标准化
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # 保存
    np.save(os.path.join(data_path, 'feature.npy'), feature_matrix_scaled)
    data_eliminate.to_csv(os.path.join(data_path, 'data_feature.csv'), index=False, encoding='utf-8-sig')

    print(f"特征矩阵形状: {feature_matrix_scaled.shape}")
    print(f"特征列名: {feature_columns}")

    return None


# 制定偏重时间维度的活跃度特征，评价用户流失
def calculate_activity_time(dataset):
    # 获取每个用户的评论时间特征
    user_temporal_features = dataset.groupby('reviewerID').agg({
        'sequence': [
            'min',
            'max',
            'nunique',  # 获取唯一值个数
            'std'
        ],
        'reviewText': 'count'
    }).round(3)  # 每一行对应一个用户

    # 重命名列
    user_temporal_features.columns = [
        'first_year', 'last_year', 'active_years', 'time_std', 'total_reviews'
    ]

    current_year = dataset['sequence'].max()  # 数据集内最新的评论时间
    max_active_years = user_temporal_features['active_years'].max()
    max_total_review = user_temporal_features['total_reviews'].max()

    # 活跃度得分 = 时间连续性权重 + 最近活跃度权重 + 活跃频率权重
    user_temporal_features['temporal_activity_score'] = (
            0.4 * (user_temporal_features['active_years'] / max_active_years) +  # 活跃连续性
            0.3 * (1 - (current_year - user_temporal_features['last_year']) /
                   np.where(current_year - user_temporal_features['first_year'] >= 1,
                            current_year - user_temporal_features['first_year'], 1)) +  # 最近活跃度
            0.3 * (user_temporal_features['total_reviews'] / max_total_review)  # 评论频率
    )

    return user_temporal_features


# 创建对业务有帮助的复合特征
def create_business_feature(data):
    # 进行用户去重和前面的一系列指标的平均化
    data_eliminate = data.groupby('reviewerID').agg(
        avg_rating=('overall', 'mean'),
        avg_sentiment=('sentiment', 'mean'),
        total_reviews=('reviewText', 'count'),
        verified_ratio=('verified', 'mean'),
        total_votes=('vote', 'sum'),
        activity=('activity', 'mean'),
    ).reset_index()

    # 计算情感标准差
    sentiment_std = data.groupby('reviewerID')['sentiment'].std().fillna(0)
    data_eliminate['sentiment_std'] = data_eliminate['reviewerID'].map(sentiment_std)

    # 用户价值得分 = 评分*购买确认*情感得分
    data_eliminate['user_value_score'] = (
            data_eliminate['avg_rating'] *
            data_eliminate['verified_ratio'] *
            (data_eliminate['avg_sentiment'] + 1)
    )

    # 用户忠诚度得分 = 购买确认*评分*活跃度
    data_eliminate['loyalty_score'] = (
            data_eliminate['verified_ratio'] *
            data_eliminate['avg_rating'] *
            data_eliminate['activity']
    )

    # 流失风险指标
    # 基础风险：负面情感 + 低评分
    base_risk = np.where(
        (data_eliminate['avg_sentiment'] < 0.1) & (data_eliminate['avg_rating'] <= 3.5),
        1, 0
    )

    # 活跃度调节因子：高活跃用户的基础风险影响更大
    activity_factor = np.where(
        data_eliminate['activity'] > data_eliminate['activity'].quantile(0.75),
        1.5,
        1.0
    )

    # 近期不活跃风险：降低情感阈值，捕捉更多潜在流失
    recency_risk = np.where(
        (data_eliminate['activity'] < data_eliminate['activity'].quantile(0.33)) &
        (data_eliminate['avg_sentiment'] < 0.3),
        1, 0
    )

    # 情感下降风险（最近评论情感低于历史平均）
    sentiment_decline_risk = np.where(
        (data_eliminate['sentiment_std'] > data_eliminate['sentiment_std'].quantile(0.66)) &
        (data_eliminate['avg_sentiment'] < data_eliminate['avg_sentiment'].median()),
        0.5, 0
    )

    # 综合风险得分 = 基础风险 * 活跃度调节 + 近期不活跃风险 + 情感下降风险
    data_eliminate['risk_score'] = (base_risk * activity_factor + recency_risk + sentiment_decline_risk).clip(0, 3)

    return data_eliminate


# 聚类与用户标签
def cluster():
    data = pd.read_csv(os.path.join(data_path, 'data_feature.csv'), encoding='utf-8-sig')

    feature_np = np.load(os.path.join(data_path, 'feature.npy'))

    # 1.肘部法则和轮廓系数发确定聚类数量
    # 肘部法则，wcss是距离平方和，k越大组内点越大，组内平方和越小
    # wcss = []
    # for k in range(1, 11):
    #    kmeans = KMeans(n_clusters=k)
    #    kmeans.fit(feature_np)
    #    wcss.append(kmeans.inertia_)
    # print(wcss)
    wcss = [99384.00000000006, 74766.35718306147, 57364.14756731535, 47942.38879599467, 40456.26756543017,
            36149.31252111676, 32735.37821135601, 30335.133286744167, 28121.912577598, 26244.202823294556]

    # 进行可视化
    plt.plot(range(1, 11), wcss)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

    # 轮廓系数，数学含义是
    # (样本点到所属集群质心的距离-距离最近的其他集群质心到该样本点的距离)/它两的最大值  越接近1则越相似
    # scores = []
    # for i in range(2,11):    # 至少两个群
    #    kmeans = KMeans(n_clusters=i)
    #    label = kmeans.fit_predict(feature_np)
    #    score = silhouette_score(feature_np,label)
    #    scores.append(score)
    # print(scores)
    scores = [0.24501494996483208, 0.2837227150724813, 0.30612483614937347, 0.321228518506661, 0.2852523602124213,
              0.27675489531070124, 0.2850733184224808, 0.28865211956574843, 0.2964727643995678]

    # 可以发现轮廓系数和肘部法则的曲线相似，选择k=5

    # 2.进行聚类
    # (1).我们有主要对我们的情感，文本，活跃度，流失风险，评分稳定性，忠诚度和用户价值进行聚类分析
    estimator = KMeans(n_clusters=5, random_state=42)
    label = estimator.fit_predict(feature_np)
    # print(f'聚类结果：{label}')
    data['kmeans'] = label

    # (2).加载评论数据，TF-IDF矩阵和词汇表，由于聚类比较少，仅处理正文内容
    data_clean = pd.read_csv(os.path.join(data_path, 'data_clean.csv'))

    # 为每条评论添加对应的聚类标签
    user_cluster = data[['reviewerID', 'kmeans']].drop_duplicates()
    data_clean_with_cluster = data_clean.merge(user_cluster, on='reviewerID', how='inner')

    # 加载TF-IDF矩阵（评论级别）
    tfidf_text = load_npz(os.path.join(data_path, 'tfidf_text.npz'))
    feature_names = np.load(os.path.join(data_path, 'names_text.npy'), allow_pickle=True)

    # 对每个聚类提取高频词
    for i in range(5):
        # 找到属于聚类i的评论在原始数据中的位置
        cluster_mask = data_clean_with_cluster['kmeans'] == i  # 获取列
        cluster_indices = cluster_mask[cluster_mask].index  # 得到索引

        if len(cluster_indices) == 0:
            print(f'聚类{i}没有评论数据')
            continue

        # 提取该聚类的TF-IDF行
        cluster_tfidf = tfidf_text[cluster_indices]

        # 按列计算平均TF-IDF值
        mean_tfidf = np.array(cluster_tfidf.mean(axis=0)).flatten()

        # 获取Top 10关键词
        top_indices = np.argsort(mean_tfidf)[-10:][::-1]
        top_words = [(feature_names[idx], mean_tfidf[idx]) for idx in top_indices]

        print(f'\n聚类{i}的高频词：')
        for word, score in top_words:
            print(f'  {word}: {score:.4f}')

    # 我们进行分组统计
    cluster_analysis = data.groupby('kmeans').agg({
        'activity': ['mean', 'std'],
        'avg_sentiment': ['mean', 'std'],
        'user_value_score': ['mean', 'std'],
        'loyalty_score': ['mean', 'std'],
        'risk_score': ['mean', 'std'],
        'sentiment_std': ['mean', 'std'],
        'avg_rating': ['mean', 'std'],
    }).round(3)
    print(cluster_analysis)

    # print(data.groupby('kmeans').size()),统计人数

    # 3.用户画像
    # 忠诚度阈值
    lty_thresholds = data['loyalty_score'].quantile([0.33, 0.67])  # 返回Series对象，索引为0.33，0.67
    loyalty_low, loyalty_high = lty_thresholds[0.33], lty_thresholds[0.67]

    # 标签制定函数
    def label_users(row):
        # 价值标签
        if row['user_value_score'] > 7.0:
            value_label = '高价值'
        elif row['user_value_score'] > 3.0:
            value_label = '中价值'
        else:
            value_label = '低价值'

        # 活跃度标签
        if row['activity'] > 0.25:
            activity_label = '高活跃'
        elif row['activity'] > 0.15:
            activity_label = '活跃'
        elif row['activity'] > 0.05:
            activity_label = '一般'
        else:
            activity_label = '低活跃'

        # 风险标签
        if row['risk_score'] >= 1.5:
            risk_label = '高风险'
        elif row['risk_score'] >= 0.5:
            risk_label = '中风险'
        else:
            risk_label = '低风险'

        # 情感标签
        if row['avg_sentiment'] > 0.5:
            sentiment_label = '积极'
        elif row['avg_sentiment'] < -0.1:
            sentiment_label = '消极'
        else:
            sentiment_label = '中性'

        # 忠诚度标签
        if row['loyalty_score'] >= loyalty_high:
            loyalty_label = '高忠诚度'
        elif row['loyalty_score'] >= loyalty_low:
            loyalty_label = '中忠诚度'
        else:
            loyalty_label = '低忠诚度'

        return f'{value_label}--{activity_label}--{risk_label}--{sentiment_label}--{loyalty_label}'

    data['label'] = data.apply(label_users, axis=1)
    # 简单统计
    # print(data['label'].value_counts())

    # 保存聚类后的结果
    data.to_csv(os.path.join(data_path, 'data_kmeans.csv'), index=False, encoding='utf-8-sig')

    return None


if __name__ == '__main__':
    # 1.数据导入
    dataset = getDF(os.path.join(base_path,'Digital_Music_5.json.gz'))
    print(dataset.head(5))
    dataset.to_csv(os.path.join(data_path, 'Digital_Music_5'))

    # 2.数据预处理
    preprocess(dataset)

    # 3.描述性统计和可视化
    describe()

    # 4.获取特征矩阵
    feature()

    # 5.聚类与用户标签
    cluster()
