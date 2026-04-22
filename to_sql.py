from sqlalchemy import create_engine, text
import pandas as pd
import os

host = 'localhost'
user = 'root'
password = '129265'
database = 'amazon_project'

engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4')

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')
df = pd.read_csv(os.path.join(data_path, 'data_clean.csv'))
df_kmeans = pd.read_csv(os.path.join(data_path, 'data_kmeans.csv'))

# print(df.columns.tolist())

if __name__ == '__main__':
    try:
        connection = engine.connect()
        print("数据库连接成功！")
        connection.close()
    except Exception as e:
        print(f"数据库连接失败：{e}")
        exit(1)

    # 1.对用户进行去重，获取每个用户的关键指标
    user_agg = df.groupby('reviewerID').agg({
        'reviewTime': ['min', 'max', 'count']
    }).reset_index()

    user_agg.columns = ['reviewerID', 'first_review', 'last_review', 'total_reviews']

    user_means = df_kmeans[['reviewerID', 'loyalty_score', 'risk_score',
                            'avg_rating', 'avg_sentiment',
                            'activity', 'user_value_score', 'sentiment_std',
                            'kmeans', 'label']].drop_duplicates(subset='reviewerID')

    # 3.合并所有用户数据
    user_all = user_agg.merge(user_means, on='reviewerID', how='left')

    # 4.重命名与数据库对应
    user_all.columns = [
        'user_id',
        'first_review',
        'last_review',
        'total_reviews',
        'avg_sentiment',
        'avg_rating',
        'activity_score',
        'user_value_score',
        'loyalty_score',
        'risk_score',
        'sentiment_stability',
        'kmeans_label',
        'label'
    ]

    user_all['first_review'] = pd.to_datetime(user_all['first_review']).dt.date
    user_all['last_review'] = pd.to_datetime(user_all['last_review']).dt.date

    # 3.评论表和风险预测表
    df_review = df[['reviewerID', 'asin', 'overall', 'vote', 'reviewTime', 'reviewText', 'summary', 'verified']]
    df_predict = pd.read_csv(os.path.join(data_path, 'predict_result.csv'))

    valid_users = user_all['user_id'].unique()
    df_reviews = df_review[df_review['reviewerID'].isin(valid_users)].copy()

    df_reviews.rename(columns={
        'reviewerID': 'user_id',
        'reviewTime': 'review_time',
        'reviewText': 'review_text'
    }, inplace=True)

    df_predict.rename(columns={
        'reviewerID': 'user_id',
    }, inplace=True)

    df_reviews['review_time'] = pd.to_datetime(df_reviews['review_time']).dt.date

    try:
        with engine.connect() as conn:
            # 禁用外键检查
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

            # 清空这些表（保留表结构）
            conn.execute(text("TRUNCATE TABLE users"))
            conn.execute(text("TRUNCATE TABLE reviews"))
            if df_predict is not None:
                conn.execute(text("TRUNCATE TABLE predict_risk"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

            # 重新启用外键检查
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        # 然后追加数据（表已存在，结构正确）
        user_all.to_sql('users', engine, index=False, if_exists='append')
        df_reviews.to_sql('reviews', engine, index=False, if_exists='append')
        df_predict.to_sql('predict_risk', engine, index=False, if_exists='append')

        # 重新启用外键检查
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        print('写入成功！')

    except Exception as e:
        print(f'数据写入失败: {e}')

