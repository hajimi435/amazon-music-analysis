import joblib
import os
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')

estimator = joblib.load(os.path.join(data_path,'risk_model.pkl'))
data = pd.read_csv(os.path.join(data_path,'data_kmeans.csv'))

features = ['avg_rating', 'avg_sentiment', 'activity', 'user_value_score', 'loyalty_score']
data_features = data[features]
data['predict_result'] = estimator.predict(data_features)

proba = estimator.predict_proba(data_features)   # 获取高风险概率
if proba.ndim == 1:
    data['risk_probability'] = proba[0]
else:
    data['risk_probability'] = proba[:, 1]

# print(data[['reviewerID', 'predict_result', 'risk_probability']].head(10))

high_risk_users = data[data['predict_result'] == 1]
# print(f'\n高风险用户数量: {len(high_risk_users)}')
# print(f'高风险用户比例: {len(high_risk_users)/len(data)*100:.2f}%')

high_risk_users.to_csv(os.path.join(data_path, 'predict_result.csv'), index=False, encoding='utf-8-sig')

print(high_risk_users.columns)