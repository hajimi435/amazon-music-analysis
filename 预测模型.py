import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')

data = pd.read_csv(os.path.join(data_path, 'data_kmeans.csv'))

features = ['avg_rating', 'avg_sentiment', 'activity', 'user_value_score', 'loyalty_score']
X = data[features]
y = (data['risk_score'] >= 1.5).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
estimator = RandomForestClassifier(class_weight='balanced')

# 设置网格搜索与交叉验证，寻找最合适的树的量
param_dict = {
    'n_estimators': [200],
    'max_depth': [15]
}

grid_search = GridSearchCV(estimator, param_grid=param_dict, cv=4)

grid_search.fit(X_train, y_train)

y_predict = grid_search.predict(X_test)

# 测试
print(classification_report(y_test,y_predict))     # 精确率，召回率等等
print('AUC:', roc_auc_score(y_test, grid_search.predict_proba(X_test)[:,1]))

# 最佳参数--(200,15)
# print('最佳参数\n:', grid_search.best_params_)
# print('最佳预估器\n:', grid_search.best_estimator_)
# print('最佳准确率\n:', grid_search.best_score_)

# 特征的重要性
importances = pd.DataFrame({'feature': features, 'importance': grid_search.best_estimator_.feature_importances_})
print(importances.sort_values('importance',ascending=True))

# 保存模型
model_path = os.path.join(data_path, 'risk_model.pkl')
joblib.dump(grid_search.best_estimator_, model_path)
print(f'\n模型已保存至: {model_path}')


def predict_risk(user_data):
    model = joblib.load(model_path)

    # 转换为DataFrame
    input_df = pd.DataFrame([user_data], columns=features)

    # 预测类别
    prediction = model.predict(input_df)[0]

    # 获取概率
    probabilities = model.predict_proba(input_df)   # 输出高/低风险的概率

    return {
        'risk_level': '高风险' if prediction == 1 else '低风险',
        'risk_probability': probabilities[1],  # 高风险概率
        'safe_probability': probabilities[0],  # 低风险概率
        'confidence': max(probabilities)  # 置信度
    }


# 测试示例
if __name__ == '__main__':
    print('\n===== 模型测试 =====')

    test_user_1 = {
        'avg_rating': 3.0,
        'avg_sentiment': 0.25,
        'activity': 0.02,
        'user_value_score': 4.0,
        'loyalty_score': 0.08
    }
    result_1 = predict_risk(test_user_1)
    print(f'\n测试用户1（疑似高风险）:')
    print(f'  预测结果: {result_1["risk_level"]}')
    print(f'  高风险概率: {result_1["risk_probability"]:.2%}')
    print(f'  置信度: {result_1["confidence"]:.2%}')
