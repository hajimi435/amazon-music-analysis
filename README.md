# 亚马逊音乐用户画像与流失预警

## 项目概述
基于 30 万条 Amazon Music 评论数据，构建用户分群与流失预测模型，并通过 Linux Shell 脚本实现自动化部署。

## 技术栈
Python 3.9 · Scikit-learn · Pandas · MySQL · Linux Shell

## 核心成果
- **用户分群**：KMeans 聚类（k=5），经 Kruskal-Wallis 检验验证（p<0.001）
- **流失预测**：随机森林模型 AUC 0.97，关键特征为“评论情感得分”
- **自动化部署**：编写 Shell 脚本，每日定时执行预测、写入 MySQL、生成可视化报告

## 主要文件说明
- `project.py`：数据清洗、特征工程
- `预测模型.py`：模型训练与评估
- `Visualization.py`：图表输出
- `to_sql.py`：写入数据库
- `run_all.sh`：Linux 环境一键执行脚本

## 快速复现
```bash
git clone https://github.com/hajimi435/amazon-music-analysis.git
cd amazon-music-analysis
pip install -r requirements.txt
python project.py
python 预测模型.py
python 实际预测.py
python Visualization.py
python to_sql.py
python ANOVA检验.py
