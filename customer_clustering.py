import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# 读取数据
user_base = pd.read_csv('user_base.csv')
user_behavior_assets = pd.read_csv('user_behavior_assets.csv')

# 数据预处理
# 合并两个数据表
df = pd.merge(user_base, user_behavior_assets, on='user_id', how='inner')

# 特征工程
# 选择用于聚类的特征
cluster_features = [
    'age',
    'monthly_trade_amount',
    'stock_portfolio_value',
    'cash_balance',
    'fixed_income_products_value',
    'equity_products_value',
    'alternative_products_value',
    'product_count',
    'stock_trade_freq',
    'credit_card_monthly_expense',
    'investment_monthly_count',
    'stock_app_open_count',
    'app_financial_view_time',
    'app_product_compare_count'
]

# 处理分类变量，进行独热编码
categorical_features = ['gender', 'occupation', 'occupation_type', 'marriage_status', 'city_level']
df_encoded = pd.get_dummies(df, columns=categorical_features)

# 合并编码后的分类特征
cluster_features_extended = cluster_features + [col for col in df_encoded.columns if col.startswith(tuple(categorical_features))]

# 处理缺失值
df_cluster = df_encoded[cluster_features_extended].fillna(0)

# 数据标准化
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df_cluster)

# 转换回DataFrame以保留列名
df_scaled = pd.DataFrame(df_scaled_array)
df_scaled.columns = [str(col) for col in cluster_features_extended]

# 使用肘部法则确定最佳聚类数
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# 根据肘部法则选择聚类数（这里我们选择4个聚类）
n_clusters = 4

# 执行K-means聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(df_scaled)

# 将聚类结果添加到原始数据
df['cluster'] = cluster_labels

# 分析各个聚类的特征
cluster_analysis = df.groupby('cluster').agg({
    'age': 'mean',
    'monthly_trade_amount': 'mean',
    'stock_portfolio_value': 'mean',
    'cash_balance': 'mean',
    'fixed_income_products_value': 'mean',
    'equity_products_value': 'mean',
    'alternative_products_value': 'mean',
    'product_count': 'mean',
    'stock_trade_freq': 'mean',
    'credit_card_monthly_expense': 'mean',
    'investment_monthly_count': 'mean',
    'stock_app_open_count': 'mean',
    'app_financial_view_time': 'mean',
    'app_product_compare_count': 'mean'
}).round(2)

print("聚类分析结果:")
# 打印完整的特征均值，确保所有列都显示
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(cluster_analysis)

# 打印每个聚类的样本数量
print("\n各聚类样本数量:")
print(df['cluster'].value_counts().sort_index())

# 计算整体数据的均值用于比较
overall_means = df[cluster_features].mean()
print("\n整体数据特征均值:")
print(overall_means)

# 使用PCA降维以便可视化
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# 创建可视化图表
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Clustering Visualization (PCA)')
plt.colorbar(scatter)

# 添加聚类中心
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.legend()
plt.show()

# 为每个聚类命名（根据业务理解和特征分析）
cluster_names_dict: Dict[int, str] = {}
for i in range(n_clusters):
    cluster_data = df[df['cluster'] == i]
    
    # 根据特征判断群组类型
    avg_age = cluster_data['age'].mean()
    avg_trade = cluster_data['monthly_trade_amount'].mean()
    avg_equity = cluster_data['equity_products_value'].mean()
    avg_fixed_income = cluster_data['fixed_income_products_value'].mean()
    avg_alternative = cluster_data['alternative_products_value'].mean()
    avg_stock = cluster_data['stock_portfolio_value'].mean()
    avg_app_usage = cluster_data['stock_app_open_count'].mean()
    avg_trade_freq = cluster_data['stock_trade_freq'].mean()
    
    # 计算总资产
    total_assets = avg_stock + avg_fixed_income + avg_equity + avg_alternative
    
    # 根据特征进行命名
    if avg_trade_freq > 25 and avg_app_usage > 50:
        cluster_names_dict[i] = "高交易活跃群体"
    elif avg_age < 45 and total_assets > 4000000:
        cluster_names_dict[i] = "高净值均衡群体"
    elif avg_equity / (total_assets + 1e-8) > 0.35:  # 权益类产品占比超过35%
        cluster_names_dict[i] = "权益偏好群体"
    else:
        cluster_names_dict[i] = "稳健投资群体"

print("\n聚类命名结果:")
for cluster_id, name in cluster_names_dict.items():
    count = len(df[df['cluster'] == cluster_id])
    print(f"聚类 {cluster_id} ({name}): {count} 位客户")

# 将聚类名称添加到数据中
def map_cluster_name(cluster_id: int) -> str:
    return cluster_names_dict[cluster_id]

cluster_name_series = df['cluster'].map(map_cluster_name)
df['cluster_name'] = cluster_name_series

# 保存带有聚类标签的数据
df.to_csv('customer_clusters.csv', index=False, encoding='utf-8-sig')

print("\n聚类结果已保存到 'customer_clusters.csv' 文件中")

# 显示每个群组的详细特征
print("\n各群组详细特征:")
for cluster_id, name in cluster_names_dict.items():
    print(f"\n{name} (聚类 {cluster_id}):")
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"  客户数量: {len(cluster_data)}")
    print(f"  平均年龄: {cluster_data['age'].mean():.1f} 岁")
    print(f"  平均月交易金额: {cluster_data['monthly_trade_amount'].mean():.2f}")
    print(f"  平均股票市值: {cluster_data['stock_portfolio_value'].mean():.2f}")
    print(f"  平均现金余额: {cluster_data['cash_balance'].mean():.2f}")
    print(f"  平均固收类产品市值: {cluster_data['fixed_income_products_value'].mean():.2f}")
    print(f"  平均权益类产品市值: {cluster_data['equity_products_value'].mean():.2f}")
    print(f"  平均另类产品市值: {cluster_data['alternative_products_value'].mean():.2f}")
    print(f"  平均产品数量: {cluster_data['product_count'].mean():.1f}")
    print(f"  平均股票交易频率: {cluster_data['stock_trade_freq'].mean():.1f}")
    print(f"  平均信用卡月消费: {cluster_data['credit_card_monthly_expense'].mean():.2f}")
    print(f"  平均投资月交易次数: {cluster_data['investment_monthly_count'].mean():.1f}")
    print(f"  平均股票App打开次数: {cluster_data['stock_app_open_count'].mean():.1f}")
    print(f"  平均App金融模块浏览时长: {cluster_data['app_financial_view_time'].mean():.1f}")
    print(f"  平均App产品对比次数: {cluster_data['app_product_compare_count'].mean():.1f}")
    
    # 计算总资产
    total_assets = (cluster_data['stock_portfolio_value'].mean() + 
                   cluster_data['fixed_income_products_value'].mean() + 
                   cluster_data['equity_products_value'].mean() + 
                   cluster_data['alternative_products_value'].mean())
    print(f"  总资产估算: {total_assets:.2f}")