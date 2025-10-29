import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """
    加载并预处理数据
    """
    print("正在加载数据...")
    # 读取数据
    user_base = pd.read_csv('user_base.csv')
    user_behavior = pd.read_csv('user_behavior_assets.csv')
    
    # 合并数据
    data = pd.merge(user_base, user_behavior, on='user_id')
    
    print(f"数据加载完成，总记录数: {len(data)}")
    return data

def feature_engineering(data):
    """
    特征工程
    """
    print("正在进行特征工程...")
    
    # 创建目标变量：当前是否为高价值客户 (100万+)
    data['is_high_value'] = (data['asset_level'] == '100万+').astype(int)
    
    # 特征选择
    numerical_features = [
        'age', 'monthly_trade_amount', 'stock_portfolio_value', 
        'cash_balance', 'fixed_income_products_value', 'equity_products_value',
        'alternative_products_value', 'product_count', 'stock_trade_freq',
        'credit_card_monthly_expense', 'investment_monthly_count', 
        'stock_app_open_count', 'app_financial_view_time', 'app_product_compare_count'
    ]
    
    categorical_features = [
        'occupation_type', 'city_level', 'lifecycle_stage', 'marriage_status'
    ]
    
    # 处理缺失值
    for col in numerical_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # 准备特征和目标变量
    feature_columns = numerical_features + categorical_features
    selected_data = data[feature_columns].copy()
    y = data['is_high_value']
    
    # 对分类变量进行标签编码
    label_encoders = {}
    for col in categorical_features:
        if col in selected_data.columns:
            le = LabelEncoder()
            selected_data[col] = le.fit_transform(selected_data[col].astype(str))
            label_encoders[col] = le
    
    X = selected_data
    
    print(f"特征工程完成，特征数量: {X.shape[1]}")
    return X, y, feature_columns, label_encoders

def train_lightgbm_model(X, y):
    """
    训练LightGBM模型
    """
    print("正在训练LightGBM模型...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 设置参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42
    }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'eval'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
    )
    
    # 预测
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("模型训练完成")
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def visualize_feature_importance(model, feature_names):
    """
    可视化特征重要性
    """
    print("正在生成特征重要性可视化图表...")
    
    # 获取特征重要性
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 打印特征重要性
    print("\n特征重要性排序 (前20):")
    print(feature_importance_df.head(20))
    
    # 保存特征重要性到CSV
    feature_importance_df.to_csv('lightgbm_feature_importance.csv', index=False, encoding='utf-8-sig')
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 10))
    top_features = feature_importance_df.head(20)
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), list(top_features['feature']))
    plt.xlabel('特征重要性 (Information Gain)')
    plt.title('LightGBM模型特征重要性排序 (前20个最重要特征)')
    plt.grid(axis='x', alpha=0.3)
    
    # 在每个条形图上显示数值
    for i, (importance, feature) in enumerate(zip(top_features['importance'], top_features['feature'].tolist())):
        plt.text(importance, i, f'{importance:.0f}', va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    评估模型性能
    """
    print("正在评估模型性能...")
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 计算AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非高价值客户', '高价值客户']))
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 绘制ROC曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线 - LightGBM')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lightgbm_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, auc

def predict_future_high_value(data, model, feature_columns, label_encoders):
    """
    预测未来可能成为高价值客户的用户
    """
    print("正在预测未来可能成为高价值客户的用户...")
    
    # 准备特征数据
    selected_data = data[feature_columns].copy()
    
    # 对分类变量进行标签编码
    categorical_features = ['occupation_type', 'city_level', 'lifecycle_stage', 'marriage_status']
    for col in categorical_features:
        if col in selected_data.columns:
            le = label_encoders[col]
            # 处理未见过的标签
            selected_data[col] = selected_data[col].astype(str).apply(
                lambda x: x if x in le.classes_ else 'Unknown'
            )
            # 如果有未知标签，添加到标签编码器中
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            selected_data[col] = le.transform(selected_data[col])
    
    # 预测概率
    probabilities = model.predict(selected_data, num_iteration=model.best_iteration)
    
    # 添加预测结果到数据中
    data['predicted_probability'] = probabilities
    
    # 选择当前还不是高价值客户但预测概率高的前100名
    non_high_value = data[data['asset_level'] != '100万+']
    top_candidates = non_high_value.nlargest(100, 'predicted_probability')
    
    # 显示结果
    print("\n未来3个月最有可能成为高价值客户的前20名用户:")
    print(top_candidates[['user_id', 'name', 'asset_level', 'predicted_probability']].head(20))
    
    # 保存结果
    top_candidates.to_csv('predicted_future_high_value_customers_lgb.csv', index=False, encoding='utf-8-sig')
    
    return top_candidates

def main():
    """
    主函数
    """
    print("=== 客户高价值潜力预测系统（LightGBM版）===")
    
    # 加载数据
    data = load_and_preprocess_data()
    
    # 特征工程
    X, y, feature_columns, label_encoders = feature_engineering(data)
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_lightgbm_model(X, y)
    
    # 输出特征重要性
    feature_importance_df = visualize_feature_importance(model, feature_columns)
    
    # 评估模型
    accuracy, auc = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # 预测未来高价值客户
    top_candidates = predict_future_high_value(data, model, feature_columns, label_encoders)
    
    print("\n=== 结果总结 ===")
    print(f"1. 模型准确率: {accuracy:.4f}")
    print(f"2. 模型AUC: {auc:.4f}")
    print(f"3. 重要特征数量: {len(feature_importance_df)}")
    print(f"4. 预测未来会成为高价值客户的用户数: {len(top_candidates)}")
    print("\n结果文件已保存:")
    print("- lightgbm_feature_importance.csv: 特征重要性")
    print("- lightgbm_feature_importance.png: 特征重要性可视化图表")
    print("- lightgbm_roc_curve.png: ROC曲线图")
    print("- predicted_future_high_value_customers_lgb.csv: 预测的未来高价值客户列表")

if __name__ == "__main__":
    main()