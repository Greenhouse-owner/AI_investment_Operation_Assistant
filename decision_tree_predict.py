import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

def train_decision_tree_model(X, y):
    """
    训练决策树模型
    """
    print("正在训练决策树模型...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练决策树模型 (最大深度为4)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    print("模型训练完成")
    return model, X_train, X_test, y_train, y_test, y_pred

def visualize_decision_tree(model, feature_names):
    """
    可视化决策树
    """
    print("正在生成决策树可视化图表...")
    
    # 文本打印决策树
    from sklearn.tree import export_text
    tree_rules = export_text(model, feature_names=feature_names)
    print("决策树结构:")
    print(tree_rules)
    
    # 保存文本到文件
    with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
        f.write(tree_rules)
    
    # 图形化可视化决策树
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=['非高价值客户', '高价值客户'], 
              filled=True, rounded=True, fontsize=10)
    plt.title('客户资产提升至100万+预测决策树 (最大深度=4)', fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tree_rules

def evaluate_model(model, X_test, y_test, y_pred):
    """
    评估模型性能
    """
    print("正在评估模型性能...")
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非高价值客户', '高价值客户']))
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排序 (前10):")
    print(feature_importance.head(10))
    
    # 保存特征重要性到CSV
    feature_importance.to_csv('decision_tree_feature_importance.csv', index=False, encoding='utf-8-sig')
    
    return accuracy, feature_importance

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
    
    # 预测
    predictions = model.predict(selected_data)
    probabilities = model.predict_proba(selected_data)[:, 1]
    
    # 添加预测结果到数据中
    data['predicted_probability'] = probabilities
    data['predicted_high_value'] = predictions
    
    # 选择当前还不是高价值客户但预测会成为高价值客户的用户
    non_high_value = data[data['asset_level'] != '100万+']
    future_high_value = non_high_value[non_high_value['predicted_high_value'] == 1]
    
    # 按预测概率排序
    future_high_value = future_high_value.nlargest(100, 'predicted_probability')
    
    # 显示结果
    print("\n未来3个月最有可能成为高价值客户的前20名用户:")
    print(future_high_value[['user_id', 'name', 'asset_level', 'predicted_probability']].head(20))
    
    # 保存结果
    future_high_value.to_csv('predicted_future_high_value_customers_dt.csv', index=False, encoding='utf-8-sig')
    
    return future_high_value

def main():
    """
    主函数
    """
    print("=== 客户高价值潜力预测系统（决策树版）===")
    
    # 加载数据
    data = load_and_preprocess_data()
    
    # 特征工程
    X, y, feature_columns, label_encoders = feature_engineering(data)
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred = train_decision_tree_model(X, y)
    
    # 可视化决策树
    tree_rules = visualize_decision_tree(model, feature_columns)
    
    # 评估模型
    accuracy, feature_importance = evaluate_model(model, X_test, y_test, y_pred)
    
    # 预测未来高价值客户
    future_high_value = predict_future_high_value(data, model, feature_columns, label_encoders)
    
    print("\n=== 结果总结 ===")
    print(f"1. 模型准确率: {accuracy:.4f}")
    print(f"2. 重要特征数量: {len(feature_importance)}")
    print(f"3. 预测未来会成为高价值客户的用户数: {len(future_high_value)}")
    print("\n结果文件已保存:")
    print("- decision_tree_rules.txt: 决策树规则文本")
    print("- decision_tree_visualization.png: 决策树可视化图表")
    print("- decision_tree_feature_importance.csv: 特征重要性")
    print("- predicted_future_high_value_customers_dt.csv: 预测的未来高价值客户列表")

if __name__ == "__main__":
    main()