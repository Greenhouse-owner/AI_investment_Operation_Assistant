import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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

def feature_engineering_01(data):
    """
    特征工程 - 使用01编码（One-Hot编码）
    """
    print("正在进行特征工程（01编码）...")
    
    # 创建目标变量：当前是否为高价值客户 (100万+)
    data['is_high_value'] = (data['asset_level'] == '100万+').astype(int)
    
    # 特征选择和处理
    # 数值特征
    numerical_features = [
        'age', 'monthly_trade_amount', 'stock_portfolio_value', 
        'cash_balance', 'fixed_income_products_value', 'equity_products_value',
        'alternative_products_value', 'product_count', 'stock_trade_freq',
        'credit_card_monthly_expense', 'investment_monthly_count', 
        'stock_app_open_count', 'app_financial_view_time', 'app_product_compare_count'
    ]
    
    # 分类特征
    categorical_features = [
        'occupation_type', 'city_level', 'lifecycle_stage', 'marriage_status'
    ]
    
    # 处理缺失值
    for col in numerical_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # 准备特征和目标变量
    feature_columns = numerical_features + categorical_features
    selected_data = data[feature_columns]
    y = data['is_high_value']
    
    # 对分类变量进行01编码（One-Hot编码）
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    X = preprocessor.fit_transform(selected_data)
    
    # 获取编码后的特征名称
    numerical_feature_names = numerical_features
    categorical_feature_names = []
    
    # 修复 NoneType 错误：检查 named_transformers_['cat'] 是否为 None
    cat_transformer = preprocessor.named_transformers_['cat']
    if cat_transformer is not None:
        for i, cat_col in enumerate(categorical_features):
            # 获取每个分类特征的唯一值（排除第一个，因为我们使用drop='first'）
            if hasattr(cat_transformer, 'categories_') and cat_transformer.categories_ is not None:
                # 确保 categories_[i] 不为 None 再进行切片操作
                if cat_transformer.categories_[i] is not None:
                    categories = cat_transformer.categories_[i][1:]
                    for category in categories:
                        categorical_feature_names.append(f"{cat_col}_{category}")
    
    feature_names = numerical_feature_names + categorical_feature_names
    
    print(f"特征工程完成，特征数量: {X.shape[1]}")
    return X, y, feature_names

def train_logistic_model(X, y):
    """
    训练逻辑回归模型
    """
    print("正在训练逻辑回归模型...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("模型训练完成")
    return model, scaler, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def visualize_coefficients(model, feature_names):
    """
    可视化逻辑回归系数
    """
    print("正在生成系数可视化图表...")
    
    # 获取系数
    coefficients = model.coef_[0]
    
    # 创建特征重要性DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # 按系数绝对值排序
    coef_df['abs_coefficient'] = np.abs(coefficients)
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
    
    # 只显示前20个最重要的特征
    top_features = coef_df.head(20)
    
    # 创建可视化图表
    plt.figure(figsize=(12, 10))
    colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
    bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
    plt.yticks(range(len(top_features)), list(top_features['feature']))
    plt.xlabel('逻辑回归系数')
    plt.title('逻辑回归模型特征系数 (前20个最重要特征) - 01编码')
    plt.grid(axis='x', alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Rectangle
    red_patch = Rectangle((0,0),1,1, color='red')
    blue_patch = Rectangle((0,0),1,1, color='blue')
    plt.legend([red_patch, blue_patch], ['负相关', '正相关'], loc='lower right')
    
    # 在每个条形图上显示数值
    for i, (coef, feature) in enumerate(zip(top_features['coefficient'], top_features['feature'].tolist())):
        plt.text(coef, i, f'{coef:.3f}', va='center', ha='left' if coef >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients_01.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存系数到CSV
    coef_df.to_csv('logistic_regression_coefficients_01.csv', index=False, encoding='utf-8-sig')
    
    return coef_df

def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    评估模型性能
    """
    print("正在评估模型性能...")
    
    # 打印分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非高价值客户', '高价值客户']))
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 计算AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc:.4f}")
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title('ROC曲线 - 01编码')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve_01.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return auc

def predict_future_high_value(data, model, scaler, feature_columns):
    """
    预测未来可能成为高价值客户的用户
    """
    print("正在预测未来可能成为高价值客户的用户...")
    
    # 准备特征数据
    X = data[feature_columns]
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测概率
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # 添加预测结果到数据中
    data['predicted_probability'] = probabilities
    
    # 选择当前还不是高价值客户但预测概率高的前100名
    non_high_value = data[data['asset_level'] != '100万+']
    top_candidates = non_high_value.nlargest(100, 'predicted_probability')
    
    # 显示结果
    print("\n未来3个月最有可能成为高价值客户的前20名用户:")
    print(top_candidates[['user_id', 'name', 'asset_level', 'predicted_probability']].head(20))
    
    # 保存结果
    top_candidates.to_csv('predicted_high_value_customers_01.csv', index=False, encoding='utf-8-sig')
    
    return top_candidates

def main():
    """
    主函数
    """
    print("=== 客户高价值潜力预测系统（01编码版）===")
    
    # 加载数据
    data = load_and_preprocess_data()
    
    # 特征工程
    X, y, feature_names = feature_engineering_01(data)
    
    # 训练模型
    model, scaler, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_logistic_model(X, y)
    
    # 输出模型准确率
    accuracy = model.score(scaler.transform(X_test), y_test)
    print(f"模型准确率: {accuracy:.4f}")
    
    # 输出逻辑回归系数
    print("\n逻辑回归系数:")
    coef_df = visualize_coefficients(model, feature_names)
    print(coef_df.head(10))
    
    # 评估模型
    auc = evaluate_model(y_test, y_pred, y_pred_proba)
    
    print("\n=== 结果总结 ===")
    print(f"1. 模型准确率: {accuracy:.4f}")
    print(f"2. 模型AUC: {auc:.4f}")
    print(f"3. 重要特征数量: {len(coef_df)}")
    print("\n结果文件已保存:")
    print("- logistic_regression_coefficients_01.csv: 逻辑回归系数")
    print("- logistic_regression_coefficients_01.png: 系数可视化图表")
    print("- roc_curve_01.png: ROC曲线图")

if __name__ == "__main__":
    main()