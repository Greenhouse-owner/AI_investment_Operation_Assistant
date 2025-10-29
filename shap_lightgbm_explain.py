import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入SHAP
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # 定义shap为None以避免未绑定变量错误
    print("警告: 未安装SHAP库，将跳过SHAP解释部分")
    print("可以通过以下命令安装: pip install shap")

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
    
    print("模型训练完成")
    return model, X_train, X_test, y_train, y_test

def shap_explain_global(model, X_train, feature_names):
    """
    SHAP全局解释：识别哪些特征整体上对"高价值客户"预测最重要
    """
    if not SHAP_AVAILABLE:
        print("SHAP库不可用，跳过全局解释")
        return None, None, None
        
    print("正在进行SHAP全局解释...")
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值（采样一部分数据以提高计算效率）
        sample_data = X_train.sample(n=min(1000, len(X_train)), random_state=42)
        shap_values = explainer.shap_values(sample_data)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, sample_data, feature_names=feature_names, show=False)
        plt.title('SHAP特征重要性全局解释', fontsize=16)
        plt.tight_layout()
        plt.savefig('shap_global_explanation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出特征重要性排序
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        print("\nSHAP特征重要性排序 (前10):")
        print(feature_importance_df.head(10))
        
        # 保存特征重要性到CSV
        feature_importance_df.to_csv('shap_feature_importance.csv', index=False, encoding='utf-8-sig')
        
        return shap_values, explainer, sample_data
    except Exception as e:
        print(f"SHAP全局解释过程中出现错误: {e}")
        return None, None, None

def shap_explain_local(model, X_train, feature_names, label_encoders, data):
    """
    SHAP局部解释：针对单个客户，解释其被预测为"未来3个月资产能否提升至100万+"的原因
    """
    if not SHAP_AVAILABLE:
        print("SHAP库不可用，跳过局部解释")
        return
        
    print("正在进行SHAP局部解释...")
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 选择几个样本客户进行局部解释
        # 选择当前还不是高价值客户但预测概率高的前几个客户
        non_high_value = data[data['asset_level'] != '100万+']
        
        # 为了演示，我们选择几个具有代表性的客户
        sample_clients = non_high_value.head(3)  # 选择前3个客户作为示例
        
        # 准备这些客户的特征数据
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
        
        for idx, client in sample_clients.iterrows():
            print(f"\n正在分析客户: {client['user_id']} ({client['name']})")
            print(f"当前资产水平: {client['asset_level']}")
            
            # 准备客户特征数据
            client_features = []
            feature_values = {}
            
            # 添加数值特征
            for col in numerical_features:
                value = client[col]
                client_features.append(value)
                feature_values[col] = value
                
            # 添加分类特征（需要进行标签编码）
            for col in categorical_features:
                raw_value = str(client[col])
                feature_values[col] = raw_value
                # 使用之前训练时的标签编码器进行编码
                if col in label_encoders:
                    le = label_encoders[col]
                    if raw_value in le.classes_:
                        encoded_value = le.transform([raw_value])[0]
                    else:
                        encoded_value = -1  # 未知类别
                    client_features.append(encoded_value)
                else:
                    client_features.append(0)
            
            # 转换为模型所需的格式
            client_X = np.array(client_features).reshape(1, -1)
            client_X_df = pd.DataFrame(client_X, columns=feature_names)
            
            # 使用模型预测
            pred_proba = model.predict(client_X_df)[0]
            print(f"预测成为高价值客户的概率: {pred_proba:.4f}")
            
            # 计算SHAP值进行局部解释
            shap_values = explainer.shap_values(client_X_df)
            
            # 创建局部解释DataFrame
            local_explanation = pd.DataFrame({
                'feature': feature_names,
                'feature_value': client_features,
                'shap_value': shap_values[0] if isinstance(shap_values, list) else shap_values
            }).sort_values('shap_value', key=abs, ascending=False)
            
            print("\n各特征对预测结果的影响:")
            print(local_explanation.head(10))
            
            # 保存局部解释结果
            local_explanation.to_csv(f"shap_local_explanation_{client['user_id']}.csv", index=False, encoding='utf-8-sig')
            
            # 绘制局部解释图
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(values=shap_values[0] if isinstance(shap_values, list) else shap_values,
                               base_values=explainer.expected_value,
                               data=client_features,
                               feature_names=feature_names),
                show=False,
                max_display=10
            )
            plt.title(f'客户 {client["user_id"]} SHAP局部解释\n(预测概率: {pred_proba:.4f})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'shap_local_explanation_{client["user_id"]}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 打印业务解读
            print("\n业务解读:")
            positive_features = local_explanation[local_explanation['shap_value'] > 0].head(3)
            negative_features = local_explanation[local_explanation['shap_value'] < 0].head(3)
            
            if not positive_features.empty:
                print("推动客户成为高价值客户的特征:")
                for _, row in positive_features.iterrows():
                    feature_name = row['feature']
                    feature_value = row['feature_value']
                    shap_value = row['shap_value']
                    print(f"  - {feature_name}: {feature_value} (影响度: +{shap_value:.4f})")
            
            if not negative_features.empty:
                print("阻碍客户成为高价值客户的特征:")
                for _, row in negative_features.iterrows():
                    feature_name = row['feature']
                    feature_value = row['feature_value']
                    shap_value = row['shap_value']
                    print(f"  - {feature_name}: {feature_value} (影响度: {shap_value:.4f})")
    except Exception as e:
        print(f"SHAP局部解释过程中出现错误: {e}")

def main():
    """
    主函数
    """
    print("=== 基于SHAP的LightGBM模型解释 ===")
    
    # 加载数据
    data = load_and_preprocess_data()
    
    # 特征工程
    X, y, feature_columns, label_encoders = feature_engineering(data)
    
    # 训练模型
    model, X_train, X_test, y_train, y_test = train_lightgbm_model(X, y)
    
    # SHAP全局解释
    shap_values, explainer, sample_data = shap_explain_global(model, X_train, feature_columns)
    
    # SHAP局部解释
    shap_explain_local(model, X_train, feature_columns, label_encoders, data)
    
    print("\n=== 解释结果总结 ===")
    if SHAP_AVAILABLE:
        print("1. 全局解释结果已保存:")
        print("   - shap_global_explanation.png: SHAP特征重要性全局解释图")
        print("   - shap_feature_importance.csv: SHAP特征重要性排序")
        print("2. 局部解释结果已保存:")
        print("   - shap_local_explanation_*.csv: 各客户SHAP局部解释数据")
        print("   - shap_local_explanation_*.png: 各客户SHAP局部解释图")
    else:
        print("由于SHAP库不可用，跳过了SHAP解释部分")
        print("建议安装SHAP库以获得更深入的模型解释: pip install shap")

if __name__ == "__main__":
    main()