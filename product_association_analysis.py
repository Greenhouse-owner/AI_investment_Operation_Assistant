import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
user_behavior_assets = pd.read_csv('user_behavior_assets.csv')

# 查看数据结构
print("数据集基本信息:")
print(user_behavior_assets.head())
print("\n数据集形状:", user_behavior_assets.shape)

# 基于产品持有标识创建产品组合数据
# 根据字段说明，以下标识字段表示用户是否持有某类产品：
# fixed_income_flag: 固收产品标识 (0/1)
# equity_flag: 权益产品标识 (0/1)
# alternative_flag: 另类产品标识 (0/1)

product_data = user_behavior_assets[['user_id', 'fixed_income_flag', 'equity_flag', 'alternative_flag']].copy()

# 重命名列以提高可读性
product_data = product_data.rename(columns={
    'fixed_income_flag': '固收类产品',
    'equity_flag': '权益类产品', 
    'alternative_flag': '另类产品'
})

print("\n产品数据概览:")
print(product_data.head())

# 统计各类产品的持有情况
print("\n各类产品持有统计:")
product_stats = product_data[['固收类产品', '权益类产品', '另类产品']].apply(lambda x: x.value_counts())
print(product_stats)

# 分析产品组合分布
print("\n产品组合分布:")
product_combinations = product_data[['固收类产品', '权益类产品', '另类产品']].value_counts()
print(product_combinations)

# 计算各项产品的实际持有率
print("\n各项产品实际持有率:")
total_customers = len(product_data)
fixed_income_rate = product_data['固收类产品'].sum() / total_customers
equity_rate = product_data['权益类产品'].sum() / total_customers
alternative_rate = product_data['另类产品'].sum() / total_customers

print(f"固收类产品持有率: {fixed_income_rate:.2%}")
print(f"权益类产品持有率: {equity_rate:.2%}")
print(f"另类产品持有率: {alternative_rate:.2%}")

# 计算理论上独立分布时的联合概率
print("\n理论上独立分布时的联合概率:")
print(f"P(固收类产品 ∩ 权益类产品) = {fixed_income_rate * equity_rate:.2%}")
print(f"P(固收类产品 ∩ 另类产品) = {fixed_income_rate * alternative_rate:.2%}")
print(f"P(权益类产品 ∩ 另类产品) = {equity_rate * alternative_rate:.2%}")
print(f"P(固收类产品 ∩ 权益类产品 ∩ 另类产品) = {fixed_income_rate * equity_rate * alternative_rate:.2%}")

# 计算实际观察到的联合概率
print("\n实际观察到的联合概率:")
actual_fixed_equity = len(product_data[(product_data['固收类产品'] == 1) & (product_data['权益类产品'] == 1)]) / total_customers
actual_fixed_alternative = len(product_data[(product_data['固收类产品'] == 1) & (product_data['另类产品'] == 1)]) / total_customers
actual_equity_alternative = len(product_data[(product_data['权益类产品'] == 1) & (product_data['另类产品'] == 1)]) / total_customers
actual_all_three = len(product_data[(product_data['固收类产品'] == 1) & (product_data['权益类产品'] == 1) & (product_data['另类产品'] == 1)]) / total_customers

print(f"实际P(固收类产品 ∩ 权益类产品) = {actual_fixed_equity:.2%}")
print(f"实际P(固收类产品 ∩ 另类产品) = {actual_fixed_alternative:.2%}")
print(f"实际P(权益类产品 ∩ 另类产品) = {actual_equity_alternative:.2%}")
print(f"实际P(固收类产品 ∩ 权益类产品 ∩ 另类产品) = {actual_all_three:.2%}")

# 分析lift值接近1的原因
print("\n=== Lift值分析 ===")
print("Lift = P(A∩B) / (P(A) × P(B))")
print(f"Lift(固收类产品, 权益类产品) = {actual_fixed_equity / (fixed_income_rate * equity_rate):.4f}")
print(f"Lift(固收类产品, 另类产品) = {actual_fixed_alternative / (fixed_income_rate * alternative_rate):.4f}")
print(f"Lift(权益类产品, 另类产品) = {actual_equity_alternative / (equity_rate * alternative_rate):.4f}")

# 构建事务数据
# 将每行转换为产品列表形式
def create_product_list(row):
    products = []
    if row['固收类产品'] == 1:
        products.append('固收类产品')
    if row['权益类产品'] == 1:
        products.append('权益类产品')
    if row['另类产品'] == 1:
        products.append('另类产品')
    return products

transactions = product_data.apply(create_product_list, axis=1).tolist()

# 使用TransactionEncoder转换数据格式
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("\n编码后的事务数据:")
print(df_encoded.head())
print("数据形状:", df_encoded.shape)

# 使用Apriori算法挖掘频繁项集
# 设置最小支持度阈值
min_support = 0.1  # 10%的支持度
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

print(f"\n频繁项集 (最小支持度={min_support}):")
print(frequent_itemsets.sort_values(by='support', ascending=False))

# 生成关联规则
# 设置最小置信度阈值
min_threshold = 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)

print(f"\n关联规则 (最小置信度={min_threshold}):")
if not rules.empty:
    # 计算lift值并显示
    rules['lift'] = rules['confidence'] / rules['consequents'].apply(
        lambda x: frequent_itemsets[frequent_itemsets['itemsets'] == x]['support'].iloc[0]
    )
    
    # 选择需要的列并排序
    rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    rules_display = rules_display.sort_values(by='lift', ascending=False)
    print(rules_display)
    
    # 分析lift值
    print(f"\nLift值分析:")
    print(f"平均lift值: {rules['lift'].mean():.4f}")
    print(f"最大lift值: {rules['lift'].max():.4f}")
    print(f"最小lift值: {rules['lift'].min():.4f}")
    
    if rules['lift'].mean() > 1.1:
        print("数据中存在较强的关联规则")
    elif rules['lift'].mean() > 1.0:
        print("数据中存在一定的关联规则")
    else:
        print("数据中关联规则较弱，产品持有行为接近独立分布")
else:
    print("未找到满足条件的关联规则")

# 可视化频繁项集
if not frequent_itemsets.empty:
    plt.figure(figsize=(10, 6))
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by='support', ascending=False).head(10)
    itemset_names = [', '.join(list(itemset)) for itemset in frequent_itemsets_sorted['itemsets']]
    plt.barh(range(len(frequent_itemsets_sorted)), frequent_itemsets_sorted['support'])
    plt.yticks(range(len(frequent_itemsets_sorted)), itemset_names)
    plt.xlabel('Support')
    plt.title('Top 10 Frequent Itemsets')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 可视化关联规则
if not rules.empty:
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=100)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules - Support vs Confidence')
    
    # 添加标签
    for i in range(len(rules)):
        plt.annotate(f'R{i+1}', (rules.iloc[i]['support'], rules.iloc[i]['confidence']))
    
    plt.tight_layout()
    plt.show()

# 详细分析产品组合
print("\n=== 产品组合详细分析 ===")

# 计算产品组合的实际分布与理论分布差异
print("\n产品组合实际分布与理论分布对比:")
print("组合\t\t\t实际概率\t理论概率\t差异")
print(f"固收+权益\t\t{actual_fixed_equity:.2%}\t\t{fixed_income_rate * equity_rate:.2%}\t\t{actual_fixed_equity - fixed_income_rate * equity_rate:.2%}")
print(f"固收+另类\t\t{actual_fixed_alternative:.2%}\t\t{fixed_income_rate * alternative_rate:.2%}\t\t{actual_fixed_alternative - fixed_income_rate * alternative_rate:.2%}")
print(f"权益+另类\t\t{actual_equity_alternative:.2%}\t\t{equity_rate * alternative_rate:.2%}\t\t{actual_equity_alternative - equity_rate * alternative_rate:.2%}")
print(f"三者组合\t\t{actual_all_three:.2%}\t\t{fixed_income_rate * equity_rate * alternative_rate:.2%}\t\t{actual_all_three - fixed_income_rate * equity_rate * alternative_rate:.2%}")

# 分析结论
print("\n=== 分析结论 ===")
if abs(actual_fixed_equity - fixed_income_rate * equity_rate) < 0.01 and \
   abs(actual_fixed_alternative - fixed_income_rate * alternative_rate) < 0.01 and \
   abs(actual_equity_alternative - equity_rate * alternative_rate) < 0.01:
    print("1. 产品持有行为接近独立分布")
    print("   - 客户是否持有某种产品与其他产品的持有情况关系不大")
    print("   - 这意味着产品之间没有明显的关联性")
else:
    print("1. 产品持有行为存在一定的依赖关系")
    print("   - 客户持有某种产品的概率会受到其他产品持有情况的影响")

if rules.empty:
    print("\n2. 未发现强关联规则")
    print("   - 当前参数设置下(min_support=0.1, min_threshold=0.5)未发现有意义的关联规则")
else:
    if rules['lift'].mean() > 1.1:
        print("\n2. 存在较强的关联规则")
        print("   - 可以基于这些规则进行产品推荐")
    elif rules['lift'].mean() > 1.0:
        print("\n2. 存在一定的关联规则")
        print("   - 关联性较弱，但仍然可以作为参考")
    else:
        print("\n2. 关联规则较弱")
        print("   - 产品持有行为接近随机分布")

print("\n3. 营销建议:")
print("   - 由于产品间关联性较弱，建议基于客户画像而非产品关联进行推荐")
print("   - 可以考虑基于客户资产水平、年龄、职业等特征进行产品推荐")
print("   - 个性化营销策略可能比基于产品关联的策略更有效")

print("\n分析完成！")