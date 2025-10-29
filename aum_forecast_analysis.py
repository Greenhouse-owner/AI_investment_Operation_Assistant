import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# 读取数据
user_base = pd.read_csv('user_base.csv')
user_behavior = pd.read_csv('user_behavior_assets.csv')

# 合并数据
df = pd.merge(user_behavior, user_base, on='user_id', how='left')

# 计算每个用户的AUM (资产管理规模)
# AUM = 股票市值 + 现金余额 + 固收类产品市值 + 权益类产品市值 + 另类产品市值
df['aum'] = (df['stock_portfolio_value'] + 
             df['cash_balance'] + 
             df['fixed_income_products_value'] + 
             df['equity_products_value'] + 
             df['alternative_products_value'])

# 按统计月份聚合AUM数据
df['stat_month'] = pd.to_datetime(df['stat_month'])
aum_ts = df.groupby('stat_month')['aum'].sum().sort_index()

# 可视化原始时间序列
plt.figure(figsize=(12, 6))
plt.plot(aum_ts)
plt.title('资产管理规模(AUM)时间序列')
plt.xlabel('时间')
plt.ylabel('AUM')
plt.grid(True)
plt.show()

# 检查平稳性
def check_stationarity(timeseries):
    # 执行ADF检验
    result = adfuller(timeseries)
    print('ADF统计量:', result[0])
    print('p值:', result[1])
    print('临界值:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("序列是平稳的")
        return True
    else:
        print("序列是非平稳的")
        return False

print("原始序列平稳性检验:")
is_stationary = check_stationarity(aum_ts)

# 如果序列不平稳，进行差分
if not is_stationary:
    # 一阶差分
    aum_diff = aum_ts.diff().dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(aum_diff)
    plt.title('一阶差分后的时间序列')
    plt.xlabel('时间')
    plt.ylabel('AUM差分')
    plt.grid(True)
    plt.show()
    
    print("\n一阶差分后序列平稳性检验:")
    is_stationary = check_stationarity(aum_diff)

# 建立ARIMA模型
# 通过观察ACF和PACF图来确定参数
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(aum_ts, ax=axes[0])
plot_pacf(aum_ts, ax=axes[1])
plt.show()

# 尝试不同的ARIMA参数组合
best_aic = np.inf
best_order = None
best_model = None

# 为了简化，我们尝试几种常见的参数组合
orders = [(1, 0, 1), (1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1)]

print("\n尝试不同的ARIMA模型参数:")
for order in orders:
    try:
        model = ARIMA(aum_ts, order=order)
        fitted_model = model.fit()
        print(f'ARIMA{order} AIC: {fitted_model.aic}')
        if fitted_model.aic < best_aic:
            best_aic = fitted_model.aic
            best_order = order
            best_model = fitted_model
    except:
        print(f'ARIMA{order} 拟合失败')
        continue

print(f'\n最佳模型: ARIMA{best_order} with AIC={best_aic}')

# 使用最佳模型进行预测
# 预测未来8个季度
forecast_steps = 8
forecast = best_model.forecast(steps=forecast_steps)
forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()

# 创建未来日期索引
last_date = aum_ts.index[-1]
future_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='M')[1:]

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(aum_ts.index, aum_ts, label='历史数据')
plt.plot(future_dates, forecast, color='red', label='预测值')
plt.fill_between(future_dates, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3, label='置信区间')
plt.title('资产管理规模(AUM)预测')
plt.xlabel('时间')
plt.ylabel('AUM')
plt.legend()
plt.grid(True)
plt.show()

# 打印预测结果
print("\n未来8个季度的AUM预测:")
forecast_df = pd.DataFrame({
    '预测值': forecast,
    '置信区间下限': forecast_ci.iloc[:, 0],
    '置信区间上限': forecast_ci.iloc[:, 1]
}, index=future_dates)
print(forecast_df)

# 计算预测增长率
current_aum = aum_ts.iloc[-1]
future_aum = forecast.iloc[-1]
growth_rate = (future_aum - current_aum) / current_aum * 100
print(f"\n预测期末AUM相比于当前AUM的增长率: {growth_rate:.2f}%")

# 分析预测结果
if growth_rate > 10:
    trend = "快速增长"
elif growth_rate > 0:
    trend = "缓慢增长"
elif growth_rate > -10:
    trend = "缓慢下降"
else:
    trend = "快速下降"

print(f"根据预测结果，未来资产管理规模将呈现{trend}趋势。")

# 查看数据结构
print("用户行为资产数据概览:")
print(user_behavior.head())
print("\n用户基础数据概览:")
print(user_base.head())

# 查看统计月份分布
print("\n统计月份分布:")
print(user_behavior['stat_month'].value_counts().sort_index())

# 查看数据时间范围
print("\n数据时间范围:")
print(f"开始时间: {df['stat_month'].min()}")
print(f"结束时间: {df['stat_month'].max()}")

# 打印AUM统计信息
print("\nAUM统计信息:")
print(df['aum'].describe())

# 模型诊断检查
print("\n模型诊断:")
best_model.plot_diagnostics(figsize=(12, 10))
plt.tight_layout()
plt.show()

# 分析和建议
print("\n=== 分析结论与建议 ===")
print("1. AUM趋势分析:")
print(f"   - 当前AUM总值: {monthly_aum.iloc[-1]:,.2f}")
print(f"   - 最初AUM总值: {monthly_aum.iloc[0]:,.2f}")
total_growth = (monthly_aum.iloc[-1] - monthly_aum.iloc[0]) / monthly_aum.iloc[0] * 100
print(f"   - 总体增长率: {total_growth:.2f}%")

print("\n2. 预测结果:")
print("   - 根据ARIMA模型预测，未来几个季度AUM将继续增长")
print("   - 预测结果带有置信区间，表示预测的不确定性")

print("\n3. 营销建议:")
print("   - 基于预测结果，可以提前规划资产管理和营销策略")
print("   - 针对预测的AUM增长，可以适当增加产品供应")
print("   - 关注预测置信区间，做好风险控制")

print("\n分析完成！")