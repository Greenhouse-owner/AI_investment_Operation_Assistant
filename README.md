# 百万客群经营项目

## 项目概述

本项目旨在通过数据驱动的方式，提升百万级用户转化率，降低营销成本与用户流失率。项目包含完整的客户数据分析、预测建模、客户分群和精准营销策略制定流程。

核心任务包括：
1. 用户分层分析（资产/年龄/职业分布）
2. 构建预测模型（AUC≥0.85）
3. 客户分群策略（如高复购、中产家庭等群体）
4. 优化线上线下触达方式

## 项目结构

```
├── data/                          # 数据文件
│   ├── user_base.csv              # 用户基础信息
│   └── user_behavior_assets.csv   # 用户行为资产信息
├── models/                        # 模型相关文件
│   ├── predict_high_value_customers.py  # 高价值客户预测模型
│   ├── customer_clustering.py     # 客户聚类分析
│   ├── decision_tree_predict.py   # 决策树预测模型
│   ├── lightgbm_predict.py        # LightGBM预测模型
│   └── lr_predict.py              # 逻辑回归预测模型
├── dashboard/                     # 可视化大屏
│   ├── dashboard.app.py           # 大屏应用
│   └── templates/
│       └── dashboard.html         # 大屏页面模板
├── analysis/                      # 分析报告
│   ├── 决策树解释.md              # 决策树模型解释
│   ├── SHAP解释.md                # SHAP模型解释
│   ├── 逻辑回归系数分析.md         # 逻辑回归系数分析
│   └── 数据表字段含义.md          # 数据字段说明
├── outputs/                       # 输出文件
│   ├── predicted_high_value_customers.csv  # 预测的高价值客户
│   ├── customer_clusters.csv      # 客户聚类结果
│   └── various charts and plots   # 各类图表文件
└── docs/                          # 项目文档
    ├── 项目说明.txt               # 项目总体说明
    └── 可视化大屏设计.md          # 大屏设计说明
```

## 数据说明

### 用户基础信息表 (user_base.csv)

| 字段名 | 含义 | 说明 |
|--------|------|------|
| user_id | 用户ID | 用户唯一标识符 |
| name | 姓名 | 用户姓名 |
| age | 年龄 | 用户年龄 |
| gender | 性别 | 用户性别（男/女） |
| occupation | 职业 | 用户职业类型 |
| occupation_type | 行业类型 | 用户所在行业领域 |
| monthly_trade_amount | 月均交易金额 | 用户每月平均交易金额 |
| open_account_date | 开户日期 | 用户账户开立时间 |
| lifecycle_stage | 生命周期阶段 | 用户所处的客户生命周期阶段 |
| marriage_status | 婚姻状况 | 用户婚姻状态 |
| city_level | 城市等级 | 用户所在城市等级分类 |
| branch_name | 支行名称 | 用户开户所在支行 |

### 用户行为与资产信息表 (user_behavior_assets.csv)

| 字段名 | 含义 | 说明 |
|--------|------|------|
| id | 记录ID | 数据记录唯一标识符 |
| user_id | 用户ID | 关联用户基本信息的用户标识符 |
| stock_portfolio_value | 股票市值 | 用户持有的股票类资产市值 |
| cash_balance | 现金余额 | 用户账户现金余额 |
| fixed_income_products_value | 固收类产品市值 | 用户持有的固定收益类产品市值 |
| equity_products_value | 权益类产品市值 | 用户持有的权益类产品市值 |
| alternative_products_value | 另类产品市值 | 用户持有的另类产品（如私募等）市值 |
| asset_level | 资产等级 | 用户资产水平分层（如50-80万、80-100万等） |
| stock_hold_flag | 股票持仓标识 | 是否持有股票产品（0/1） |
| fixed_income_flag | 固收产品标识 | 是否持有固收类产品（0/1） |
| equity_flag | 权益产品标识 | 是否持有权益类产品（0/1） |
| alternative_flag | 另类产品标识 | 是否持有另类产品（0/1） |
| product_count | 产品数量 | 用户持有的产品总数 |
| stock_trade_freq | 股票交易频率 | 用户股票交易频次 |
| credit_card_monthly_expense | 信用卡月均消费 | 用户信用卡月均消费金额 |
| investment_monthly_count | 投资月交易次数 | 用户每月平均投资交易次数 |
| stock_app_open_count | 股票App打开次数 | 用户股票App月度打开次数 |
| app_financial_view_time | App金融模块浏览时长 | 用户在App金融模块的月度浏览时长 |
| app_product_compare_count | App产品对比次数 | 用户在App中进行产品对比的月度次数 |
| last_app_login_time | 最近App登录时间 | 用户最近一次登录App的时间 |
| last_contact_time | 最近联系时间 | 最近一次与用户联系的时间 |
| contact_result | 联系结果 | 与用户联系后的结果状态 |
| marketing_cool_period | 营销冷却期 | 营销活动后的冷却时间设置 |
| stat_month | 统计月份 | 数据统计的月份 |

## 核心功能模块

### 1. 客户高价值潜力预测
基于逻辑回归、决策树、LightGBM等多种机器学习模型，预测客户未来成为高价值客户（资产100万+）的可能性。

主要文件：
- `predict_high_value_customers.py` - 逻辑回归模型
- `decision_tree_predict.py` - 决策树模型
- `lightgbm_predict.py` - LightGBM模型
- `lr_predict.py` - 逻辑回归模型（另一种实现）

输出文件：
- `predicted_high_value_customers.csv` - 预测结果
- `logistic_regression_coefficients.png` - 逻辑回归系数图
- `roc_curve.png` - ROC曲线图

### 2. 客户聚类分析
使用K-means算法对客户进行聚类，识别不同类型的客户群体。

主要文件：
- `customer_clustering.py` - 客户聚类分析

输出文件：
- `customer_clusters.csv` - 聚类结果
- 聚类可视化图表

### 3. 模型解释分析
提供多种模型解释方法，增强模型的可解释性和业务应用价值。

分析文档：
- `决策树解释.md` - 决策树模型解释
- `SHAP解释.md` - SHAP模型解释
- `逻辑回归系数分析.md` - 逻辑回归系数分析

### 4. 可视化大屏
提供实时的客户数据分析可视化大屏，展示关键业务指标。

主要文件：
- `dashboard.app.py` - 大屏应用后端
- `templates/dashboard.html` - 大屏前端页面

大屏功能：
- 用户画像分析（雷达图）
- 资产行为分析（堆叠柱状图）
- 营销效果分析（折线图）
- 综合指标展示（仪表盘）
- 核心指标概览（数字卡片）

## 使用方法

### 启动可视化大屏
```bash
python dashboard.app.py
```
访问 http://localhost:5000 查看可视化大屏

### 运行高价值客户预测
```bash
python predict_high_value_customers.py
```

### 运行客户聚类分析
```bash
python customer_clustering.py
```

## 项目目标

1. **数据分析**：可视化资产分层、高潜力用户画像，分析行为与资产相关性
2. **智能建模**：预测百万级用户（SHAP解释关键因子），聚类分群并定制策略
3. **精准营销**：动态更新高潜力名单，结合APP推送、电话外呼等渠道，设定转化率监控与预警机制

## 业务价值

通过本项目可以实现：
- 精准识别高潜力客户群体
- 制定个性化客户服务策略
- 优化产品推荐和营销活动
- 提升客户转化率和满意度
- 降低营销成本和客户流失率