import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine, text
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
# dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 银行客户数据助手 system prompt 和函数描述 ======
system_prompt = """我是银行客户数据助手，以下是关于银行客户数据表相关的字段，我可能会编写对应的SQL，对数据进行查询
-- 银行客户数据表 customer_data
CREATE TABLE `customer_data`  (
  `customer_id` varchar(10) NOT NULL COMMENT '客户编号',
  `gender` char(1) NULL DEFAULT NULL COMMENT '性别: M-男, F-女',
  `age` int NULL DEFAULT NULL COMMENT '年龄',
  `occupation` varchar(20) NULL DEFAULT NULL COMMENT '职业',
  `marital_status` varchar(10) NULL DEFAULT NULL COMMENT '婚姻状况: 已婚、未婚、离异',
  `city_level` varchar(10) NULL DEFAULT NULL COMMENT '城市等级: 一线、二线、三线',
  `account_open_date` varchar(10) NULL DEFAULT NULL COMMENT '开户日期',
  `total_aum` decimal(18, 2) NULL DEFAULT NULL COMMENT '总资产管理规模',
  `deposit_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '存款余额',
  `wealth_management_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '理财余额',
  `fund_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '基金余额',
  `insurance_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '保险余额',
  `deposit_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '存款月均余额',
  `wealth_management_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '理财月均余额',
  `fund_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '基金月均余额',
  `insurance_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '保险月均余额',
  `monthly_transaction_count` decimal(10, 2) NULL DEFAULT NULL COMMENT '月均交易次数',
  `monthly_transaction_amount` decimal(18, 2) NULL DEFAULT NULL COMMENT '月均交易金额',
  `last_transaction_date` varchar(10) NULL DEFAULT NULL COMMENT '最近交易日期',
  `mobile_bank_login_count` int NULL DEFAULT NULL COMMENT '手机银行登录次数',
  `branch_visit_count` int NULL DEFAULT NULL COMMENT '网点访问次数',
  `last_mobile_login` varchar(10) NULL DEFAULT NULL COMMENT '最近手机银行登录日期',
  `last_branch_visit` varchar(10) NULL DEFAULT NULL COMMENT '最近网点访问日期',
  `customer_tier` varchar(10) NULL DEFAULT NULL COMMENT '客户等级: 普通、潜力、临界、高净值',
  PRIMARY KEY (`customer_id`),
  INDEX `idx_customer_tier`(`customer_tier`),
  INDEX `idx_age`(`age`),
  INDEX `idx_total_aum`(`total_aum`),
  INDEX `idx_occupation`(`occupation`),
  INDEX `idx_city_level`(`city_level`),
  INDEX `idx_account_open_date`(`account_open_date`),
  INDEX `idx_last_transaction_date`(`last_transaction_date`)
) ENGINE = InnoDB COMMENT = '银行客户数据表';

我将回答用户关于银行客户数据相关的问题

每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
"""

functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
# 用于存储每个会话的 DataFrame，避免多用户数据串扰
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    """
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def call(self, params: str | dict, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        if isinstance(params, str):
            args = json.loads(params)
        else:
            args = params
        sql_input = args['sql_input']
        database = args.get('database', 'bank2')
        # 从环境变量获取数据库连接信息
        db_user = os.getenv('DB_USER', 'your_db_username')
        db_password = os.getenv('DB_PASSWORD', 'your_db_password')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '3306')
        
        engine = create_engine(
            f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{database}?charset=utf8mb4',
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        try:
            df = pd.read_sql(text(sql_input), engine)
            md = df.head(10).to_markdown(index=False)
            # 自动创建目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'bar_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            # 生成图表
            generate_chart_png(df, save_path)
            img_path = os.path.join('image_show', filename)
            img_md = f'![柱状图]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

# ========== 通用可视化函数 ========== 
def generate_chart_png(df_sql, save_path):
    columns = df_sql.columns
    x = np.arange(len(df_sql))
    # 获取object类型
    object_columns = df_sql.select_dtypes(include='O').columns.tolist()
    if columns[0] in object_columns:
        object_columns.remove(columns[0])
    num_columns = df_sql.select_dtypes(exclude='O').columns.tolist()
    if len(object_columns) > 0:
        # 对数据进行透视，以便为每个日期和销售渠道创建堆积柱状图
        pivot_df = df_sql.pivot_table(index=columns[0], columns=object_columns, 
                                      values=num_columns, 
                                      fill_value=0)
        # 绘制堆积柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        # 为每个销售渠道和票类型创建柱状图
        bottoms = None
        for col in pivot_df.columns:
            ax.bar(pivot_df.index, pivot_df[col], bottom=bottoms, label=str(col))
            if bottoms is None:
                bottoms = pivot_df[col].copy()
            else:
                bottoms += pivot_df[col]
    else:
        print('进入到else...')
        bottom = np.zeros(len(df_sql))
        for column in columns[1:]:
            plt.bar(x, df_sql[column], bottom=bottom, label=column)
            bottom += df_sql[column]
        plt.xticks(x, df_sql[columns[0]])
    plt.legend()
    plt.title("银行客户数据分析")
    plt.xlabel(columns[0])
    plt.ylabel("数值")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ====== 初始化银行客户数据助手服务 ======
def init_agent_service():
    """初始化银行客户数据助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='银行客户数据助手',
            description='银行客户数据查询与分析',
            system_message=system_prompt,
            function_list=['exc_sql'],  # 移除绘图工具
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型银行客户数据查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '查询不同客户等级的数量分布情况',
                '分析各职业客户的平均总资产和存款余额',
                '统计不同年龄段客户的月均交易次数和金额',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）