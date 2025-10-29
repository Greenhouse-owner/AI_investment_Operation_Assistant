import pandas as pd

def read_user_base_sample():
    """
    读取user_base.csv的前5行数据并清晰显示
    """
    print("=" * 60)
    print("读取 user_base.csv 文件的前5行数据")
    print("=" * 60)
    
    # 读取用户基础信息文件的前5行
    df = pd.read_csv('user_base.csv', nrows=5)
    
    # 显示每行数据
    for index in range(len(df)):
        row = df.iloc[index]
        print(f"\n--- 第 {index+1} 行数据 ---")
        for col_name, value in row.items():
            print(f"{col_name:>20}: {value}")
    
    print("\n")
    

def read_user_behavior_sample():
    """
    读取user_behavior_assets.csv的前5行数据并清晰显示
    """
    print("=" * 60)
    print("读取 user_behavior_assets.csv 文件的前5行数据")
    print("=" * 60)
    
    # 读取用户行为资产文件的前5行
    df = pd.read_csv('user_behavior_assets.csv', nrows=5)
    
    # 显示每行数据
    for index in range(len(df)):
        row = df.iloc[index]
        print(f"\n--- 第 {index+1} 行数据 ---")
        for col_name, value in row.items():
            # 对于时间戳字段，保持完整显示
            if 'time' in str(col_name).lower() and isinstance(value, str) and len(value) > 16:
                print(f"{col_name:>25}: {value}")
            else:
                print(f"{col_name:>25}: {value}")
    
    print("\n")


def read_csv_sample():
    """
    主函数，读取并显示两个CSV文件的样本数据
    """
    read_user_base_sample()
    read_user_behavior_sample()


if __name__ == "__main__":
    read_csv_sample()