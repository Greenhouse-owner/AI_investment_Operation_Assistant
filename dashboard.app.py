from flask import Flask, render_template, jsonify
import csv
import os

# 设置模板目录的绝对路径
template_dir = os.path.abspath('templates')
app = Flask(__name__, template_folder=template_dir)

# 模拟数据文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_BASE_FILE = os.path.join(BASE_DIR, 'user_base.csv')
USER_BEHAVIOR_ASSETS_FILE = os.path.join(BASE_DIR, 'user_behavior_assets.csv')

def read_csv_data(file_path):
    """读取CSV数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def process_user_profile_data():
    """处理用户画像数据"""
    user_data = read_csv_data(USER_BASE_FILE)
    
    # 统计各维度数据（简化处理）
    age_groups = {'青年': 0, '中年': 0, '老年': 0}
    occupations = {}
    city_levels = {}
    marriage_stats = {}
    lifecycle_stages = {}
    
    for user in user_data:
        # 年龄分组
        age = int(user['age'])
        if age < 35:
            age_groups['青年'] += 1
        elif age < 55:
            age_groups['中年'] += 1
        else:
            age_groups['老年'] += 1
            
        # 职业统计
        occ = user['occupation_type']
        occupations[occ] = occupations.get(occ, 0) + 1
        
        # 城市等级统计
        city = user['city_level']
        city_levels[city] = city_levels.get(city, 0) + 1
        
        # 婚姻状态统计
        marriage = user['marriage_status']
        marriage_stats[marriage] = marriage_stats.get(marriage, 0) + 1
        
        # 生命周期阶段统计
        lifecycle = user['lifecycle_stage']
        lifecycle_stages[lifecycle] = lifecycle_stages.get(lifecycle, 0) + 1
    
    # 构造返回数据
    result = {
        'indicator': [
            {'name': '年龄分布', 'max': len(user_data)},
            {'name': '职业类型', 'max': len(user_data)},
            {'name': '城市等级', 'max': len(user_data)},
            {'name': '婚姻状况', 'max': len(user_data)},
            {'name': '生命周期', 'max': len(user_data)}
        ],
        'data': [
            {
                'name': '用户画像',
                'value': [
                    max(age_groups.values()),
                    max(occupations.values()) if occupations else 0,
                    max(city_levels.values()) if city_levels else 0,
                    max(marriage_stats.values()) if marriage_stats else 0,
                    max(lifecycle_stages.values()) if lifecycle_stages else 0
                ]
            }
        ]
    }
    return result

def process_asset_behavior_data():
    """处理资产行为数据"""
    asset_data = read_csv_data(USER_BEHAVIOR_ASSETS_FILE)
    
    # 按资产层级分组统计
    asset_levels = {}
    for record in asset_data:
        level = record['asset_level']
        if level not in asset_levels:
            asset_levels[level] = {
                'cash': 0,
                'fixed_income': 0,
                'equity': 0,
                'alternative': 0,
                'count': 0
            }
        
        asset_levels[level]['cash'] += float(record['cash_balance']) if record['cash_balance'] else 0
        asset_levels[level]['fixed_income'] += float(record['fixed_income_products_value']) if record['fixed_income_products_value'] else 0
        asset_levels[level]['equity'] += float(record['equity_products_value']) if record['equity_products_value'] else 0
        asset_levels[level]['alternative'] += float(record['alternative_products_value']) if record['alternative_products_value'] else 0
        asset_levels[level]['count'] += 1
    
    # 构造返回数据
    sorted_levels = sorted(asset_levels.keys())
    result = {
        'legend': ['现金余额', '固收类产品', '权益类产品', '另类产品'],
        'xAxis': sorted_levels,
        'series': [
            {
                'name': '现金余额',
                'type': 'bar',
                'stack': '总量',
                'data': [round(asset_levels[level]['cash'] / asset_levels[level]['count'], 2) for level in sorted_levels]
            },
            {
                'name': '固收类产品',
                'type': 'bar',
                'stack': '总量',
                'data': [round(asset_levels[level]['fixed_income'] / asset_levels[level]['count'], 2) for level in sorted_levels]
            },
            {
                'name': '权益类产品',
                'type': 'bar',
                'stack': '总量',
                'data': [round(asset_levels[level]['equity'] / asset_levels[level]['count'], 2) for level in sorted_levels]
            },
            {
                'name': '另类产品',
                'type': 'bar',
                'stack': '总量',
                'data': [round(asset_levels[level]['alternative'] / asset_levels[level]['count'], 2) for level in sorted_levels]
            }
        ]
    }
    return result

def process_marketing_effect_data():
    """处理营销效果数据"""
    asset_data = read_csv_data(USER_BEHAVIOR_ASSETS_FILE)
    
    # 按统计月份分组
    monthly_data = {}
    for record in asset_data:
        month = record['stat_month']
        if month not in monthly_data:
            monthly_data[month] = {
                'contact_success': 0,
                'total_contacts': 0,
                'converted': 0,
                'total': 0
            }
        
        # 统计联系成功率
        if record['contact_result'] == '成功':
            monthly_data[month]['contact_success'] += 1
        if record['contact_result'] in ['成功', '失败', '未接通', '拒绝']:
            monthly_data[month]['total_contacts'] += 1
            
        # 统计转化相关（这里简化处理）
        monthly_data[month]['converted'] += int(record['stock_hold_flag']) if record['stock_hold_flag'] else 0
        monthly_data[month]['total'] += 1
    
    # 排序月份
    sorted_months = sorted(monthly_data.keys())
    
    # 计算成功率和转化率
    success_rates = []
    conversion_rates = []
    
    for month in sorted_months:
        data = monthly_data[month]
        success_rate = (data['contact_success'] / data['total_contacts'] * 100) if data['total_contacts'] > 0 else 0
        conversion_rate = (data['converted'] / data['total'] * 100) if data['total'] > 0 else 0
        success_rates.append(round(success_rate, 2))
        conversion_rates.append(round(conversion_rate, 2))
    
    result = {
        'legend': ['联系成功率', '转化率'],
        'xAxis': sorted_months,
        'series': [
            {
                'name': '联系成功率',
                'type': 'line',
                'data': success_rates
            },
            {
                'name': '转化率',
                'type': 'line',
                'data': conversion_rates
            }
        ]
    }
    return result

def process_comprehensive_metrics():
    """处理综合指标数据"""
    user_data = read_csv_data(USER_BASE_FILE)
    asset_data = read_csv_data(USER_BEHAVIOR_ASSETS_FILE)
    
    # 总用户数
    total_users = len(user_data)
    
    # 高价值用户数 (资产100万+)
    high_value_users = sum(1 for record in asset_data if record['asset_level'] == '100万+')
    
    # 平均用户资产 (简化计算)
    total_assets = sum(
        (float(record['cash_balance'] or 0) +
         float(record['fixed_income_products_value'] or 0) +
         float(record['equity_products_value'] or 0) +
         float(record['alternative_products_value'] or 0))
        for record in asset_data
    )
    avg_assets = total_assets / len(asset_data) if asset_data else 0
    
    # 营销活动转化率 (简化)
    converted_users = sum(1 for record in asset_data if record['contact_result'] == '成功')
    conversion_rate = (converted_users / len(asset_data) * 100) if asset_data else 0
    
    # 用户活跃度 (基于APP打开次数)
    active_users = sum(1 for record in asset_data if int(record['stock_app_open_count'] or 0) > 10)
    user_activity = (active_users / len(asset_data) * 100) if asset_data else 0
    
    result = {
        'total_users': total_users,
        'high_value_users': high_value_users,
        'avg_assets': round(avg_assets, 2),
        'conversion_rate': round(conversion_rate, 2),
        'user_activity': round(user_activity, 2)
    }
    return result

def process_core_overview():
    """处理核心指标概览数据"""
    user_data = read_csv_data(USER_BASE_FILE)
    asset_data = read_csv_data(USER_BEHAVIOR_ASSETS_FILE)
    
    # 当前总用户数
    total_users = len(user_data)
    
    # 总资产管理规模
    total_assets = sum(
        (float(record['cash_balance'] or 0) +
         float(record['fixed_income_products_value'] or 0) +
         float(record['equity_products_value'] or 0) +
         float(record['alternative_products_value'] or 0))
        for record in asset_data
    )
    
    # 本月新增用户数 (简化处理)
    new_users = sum(1 for user in user_data if user['lifecycle_stage'] == '新客户')
    
    # 用户平均生命周期价值 (简化处理)
    avg_ltv = total_assets / len(user_data) if user_data else 0
    
    result = {
        'total_users': total_users,
        'total_assets': round(total_assets, 2),
        'new_users': new_users,
        'avg_ltv': round(avg_ltv, 2)
    }
    return result

@app.route('/')
def dashboard():
    """主页面"""
    return render_template('dashboard.html')

@app.route('/api/user_profile')
def user_profile_data():
    """用户画像分析数据接口"""
    try:
        result = process_user_profile_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/asset_behavior')
def asset_behavior_data():
    """资产行为分析数据接口"""
    try:
        result = process_asset_behavior_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/marketing_effect')
def marketing_effect_data():
    """营销效果分析数据接口"""
    try:
        result = process_marketing_effect_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comprehensive_metrics')
def comprehensive_metrics_data():
    """综合指标数据接口"""
    try:
        result = process_comprehensive_metrics()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/core_overview')
def core_overview_data():
    """核心指标概览数据接口"""
    try:
        result = process_core_overview()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)