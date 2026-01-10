#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成20种表格类型，其中10种是常见的（不需要学习的），10种是复杂的（需要学习的）
"""

import os
import pandas as pd
import random
import string

# 定义表格生成函数
def generate_table(columns, rows, is_common=True):
    """
    生成表格数据
    """
    data = []
    for _ in range(rows):
        row = {}
        for col in columns:
            if is_common:
                # 常见表格：使用简单的数据
                if col == '名称':
                    row[col] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                elif col == '数值1':
                    row[col] = random.randint(1, 100)
                elif col == '数值2':
                    row[col] = random.randint(100, 1000)
                elif col == '数值3':
                    row[col] = random.randint(1000, 10000)
                elif col == '日期':
                    row[col] = pd.Timestamp.now().strftime('%Y-%m-%d')
                else:
                    row[col] = random.choice(['A', 'B', 'C', 'D', 'E'])
            else:
                # 复杂表格：使用更复杂的数据结构
                if col == '复杂名称':
                    row[col] = f"复杂_{''.join(random.choices(string.ascii_uppercase + string.digits, k=12))}"
                elif col == '多层级数值1':
                    row[col] = {'value': random.randint(1, 100), 'unit': random.choice(['kg', 'm', 's', '°C'])}
                elif col == '多层级数值2':
                    row[col] = [random.randint(100, 1000) for _ in range(5)]
                elif col == '时间序列数据':
                    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D').tolist()
                    row[col] = random.choice(dates).strftime('%Y-%m-%d %H:%M:%S')
                elif col == '分类数据':
                    row[col] = random.choice(['类别A', '类别B', '类别C', '类别D', '类别E'])
                else:
                    row[col] = {'nested': {'key': 'value', 'data': [random.randint(1, 100) for _ in range(3)]}}
        data.append(row)
    return data

def save_table_to_excel(data, filename, sheet_name):
    """
    保存表格数据到Excel文件
    """
    df = pd.DataFrame(data)
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"✅ 表格已保存到 {filename}，工作表: {sheet_name}")

def main():
    """
    主函数，生成20种表格类型
    """
    # 确保输出目录存在
    output_dir = './generated_tables'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '20_table_types.xlsx')
    
    # 创建Excel文件
    import openpyxl
    # 使用openpyxl直接创建工作簿
    wb = openpyxl.Workbook()
    wb.save(output_file)
    wb.close()
    
    print("📊 开始生成20种表格类型...")
    
    # 10种不需要学习的常见表格
    common_tables = [
        {"name": "简单数值表", "columns": ["名称", "数值1", "数值2", "日期"]},
        {"name": "基础分类表", "columns": ["分类", "数值", "单位"]},
        {"name": "销售数据表", "columns": ["产品", "销量", "单价", "金额"]},
        {"name": "员工信息表", "columns": ["姓名", "部门", "工资", "入职日期"]},
        {"name": "库存管理表", "columns": ["商品", "库存", "进货价", "售价"]},
        {"name": "订单明细表", "columns": ["订单号", "产品", "数量", "金额"]},
        {"name": "学生成绩表", "columns": ["姓名", "科目", "成绩", "班级"]},
        {"name": "客户信息表", "columns": ["客户ID", "姓名", "性别", "年龄"]},
        {"name": "财务报表", "columns": ["项目", "收入", "支出", "利润"]},
        {"name": "生产计划表", "columns": ["产品", "计划产量", "实际产量", "完成率"]}
    ]
    
    # 10种需要学习的复杂表格
    complex_tables = [
        {"name": "多层级数据表", "columns": ["复杂名称", "多层级数值1", "多层级数值2", "时间序列数据"]},
        {"name": "嵌套结构表", "columns": ["父类别", "子类别", "数值", "分类数据"]},
        {"name": "关联分析表", "columns": ["主表", "关联表", "计算规则", "结果列"]},
        {"name": "数据透视表", "columns": ["行字段", "列字段", "值字段", "筛选条件"]},
        {"name": "预测模型表", "columns": ["特征", "权重", "偏置", "预测值"]},
        {"name": "统计分析表", "columns": ["指标", "均值", "中位数", "标准差", "最大值", "最小值"]},
        {"name": "时间序列预测表", "columns": ["日期", "实际值", "预测值", "误差"]},
        {"name": "机器学习特征表", "columns": ["特征名称", "特征类型", "重要性", "处理方法"]},
        {"name": "数据清洗规则表", "columns": ["字段", "规则", "转换方法", "输出格式"]},
        {"name": "复杂计算表", "columns": ["计算逻辑", "依赖字段", "输出结果", "验证规则"]}
    ]
    
    # 生成常见表格
    for i, table in enumerate(common_tables, 1):
        print(f"\n生成常见表格 {i}/10: {table['name']}")
        data = generate_table(table["columns"], 10, is_common=True)
        save_table_to_excel(data, output_file, f"common_{i}_{table['name'][:5]}")
    
    # 生成复杂表格
    for i, table in enumerate(complex_tables, 1):
        print(f"\n生成复杂表格 {i}/10: {table['name']}")
        data = generate_table(table["columns"], 10, is_common=False)
        save_table_to_excel(data, output_file, f"complex_{i}_{table['name'][:5]}")
    
    print(f"\n🎉 20种表格类型已生成完成！")
    print(f"📁 输出文件: {output_file}")

if __name__ == "__main__":
    main()
