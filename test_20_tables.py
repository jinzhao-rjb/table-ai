#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用生成的20个表格进行需求测试
"""

import os
import sys
import pandas as pd
from datetime import datetime
import openpyxl

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入MultiColumnProcessor
from src.modules.multi_column_processor import MultiColumnProcessor

def test_20_tables_requirements():
    """
    使用20个表格进行需求测试
    """
    # 初始化处理器
    processor = MultiColumnProcessor()
    
    # 表格文件路径
    table_file = os.path.join(project_root, "generated_tables", "20_table_types.xlsx")
    
    # 测试结果目录
    results_dir = os.path.join(project_root, "20_tables_test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试结果记录
    test_results = []
    
    print("开始使用20个表格进行需求测试...")
    print("=" * 60)
    
    # 打开Excel文件
    wb = openpyxl.load_workbook(table_file)
    sheet_names = wb.sheetnames
    
    # 过滤掉默认的Sheet
    sheet_names = [sheet for sheet in sheet_names if sheet != 'Sheet']
    
    # 为不同类型的表格生成不同的需求
    requirement_templates = {
        "common": [
            "计算每列的总和",
            "计算每列的平均值",
            "计算每行的总和",
            "计算每行的平均值",
            "找出每列的最大值",
            "找出每列的最小值",
            "计算表格的总平均值",
            "计算每列数据的范围（最大值-最小值）",
            "计算每列数据的方差",
            "计算每列数据的标准差"
        ],
        "complex": [
            "分析多层级数据的关系",
            "处理嵌套结构数据",
            "关联分析不同表的数据",
            "生成数据透视表",
            "预测未来趋势",
            "统计分析数据分布",
            "分析时间序列数据",
            "处理机器学习特征",
            "应用数据清洗规则",
            "执行复杂计算逻辑"
        ]
    }
    
    # 逐个测试工作表
    for i, sheet_name in enumerate(sheet_names, 1):
        print(f"\n测试工作表 {i}/{len(sheet_names)}: {sheet_name}")
        print("-" * 40)
        
        try:
            # 加载工作表数据
            df = pd.read_excel(table_file, sheet_name=sheet_name)
            
            # 记录原始数据信息
            original_shape = df.shape
            original_columns = list(df.columns)
            
            # 判断表格类型
            table_type = "common" if sheet_name.startswith("common_") else "complex"
            
            # 随机选择一个需求模板
            import random
            requirement = random.choice(requirement_templates[table_type])
            print(f"需求: {requirement}")
            
            # 生成数据上下文
            data_context = processor._generate_data_context(df)
            
            # 生成函数
            functions = processor.generate_multi_column_functions(requirement, data_context)
            
            if functions:
                print(f"生成了 {len(functions)} 个函数")
                
                # 应用函数处理数据
                processed_df = processor.process_data(df, functions, requirement)
                
                # 记录处理结果
                processed_shape = processed_df.shape
                new_columns = list(set(processed_df.columns) - set(original_columns))
                
                print(f"处理结果: {processed_shape} 行 x {processed_shape[1]} 列")
                print(f"新增列: {new_columns}")
                
                # 保存处理后的数据
                output_file = os.path.join(results_dir, f"processed_{sheet_name}.xlsx")
                processed_df.to_excel(output_file, index=False)
                
                # 记录测试结果
                test_results.append({
                    "sheet_name": sheet_name,
                    "table_type": table_type,
                    "requirement": requirement,
                    "original_shape": original_shape,
                    "processed_shape": processed_shape,
                    "new_columns": new_columns,
                    "functions_count": len(functions),
                    "success": True,
                    "output_file": output_file,
                    "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                print("✅ 成功")
                
            else:
                print("未生成有效的函数")
                test_results.append({
                    "sheet_name": sheet_name,
                    "table_type": table_type,
                    "requirement": requirement,
                    "original_shape": original_shape,
                    "processed_shape": original_shape,
                    "new_columns": [],
                    "functions_count": 0,
                    "success": False,
                    "error": "未生成有效的函数",
                    "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                print("❌ 失败")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            test_results.append({
                "sheet_name": sheet_name,
                "table_type": table_type if 'table_type' in locals() else "",
                "requirement": requirement if 'requirement' in locals() else "",
                "original_shape": original_shape if 'original_shape' in locals() else (0, 0),
                "processed_shape": original_shape if 'original_shape' in locals() else (0, 0),
                "new_columns": [],
                "functions_count": 0,
                "success": False,
                "error": str(e),
                "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # 保存测试结果
    import json
    results_file = os.path.join(results_dir, "20_tables_test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    # 生成测试摘要
    summary_file = os.path.join(results_dir, "20_tables_test_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("20个表格需求测试摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"表格文件: {table_file}\n")
        f.write(f"测试工作表数量: {len(sheet_names)}\n")
        
        # 统计成功和失败数量
        success_count = sum(1 for r in test_results if r['success'])
        failure_count = len(test_results) - success_count
        
        f.write(f"成功数量: {success_count}\n")
        f.write(f"失败数量: {failure_count}\n")
        f.write(f"成功率: {success_count/len(test_results)*100:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        
        # 按表格类型分类统计
        common_count = sum(1 for r in test_results if r['table_type'] == 'common')
        complex_count = sum(1 for r in test_results if r['table_type'] == 'complex')
        
        common_success = sum(1 for r in test_results if r['table_type'] == 'common' and r['success'])
        complex_success = sum(1 for r in test_results if r['table_type'] == 'complex' and r['success'])
        
        f.write("按表格类型统计:\n")
        f.write(f"常见表格: {common_count} 个，成功: {common_success} 个，成功率: {common_success/common_count*100:.1f}%\n")
        f.write(f"复杂表格: {complex_count} 个，成功: {complex_success} 个，成功率: {complex_success/complex_count*100:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        
        # 详细结果
        f.write("详细结果:\n")
        for r in test_results:
            status = "✅ 成功" if r['success'] else "❌ 失败"
            f.write(f"{status} {r['sheet_name']} - {r['requirement']}\n")
            f.write(f"   类型: {r['table_type']}, 原始: {r['original_shape']}, 处理后: {r['processed_shape']}, 新增列: {len(r['new_columns'])}\n")
            if not r['success']:
                f.write(f"   错误: {r['error']}\n")
            f.write("\n")
    
    print(f"\n测试完成！")
    print(f"测试结果保存在: {results_dir}")
    print(f"成功: {success_count}/{len(test_results)} ({success_count/len(test_results)*100:.1f}%)")
    
    return test_results

if __name__ == "__main__":
    test_20_tables_requirements()
