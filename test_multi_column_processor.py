#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用multi_column_processor测试生成的表格
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.modules.multi_column_processor import MultiColumnProcessor

def test_multi_column_processor():
    """
    使用MultiColumnProcessor测试生成的表格
    """
    # 初始化处理器
    processor = MultiColumnProcessor()
    
    # 表格目录
    table_dir = os.path.join(project_root, "generated_tables")
    
    # 获取所有Excel文件
    excel_files = [f for f in os.listdir(table_dir) if f.endswith('.xlsx')]
    
    # 测试结果目录
    results_dir = os.path.join(project_root, "multi_column_test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试结果记录
    test_results = []
    
    print("开始使用MultiColumnProcessor测试表格...")
    print(f"共找到{len(excel_files)}个Excel文件")
    print("=" * 60)
    
    # 测试需求列表
    requirements = [
        "计算每列的总和",
        "计算每列的平均值",
        "计算每列的最大值和最小值",
        "计算每列的中位数",
        "计算每行的总和",
        "计算每行的平均值",
        "计算表格的总平均值",
        "计算每列数据的范围（最大值-最小值）",
        "计算每列数据的方差",
        "计算每列数据的标准差"
    ]
    
    for i, file in enumerate(excel_files, 1):
        print(f"\n测试文件 {i}/{len(excel_files)}: {file}")
        print("-" * 40)
        
        file_path = os.path.join(table_dir, file)
        
        try:
            # 加载数据
            df = pd.read_excel(file_path)
            
            # 记录原始数据信息
            original_shape = df.shape
            original_columns = list(df.columns)
            
            # 生成数据上下文
            data_context = processor._generate_data_context(df)
            
            # 随机选择一个需求
            import random
            requirement = random.choice(requirements)
            
            print(f"测试需求: {requirement}")
            print(f"原始数据: {original_shape} 行 x {original_shape[1]} 列")
            
            # 生成函数
            print("生成函数中...")
            functions = processor.generate_multi_column_functions(requirement, data_context)
            
            if functions:
                print(f"生成了 {len(functions)} 个函数")
                
                # 显示函数名称和描述，不显示实现
                for j, func in enumerate(functions):
                    print(f"函数 {j+1}: {func.get('name', 'unknown')} - {func.get('description', '无描述')}")
                
                # 应用函数处理数据
                print("应用函数处理数据中...")
                processed_df = processor.process_data(df, functions, requirement)
                
                # 记录处理结果
                processed_shape = processed_df.shape
                new_columns = list(set(processed_df.columns) - set(original_columns))
                
                print(f"处理结果: {processed_shape} 行 x {processed_shape[1]} 列")
                print(f"新增列: {new_columns}")
                
                # 保存处理后的数据
                output_file = os.path.join(results_dir, f"processed_{file}")
                processed_df.to_excel(output_file, index=False)
                
                # 记录测试结果
                test_results.append({
                    "file": file,
                    "requirement": requirement,
                    "original_shape": original_shape,
                    "processed_shape": processed_shape,
                    "new_columns": new_columns,
                    "functions_count": len(functions),
                    "functions": [{
                        "name": func.get('name'),
                        "description": func.get('description')
                    } for func in functions],
                    "success": True,
                    "output_file": output_file,
                    "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                print("未生成有效的函数")
                test_results.append({
                    "file": file,
                    "requirement": requirement,
                    "original_shape": original_shape,
                    "processed_shape": original_shape,
                    "new_columns": [],
                    "functions_count": 0,
                    "functions": [],
                    "success": False,
                    "error": "未生成有效的函数",
                    "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
        except Exception as e:
            print(f"测试失败: {e}")
            test_results.append({
                "file": file,
                "requirement": requirement if 'requirement' in locals() else "",
                "original_shape": original_shape if 'original_shape' in locals() else (0, 0),
                "processed_shape": original_shape if 'original_shape' in locals() else (0, 0),
                "new_columns": [],
                "functions_count": 0,
                "functions": [],
                "success": False,
                "error": str(e),
                "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # 保存测试结果
    import json
    results_file = os.path.join(results_dir, "multi_column_test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    # 生成测试摘要
    summary_file = os.path.join(results_dir, "multi_column_test_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MultiColumnProcessor 测试摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试文件数量: {len(excel_files)}\n")
        
        # 统计成功和失败数量
        success_count = sum(1 for r in test_results if r['success'])
        failure_count = len(test_results) - success_count
        
        f.write(f"成功数量: {success_count}\n")
        f.write(f"失败数量: {failure_count}\n")
        f.write(f"成功率: {success_count/len(excel_files)*100:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        
        # 按文件类型分类统计
        common_count = sum(1 for r in test_results if r['file'].startswith('common_'))
        complex_count = sum(1 for r in test_results if r['file'].startswith('complex_'))
        
        common_success = sum(1 for r in test_results if r['file'].startswith('common_') and r['success'])
        complex_success = sum(1 for r in test_results if r['file'].startswith('complex_') and r['success'])
        
        f.write("按表格类型统计:\n")
        f.write(f"常见表格: {common_count} 个，成功: {common_success} 个，成功率: {common_success/common_count*100:.1f}%\n")
        f.write(f"复杂表格: {complex_count} 个，成功: {complex_success} 个，成功率: {complex_success/complex_count*100:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        
        # 详细结果
        f.write("详细结果:\n")
        for r in test_results:
            status = "✅ 成功" if r['success'] else "❌ 失败"
            f.write(f"{status} {r['file']} - {r['requirement']}\n")
            f.write(f"   原始: {r['original_shape']}, 处理后: {r['processed_shape']}, 新增列: {len(r['new_columns'])}\n")
            if not r['success']:
                f.write(f"   错误: {r['error']}\n")
            f.write("\n")
    
    print(f"\n测试完成！")
    print(f"测试结果保存在: {results_dir}")
    print(f"成功: {success_count}/{len(excel_files)} ({success_count/len(excel_files)*100:.1f}%)")
    
    return test_results

if __name__ == "__main__":
    test_multi_column_processor()
