#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用multi_column_processor测试生成的表格（简化版，不依赖AI服务）
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 模拟AI服务的基础类，避免依赖缺失的qwen_learning模块
class MockAIService:
    def generate_functions(self, prompt, data_context):
        # 模拟生成的函数，直接返回一些简单的计算函数
        return [
            {
                "name": "calculate_sum",
                "description": "计算每列的总和",
                "implementation": "import pandas as pd\nimport numpy as np\ndef calculate_sum(df):\n    for col in df.columns:\n        df[f'{col}_总和'] = df[col].sum()\n    return df",
                "required_columns": list(data_context.get("columns", []))
            },
            {
                "name": "calculate_average",
                "description": "计算每列的平均值",
                "implementation": "import pandas as pd\nimport numpy as np\ndef calculate_average(df):\n    for col in df.columns:\n        df[f'{col}_平均值'] = df[col].mean()\n    return df",
                "required_columns": list(data_context.get("columns", []))
            }
        ]

# 简化的测试函数
def test_multi_column_processor_simple():
    """
    使用简化的方式测试MultiColumnProcessor的核心功能
    """
    print("DEBUG: 进入test_multi_column_processor_simple函数")
    
    # 表格目录
    table_dir = os.path.join(project_root, "generated_tables")
    print(f"DEBUG: 表格目录: {table_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(table_dir):
        print(f"ERROR: 表格目录不存在: {table_dir}")
        return
    
    # 获取所有Excel文件
    print(f"DEBUG: 列出目录内容: {os.listdir(table_dir)}")
    excel_files = [f for f in os.listdir(table_dir) if f.endswith('.xlsx')]
    
    # 测试结果目录
    results_dir = os.path.join(project_root, "multi_column_simple_test_results")
    print(f"DEBUG: 测试结果目录: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试结果记录
    test_results = []
    
    print("开始使用简化版MultiColumnProcessor测试表格...")
    print(f"共找到{len(excel_files)}个Excel文件")
    print("=" * 60)
    
    # 测试需求列表
    requirements = [
        "计算每列的总和",
        "计算每列的平均值",
        "计算每行的总和",
        "计算每行的平均值"
    ]
