#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
函数生成与调用测试脚本
1. 测试从Redis获取现有函数
2. 测试生成新函数
3. 测试调用函数
4. 保存测试结果
"""

import os
import sys
import random
import json
from datetime import datetime
import importlib.util
import tempfile

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.dual_redis_db import DualRedisDB

class FunctionGenerationTester:
    def __init__(self):
        self.dual_redis = DualRedisDB()
        self.test_results_dir = os.path.join(project_root, "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # 保存测试结果的文件
        self.test_results_file = os.path.join(self.test_results_dir, "function_test_results.txt")
        
        # 测试用的简单数据
        self.test_data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    
    def get_existing_functions(self, count=10):
        """
        从Redis获取现有的函数
        """
        # 获取所有函数ID
        func_list = self.dual_redis.logic_conn.lrange("qwen:func_list", 0, -1)
        
        # 随机选择count个函数
        if len(func_list) < count:
            print(f"警告：现有函数数量不足{count}个，只能使用{len(func_list)}个")
            return func_list
        
        return random.sample(func_list, count)
    
    def get_function_details(self, func_id):
        """
        获取函数详细信息
        """
        func_key = f"qwen:funcs:{func_id}"
        return self.dual_redis.logic_conn.hgetall(func_key)
    
    def generate_new_functions(self, count=10):
        """
        生成新的函数
        """
        # 预定义的函数模板
        function_templates = [
            {
                "name": "calculate_sum",
                "code": "def calculate_sum(numbers):\n    return sum(numbers)",
                "description": "计算列表的和"
            },
            {
                "name": "calculate_average",
                "code": "def calculate_average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0",
                "description": "计算列表的平均值"
            },
            {
                "name": "find_max",
                "code": "def find_max(numbers):\n    return max(numbers) if numbers else None",
                "description": "查找列表中的最大值"
            },
            {
                "name": "find_min",
                "code": "def find_min(numbers):\n    return min(numbers) if numbers else None",
                "description": "查找列表中的最小值"
            },
            {
                "name": "reverse_list",
                "code": "def reverse_list(numbers):\n    return numbers[::-1]",
                "description": "反转列表"
            },
            {
                "name": "sort_list",
                "code": "def sort_list(numbers):\n    return sorted(numbers)",
                "description": "排序列表"
            },
            {
                "name": "filter_even",
                "code": "def filter_even(numbers):\n    return [num for num in numbers if num % 2 == 0]",
                "description": "过滤出偶数"
            },
            {
                "name": "filter_odd",
                "code": "def filter_odd(numbers):\n    return [num for num in numbers if num % 2 != 0]",
                "description": "过滤出奇数"
            },
            {
                "name": "multiply_by_constant",
                "code": "def multiply_by_constant(numbers, constant=2):\n    return [num * constant for num in numbers]",
                "description": "将列表中的每个元素乘以常量"
            },
            {
                "name": "flatten_list",
                "code": "def flatten_list(nested_list):\n    flattened = []\n    for item in nested_list:\n        if isinstance(item, list):\n            flattened.extend(flatten_list(item))\n        else:\n            flattened.append(item)\n    return flattened",
                "description": "扁平化嵌套列表"
            }
        ]
        
        new_functions = []
        for i in range(count):
            # 随机选择一个模板
            template = random.choice(function_templates)
            
            # 生成函数ID
            import time
            func_id = f"fn:{int(time.time()*1000)}_{i}"
            
            # 创建函数数据
            func_data = {
                "name": f"{template['name']}_{i}",
                "code": template['code'],
                "context": json.dumps({"description": template['description'], "requirements": template['description']}),
                "created_at": datetime.now().ctime(),
                "score": 80,
                "use_count": 0,
                "weight": 1,
                "is_golden": 1
            }
            
            # 保存到Redis
            func_key = f"qwen:funcs:{func_id}"
            self.dual_redis.logic_conn.hset(func_key, mapping=func_data)
            self.dual_redis.logic_conn.lpush("qwen:func_list", func_id)
            
            new_functions.append(func_id)
        
        return new_functions
    
    def execute_function(self, func_code, func_name, data):
        """
        执行函数
        """
        try:
            # 创建临时模块
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(func_code)
                temp_file = f.name
            
            # 加载模块
            spec = importlib.util.spec_from_file_location("temp_module", temp_file)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            
            # 获取函数
            func = getattr(temp_module, func_name)
            
            # 执行函数
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                # 如果是二维列表，展平后传递
                flattened_data = [item for sublist in data for item in sublist]
                result = func(flattened_data)
            else:
                result = func(data)
            
            # 删除临时文件
            os.unlink(temp_file)
            
            return True, result
        except Exception as e:
            # 删除临时文件
            if 'temp_file' in locals():
                os.unlink(temp_file)
            return False, str(e)
    
    def test_existing_functions(self, count=10):
        """
        测试现有函数
        """
        print("测试现有函数...")
        existing_funcs = self.get_existing_functions(count)
        results = []
        
        for func_id in existing_funcs:
            func_details = self.get_function_details(func_id)
            if not func_details:
                results.append({"func_id": func_id, "success": False, "message": "函数详情获取失败"})
                continue
            
            func_name = func_details.get("name", "unknown")
            func_code = func_details.get("code", "")
            
            print(f"  测试现有函数: {func_name} (ID: {func_id})")
            
            if not func_code:
                results.append({"func_id": func_id, "name": func_name, "success": False, "message": "函数代码为空"})
                continue
            
            # 执行函数
            success, result = self.execute_function(func_code, func_name.split('_')[0], self.test_data)
            
            results.append({
                "func_id": func_id,
                "name": func_name,
                "success": success,
                "result": result if success else str(result),
                "code": func_code
            })
        
        return results
    
    def test_new_functions(self, count=10):
        """
        测试新生成的函数
        """
        print("测试新生成的函数...")
        new_funcs = self.generate_new_functions(count)
        results = []
        
        for func_id in new_funcs:
            func_details = self.get_function_details(func_id)
            if not func_details:
                results.append({"func_id": func_id, "success": False, "message": "函数详情获取失败"})
                continue
            
            func_name = func_details.get("name", "unknown")
            func_code = func_details.get("code", "")
            
            print(f"  测试新函数: {func_name} (ID: {func_id})")
            
            if not func_code:
                results.append({"func_id": func_id, "name": func_name, "success": False, "message": "函数代码为空"})
                continue
            
            # 执行函数
            base_func_name = func_name.split('_')[0]
            success, result = self.execute_function(func_code, base_func_name, self.test_data)
            
            results.append({
                "func_id": func_id,
                "name": func_name,
                "success": success,
                "result": result if success else str(result),
                "code": func_code
            })
        
        return results
    
    def run_test(self):
        """
        运行完整测试
        """
        print("开始函数生成与调用测试...")
        
        # 测试现有函数
        existing_results = self.test_existing_functions(10)
        
        # 测试新生成的函数
        new_results = self.test_new_functions(10)
        
        # 保存测试结果
        self.save_test_results(existing_results, new_results)
        
        print("测试完成！")
        print(f"测试结果保存在: {self.test_results_file}")
        
        return existing_results, new_results
    
    def save_test_results(self, existing_results, new_results):
        """
        保存测试结果到文件
        """
        with open(self.test_results_file, "w", encoding="utf-8") as f:
            f.write(f"函数生成与调用测试结果\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*70 + "\n\n")
            
            # 现有函数测试结果
            f.write("1. 现有函数测试结果\n")
            f.write(f"-"*50 + "\n")
            for i, result in enumerate(existing_results, 1):
                status = "✓ 成功" if result["success"] else "✗ 失败"
                f.write(f"{i}. {result['name']} (ID: {result['func_id']}) - {status}\n")
                if result["success"]:
                    f.write(f"   结果: {result['result']}\n")
                else:
                    f.write(f"   错误: {result['result']}\n")
                f.write(f"   代码: {result['code'][:100]}...\n")
            f.write("\n")
            
            # 新函数测试结果
            f.write("2. 新生成函数测试结果\n")
            f.write(f"-"*50 + "\n")
            for i, result in enumerate(new_results, 1):
                status = "✓ 成功" if result["success"] else "✗ 失败"
                f.write(f"{i}. {result['name']} (ID: {result['func_id']}) - {status}\n")
                if result["success"]:
                    f.write(f"   结果: {result['result']}\n")
                else:
                    f.write(f"   错误: {result['result']}\n")
                f.write(f"   代码: {result['code'][:100]}...\n")
            f.write("\n")
            
            # 统计结果
            existing_success = sum(1 for r in existing_results if r["success"])
            new_success = sum(1 for r in new_results if r["success"])
            
            f.write("3. 测试统计\n")
            f.write(f"-"*50 + "\n")
            f.write(f"现有函数: {existing_success}/{len(existing_results)} 成功\n")
            f.write(f"新生成函数: {new_success}/{len(new_results)} 成功\n")
            total_success = existing_success + new_success
            total_tests = len(existing_results) + len(new_results)
            f.write(f"总成功率: {total_success/total_tests*100:.1f}%\n")

if __name__ == "__main__":
    tester = FunctionGenerationTester()
    tester.run_test()
