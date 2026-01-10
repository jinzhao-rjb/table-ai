import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.modules.api_manager import APIManager
from src.utils.config import config_manager

def test_function_generation():
    """
    测试函数调用功能，验证generate_functions方法是否正常工作
    """
    print("开始测试函数调用功能...")
    
    # 初始化API管理器
    api_manager = APIManager(
        api_key=config_manager.get("ai.api_key", ""),
        model="qwen-plus",
        api_type="qwen"
    )
    
    # 测试用例1：简单的数据处理需求
    print("\n测试用例1：简单的数据处理需求")
    prompt = "计算每个产品的毛利率，毛利率 = (销售额 - 成本) / 销售额 * 100%"
    data_context = {
        "columns": ["产品名称", "销售额", "成本"],
        "sample_data": [
            ["产品A", 100, 60],
            ["产品B", 200, 150],
            ["产品C", 150, 90]
        ]
    }
    
    success, functions, error = api_manager.generate_functions(prompt, data_context)
    
    if success:
        print(f"✓ 测试用例1成功，生成了{len(functions)}个函数")
        for i, func in enumerate(functions):
            print(f"\n函数{i+1}：{func['name']}")
            print(f"描述：{func['description']}")
            print(f"实现：{func['implementation'][:200]}...")
    else:
        print(f"✗ 测试用例1失败：{error}")
    
    # 测试用例2：更复杂的数据处理需求
    print("\n测试用例2：更复杂的数据处理需求")
    prompt = "计算月度销售总额、平均销售额、最高销售额和最低销售额，并添加排名列"
    data_context = {
        "columns": ["月份", "销售额", "产品类别"],
        "sample_data": [
            ["1月", 15000, "电子产品"],
            ["2月", 23000, "电子产品"],
            ["3月", 18000, "服装"],
            ["4月", 32000, "电子产品"],
            ["5月", 27000, "服装"]
        ]
    }
    
    success, functions, error = api_manager.generate_functions(prompt, data_context)
    
    if success:
        print(f"✓ 测试用例2成功，生成了{len(functions)}个函数")
        for i, func in enumerate(functions):
            print(f"\n函数{i+1}：{func['name']}")
            print(f"描述：{func['description']}")
            print(f"实现：{func['implementation'][:200]}...")
    else:
        print(f"✗ 测试用例2失败：{error}")
    
    # 测试用例3：测试API连接
    print("\n测试用例3：测试API连接")
    success, message = api_manager.test_connection()
    if success:
        print(f"✓ API连接测试成功：{message}")
    else:
        print(f"✗ API连接测试失败：{message}")
    
    print("\n函数调用功能测试完成！")

if __name__ == "__main__":
    test_function_generation()