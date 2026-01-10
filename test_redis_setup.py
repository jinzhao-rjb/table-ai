#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Redis连接和基本功能
"""

import os
import sys
import redis

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.dual_redis_db import DualRedisDB

def test_redis_connection():
    """
    测试Redis连接
    """
    print("测试Redis连接...")
    
    try:
        # 初始化DualRedisDB
        dual_redis = DualRedisDB()
        print("✅ DualRedisDB初始化成功")
        
        # 测试任务库连接
        try:
            task_conn = dual_redis.task_conn
            task_conn.ping()
            print("✅ 任务库(task_conn)连接成功")
        except Exception as e:
            print(f"❌ 任务库(task_conn)连接失败: {e}")
        
        # 测试逻辑库连接
        try:
            db_conn = dual_redis.logic_conn
            db_conn.ping()
            print("✅ 逻辑库(logic_conn)连接成功")
        except Exception as e:
            print(f"❌ 逻辑库(logic_conn)连接失败: {e}")
        
        # 测试基本操作
        test_key = "test_redis_setup"
        test_value = "hello_redis"
        
        # 设置测试值
        db_conn.set(test_key, test_value)
        print(f"✅ 设置键 {test_key} = {test_value} 成功")
        
        # 获取测试值
        retrieved_value = db_conn.get(test_key)
        if retrieved_value:
            if retrieved_value == test_value:
                print(f"✅ 获取键 {test_key} 成功，值为: {retrieved_value}")
            else:
                print(f"❌ 获取键 {test_key} 失败，期望值: {test_value}，实际值: {retrieved_value}")
        else:
            print(f"❌ 获取键 {test_key} 失败，返回值为None")
        
        # 删除测试值
        db_conn.delete(test_key)
        print(f"✅ 删除键 {test_key} 成功")
        
        # 验证删除
        retrieved_value = db_conn.get(test_key)
        if retrieved_value is None:
            print(f"✅ 验证键 {test_key} 已删除")
        else:
            print(f"❌ 键 {test_key} 未被成功删除")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_learning():
    """
    测试qwen_learning模块
    """
    print("\n测试qwen_learning模块...")
    
    try:
        from src.modules.qwen_learning import get_qwen_learning
        
        # 获取qwen_learning实例
        qwen_learning = get_qwen_learning()
        print("✅ qwen_learning实例获取成功")
        
        # 测试从Redis加载数据
        qwen_learning.load_learning_data()
        print("✅ 从Redis加载学习数据成功")
        
        # 测试保存学习数据
        qwen_learning.save_learning_data()
        print("✅ 学习数据保存到Redis成功")
        
        # 测试生成增强提示词
        test_prompt = "计算每列的总和"
        test_context = {
            "columns": ["列1", "列2", "列3"],
            "data_types": {"列1": "int64", "列2": "int64", "列3": "int64"},
            "data_shape": (10, 3)
        }
        
        enhanced_prompt = qwen_learning.generate_enhanced_prompt(test_prompt, test_context)
        print("✅ 生成增强提示词成功")
        print(f"增强提示词: {enhanced_prompt[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ qwen_learning模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Redis连接和qwen_learning模块测试")
    print("=" * 50)
    
    # 测试Redis连接
    redis_success = test_redis_connection()
    
    # 测试qwen_learning模块
    qwen_success = test_qwen_learning()
    
    print("\n" + "=" * 50)
    if redis_success and qwen_success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查配置")
