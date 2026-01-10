#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis连接测试脚本
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.dual_redis_db import DualRedisDB
from src.utils.config import config_manager

def main():
    """
    主函数：测试Redis连接
    """
    print("开始测试Redis连接...")
    
    # 测试配置管理器是否能正确获取Redis配置
    print("1. 测试配置管理器获取Redis配置")
    redis_config = config_manager.get("redis")
    print(f"   Redis配置: {redis_config}")
    
    # 测试DualRedisDB连接
    print("2. 测试DualRedisDB连接")
    dual_redis = DualRedisDB()
    
    # 测试任务库连接
    print("3. 测试任务库连接")
    try:
        task_ping = dual_redis.task_conn.ping()
        print(f"   任务库(DB {redis_config.get('task_db', 0)}) ping: {task_ping}")
    except Exception as e:
        print(f"   任务库连接失败: {e}")
        return False
    
    # 测试逻辑库连接
    print("4. 测试逻辑库连接")
    try:
        logic_ping = dual_redis.logic_conn.ping()
        print(f"   逻辑库(DB {redis_config.get('logic_db', 1)}) ping: {logic_ping}")
    except Exception as e:
        print(f"   逻辑库连接失败: {e}")
        return False
    
    # 测试基本操作
    print("5. 测试基本Redis操作")
    test_key = "test_connection"
    test_value = "connection_test_success"
    
    try:
        # 写入测试数据
        dual_redis.task_conn.set(test_key, test_value)
        print(f"   写入测试数据: {test_key} -> {test_value}")
        
        # 读取测试数据
        read_value = dual_redis.task_conn.get(test_key)
        print(f"   读取测试数据: {test_key} -> {read_value}")
        
        # 删除测试数据
        dual_redis.task_conn.delete(test_key)
        print(f"   删除测试数据: {test_key}")
        
        # 验证删除
        deleted_value = dual_redis.task_conn.get(test_key)
        print(f"   验证删除: {test_key} -> {deleted_value}")
        
        if read_value == test_value and deleted_value is None:
            print("   基本操作测试成功")
        else:
            print("   基本操作测试失败")
            return False
    except Exception as e:
        print(f"   基本操作测试失败: {e}")
        return False
    
    print("\nRedis连接测试成功！项目已正确连接到Redis服务器。")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
