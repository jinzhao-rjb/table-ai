#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置加载是否正确
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import config_manager

# 打印当前配置
print("当前配置:")
print(f"API Key: {config_manager.get('ai.api_key')}")
print(f"Model: {config_manager.get('ai.model')}")
print(f"API Type: {config_manager.get('ai.api_type')}")
print(f"Redis Host: {config_manager.get('redis.host')}")
print(f"Redis Port: {config_manager.get('redis.port')}")

# 测试环境变量覆盖
print("\n=== 测试环境变量覆盖 ===")
os.environ['AI_API_KEY'] = 'test_env_api_key'
os.environ['AI_MODEL'] = 'test_env_model'
os.environ['AI_API_TYPE'] = 'test_env_type'
os.environ['REDIS_HOST'] = 'test_env_redis'
os.environ['REDIS_PORT'] = '6380'

# 重新初始化配置管理器
from src.utils.config import ConfigManager
config_manager_test = ConfigManager()

print(f"API Key (环境变量): {config_manager_test.get('ai.api_key')}")
print(f"Model (环境变量): {config_manager_test.get('ai.model')}")
print(f"API Type (环境变量): {config_manager_test.get('ai.api_type')}")
print(f"Redis Host (环境变量): {config_manager_test.get('redis.host')}")
print(f"Redis Port (环境变量): {config_manager_test.get('redis.port')}")

print("\n测试完成!")
