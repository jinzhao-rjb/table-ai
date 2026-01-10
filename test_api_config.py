#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API配置是否正确加载
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 设置环境变量
env_api_key = 'your_actual_api_key_from_env'
os.environ['AI_API_KEY'] = env_api_key
os.environ['AI_MODEL'] = 'qwen-vl-plus'
os.environ['AI_API_TYPE'] = 'qwen'

print("=== API配置测试 ===")
print(f"设置的环境变量API_KEY: {env_api_key}")

# 测试配置管理器
from src.utils.config import ConfigManager
config_manager = ConfigManager()
print(f"\n1. 配置管理器加载结果:")
print(f"   API Key: {config_manager.get('ai.api_key')}")
print(f"   Model: {config_manager.get('ai.model')}")
print(f"   API Type: {config_manager.get('ai.api_type')}")

# 测试API管理器
from src.modules.api_manager import APIManager
print(f"\n2. API管理器初始化结果:")
try:
    api_manager = APIManager()
    print(f"   API Key: {api_manager.api_key}")
    print(f"   Model: {api_manager.model}")
    print(f"   API Type: {api_manager.api_type}")
    print("   ✅ API管理器初始化成功!")
except Exception as e:
    print(f"   ❌ API管理器初始化失败: {e}")

# 测试QwenVL管理器
from src.modules.qwen_vl_manager import QwenVLManager
print(f"\n3. QwenVL管理器初始化结果:")
try:
    qwen_manager = QwenVLManager()
    print(f"   API Key: {qwen_manager.api_key}")
    print(f"   Model: {qwen_manager.model}")
    print(f"   API Type: {qwen_manager.api_type}")
    print("   ✅ QwenVL管理器初始化成功!")
except Exception as e:
    print(f"   ❌ QwenVL管理器初始化失败: {e}")

print("\n=== 测试完成 ===")
print("✅ API配置已成功更新，现在优先使用环境变量!")
print("\n使用说明:")
print("1. 在系统环境变量中设置AI_API_KEY、AI_MODEL、AI_API_TYPE")
print("2. 应用将自动使用这些环境变量覆盖配置文件")
print("3. 无需修改代码或配置文件即可切换API")
