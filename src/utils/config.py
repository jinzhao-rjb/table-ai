#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
"""

import os
import json

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        # 默认配置
        self.config = {
            # Redis配置
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            # AI服务配置
            "ai": {
                "api_key": "",
                "model": "qwen-vl-max",
                "api_type": "qwen"
            },
            # 表格提取配置
            "table_extraction": {
                "yolo_model_path": "D:\\office\\office-lazy-tool\\weights\\best.pt",
                "output_folder": "output",
                "max_workers": 4
            },
            # API速率限制配置
            "rate_limiting": {
                "max_requests_per_minute": 30
            }
        }
        
        # 加载外部配置文件 - 支持从项目根目录或src目录运行
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        
        # 尝试从项目根目录加载配置文件
        config_file = os.path.join(project_root, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                external_config = json.load(f)
                # 合并配置
                self._merge_dict(self.config, external_config)
        else:
            # 回退到当前目录
            config_file = "config.json"
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    external_config = json.load(f)
                    # 合并配置
                    self._merge_dict(self.config, external_config)

        # AI服务配置
        if os.environ.get("AI_API_KEY"):
            self.config["ai"]["api_key"] = os.environ.get("AI_API_KEY")
        if os.environ.get("AI_MODEL"):
            self.config["ai"]["model"] = os.environ.get("AI_MODEL")
        if os.environ.get("AI_API_TYPE"):
            self.config["ai"]["api_type"] = os.environ.get("AI_API_TYPE")
        
        # Redis配置
        if os.environ.get("REDIS_HOST"):
            self.config["redis"]["host"] = os.environ.get("REDIS_HOST")
        if os.environ.get("REDIS_PORT"):
            self.config["redis"]["port"] = int(os.environ.get("REDIS_PORT"))
        if os.environ.get("REDIS_DB"):
            self.config["redis"]["db"] = int(os.environ.get("REDIS_DB"))
    
    def _merge_dict(self, dest, src):
        """合并字典"""
        for key, value in src.items():
            if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
                self._merge_dict(dest[key], value)
            else:
                dest[key] = value
    
    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split(".")
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """设置配置值"""
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

# 创建全局配置实例
config_manager = ConfigManager()
