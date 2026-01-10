#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据迁移脚本：将SQLite数据库中的数据迁移到Redis中
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.dual_redis_db import DualRedisDB
from src.utils.qwen_db import QwenDB

def main():
    """
    主函数：执行数据迁移
    """
    print("开始数据迁移：SQLite -> Redis")
    
    # 获取SQLite数据库文件路径
    qwen_db = QwenDB()
    sqlite_db_path = qwen_db.db_file
    print(f"SQLite数据库路径：{sqlite_db_path}")
    
    # 检查SQLite数据库是否存在
    if not os.path.exists(sqlite_db_path):
        print(f"错误：SQLite数据库文件不存在：{sqlite_db_path}")
        return
    
    # 初始化DualRedisDB
    dual_redis = DualRedisDB()
    
    # 执行数据迁移
    print("正在执行数据迁移...")
    success = dual_redis.migrate_from_sqlite(sqlite_db_path)
    
    if success:
        print("数据迁移成功！")
    else:
        print("数据迁移失败！")

if __name__ == "__main__":
    main()
