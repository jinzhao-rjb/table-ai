#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis任务监控脚本
用于实时监控表格提取任务的进度
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
import redis
from src.utils.config import config_manager


def main():
    """
    主函数
    """
    # 连接Redis
    redis_host = config_manager.get("redis.host", "localhost")
    redis_port = config_manager.get("redis.port", 6379)
    redis_db = config_manager.get("redis.db", 0)
    
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        # 测试连接
        r.ping()
        print(f"成功连接到Redis: {redis_host}:{redis_port}/{redis_db}")
    except Exception as e:
        print(f"连接Redis失败: {e}")
        return
    
    print("\n开始监控表格提取任务进度...")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            # 获取任务状态
            status = r.hgetall('table_extraction_status')
            
            # 确保所有状态字段都存在
            total = status.get('total', '0')
            success = status.get('success', '0')
            failed = status.get('failed', '0')
            pending = status.get('pending', '0')
            
            # 获取队列长度
            queue_length = r.llen('table_extraction_tasks')
            processing_length = r.llen('table_extraction_processing')
            
            # 计算成功率
            total_int = int(total)
            success_int = int(success)
            failed_int = int(failed)
            
            success_rate = 0.0
            if total_int > 0:
                success_rate = (success_int / total_int) * 100
            
            # 清除屏幕并打印进度
            print("\033c", end="")  # 清除屏幕
            print("表格提取任务监控")
            print("=" * 40)
            print(f"总任务数: {total}")
            print(f"成功: {success} ({success_rate:.1f}%)")
            print(f"失败: {failed}")
            print(f"待处理: {pending}")
            print(f"队列中: {queue_length}")
            print(f"正在处理: {processing_length}")
            print("=" * 40)
            print(f"活跃任务: {processing_length}")
            print(f"剩余任务: {queue_length + processing_length}")
            print("\n按 Ctrl+C 停止监控")
            
            # 等待1秒后刷新
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n监控已停止")


if __name__ == "__main__":
    main()