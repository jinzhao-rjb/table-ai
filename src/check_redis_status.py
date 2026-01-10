#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 Redis 中的任务状态
"""

import redis
import datetime

# 连接 Redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # 检查原有任务状态
    print("=== 原有任务状态 ===")
    task_status = r.hgetall('task_status')
    if task_status:
        total = task_status.get('total', 0)
        success = task_status.get('success', 0)
        failed = task_status.get('failed', 0)
        pending = task_status.get('pending', 0)
        
        print(f"总计: {total}")
        print(f"成功: {success}")
        print(f"失败: {failed}")
        print(f"待处理: {pending}")
    else:
        print("没有找到原有任务状态")
    
    # 检查新测试任务状态
    print("\n=== 新测试任务状态 ===")
    database_test_status = r.hgetall('database_test_task_status')
    if database_test_status:
        total = database_test_status.get('total', 0)
        success = database_test_status.get('success', 0)
        failed = database_test_status.get('failed', 0)
        pending = database_test_status.get('pending', 0)
        
        print(f"总计: {total}")
        print(f"成功: {success}")
        print(f"失败: {failed}")
        print(f"待处理: {pending}")
    else:
        print("没有找到新测试任务状态")
    
    # 检查当前时间
    print(f"\n当前时间: {datetime.datetime.now()}")
    
except Exception as e:
    print(f"连接 Redis 失败: {e}")
