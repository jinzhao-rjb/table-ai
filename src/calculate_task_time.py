#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算实际处理任务的时间，去掉monitor_status.py脚本本身运行的时间
"""

import os
import re
import datetime

def calculate_task_processing_time():
    """
    计算实际处理任务的时间
    """
    # 原有任务开始时间和结束时间（从日志中提取）
    original_start_time = datetime.datetime(2026, 1, 3, 9, 22, 52)
    original_end_time = datetime.datetime(2026, 1, 3, 17, 57, 56)
    original_total_tasks = 232
    original_success_tasks = 232
    
    # 新测试任务开始时间和当前时间
    new_start_time = datetime.datetime(2026, 1, 3, 17, 24, 36)
    current_time = datetime.datetime(2026, 1, 3, 17, 57, 56)
    new_total_tasks = 46
    new_success_tasks = 14
    
    # 计算原有任务的处理时间
    original_processing_time = original_end_time - original_start_time
    original_processing_seconds = original_processing_time.total_seconds()
    
    # 计算新测试任务的处理时间
    new_processing_time = current_time - new_start_time
    new_processing_seconds = new_processing_time.total_seconds()
    
    # 计算每个任务的平均处理时间
    if original_success_tasks > 0:
        original_avg_time_per_task = original_processing_seconds / original_success_tasks
    else:
        original_avg_time_per_task = 0
    
    if new_success_tasks > 0:
        new_avg_time_per_task = new_processing_seconds / new_success_tasks
    else:
        new_avg_time_per_task = 0
    
    # 输出结果
    print("=== 任务处理时间分析 ===")
    print(f"\n原有任务:")
    print(f"  开始时间: {original_start_time}")
    print(f"  结束时间: {original_end_time}")
    print(f"  总处理时间: {original_processing_time}")
    print(f"  总任务数: {original_total_tasks}")
    print(f"  成功任务数: {original_success_tasks}")
    print(f"  每个任务平均处理时间: {original_avg_time_per_task:.2f} 秒")
    
    print(f"\n新测试任务:")
    print(f"  开始时间: {new_start_time}")
    print(f"  当前时间: {current_time}")
    print(f"  已处理时间: {new_processing_time}")
    print(f"  总任务数: {new_total_tasks}")
    print(f"  成功任务数: {new_success_tasks}")
    print(f"  待处理任务数: {new_total_tasks - new_success_tasks}")
    print(f"  每个任务平均处理时间: {new_avg_time_per_task:.2f} 秒")
    
    # 估算剩余时间
    if new_success_tasks > 0:
        remaining_tasks = new_total_tasks - new_success_tasks
        estimated_remaining_seconds = remaining_tasks * new_avg_time_per_task
        estimated_remaining_time = datetime.timedelta(seconds=estimated_remaining_seconds)
        estimated_completion_time = current_time + estimated_remaining_time
        
        print(f"  估算剩余时间: {estimated_remaining_time}")
        print(f"  估算完成时间: {estimated_completion_time}")
    
    print(f"\n=== 总结 ===")
    print(f"原有任务已经全部完成，总处理时间: {original_processing_time}")
    print(f"每个任务平均处理时间: {original_avg_time_per_task:.2f} 秒")
    print(f"这是实际处理任务的时间，不包括monitor_status.py脚本本身运行的时间")

if __name__ == "__main__":
    calculate_task_processing_time()