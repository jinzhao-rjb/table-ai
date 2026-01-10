#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测量风险文件名处理的执行时间
"""

import time
import random
import os

# 模拟风险文件名处理的代码
def process_risk_filename(img_path, risk_warning, risk_reasons):
    """
    模拟风险文件名处理
    """
    base_name = os.path.basename(img_path)
    file_name = os.path.splitext(base_name)[0] + "_table.xlsx"
    if risk_warning:
        file_name = os.path.splitext(base_name)[0] + "_table_RISK.xlsx"
        # 模拟日志记录
        warning_message = f"风险预警: {img_path} - {'; '.join(risk_reasons)}"
    
    return file_name

# 测量执行时间
def measure_execution_time():
    """
    测量风险文件名处理的执行时间
    """
    # 生成测试数据
    test_cases = 100000  # 测试10万次
    
    # 生成模拟数据
    img_paths = [f"input_images/image_{i}.jpg" for i in range(test_cases)]
    risk_warnings = [random.random() > 0.7 for _ in range(test_cases)]  # 30%的概率有风险
    risk_reasons_list = [
        ["YOLO置信度过低 (0.65)", "HTML不完整（缺少</table>结尾）"] if risk else []
        for risk in risk_warnings
    ]
    
    # 开始测量
    start_time = time.time()
    
    # 执行风险文件名处理
    for i in range(test_cases):
        process_risk_filename(img_paths[i], risk_warnings[i], risk_reasons_list[i])
    
    # 结束测量
    end_time = time.time()
    
    # 计算结果
    total_time = end_time - start_time
    avg_time_per_process = total_time / test_cases
    
    # 输出结果
    print(f"=== 风险文件名处理执行时间测量 ===")
    print(f"测试次数: {test_cases}")
    print(f"总执行时间: {total_time:.6f} 秒")
    print(f"每次处理平均执行时间: {avg_time_per_process:.9f} 秒")
    print(f"每次处理平均执行时间: {avg_time_per_process * 1000:.6f} 毫秒")
    print(f"每次处理平均执行时间: {avg_time_per_process * 1000000:.3f} 微秒")
    
    # 计算对整体处理时间的影响
    total_tasks = 232  # 原有任务数
    total_risk_filename_time = total_tasks * avg_time_per_process
    print(f"\n对原有任务的影响:")
    print(f"总任务数: {total_tasks}")
    print(f"风险文件名处理总耗时: {total_risk_filename_time:.6f} 秒")
    print(f"风险文件名处理总耗时: {total_risk_filename_time * 1000:.3f} 毫秒")
    
    # 结论
    print(f"\n=== 结论 ===")
    print(f"风险文件名处理的执行时间非常短，平均每次处理仅需要 {avg_time_per_process * 1000:.3f} 毫秒")
    print(f"对于232个任务，风险文件名处理总耗时仅为 {total_risk_filename_time * 1000:.3f} 毫秒")
    print(f"这个时间可以忽略不计，对整体处理时间几乎没有影响")
    print(f"\n注意：'风险表格生成' 实际上只是在文件名中添加了 '_RISK' 后缀，并没有额外的表格生成过程")

if __name__ == "__main__":
    measure_execution_time()