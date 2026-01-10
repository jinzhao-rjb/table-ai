#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测量风险预警检查代码的执行时间
"""

import time
import random

# 模拟风险预警检查的代码
def risk_warning_check(yolo_confidence, html_content):
    """
    模拟风险预警检查
    """
    risk_warning = False
    risk_reasons = []
    
    # 检查YOLO置信度
    if yolo_confidence < 0.7:
        risk_warning = True
        risk_reasons.append(f"YOLO置信度过低 ({yolo_confidence:.2f})")
    
    # 检查HTML完整性
    if '</table>' not in html_content.lower():
        risk_warning = True
        risk_reasons.append("HTML不完整（缺少</table>结尾）")
    
    return risk_warning, risk_reasons

# 测量执行时间
def measure_execution_time():
    """
    测量风险预警检查的执行时间
    """
    # 生成测试数据
    test_cases = 100000  # 测试10万次
    
    # 生成模拟数据
    yolo_confidences = [random.uniform(0.5, 1.0) for _ in range(test_cases)]
    html_contents = [
        '<table><tr><td>test</td></tr></table>' if random.random() > 0.1 else '<table><tr><td>test</td></tr>'
        for _ in range(test_cases)
    ]
    
    # 开始测量
    start_time = time.time()
    
    # 执行风险预警检查
    for i in range(test_cases):
        risk_warning_check(yolo_confidences[i], html_contents[i])
    
    # 结束测量
    end_time = time.time()
    
    # 计算结果
    total_time = end_time - start_time
    avg_time_per_check = total_time / test_cases
    
    # 输出结果
    print(f"=== 风险预警检查执行时间测量 ===")
    print(f"测试次数: {test_cases}")
    print(f"总执行时间: {total_time:.6f} 秒")
    print(f"每次检查平均执行时间: {avg_time_per_check:.9f} 秒")
    print(f"每次检查平均执行时间: {avg_time_per_check * 1000:.6f} 毫秒")
    print(f"每次检查平均执行时间: {avg_time_per_check * 1000000:.3f} 微秒")
    
    # 计算对整体处理时间的影响
    total_tasks = 232  # 原有任务数
    total_risk_check_time = total_tasks * avg_time_per_check
    print(f"\n对原有任务的影响:")
    print(f"总任务数: {total_tasks}")
    print(f"风险检查总耗时: {total_risk_check_time:.6f} 秒")
    print(f"风险检查总耗时: {total_risk_check_time * 1000:.3f} 毫秒")
    
    # 结论
    print(f"\n=== 结论 ===")
    print(f"风险预警检查的执行时间非常短，平均每次检查仅需要 {avg_time_per_check * 1000:.3f} 毫秒")
    print(f"对于232个任务，风险检查总耗时仅为 {total_risk_check_time * 1000:.3f} 毫秒")
    print(f"这个时间可以忽略不计，对整体处理时间几乎没有影响")

if __name__ == "__main__":
    measure_execution_time()