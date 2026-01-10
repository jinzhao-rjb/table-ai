#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控 Redis 中的任务状态
"""

import redis
import time
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    主函数：实时监控 Redis 中的任务状态
    """
    # 连接 Redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    logger.info("开始监控 Redis 任务状态...")
    logger.info("按 Ctrl+C 停止监控")
    logger.info("=" * 60)
    
    try:
        while True:
            # 获取任务状态
            status = r.hgetall('task_status')
            
            # 提取状态信息
            total = int(status.get('total', 0))
            success = int(status.get('success', 0))
            failed = int(status.get('failed', 0))
            pending = int(status.get('pending', 0))
            
            # 计算百分比
            if total > 0:
                success_pct = (success / total) * 100
                failed_pct = (failed / total) * 100
                pending_pct = (pending / total) * 100
            else:
                success_pct = 0
                failed_pct = 0
                pending_pct = 0
            
            # 清除屏幕并显示状态
            print("\033c", end="")  # 清除屏幕
            print("=" * 60)
            print("实时任务状态监控")
            print("=" * 60)
            print(f"总计任务数: {total}")
            print(f"成功: {success} ({success_pct:.1f}%)")
            print(f"失败: {failed} ({failed_pct:.1f}%)")
            print(f"待处理: {pending} ({pending_pct:.1f}%)")
            print("=" * 60)
            print(f"状态: Total: {total} | Success: {success} | Failed: {failed} | Pending: {pending}")
            print("=" * 60)
            print("更新时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("按 Ctrl+C 停止监控")
            
            # 每秒更新一次
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("监控已停止")
    except Exception as e:
        logger.error(f"监控时发生异常: {e}")

if __name__ == "__main__":
    main()