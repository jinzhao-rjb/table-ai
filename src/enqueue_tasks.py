#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将图片路径和哈希值存入 Redis 的 LIST 结构中
"""

import os
import hashlib
import redis
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_image_hash(img_path):
    """
    计算图片的 MD5 哈希值
    
    Args:
        img_path: 图片路径
    
    Returns:
        str: 图片的 MD5 哈希值
    """
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            img_hash = hashlib.md5(img_bytes).hexdigest()
        return img_hash
    except Exception as e:
        logger.error(f"计算图片哈希值失败: {img_path}, 错误: {e}")
        return None

def main():
    """
    主函数：将图片路径和哈希值存入 Redis
    """
    # 设置输入路径
    input_folder = "input_images"  # 替换为你的输入文件夹路径
    
    # 连接 Redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # 清空现有的任务队列和结果缓存（可选）
    r.delete('image_tasks')
    r.delete('result_cache')
    r.delete('task_status')
    
    # 获取图片列表
    image_paths = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(input_folder, file))
    
    # 限制为 300 张图片
    image_paths = image_paths[:300]
    
    logger.info(f"找到 {len(image_paths)} 张图片，将存入 Redis")
    
    # 将图片路径和哈希值存入 Redis 的 LIST 结构
    for img_path in image_paths:
        img_hash = calculate_image_hash(img_path)
        if img_hash:
            # 构造任务数据
            task_data = f"{img_path}#{img_hash}"
            # 将任务存入 Redis LIST
            r.rpush('image_tasks', task_data)
            logger.info(f"任务已入队: {img_path} - {img_hash}")
    
    # 设置初始任务状态
    total = len(image_paths)
    r.hset('task_status', 'total', total)
    r.hset('task_status', 'success', 0)
    r.hset('task_status', 'failed', 0)
    r.hset('task_status', 'pending', total)
    
    logger.info(f"任务入队完成！总共 {total} 个任务")
    logger.info(f"Redis 键：")
    logger.info(f"- image_tasks: 任务队列（LIST）")
    logger.info(f"- result_cache: 结果缓存（HASH）")
    logger.info(f"- task_status: 任务状态（HASH）")

if __name__ == "__main__":
    main()