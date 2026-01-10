#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双 Redis 数据库管理器
用于实现任务型数据和知识型数据的分离管理
"""

import redis
import json
import time
import hashlib
import logging
import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("DualRedisDB")


class DualRedisDB:
    """
    双 Redis 数据库管理器
    - DB 0: 用于表格提取任务队列（高频读写、任务状态、OCR 缓存）
    - DB 1: 用于函数库与学习知识（持久化代码、用户反馈、需求匹配）
    """
    
    # Pub/Sub 通道定义
    STATUS_CHANNEL = "table_extraction_status"
    TASK_CHANNEL = "task_updates"
    
    def __init__(self):
        """
        初始化两个 Redis 连接
        """
        # 从配置文件获取 Redis 配置
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from src.utils.config import config_manager
        
        # 获取 Redis 配置
        redis_config = config_manager.get("redis", {})
        host = redis_config.get("host", "localhost")
        port = redis_config.get("port", 6379)
        task_db = redis_config.get("task_db", 0)
        logic_db = redis_config.get("logic_db", 1)
        
        # 初始化两个连接
        self.task_conn = redis.Redis(
            host=host, port=port, db=task_db, decode_responses=True
        )
        self.logic_conn = redis.Redis(
            host=host, port=port, db=logic_db, decode_responses=True
        )
        
        # 测试连接
        try:
            self.task_conn.ping()
            self.logic_conn.ping()
            logger.info(f"双 Redis 数据库连接成功: 任务库(DB {task_db}), 函数库(DB {logic_db})")
        except Exception as e:
            logger.error(f"双 Redis 数据库连接失败: {e}")
    
    def publish_status(self, message: Dict[str, Any]):
        """
        发布状态消息到 Redis Pub/Sub 通道
        
        Args:
            message: 要发布的消息
        """
        try:
            # 使用任务库发布状态更新
            self.task_conn.publish(self.STATUS_CHANNEL, json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发布状态消息失败: {e}")
    
    def publish_task_update(self, task_id: str, status: str, **kwargs):
        """
        发布任务更新消息
        
        Args:
            task_id: 任务ID
            status: 任务状态
            **kwargs: 其他任务相关信息
        """
        try:
            message = {
                "task_id": task_id,
                "status": status,
                "timestamp": time.time(),
                **kwargs
            }
            self.task_conn.publish(self.TASK_CHANNEL, json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发布任务更新消息失败: {e}")
    
    def get_pubsub(self):
        """
        获取 Redis Pub/Sub 对象
        
        Returns:
            redis.client.PubSub: Redis Pub/Sub 对象
        """
        return self.task_conn.pubsub()
    
    # --- 逻辑库 (原 QwenDB 功能) ---    
    def save_function(self, func_name: str, code: str, context: Dict[str, Any]):
        """
        保存函数到逻辑库
        
        Args:
            func_name: 函数名称
            code: 函数代码
            context: 函数上下文信息
            
        Returns:
            func_id: 生成的函数 ID，保存失败返回 None
        """
        from src.utils.ast_security_checker import ASTSecurityChecker
        
        # 初始化AST安全检查器
        security_checker = ASTSecurityChecker()
        
        # 检查代码安全性
        security_report = security_checker.get_security_report(code)
        if not security_report["is_safe"]:
            # 记录安全问题
            logger.warning(f"函数 {func_name} 包含安全问题，拒绝保存:")
            for issue in security_report["issues"]:
                logger.warning(f"  行 {issue['line']} [{issue['risk']}]: {issue['desc']} - {issue['code']}")
            return None
        
        func_id = f"fn:{int(time.time()*1000)}"
        data = {
            "name": func_name,
            "code": code,
            "context": json.dumps(context, ensure_ascii=False),
            "created_at": time.ctime(),
            "score": 80,  # 初始评分
            "use_count": 0,
            "weight": 1,  # 初始权重，用于RLHF
            "is_golden": 1
        }
        self.logic_conn.hset(f"qwen:funcs:{func_id}", mapping=data)
        self.logic_conn.lpush("qwen:func_list", func_id)
        
        # 为需求关键词创建索引
        if context.get("requirements"):
            for keyword in context["requirements"].split():
                self.logic_conn.sadd(f"qwen:keywords:{keyword}", func_id)
        
        logger.info(f"函数已保存到 Redis 逻辑库: {func_name}")
        return func_id
    
    def get_function(self, func_id: str) -> Optional[Dict[str, Any]]:
        """
        获取函数详情
        
        Args:
            func_id: 函数 ID
            
        Returns:
            函数详情字典，不存在则返回 None
        """
        data = self.logic_conn.hgetall(f"qwen:funcs:{func_id}")
        if data:
            # 解析 JSON 字段
            if "context" in data:
                data["context"] = json.loads(data["context"])
            return data
        return None
    
    def get_functions_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        根据关键词获取函数
        
        Args:
            keyword: 关键词
            limit: 返回数量限制
            
        Returns:
            函数列表
        """
        func_ids = self.logic_conn.smembers(f"qwen:keywords:{keyword}")
        functions = []
        
        for func_id in func_ids:
            func_data = self.get_function(func_id)
            if func_data:
                functions.append(func_data)
                if len(functions) >= limit:
                    break
        
        # 按权重、评分和使用次数排序，优先使用权重高的函数（RLHF）
        functions.sort(key=lambda x: (int(x.get("weight", 1)), int(x.get("score", 0)), int(x.get("use_count", 0))), reverse=True)
        return functions
    
    def update_function_score(self, func_id: str, delta: int):
        """
        更新函数评分
        
        Args:
            func_id: 函数 ID
            delta: 评分变化值
        """
        current_score = int(self.logic_conn.hget(f"qwen:funcs:{func_id}", "score") or 80)
        new_score = max(0, min(100, current_score + delta))
        self.logic_conn.hset(f"qwen:funcs:{func_id}", "score", new_score)
        
        # 更新黄金标志
        if new_score >= 80:
            self.logic_conn.hset(f"qwen:funcs:{func_id}", "is_golden", 1)
        else:
            self.logic_conn.hset(f"qwen:funcs:{func_id}", "is_golden", 0)
        
        logger.info(f"函数评分已更新: {func_id}, 变化值: {delta}, 新评分: {new_score}")
    
    def increment_function_use_count(self, func_id: str):
        """
        增加函数使用次数
        
        Args:
            func_id: 函数 ID
        """
        self.logic_conn.hincrby(f"qwen:funcs:{func_id}", "use_count", 1)
    
    def update_function_weight(self, func_id: str, delta: int = 1):
        """
        更新函数权重（用于人类反馈强化学习 RLHF）
        
        Args:
            func_id: 函数 ID
            delta: 权重变化值，默认+1（当用户点击"结果正确"时调用）
        """
        self.logic_conn.hincrby(f"qwen:funcs:{func_id}", "weight", delta)
        logger.info(f"函数权重已更新: {func_id}, 变化值: {delta}")
    
    def save_data(self, data_type: str, content: Any):
        """
        保存数据到逻辑库（兼容原 QwenDB.save_data 方法）
        
        Args:
            data_type: 数据类型
            content: 数据内容
            
        Returns:
            bool: 保存成功返回 True
        """
        try:
            # 清理数据，确保所有类型都能被 JSON 序列化
            def clean_data(data):
                if isinstance(data, dict):
                    return {key: clean_data(value) for key, value in data.items()}
                elif isinstance(data, list):
                    return [clean_data(item) for item in data]
                elif isinstance(data, (datetime.datetime, datetime.date)):
                    return data.isoformat()
                elif isinstance(data, (int, float, str, bool, type(None))):
                    return data
                else:
                    return str(data)
            
            cleaned_content = clean_data(content)
            content_json = json.dumps(cleaned_content, ensure_ascii=False)
            
            # 计算内容哈希值
            content_hash = hashlib.md5(content_json.encode('utf-8')).hexdigest()
            
            data_id = f"data:{int(time.time()*1000)}"
            data = {
                "content": content_json,
                "data_type": data_type,
                "created_at": time.ctime(),
                "content_hash": content_hash
            }
            
            self.logic_conn.hset(f"qwen:data:{data_id}", mapping=data)
            self.logic_conn.lpush(f"qwen:data_list:{data_type}", data_id)
            
            logger.info(f"数据已保存到 Redis 逻辑库: {data_type}")
            return True
        except Exception as e:
            logger.error(f"保存数据到 Redis 逻辑库失败: {e}")
            return False
    
    def save_batch_data(self, batch_data: List[Dict[str, Any]]):
        """
        批量保存数据到逻辑库（兼容原 QwenDB.save_batch_data 方法）
        
        Args:
            batch_data: 数据列表，每个元素包含 data_type 和 content 字段
            
        Returns:
            bool: 保存成功返回 True
        """
        try:
            success_count = 0
            for data in batch_data:
                data_type = data.get("data_type")
                content = data.get("content")
                if data_type and content:
                    if self.save_data(data_type, content):
                        success_count += 1
            
            logger.info(f"批量数据保存成功: {success_count}/{len(batch_data)} 条数据已保存到 Redis 逻辑库")
            return success_count > 0
        except Exception as e:
            logger.error(f"批量保存数据到 Redis 逻辑库失败: {e}")
            return False
    
    def get_功勋代码(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取功勋代码列表（兼容原 QwenDB.get_功勋代码 方法）
        
        Args:
            limit: 返回数量限制
            
        Returns:
            功勋代码列表
        """
        # 从逻辑库获取高评分、高使用次数的函数
        func_ids = self.logic_conn.lrange("qwen:func_list", 0, limit * 2)  # 获取更多数据，然后筛选
        functions = []
        
        for func_id in func_ids:
            func_data = self.get_function(func_id)
            if func_data:
                # 只返回评分 >= 60 的函数
                if int(func_data.get("score", 0)) >= 60:
                    functions.append(func_data)
                    if len(functions) >= limit:
                        break
        
        # 按权重、评分和使用次数排序，优先使用权重高的函数（RLHF）
        functions.sort(key=lambda x: (int(x.get("weight", 1)), int(x.get("score", 0)), int(x.get("use_count", 0))), reverse=True)
        return functions
    
    def update_score(self, func_id: str, delta: int):
        """
        更新函数评分（兼容原 QwenDB.update_score 方法）
        
        Args:
            func_id: 函数 ID
            delta: 评分变化值
        """
        return self.update_function_score(func_id, delta)
    
    # --- 任务库 (原表格提取功能) ---    
    def add_task(self, image_path: str, priority: int = 5) -> str:
        """
        添加任务到优先级队列
        
        Args:
            image_path: 图片路径
            priority: 任务优先级，范围 1-10，数字越小优先级越高
            
        Returns:
            task_id: 生成的任务 ID
        """
        # 确保优先级在有效范围内
        priority = max(1, min(10, priority))
        
        task_id = f"task:{hash(image_path)}"
        
        # 使用有序集合实现优先级队列
        # 优先级分数 = 优先级 * -1（因为zrangebyscore默认从小到大排序）
        score = -priority
        self.task_conn.zadd("table_extraction_tasks_priority", {image_path: score})
        
        self.task_conn.hset(f"status:{task_id}", "state", "pending")
        self.task_conn.hset(f"status:{task_id}", "created_at", time.ctime())
        self.task_conn.hset(f"status:{task_id}", "priority", priority)
        
        # 更新状态统计
        self.task_conn.hincrby("table_extraction_status", "total", 1)
        self.task_conn.hincrby("table_extraction_status", "pending", 1)
        
        logger.info(f"任务已添加到 Redis 优先级队列: {image_path}, 优先级: {priority}")
        return task_id
    
    def update_task_status(self, task_id: str, state: str):
        """
        更新任务状态
        
        Args:
            task_id: 任务 ID
            state: 任务状态 (pending, processing, completed, failed)
        """
        current_state = self.task_conn.hget(f"status:{task_id}", "state")
        
        # 更新状态哈希
        self.task_conn.hset(f"status:{task_id}", "state", state)
        self.task_conn.hset(f"status:{task_id}", "updated_at", time.ctime())
        
        # 更新状态统计
        if current_state != state:
            if current_state:
                self.task_conn.hincrby("table_extraction_status", current_state, -1)
            self.task_conn.hincrby("table_extraction_status", state, 1)
        
        logger.info(f"任务状态已更新: {task_id} -> {state}")
        
        # 发布任务状态更新
        task_status = self.get_task_status(task_id)
        self.publish_task_update(task_id, state, details=task_status)
        
        # 发布整体状态更新
        status_summary = self.get_status_summary()
        self.publish_status({
            "type": "STATUS_UPDATE",
            "data": status_summary,
            "timestamp": time.time()
        })
    
    def cache_ocr_result(self, image_hash: str, html_content: str):
        """
        缓存 OCR 结果
        
        Args:
            image_hash: 图片哈希值
            html_content: HTML 内容
        """
        # 设置过期时间（24小时）
        self.task_conn.setex(f"ocr_cache:{image_hash}", 86400, html_content)
        logger.info(f"OCR 结果已缓存到 Redis 任务库: {image_hash}")
    
    def get_ocr_cache(self, image_hash: str) -> Optional[str]:
        """
        获取 OCR 缓存
        
        Args:
            image_hash: 图片哈希值
            
        Returns:
            HTML 内容，不存在则返回 None
        """
        return self.task_conn.get(f"ocr_cache:{image_hash}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务状态字典，不存在则返回 None
        """
        return self.task_conn.hgetall(f"status:{task_id}")
    
    def get_status_summary(self) -> Dict[str, int]:
        """
        获取状态汇总
        
        Returns:
            状态汇总字典
        """
        status = self.task_conn.hgetall("table_extraction_status")
        return {
            "total": int(status.get("total", 0)),
            "pending": int(status.get("pending", 0)),
            "processing": int(status.get("processing", 0)),
            "completed": int(status.get("completed", 0)),
            "failed": int(status.get("failed", 0))
        }
    
    def get_task_from_priority_queue(self) -> Optional[str]:
        """
        从优先级队列获取下一个任务
        
        Returns:
            str: 任务数据，格式为 "img_path#img_hash"，如果队列为空返回 None
        """
        # 从有序集合中获取最高优先级的任务（分数最小的）
        # ZRANGEBYSCORE 返回分数在指定范围内的元素，按分数从小到大排序
        # 取第一个元素，即优先级最高的任务
        tasks = self.task_conn.zrangebyscore("table_extraction_tasks_priority", "-inf", "+inf", start=0, num=1)
        
        if tasks:
            task_data = tasks[0]
            # 从有序集合中移除该任务
            self.task_conn.zrem("table_extraction_tasks_priority", task_data)
            return task_data
        
        return None
    
    def add_task_to_queue(self, task_data: str, priority: int = 5):
        """
        添加任务到优先级队列，用于直接添加任务数据
        
        Args:
            task_data: 任务数据，格式为 "img_path#img_hash"
            priority: 任务优先级，范围 1-10，数字越小优先级越高
        """
        # 确保优先级在有效范围内
        priority = max(1, min(10, priority))
        
        # 使用有序集合实现优先级队列
        # 优先级分数 = 优先级 * -1（因为zrangebyscore默认从小到大排序）
        score = -priority
        self.task_conn.zadd("table_extraction_tasks_priority", {task_data: score})
    
    def clear_task_queue(self):
        """
        清空任务队列
        """
        self.task_conn.delete("table_queue")
        self.task_conn.delete("table_extraction_status")
        
        # 删除所有状态哈希
        keys = self.task_conn.keys("status:*")
        if keys:
            self.task_conn.delete(*keys)
        
        # 删除所有 OCR 缓存
        cache_keys = self.task_conn.keys("ocr_cache:*")
        if cache_keys:
            self.task_conn.delete(*cache_keys)
        
        logger.info("Redis 任务库已清空")
    
    def clear_logic_db(self):
        """
        清空逻辑库
        """
        keys = self.logic_conn.keys("qwen:*")
        if keys:
            self.logic_conn.delete(*keys)
        logger.info("Redis 逻辑库已清空")
    
    def migrate_from_sqlite(self, sqlite_db_path: str):
        """
        从 SQLite 数据库迁移旧数据到 Redis 逻辑库
        
        Args:
            sqlite_db_path: SQLite 数据库文件路径
        """
        try:
            import sqlite3
            logger.info(f"开始从 SQLite 数据库迁移数据: {sqlite_db_path}")
            
            # 连接旧的 SQLite 数据库
            conn = sqlite3.connect(sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"找到 {len(tables)} 个表: {[table['name'] for table in tables]}")
            
            # 迁移 qwen_ai_data 表（主要存储函数和历史数据）
            cursor.execute("SELECT * FROM qwen_ai_data;")
            rows = cursor.fetchall()
            logger.info(f"找到 {len(rows)} 条数据在 qwen_ai_data 表中")
            
            migrated_count = 0
            for row in rows:
                try:
                    data_id = row['id']
                    data_type = row['data_type']
                    content_json = row['content']
                    created_at = row['created_at']
                    updated_at = row['updated_at']
                    
                    # 解析内容
                    content = json.loads(content_json)
                    
                    # 根据数据类型进行迁移
                    if data_type == 'history_functions':
                        # 迁移函数数据
                        functions = content.get('functions', [])
                        for func in functions:
                            func_name = func.get('name', f"function_{data_id}")
                            code = func.get('implementation', '')
                            context = {
                                'prompt': content.get('prompt', ''),
                                'enhanced_prompt': content.get('enhanced_prompt', ''),
                                'data_context': content.get('data_context', {}),
                                'created_at': created_at
                            }
                            
                            if code:  # 只迁移有代码的函数
                                self.save_function(func_name, code, context)
                                migrated_count += 1
                    else:
                        # 迁移其他类型数据
                        self.save_data(data_type, content)
                        migrated_count += 1
                except Exception as e:
                    logger.error(f"迁移第 {row['id']} 条数据失败: {e}")
                    continue
            
            conn.close()
            logger.info(f"数据迁移完成: 共迁移 {migrated_count}/{len(rows)} 条数据到 Redis 逻辑库")
            return True
        except Exception as e:
            logger.error(f"从 SQLite 数据库迁移数据失败: {e}")
            return False
