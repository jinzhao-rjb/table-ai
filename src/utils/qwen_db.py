#!/usr/bin/env python3
"""
千问AI服务的数据库连接和操作模块
使用SQLite嵌入式数据库，无需启动服务，无账号密码
"""

import logging
import json
import datetime
import sqlite3
import threading
from typing import List, Dict, Any, Optional

# 导入配置管理器
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.config import config_manager

logger = logging.getLogger("QwenDB")


class QwenDB:
    """
    千问AI服务的数据库连接和操作类（SQLite版本）
    """
    
    def __init__(self):
        """
        初始化数据库连接
        """
        # SQLite数据库文件路径
        self.db_dir = os.path.join(os.path.expanduser("~"), ".office-lazy-tool")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_file = os.path.join(self.db_dir, "qwen_ai.db")
        
        # 添加线程锁，确保线程安全
        self._lock = threading.Lock()
        
        # 自动创建表，添加错误处理
        try:
            self._create_table()
        except Exception as e:
            logger.error(f"千问AI数据库初始化失败: {e}")
            # 初始化失败不影响应用运行，后续使用时会再次尝试连接
            pass
    
    def execute_query(self, sql, params=None):
        """
        执行SQL查询，实现自动重连和上下文管理
        
        Args:
            sql: SQL语句
            params: SQL参数
            
        Returns:
            查询结果，如果执行失败返回None
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params or ())
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return result
        
        try:
            return self._safe_execute(_execute)
        except Exception as e:
            logger.error(f"执行SQL失败: {e}")
            logger.error(f"SQL: {sql}")
            if params:
                logger.error(f"Params: {params}")
            return None

    def _get_connection(self):
        """
        获取SQLite数据库连接
        
        Returns:
            可用的数据库连接
        """
        # SQLite连接非常轻量，每次需要时创建新连接
        conn = sqlite3.connect(
            self.db_file,
            timeout=5,  # 防止并发写入冲突
            isolation_level=None,  # 自动提交模式
            check_same_thread=False  # 允许跨线程使用
        )
        # 设置游标返回字典格式
        conn.row_factory = sqlite3.Row
        return conn
    
    def _safe_execute(self, func, *args, **kwargs):
        """
        安全执行数据库操作，确保连接有效
        
        Args:
            func: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        with self._lock:
            for attempt in range(3):  # 最多尝试3次
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"数据库执行失败 (尝试 {attempt+1}/3): {e}")
                    # 等待一段时间后重试
                    import time
                    time.sleep(0.5)
                    if attempt == 2:  # 最后一次尝试失败
                        raise
        return None
    
    def _close(self):
        """
        关闭数据库连接（SQLite连接会在with块结束时自动关闭）
        """
        pass
    
    def _create_table(self):
        """
        创建千问AI数据表
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 先创建表的基础结构，不包含可能缺失的字段
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS qwen_ai_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # 检查并添加缺失的字段（兼容旧版本表结构）
            cursor.execute("PRAGMA table_info(qwen_ai_data)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # 需要添加的字段列表
            fields_to_add = [
                ("prompt_hash", "VARCHAR(32)"),
                ("is_golden", "INTEGER NOT NULL DEFAULT 0"),
                ("logic_tag", "VARCHAR(50)"),
                ("score", "INTEGER NOT NULL DEFAULT 0"),
                ("use_count", "INTEGER NOT NULL DEFAULT 0")
            ]
            
            # 逐个添加缺失的字段
            for field_name, field_type in fields_to_add:
                if field_name not in columns:
                    cursor.execute(f"ALTER TABLE qwen_ai_data ADD COLUMN {field_name} {field_type}")
                    logger.info(f"添加缺失字段: {field_name} {field_type}")
            
            # 添加向量化索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_data_type ON qwen_ai_data(data_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_type_golden ON qwen_ai_data(data_type, is_golden)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_type_created ON qwen_ai_data(data_type, created_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_prompt_hash ON qwen_ai_data(prompt_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_logic_tag ON qwen_ai_data(logic_tag)')
            
            cursor.close()
            conn.close()
            logger.info("千问AI数据表创建成功")
        except Exception as e:
            logger.error(f"千问AI数据表创建失败: {e}")
            # 初始化失败不影响应用运行，后续使用时会再次尝试
            pass
    
    def _clean_data(self, data: Any) -> Any:
        """
        清理数据，确保所有类型都能被JSON序列化
        
        Args:
            data: 需要清理的数据
            
        Returns:
            清理后的数据
        """
        if isinstance(data, dict):
            return {key: self._clean_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._clean_data(item) for item in data]
        elif isinstance(data, datetime.datetime):
            return data.isoformat()
        elif isinstance(data, datetime.date):
            return data.isoformat()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # 其他类型转换为字符串
            return str(data)
    
    def _validate_code_syntax(self, content: Dict[str, Any]) -> bool:
        """
        验证代码语法，检查是否存在未定义的变量（如pd）
        
        Args:
            content: 包含代码的字典
            
        Returns:
            语法有效返回True，否则返回False
        """
        import ast
        
        try:
            # 获取函数实现代码
            functions = content.get('functions', [])
            for func in functions:
                func_impl = func.get('implementation', '')
                if not func_impl:
                    continue
                
                # 检查是否使用了pd或np但没有导入
                uses_pandas = 'pd.' in func_impl or 'pandas' in func_impl
                uses_numpy = 'np.' in func_impl or 'numpy' in func_impl
                
                has_pandas_import = 'import pandas' in func_impl or 'import pandas as pd' in func_impl
                has_numpy_import = 'import numpy' in func_impl or 'import numpy as np' in func_impl
                
                # 检查语法正确性
                try:
                    ast.parse(func_impl)
                except SyntaxError as e:
                    logger.warning(f"代码语法错误: {e}")
                    return False
                
                # 检查是否使用了未导入的库
                if (uses_pandas and not has_pandas_import) or (uses_numpy and not has_numpy_import):
                    logger.warning(f"代码使用了未导入的库: pandas={'是' if uses_pandas else '否'}, numpy={'是' if uses_numpy else '否'}")
                    # 不立即返回False，而是在保存时注入导入语句
        
        except Exception as e:
            logger.error(f"验证代码语法失败: {e}")
            return False
        
        return True
    
    def save_data(self, data_type: str, content: Any):
        """
        保存数据到数据库，包含去重性约束和语法验证
        
        Args:
            data_type: 数据类型（history_functions, feedback_history, error_history）
            content: 数据内容
            
        Returns:
            保存成功返回True，否则返回False
        """
        import hashlib
        
        # 【新增】如果是函数数据，先自动检查并修复 import
        if data_type == "history_functions":
            content = self._auto_fix_imports(content)
        
        # 清理数据，确保所有类型都能被JSON序列化
        cleaned_content = self._clean_data(content)
        
        # 将数据转换为JSON格式
        content_json = json.dumps(cleaned_content, ensure_ascii=False)
        
        # 计算prompt内容的MD5哈希值，用于去重
        prompt_hash = hashlib.md5(content_json.encode('utf-8')).hexdigest()
        
        # 提取logic_tag和score
        logic_tag = cleaned_content.get('logic_tag', '')
        score = cleaned_content.get('score', 0)
        is_golden = cleaned_content.get('is_golden', 0)
        
        # 验证代码语法，只有通过语法检查的代码才能获得is_golden=1
        if data_type == 'history_functions' and is_golden == 1:
            if not self._validate_code_syntax(cleaned_content):
                # 语法验证失败，降级为普通数据
                is_golden = 0
                logger.warning(f"代码语法验证失败，将is_golden从1降级为0")
        
        # 根据函数执行结果调整评分
        success = cleaned_content.get('success', False)
        if not success and data_type == 'history_functions':
            # 失败的函数降低评分
            score = max(0, score - 5)
            logger.debug(f"函数执行失败，评分从 {cleaned_content.get('score', 0)} 降低到 {score}")
        
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 检查是否已存在相同的prompt_hash
            cursor.execute("SELECT id, score, use_count FROM qwen_ai_data WHERE prompt_hash = ? AND data_type = ?", (prompt_hash, data_type))
            existing = cursor.fetchone()
            
            if existing:
                # 更新现有记录，失败的函数降低评分
                new_score = score if not existing['score'] else existing['score']
                if not success and data_type == 'history_functions':
                    new_score = max(0, existing['score'] - 5)
                
                update_sql = "UPDATE qwen_ai_data SET content = ?, updated_at = CURRENT_TIMESTAMP, is_golden = ?, logic_tag = ?, score = ? WHERE id = ?"
                cursor.execute(update_sql, (content_json, is_golden, logic_tag, new_score, existing['id']))
                logger.info(f"千问AI数据已存在，更新成功，类型: {data_type}, ID: {existing['id']}")
            else:
                # 插入新记录
                insert_sql = "INSERT INTO qwen_ai_data (data_type, content, prompt_hash, is_golden, logic_tag, score, updated_at) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
                cursor.execute(insert_sql, (data_type, content_json, prompt_hash, is_golden, logic_tag, score))
                logger.info(f"千问AI数据保存成功，类型: {data_type}")
            
            cursor.close()
            conn.close()
            return True
        
        try:
            return self._safe_execute(_execute)
        except Exception as e:
            logger.error(f"千问AI数据保存失败: {e}")
            return False
    
    def save_batch_data(self, data_list: List[Dict[str, Any]]):
        """
        批量保存数据到数据库，包含去重性约束和语法验证
        
        Args:
            data_list: 数据列表，每个元素包含data_type和content字段
            
        Returns:
            保存成功返回True，否则返回False
        """
        import hashlib
        
        # 准备批量数据，先去重
        unique_data = {}
        for data in data_list:
            data_type = data.get("data_type")
            content = data.get("content")
            if data_type and content:
                # 清理数据，确保所有类型都能被JSON序列化
                cleaned_content = self._clean_data(content)
                content_json = json.dumps(cleaned_content, ensure_ascii=False)
                
                # 计算prompt_hash
                prompt_hash = hashlib.md5(content_json.encode('utf-8')).hexdigest()
                
                # 提取字段
                is_golden = cleaned_content.get('is_golden', 0)
                logic_tag = cleaned_content.get('logic_tag', '')
                score = cleaned_content.get('score', 0)
                
                # 验证代码语法，只有通过语法检查的代码才能获得is_golden=1
                if data_type == 'history_functions' and is_golden == 1:
                    if not self._validate_code_syntax(cleaned_content):
                        # 语法验证失败，降级为普通数据
                        is_golden = 0
                        logger.warning(f"代码语法验证失败，将is_golden从1降级为0")
                
                # 根据函数执行结果调整评分
                success = cleaned_content.get('success', False)
                if not success and data_type == 'history_functions':
                    # 失败的函数降低评分
                    score = max(0, score - 5)
                
                # 使用prompt_hash作为键，确保唯一性
                key = f"{data_type}:{prompt_hash}"
                unique_data[key] = {
                    "data_type": data_type,
                    "content": content_json,
                    "prompt_hash": prompt_hash,
                    "is_golden": is_golden,
                    "logic_tag": logic_tag,
                    "score": score,
                    "success": success
                }
        
        batch_data = list(unique_data.values())
        
        if not batch_data:
            return False
        
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for data in batch_data:
                # 检查是否已存在相同的prompt_hash
                cursor.execute("SELECT id, score FROM qwen_ai_data WHERE prompt_hash = ? AND data_type = ?", 
                              (data["prompt_hash"], data["data_type"]))
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有记录，失败的函数降低评分
                    new_score = data["score"]
                    if not data["success"] and data["data_type"] == 'history_functions':
                        new_score = max(0, existing['score'] - 5)
                    
                    update_sql = "UPDATE qwen_ai_data SET content = ?, updated_at = CURRENT_TIMESTAMP, is_golden = ?, logic_tag = ?, score = ? WHERE id = ?"
                    cursor.execute(update_sql, (
                        data["content"], data["is_golden"], data["logic_tag"], new_score, existing['id']))
                else:
                    # 插入新记录
                    insert_sql = "INSERT INTO qwen_ai_data (data_type, content, prompt_hash, is_golden, logic_tag, score, updated_at) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)"
                    cursor.execute(insert_sql, (
                        data["data_type"], data["content"], data["prompt_hash"], data["is_golden"], data["logic_tag"], data["score"]))
            
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            logger.info(f"千问AI批量数据保存成功，去重后数量: {len(batch_data)}")
            return result
        except Exception as e:
            logger.error(f"千问AI批量数据保存失败: {e}")
            return False
    
    def get_data(self, data_type: str, limit: int = 50, priority_golden: bool = True) -> List[Dict[str, Any]]:
        """
        从数据库获取数据，优先返回黄金数据和功勋代码
        
        Args:
            data_type: 数据类型
            limit: 返回的数据数量限制
            priority_golden: 是否优先返回黄金数据
            
        Returns:
            数据列表
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if priority_golden:
                # 强化功勋定义的权重：
                # 1. 优先返回黄金数据
                # 2. 同时考虑score和use_count，给use_count高的代码增加实战加成
                # 3. 使用score * (1 + use_count/100)作为综合排序依据
                select_sql = "SELECT * FROM qwen_ai_data WHERE data_type = ? AND is_golden = 1 ORDER BY (score * (1 + use_count/100)) DESC, use_count DESC, score DESC LIMIT ?"
                cursor.execute(select_sql, (data_type, limit))
                golden_results = cursor.fetchall()
                
                # 如果黄金数据不足，补充普通数据，同样考虑use_count
                if len(golden_results) < limit:
                    remaining = limit - len(golden_results)
                    select_sql = "SELECT * FROM qwen_ai_data WHERE data_type = ? AND is_golden = 0 ORDER BY (score * (1 + use_count/100)) DESC, use_count DESC, created_at DESC LIMIT ?"
                    cursor.execute(select_sql, (data_type, remaining))
                    golden_results.extend(cursor.fetchall())
                
                cursor.close()
                conn.close()
                return golden_results
            else:
                # 不优先黄金数据，但同样考虑use_count的实战加成
                select_sql = "SELECT * FROM qwen_ai_data WHERE data_type = ? ORDER BY (score * (1 + use_count/100)) DESC, created_at DESC LIMIT ?"
                cursor.execute(select_sql, (data_type, limit))
                result = cursor.fetchall()
                cursor.close()
                conn.close()
                return result
        
        try:
            results = self._safe_execute(_execute)
            if results is None:
                return []
            
            # 将JSON转换为Python对象
            data_list = []
            for result in results:
                try:
                    content = json.loads(result["content"])
                    data_item = {
                        "id": result["id"],
                        "data_type": result["data_type"],
                        "content": content,
                        "is_golden": result["is_golden"],
                        "logic_tag": result["logic_tag"],
                        "score": result["score"],
                        "use_count": result["use_count"],
                        "created_at": result["created_at"],
                        "updated_at": result["updated_at"]
                    }
                    data_list.append(data_item)
                    
                    # 自动增加使用次数
                    self.increment_use_count(result["id"])
                except json.JSONDecodeError as e:
                    logger.error(f"千问AI数据解析失败: {e}")
                    continue
            
            logger.info(f"千问AI数据获取成功，类型: {data_type}, 数量: {len(data_list)}")
            return data_list
        except Exception as e:
            logger.error(f"千问AI数据获取失败: {e}")
            return []
    
    def get_all_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        从数据库获取所有类型的数据
        
        Args:
            limit: 返回的数据数量限制
            
        Returns:
            数据列表
        """
        def _execute():
            select_sql = "SELECT * FROM qwen_ai_data ORDER BY created_at DESC LIMIT ?"
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(select_sql, (limit,))
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return result
        
        try:
            results = self._safe_execute(_execute)
            if results is None:
                return []
            
            # 将JSON转换为Python对象
            data_list = []
            for result in results:
                try:
                    content = json.loads(result["content"])
                    data_list.append({
                        "id": result["id"],
                        "data_type": result["data_type"],
                        "content": content,
                        "created_at": result["created_at"],
                        "updated_at": result["updated_at"]
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"千问AI数据解析失败: {e}")
                    continue
            
            logger.info(f"千问AI所有数据获取成功，数量: {len(data_list)}")
            return data_list
        except Exception as e:
            logger.error(f"千问AI所有数据获取失败: {e}")
            return []
    
    def get_latest_best(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取最新最好的history_functions记录（优化版）
        
        Args:
            limit: 返回的数据数量限制
            
        Returns:
            数据列表
        """
        def _execute():
            # 优化：只查询is_golden为true的记录，减少查询数据量
            select_sql = "SELECT * FROM qwen_ai_data WHERE data_type = 'history_functions' AND is_golden = 1 ORDER BY created_at DESC LIMIT ?"
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(select_sql, (limit,))
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return result
        
        try:
            results = self._safe_execute(_execute)
            if results is None:
                return []
            
            # 将JSON转换为Python对象，只保留content字段
            data_list = []
            for result in results:
                try:
                    content = json.loads(result["content"])
                    # 只保留成功的函数
                    if content.get("success", False):
                        data_list.append(content)
                        if len(data_list) >= limit:
                            break
                except json.JSONDecodeError as e:
                    logger.error(f"千问AI数据解析失败: {e}")
                    continue
            
            logger.info(f"千问AI获取最新最佳记录成功，数量: {len(data_list)}")
            return data_list
        except Exception as e:
            logger.error(f"千问AI获取最新最佳记录失败: {e}")
            return []
    
    def delete_data(self, data_type: str, before_date: datetime.datetime = None) -> bool:
        """
        删除指定类型的数据
        
        Args:
            data_type: 数据类型
            before_date: 删除指定日期之前的数据
            
        Returns:
            删除成功返回True，否则返回False
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            if before_date:
                delete_sql = "DELETE FROM qwen_ai_data WHERE data_type = ? AND created_at < ?"
                cursor.execute(delete_sql, (data_type, before_date))
            else:
                delete_sql = "DELETE FROM qwen_ai_data WHERE data_type = ?"
                cursor.execute(delete_sql, (data_type,))
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            logger.info(f"千问AI数据删除成功，类型: {data_type}")
            return result
        except Exception as e:
            logger.error(f"千问AI数据删除失败: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """
        清空所有数据
        
        Returns:
            清空成功返回True，否则返回False
        """
        with self._lock:
            try:
                clear_sql = "DELETE FROM qwen_ai_data"
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(clear_sql)
                cursor.close()
                conn.close()
                
                logger.info("千问AI所有数据清空成功")
                return True
            except Exception as e:
                logger.error(f"千问AI所有数据清空失败: {e}")
                return False
    
    def get_data_count(self, data_type: str = None) -> int:
        """
        获取数据数量
        
        Args:
            data_type: 数据类型，为空则获取所有类型数据数量
            
        Returns:
            数据数量
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            if data_type:
                count_sql = "SELECT COUNT(*) as count FROM qwen_ai_data WHERE data_type = ?"
                cursor.execute(count_sql, (data_type,))
                result = cursor.fetchone()
            else:
                count_sql = "SELECT COUNT(*) as count FROM qwen_ai_data"
                cursor.execute(count_sql)
                result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result
        
        try:
            result = self._safe_execute(_execute)
            count = result["count"] if result else 0
            logger.info(f"千问AI数据数量获取成功，类型: {data_type or 'all'}, 数量: {count}")
            return count
        except Exception as e:
            logger.error(f"千问AI数据数量获取失败: {e}")
            return 0
    
    def update_data(self, data_id: int, updates: Dict[str, Any]) -> bool:
        """
        更新数据库记录
        
        Args:
            data_id: 记录ID
            updates: 要更新的字段和值
            
        Returns:
            更新成功返回True，否则返回False
        """
        def _execute():
            if not updates:
                return True
            
            # 构建UPDATE语句
            set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
            update_sql = f"UPDATE qwen_ai_data SET {set_clause} WHERE id = ?"
            
            # 准备参数
            params = list(updates.values()) + [data_id]
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(update_sql, params)
            affected_rows = cursor.rowcount
            cursor.close()
            conn.close()
            
            return affected_rows > 0
        
        try:
            result = self._safe_execute(_execute)
            if result:
                logger.info(f"千问AI数据更新成功，ID: {data_id}, 更新字段: {list(updates.keys())}")
            return result
        except Exception as e:
            logger.error(f"千问AI数据更新失败，ID: {data_id}: {e}")
            return False
    
    def update_golden_data(self, data_ids: List[int]) -> bool:
        """
        批量更新黄金数据标记
        
        Args:
            data_ids: 要标记为黄金数据的记录ID列表
            
        Returns:
            更新成功返回True，否则返回False
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 先将所有记录标记为非黄金数据
            cursor.execute("UPDATE qwen_ai_data SET is_golden = 0 WHERE data_type = 'history_functions'")
            
            # 再将指定记录标记为黄金数据
            if data_ids:
                placeholders = ", ".join(["?"] * len(data_ids))
                cursor.execute(f"UPDATE qwen_ai_data SET is_golden = 1 WHERE id IN ({placeholders})", data_ids)
            
            cursor.close()
            conn.close()
            return True
        
        try:
            return self._safe_execute(_execute)
        except Exception as e:
            logger.error(f"千问AI黄金数据更新失败: {e}")
            return False
    
    def update_score(self, item_id: int, delta: int):
        """
        更新代码评分和使用次数
        delta: 分数变化值（如成功+5，失败-20）
        分值低于80自动取消黄金标志
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 调试：获取当前分数
            cursor.execute("SELECT score, use_count, is_golden FROM qwen_ai_data WHERE id = ?", (item_id,))
            current = cursor.fetchone()
            if not current:
                logger.debug(f"未找到ID为 {item_id} 的记录")
                return False
            
            current_score = current['score']
            current_use_count = current['use_count']
            current_is_golden = current['is_golden']
            logger.debug(f"当前记录: ID={item_id}, 分数={current_score}, 使用次数={current_use_count}, 黄金标志={current_is_golden}")
            
            # 计算新分数
            new_score = current_score + delta
            logger.debug(f"分数变化: {delta}, 计算后新分数: {new_score}")
            
            # 确保分数在0-100之间
            new_score = max(0, min(100, new_score))
            logger.debug(f"限制后新分数: {new_score}")
            
            # 计算新的黄金标志
            new_is_golden = 1 if new_score >= 80 else 0
            logger.debug(f"新黄金标志: {new_is_golden}")
            
            # 更新评分，并根据分数决定是否保留黄金标志
            sql = """
            UPDATE qwen_ai_data
            SET score = ?,
                is_golden = ?
            WHERE id = ?
            """
            cursor.execute(sql, (new_score, new_is_golden, item_id))
            rows_affected = cursor.rowcount
            logger.debug(f"更新影响行数: {rows_affected}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return rows_affected > 0
        
        try:
            result = self._safe_execute(_execute)
            if result:
                logger.info(f"千问AI数据评分更新成功，ID: {item_id}, 分数变化: {delta}")
            return result
        except Exception as e:
            logger.error(f"更新评分失败: {e}")
            return False
    
    def _auto_fix_imports(self, content: Any) -> Any:
        """自动修复代码中的import缺失问题，兼容多种存储结构"""
        if not isinstance(content, dict):
            return content

        # 兼容结构 A: content直接包含 'code' 键
        if "code" in content:
            code = content["code"]
            if "pd." in code and "import pandas" not in code:
                content["code"] = "import pandas as pd\nimport numpy as np\n" + code
                logger.info("已通过 'code' 键修复 import 缺失")

        # 兼容结构 B: content包含 'functions' 列表
        if "functions" in content and isinstance(content["functions"], list):
            for func in content["functions"]:
                impl = func.get("implementation", "")
                if "pd." in impl and "import pandas" not in impl:
                    func["implementation"] = "import pandas as pd\nimport numpy as np\n" + impl
                    logger.info("已通过 'functions' 列表修复 import 缺失")
                    
        return content
    
    def punish_code(self, code_id: int) -> bool:
        """
        实战报错时严重扣分并降低优先级
        
        Args:
            code_id: 要惩罚的代码ID
            
        Returns:
            惩罚成功返回True，否则返回False
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 严重扣分并取消黄金数据标记
            cursor.execute("UPDATE qwen_ai_data SET score = MAX(0, score - 20), is_golden = 0 WHERE id = ?", (code_id,))
            
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            if result:
                logger.info(f"千问AI数据惩罚成功，ID: {code_id}, 扣除20分并取消黄金数据标记")
            return result
        except Exception as e:
            logger.error(f"千问AI数据惩罚失败，ID: {code_id}: {e}")
            return False
    
    def update_score_by_content(self, content: Any, score_change: int) -> bool:
        """
        根据内容动态调整评分
        
        Args:
            content: 要调整评分的数据内容
            score_change: 评分变化值（正数增加，负数减少）
            
        Returns:
            更新成功返回True，否则返回False
        """
        import hashlib
        
        # 计算content的MD5哈希
        cleaned_content = self._clean_data(content)
        content_json = json.dumps(cleaned_content, ensure_ascii=False)
        prompt_hash = hashlib.md5(content_json.encode('utf-8')).hexdigest()
        
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 查找匹配的记录并调整评分
            cursor.execute("UPDATE qwen_ai_data SET score = MAX(0, MIN(100, score + ?)) WHERE prompt_hash = ?", (score_change, prompt_hash))
            
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            if result:
                logger.info(f"千问AI数据评分调整成功，哈希: {prompt_hash[:8]}..., 变化值: {score_change}")
            return result
        except Exception as e:
            logger.error(f"千问AI数据评分调整失败: {e}")
            return False
    
    def increment_use_count(self, data_id: int) -> bool:
        """
        增加数据的使用次数
        
        Args:
            data_id: 要增加使用次数的数据ID
            
        Returns:
            更新成功返回True，否则返回False
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 增加使用次数
            cursor.execute("UPDATE qwen_ai_data SET use_count = use_count + 1 WHERE id = ?", (data_id,))
            
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            if result:
                logger.debug(f"千问AI数据使用次数增加成功，ID: {data_id}")
            return result
        except Exception as e:
            logger.error(f"千问AI数据使用次数增加失败，ID: {data_id}: {e}")
            return False
    
    def increment_use_count_by_content(self, content: Any) -> bool:
        """
        根据内容增加数据的使用次数
        
        Args:
            content: 要增加使用次数的数据内容
            
        Returns:
            更新成功返回True，否则返回False
        """
        import hashlib
        
        # 计算content的MD5哈希
        cleaned_content = self._clean_data(content)
        content_json = json.dumps(cleaned_content, ensure_ascii=False)
        prompt_hash = hashlib.md5(content_json.encode('utf-8')).hexdigest()
        
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 查找匹配的记录并增加使用次数
            cursor.execute("UPDATE qwen_ai_data SET use_count = use_count + 1 WHERE prompt_hash = ?", (prompt_hash,))
            
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            if result:
                logger.debug(f"千问AI数据使用次数增加成功，哈希: {prompt_hash[:8]}...")
            return result
        except Exception as e:
            logger.error(f"千问AI数据使用次数增加失败: {e}")
            return False
    
    def get_功勋代码(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取实战中的功勋代码（score高且use_count大的数据）
        
        Args:
            limit: 返回的数据数量限制
            
        Returns:
            功勋代码列表
        """
        def _execute():
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 优先返回score高且use_count大的数据，只返回score >= 60的代码（末位淘汰）
            select_sql = "SELECT * FROM qwen_ai_data WHERE data_type = 'history_functions' AND score >= 60 ORDER BY score DESC, use_count DESC LIMIT ?"
            cursor.execute(select_sql, (limit,))
            result = cursor.fetchall()
            
            cursor.close()
            conn.close()
            return result
        
        try:
            results = self._safe_execute(_execute)
            if results is None:
                return []
            
            # 将JSON转换为Python对象
            data_list = []
            for result in results:
                try:
                    content = json.loads(result["content"])
                    data_item = {
                        "id": result["id"],
                        "data_type": result["data_type"],
                        "content": content,
                        "is_golden": result["is_golden"],
                        "logic_tag": result["logic_tag"],
                        "score": result["score"],
                        "use_count": result["use_count"],
                        "created_at": result["created_at"],
                        "updated_at": result["updated_at"]
                    }
                    data_list.append(data_item)
                except json.JSONDecodeError as e:
                    logger.error(f"千问AI数据解析失败: {e}")
                    continue
            
            logger.info(f"千问AI获取功勋代码成功，数量: {len(data_list)}")
            return data_list
        except Exception as e:
            logger.error(f"千问AI获取功勋代码失败: {e}")
            return []
    
    def backup_to_json(self, file_path: str) -> bool:
        """
        将数据库中的所有数据备份到JSON文件
        
        Args:
            file_path: 备份文件路径
            
        Returns:
            备份成功返回True，否则返回False
        """
        try:
            # 获取所有数据
            all_data = self.get_all_data(limit=0)  # limit=0表示获取所有数据
            
            # 准备备份数据
            backup_data = {
                "backup_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_count": len(all_data),
                "data": all_data
            }
            
            # 写入JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"千问AI数据库JSON备份成功，文件路径: {file_path}")
            return True
        except Exception as e:
            logger.error(f"千问AI数据库JSON备份失败: {e}")
            return False
    
    def restore_from_json(self, file_path: str, clear_existing: bool = True) -> bool:
        """
        从JSON文件恢复数据库数据
        
        Args:
            file_path: 备份文件路径
            clear_existing: 是否清除现有数据，默认为True
            
        Returns:
            恢复成功返回True，否则返回False
        """
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # 清除现有数据（如果需要）
            if clear_existing:
                if not self.clear_all_data():
                    return False
            
            # 准备批量插入数据
            data_list = []
            for item in backup_data.get("data", []):
                data_list.append({
                    "data_type": item["data_type"],
                    "content": item["content"]
                })
            
            # 批量插入数据
            if data_list:
                if not self.save_batch_data(data_list):
                    return False
            
            logger.info(f"千问AI数据库JSON恢复成功，文件路径: {file_path}, 恢复数据数量: {len(data_list)}")
            return True
        except Exception as e:
            logger.error(f"千问AI数据库JSON恢复失败: {e}")
            return False
    
    def backup_to_sql(self, file_path: str) -> bool:
        """
        将数据库中的所有数据备份到SQL文件
        
        Args:
            file_path: 备份文件路径
            
        Returns:
            备份成功返回True，否则返回False
        """
        with self._lock:
            try:
                # 创建SQL备份文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 写入备份头部
                    f.write(f"-- 千问AI数据库备份\n")
                    f.write(f"-- 备份时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"-- 数据库类型: SQLite\n\n")
                    
                    # 写入表结构
                    f.write("-- 表结构\n")
                    create_table_sql = """
                    CREATE TABLE IF NOT EXISTS qwen_ai_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_golden INTEGER NOT NULL DEFAULT 0
                    );
                    """
                    f.write(create_table_sql)
                    f.write("\n\n")
                    
                    # 写入索引
                    f.write("-- 索引\n")
                    f.write("CREATE INDEX IF NOT EXISTS idx_qwen_ai_data_type ON qwen_ai_data(data_type);\n")
                    f.write("CREATE INDEX IF NOT EXISTS idx_qwen_ai_is_golden ON qwen_ai_data(is_golden);\n")
                    f.write("CREATE INDEX IF NOT EXISTS idx_qwen_ai_type_golden ON qwen_ai_data(data_type, is_golden);\n\n")
                    
                    # 写入数据
                    f.write("-- 数据\n")
                    select_all_sql = "SELECT * FROM qwen_ai_data"
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    cursor.execute(select_all_sql)
                    results = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    for result in results:
                        # 转义内容中的单引号
                        escaped_content = result['content'].replace("'", "''")
                        insert_sql = f"INSERT INTO qwen_ai_data (data_type, content, created_at, updated_at, is_golden) VALUES ('{result['data_type']}', '{escaped_content}', '{result['created_at']}', '{result['updated_at']}', {result['is_golden']});\n"
                        f.write(insert_sql)
                
                logger.info(f"千问AI数据库SQL备份成功，文件路径: {file_path}")
                return True
            except Exception as e:
                logger.error(f"千问AI数据库SQL备份失败: {e}")
                return False
    
    def restore_from_sql(self, file_path: str, clear_existing: bool = True) -> bool:
        """
        从SQL文件恢复数据库数据
        
        Args:
            file_path: 备份文件路径
            clear_existing: 是否清除现有数据，默认为True
            
        Returns:
            恢复成功返回True，否则返回False
        """
        with self._lock:
            try:
                # 清除现有数据（如果需要）
                if clear_existing:
                    if not self.clear_all_data():
                        return False
                
                # 读取SQL文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                # 执行SQL语句
                # 更可靠的SQL语句解析，使用正则表达式处理字符串中的分号
                import re
                
                # 匹配SQL语句结束符，考虑字符串中的分号
                sql_pattern = r';\s*(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)'
                sql_statements = re.split(sql_pattern, sql_content)
                
                # 执行所有SQL语句
                conn = self._get_connection()
                cursor = conn.cursor()
                for sql in sql_statements:
                    sql = sql.strip()
                    if sql and not sql.startswith('--'):
                        cursor.execute(sql)
                cursor.close()
                conn.close()
                
                logger.info(f"千问AI数据库SQL恢复成功，文件路径: {file_path}")
                return True
            except Exception as e:
                logger.error(f"千问AI数据库SQL恢复失败: {e}")
                return False
    
    def __del__(self):
        """
        析构函数，SQLite不需要特殊处理
        """
        pass
