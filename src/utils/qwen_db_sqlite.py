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
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS qwen_ai_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_golden INTEGER NOT NULL DEFAULT 0
            )
            """
            cursor.execute(create_table_sql)
            # 添加索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_data_type ON qwen_ai_data(data_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_is_golden ON qwen_ai_data(is_golden)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qwen_ai_type_golden ON qwen_ai_data(data_type, is_golden)')
            cursor.close()
            conn.close()
            logger.info("千问AI数据表创建成功")
        except Exception as e:
            logger.error(f"千问AI数据表创建失败: {e}")
            raise
    
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
    
    def save_data(self, data_type: str, content: Any):
        """
        保存数据到数据库
        
        Args:
            data_type: 数据类型（history_functions, feedback_history, error_history）
            content: 数据内容
            
        Returns:
            保存成功返回True，否则返回False
        """
        # 清理数据，确保所有类型都能被JSON序列化
        cleaned_content = self._clean_data(content)
        
        # 将数据转换为JSON格式
        content_json = json.dumps(cleaned_content, ensure_ascii=False)
        
        def _execute():
            # 插入数据
            insert_sql = "INSERT INTO qwen_ai_data (data_type, content, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)"
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(insert_sql, (data_type, content_json))
            cursor.close()
            conn.close()
            logger.info(f"千问AI数据保存成功，类型: {data_type}")
            return True
        
        try:
            return self._safe_execute(_execute)
        except Exception as e:
            logger.error(f"千问AI数据保存失败: {e}")
            return False
    
    def save_batch_data(self, data_list: List[Dict[str, Any]]):
        """
        批量保存数据到数据库
        
        Args:
            data_list: 数据列表，每个元素包含data_type和content字段
            
        Returns:
            保存成功返回True，否则返回False
        """
        # 准备批量插入数据
        batch_data = []
        for data in data_list:
            data_type = data.get("data_type")
            content = data.get("content")
            if data_type and content:
                # 清理数据，确保所有类型都能被JSON序列化
                cleaned_content = self._clean_data(content)
                content_json = json.dumps(cleaned_content, ensure_ascii=False)
                batch_data.append((data_type, content_json))
        
        if not batch_data:
            return False
        
        def _execute():
            insert_sql = "INSERT INTO qwen_ai_data (data_type, content, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)"
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.executemany(insert_sql, batch_data)
            cursor.close()
            conn.close()
            return True
        
        try:
            result = self._safe_execute(_execute)
            logger.info(f"千问AI批量数据保存成功，数量: {len(batch_data)}")
            return result
        except Exception as e:
            logger.error(f"千问AI批量数据保存失败: {e}")
            return False
    
    def get_data(self, data_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        从数据库获取数据
        
        Args:
            data_type: 数据类型
            limit: 返回的数据数量限制
            
        Returns:
            数据列表
        """
        def _execute():
            select_sql = "SELECT * FROM qwen_ai_data WHERE data_type = ? ORDER BY created_at DESC LIMIT ?"
            conn = self._get_connection()
            cursor = conn.cursor()
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
