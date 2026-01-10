#!/usr/bin/env python3
"""
千问AI学习算法模块
提供基于历史数据的学习、反馈驱动的学习和错误驱动的学习
移除了tensorflow深度学习相关功能，仅保留传统机器学习算法
"""

import logging
import time
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# 导入配置管理器
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.dual_redis_db import DualRedisDB

logger = logging.getLogger("QwenLearning")

# 不使用深度学习库
DEEP_LEARNING_AVAILABLE = False


class QwenLearning:
    """千问AI学习算法类"""

    def __init__(self):
        """初始化学习算法"""
        self.logger = logging.getLogger("QwenLearning")
        self.logger.info("初始化QwenLearning...")
        
        # 学习相关配置 - 先初始化配置参数
        self.max_history_size = 1000  # 最大历史记录数量
        self.feedback_threshold = 0.5  # 反馈阈值
        self.error_threshold = 0.3  # 错误阈值
        
        # 2. 定义Redis存储的Key
        self.redis_key = "qwen_learning_memory"
        
        # 初始化Redis连接
        self.dual_redis = None
        self.qwen_db = None
        self._initialize_db()
        
        # 初始化学习数据
        self.learning_data = {
            "history_functions": [],
            "feedback_history": [],
            "error_history": []
        }
        
        # 加载历史学习数据
        self.load_learning_data()
        
        # 【改动 C：精简TF-IDF的特征】
        # 加入stop_words，去掉废话词，让匹配更关注硬核逻辑词
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True,
            stop_words=["请", "帮我", "计算", "处理", "表格", "数据", "函数", "生成", "Excel", "文件"]
        )
        
        # 初始化学习模型
        self._init_learning_model()
        
    def _initialize_db(self):
        """连接到 Redis 第二个数据库 (logic_conn)"""
        try:
            self.dual_redis = DualRedisDB()
            # 关键：使用 logic_conn 作为数据存储库，区分于 task_conn 任务队列库
            self.qwen_db = self.dual_redis.logic_conn
            self.logger.info("QwenLearning 成功接入 Redis 数据后端")
        except Exception as e:
            self.logger.error(f"Redis 连接失败: {e}")
    
    def load_learning_data(self):
        """从 Redis 读取学习记忆"""
        if self.qwen_db:
            data = self.qwen_db.get(self.redis_key)
            if data:
                try:
                    self.learning_data = json.loads(data)
                    self.logger.info("从Redis加载学习数据成功")
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析Redis学习数据失败: {e}")
    
    def _init_learning_model(self):
        """初始化学习模型"""
        self.logger.info("初始化学习模型...")
        # 仅使用TF-IDF向量器，不使用深度学习模型
        self.logger.info("学习模型初始化完成")
        
    def save_learning_data(self):
        """将学习记忆序列化存入 Redis"""
        if self.qwen_db:
            try:
                # 限制历史长度防止 Redis Key 过大
                for key in self.learning_data:
                    if isinstance(self.learning_data[key], list):
                        self.learning_data[key] = self.learning_data[key][-self.max_history_size:]
                
                self.qwen_db.set(self.redis_key, json.dumps(self.learning_data))
                self.logger.debug("学习数据已同步至Redis")
            except Exception as e:
                self.logger.error(f"同步学习数据至 Redis 失败: {e}")
    
    def learn_from_history(self, prompt: str, enhanced_prompt: str, data_context: Dict[str, Any], 
                         functions: List[Dict[str, Any]], success: bool, attempt: int = 1,
                         note: str = "") -> bool:
        """从历史记录中学习
        
        Args:
            prompt: 原始提示词
            enhanced_prompt: 增强后的提示词
            data_context: 数据上下文
            functions: 生成的函数列表
            success: 是否生成成功
            attempt: 尝试次数
            note: 备注信息
            
        Returns:
            学习是否成功
        """
        try:
            # 保存到学习数据
            history_item = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "data_context": data_context,
                "functions": functions,
                "success": success,
                "attempt": attempt,
                "note": note,
                "timestamp": time.time()
            }
            
            # 添加到内存中的历史记录
            self.learning_data["history_functions"].append(history_item)
            
            # 限制历史记录数量
            if len(self.learning_data["history_functions"]) > self.max_history_size:
                self.learning_data["history_functions"] = self.learning_data["history_functions"][-self.max_history_size:]
            
            # 保存到Redis
            self.save_learning_data()
            
            self.logger.info("从历史记录中学习成功")
            return True
        except Exception as e:
            self.logger.error(f"从历史记录中学习失败: {e}")
            return False
    
    def learn_from_feedback(self, function_id: str, feedback: str, improvement: str = "") -> bool:
        """从反馈中学习
        
        Args:
            function_id: 函数ID
            feedback: 反馈内容
            improvement: 改进建议
            
        Returns:
            学习是否成功
        """
        try:
            # 保存到学习数据
            feedback_item = {
                "function_id": function_id,
                "feedback": feedback,
                "improvement": improvement,
                "timestamp": time.time()
            }
            
            # 添加到内存中的反馈历史
            self.learning_data["feedback_history"].append(feedback_item)
            
            # 限制反馈历史数量
            if len(self.learning_data["feedback_history"]) > self.max_history_size:
                self.learning_data["feedback_history"] = self.learning_data["feedback_history"][-self.max_history_size:]
            
            # 保存到Redis
            self.save_learning_data()
            
            self.logger.info("从反馈中学习成功")
            return True
        except Exception as e:
            self.logger.error(f"从反馈中学习失败: {e}")
            return False
    
    def learn_from_error(self, prompt: str, enhanced_prompt: str, data_context: Dict[str, Any],
                        error: str, traceback: str, attempt: int = 1,
                        note: str = "") -> bool:
        """从错误中学习
        
        Args:
            prompt: 原始提示词
            enhanced_prompt: 增强后的提示词
            data_context: 数据上下文
            error: 错误信息
            traceback: 错误堆栈
            attempt: 尝试次数
            note: 备注信息
            
        Returns:
            学习是否成功
        """
        try:
            # 保存到学习数据
            error_item = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "data_context": data_context,
                "error": error,
                "traceback": traceback,
                "attempt": attempt,
                "note": note,
                "timestamp": time.time()
            }
            
            # 添加到内存中的错误历史
            self.learning_data["error_history"].append(error_item)
            
            # 限制错误历史数量
            if len(self.learning_data["error_history"]) > self.max_history_size:
                self.learning_data["error_history"] = self.learning_data["error_history"][-self.max_history_size:]
            
            # 保存到Redis
            self.save_learning_data()
            
            self.logger.info("从错误中学习成功")
            return True
        except Exception as e:
            self.logger.error(f"从错误中学习失败: {e}")
            return False
    
    def analyze_and_patch(self, error_msg):
        """
        针对特定错误，在 Redis 中建立‘强规则’
        
        Args:
            error_msg: 错误信息
            
        Returns:
            修正建议，如果没有匹配的规则则返回 None
        """
        if "MultiIndex" in error_msg and "index=False" in error_msg:
            return "CRITICAL: 当存在 MultiIndex 时，必须设置 index=True，或平铺表头。"
        
        if "Timedelta" in error_msg:
            return "CRITICAL: 对时间差进行运算前，必须使用 .dt.total_seconds() 转换为数值。"
            
        if "dtype('<U4')" in error_msg or "UFuncNoLoopError" in error_msg:
            return "CRITICAL: 检测到字符串与数字混合，请先执行 df.fillna(0) 并强制转换类型。"
            
        return None
    
    def generate_enhanced_prompt(self, prompt: str, data_context: Dict[str, Any]) -> str:
        """
        生成增强的提示词
        
        Args:
            prompt: 原始提示词
            data_context: 数据上下文
            
        Returns:
            增强后的提示词
        """
        try:
            # 基础增强提示词
            enhanced_prompt = prompt
            
            # 添加数据上下文信息
            if data_context:
                enhanced_prompt += "\n\n数据上下文："
                enhanced_prompt += json.dumps(data_context, ensure_ascii=False, indent=2)
            
            # 【改动 A：增加“负向反馈”权重】
            # 优先提取之前在这里跌倒过的记录
            relevant_errors = self.get_relevant_errors(prompt, limit=3)
            if relevant_errors:
                enhanced_prompt += "\n\n⚠️ 历史错误警告："
                for i, error_item in enumerate(relevant_errors):
                    error_msg = error_item.get("error", "")
                    # 提取错误类型
                    error_type = error_msg.split(":")[0] if ":" in error_msg else "未知错误"
                    # 分析错误并提供补丁建议
                    patch_advice = self.analyze_and_patch(error_msg)
                    if patch_advice:
                        enhanced_prompt += f"\n{i+1}. {patch_advice}"
                    else:
                        enhanced_prompt += f"\n{i+1}. 该操作曾出现过 [{error_type}]，请务必避免类似错误"
            
            # 添加成功的历史学习经验
            relevant_history = self.get_relevant_history(prompt, limit=3)
            if relevant_history:
                enhanced_prompt += "\n\n✅ 相关成功经验："
                for i, item in enumerate(relevant_history):
                    enhanced_prompt += f"\n{i+1}. " + item.get("prompt", "")[:100] + "..."
            
            # 针对需求类型添加预防性规则
            if any(keyword in prompt for keyword in ["时间", "日期", "Timedelta"]):
                enhanced_prompt += "\n\n⚠️ 时间处理规则：对时间差进行运算前，必须使用 .dt.total_seconds() 转换为数值。"
            
            if any(keyword in prompt for keyword in ["嵌套", "多层", "MultiIndex"]):
                enhanced_prompt += "\n\n⚠️ 嵌套结构规则：当存在 MultiIndex 时，必须设置 index=True，或平铺表头。"
            
            if any(keyword in prompt for keyword in ["计算", "金额", "数量"]):
                enhanced_prompt += "\n\n⚠️ 数值计算规则：检测到字符串与数字混合，请先执行 df.fillna(0) 并强制转换类型。"
            
            return enhanced_prompt
        except Exception as e:
            self.logger.error(f"生成增强提示词失败: {e}")
            return prompt
    
    def get_relevant_errors(self, prompt: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取与当前提示词相关的错误记录
        
        Args:
            prompt: 提示词
            limit: 返回的错误记录数量
            
        Returns:
            相关的错误记录列表
        """
        try:
            if not self.learning_data["error_history"]:
                return []
            
            # 提取历史错误提示词
            error_prompts = [item.get("prompt", "") for item in self.learning_data["error_history"]]
            
            if not error_prompts:
                return []
            
            # 使用TF-IDF计算相似度
            all_texts = [prompt] + error_prompts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # 按相似度排序，获取最相关的错误记录
            similar_indices = similarity_scores.argsort()[::-1][:limit]
            relevant_errors = [self.learning_data["error_history"][index] for index in similar_indices]
            
            return relevant_errors
        except Exception as e:
            self.logger.error(f"获取相关错误记录失败: {e}")
            return []
    
    def get_relevant_history(self, prompt: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取相关的历史记录
        
        Args:
            prompt: 提示词
            limit: 返回的历史记录数量
            
        Returns:
            相关的历史记录列表
        """
        try:
            if not self.learning_data["history_functions"]:
                return []
            
            # 提取历史提示词
            history_prompts = [item.get("prompt", "") for item in self.learning_data["history_functions"] if item.get("success", False)]
            
            if not history_prompts:
                return []
            
            # 使用TF-IDF计算相似度
            all_texts = [prompt] + history_prompts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # 按相似度排序，获取最相关的历史记录
            similar_indices = similarity_scores.argsort()[::-1][:limit]
            relevant_history = [self.learning_data["history_functions"][index] for index in similar_indices]
            
            return relevant_history
        except Exception as e:
            self.logger.error(f"获取相关历史记录失败: {e}")
            return []
    
    def update_model(self):
        """更新学习模型"""
        self.logger.info("更新学习模型...")
        # 仅更新TF-IDF向量器
        try:
            # 提取历史提示词
            history_prompts = [item.get("prompt", "") for item in self.learning_data["history_functions"] if item.get("success", False)]
            
            if history_prompts:
                # 更新TF-IDF向量器
                self.tfidf_vectorizer.fit(history_prompts)
                self.logger.info("TF-IDF向量器更新成功")
        except Exception as e:
            self.logger.error(f"更新学习模型失败: {e}")
    
    def clear_learning_data(self):
        """清除学习数据"""
        try:
            # 清除内存中的学习数据
            self.learning_data = {
                "history_functions": [],
                "feedback_history": [],
                "error_history": []
            }
            
            # 清除数据库中的学习数据
            if self.qwen_db:
                self.qwen_db.clear_all_data()
            
            self.logger.info("学习数据清除成功")
        except Exception as e:
            self.logger.error(f"清除学习数据失败: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息
        
        Returns:
            学习统计信息字典
        """
        try:
            # 计算成功生成率
            total_functions = len(self.learning_data["history_functions"])
            if total_functions == 0:
                success_rate = 0
            else:
                success_count = sum(1 for item in self.learning_data["history_functions"] if item.get("success", False))
                success_rate = success_count / total_functions * 100
            
            # 计算平均尝试次数
            if total_functions == 0:
                avg_attempts = 0
            else:
                total_attempts = sum(item.get("attempt", 1) for item in self.learning_data["history_functions"])
                avg_attempts = total_attempts / total_functions
            
            statistics = {
                "total_functions": total_functions,
                "success_count": success_count,
                "success_rate": round(success_rate, 2),
                "avg_attempts": round(avg_attempts, 2),
                "feedback_count": len(self.learning_data["feedback_history"]),
                "error_count": len(self.learning_data["error_history"]),
                "last_updated": time.time()
            }
            
            return statistics
        except Exception as e:
            self.logger.error(f"获取学习统计信息失败: {e}")
            return {}
    
    def learn_from_iteration(self, requirement, code, error_msg=None, success=False):
        """
        每次迭代后的学习函数：
        如果成功，存为黄金案例；如果失败，存入错误索引。
        """
        try:
            if success:
                # 记录成功案例供 Prompt 引用
                self.learning_data["history_functions"].append({
                    "prompt": requirement,
                    "enhanced_prompt": requirement,
                    "data_context": {},
                    "functions": [{"code": code}],
                    "success": True,
                    "attempt": 1,
                    "timestamp": time.time()
                })
            else:
                # 记录错误原因
                self.learning_data["error_history"].append({
                    "prompt": requirement,
                    "enhanced_prompt": requirement,
                    "data_context": {},
                    "error": error_msg,
                    "traceback": "",
                    "attempt": 1,
                    "timestamp": time.time()
                })
            
            # 同步回 Redis
            self.save_learning_data()
            self.logger.info("从迭代中学习成功")
            return True
        except Exception as e:
            self.logger.error(f"从迭代中学习失败: {e}")
            return False
    
    def learn_from_failure(self, requirement, error_msg):
        """
        将失败的需求和错误原因存入 Redis
        
        Args:
            requirement: 用户需求
            error_msg: 错误原因
        """
        try:
            # 使用单独的键存储持久化错误记忆
            persistent_key = "persistent_error_knowledge"
            
            # 获取现有数据
            knowledge = self.qwen_db.get(persistent_key)
            data = json.loads(knowledge) if knowledge else {}
            
            # 更新错误记忆
            data[requirement] = {
                "error_msg": error_msg,
                "timestamp": time.time()
            }
            
            # 保存回 Redis
            self.qwen_db.set(persistent_key, json.dumps(data))
            self.logger.info("持久化错误记忆保存成功")
            
            # 同时也将错误存入常规错误历史，以便在 generate_enhanced_prompt 中使用
            self.learning_data["error_history"].append({
                "prompt": requirement,
                "enhanced_prompt": requirement,
                "data_context": {},
                "error": error_msg,
                "traceback": "",
                "attempt": 1,
                "timestamp": time.time()
            })
            
            # 同步回 Redis
            self.save_learning_data()
            
            return True
        except Exception as e:
            self.logger.error(f"保存持久化错误记忆失败: {e}")
            return False
    
    def get_error_memory(self, requirement):
        """
        从 Redis 获取指定需求的历史错误记忆
        
        Args:
            requirement: 用户需求
            
        Returns:
            历史错误信息，如果没有则返回 None
        """
        try:
            # 使用单独的键存储持久化错误记忆
            persistent_key = "persistent_error_knowledge"
            
            # 获取现有数据
            knowledge = self.qwen_db.get(persistent_key)
            if knowledge:
                data = json.loads(knowledge)
                if requirement in data:
                    return data[requirement]["error_msg"]
            
            return None
        except Exception as e:
            self.logger.error(f"获取错误记忆失败: {e}")
            return None
    
    def save_success_case(self, requirement, fixed_code):
        """
        保存手动修复的成功案例到 Redis
        
        Args:
            requirement: 用户需求
            fixed_code: 手动修复后的代码
        """
        try:
            # 保存到学习数据
            self.learning_data["history_functions"].append({
                "prompt": requirement,
                "enhanced_prompt": requirement,
                "data_context": {},
                "functions": [{"code": fixed_code}],
                "success": True,
                "attempt": 1,
                "timestamp": time.time(),
                "is_manual_fix": True
            })
            
            # 同步回 Redis
            self.save_learning_data()
            
            # 同时保存到成功案例集合
            success_cases_key = "success_cases"
            success_cases = self.qwen_db.get(success_cases_key)
            if success_cases:
                success_cases = json.loads(success_cases)
            else:
                success_cases = []
            
            success_cases.append({
                "requirement": requirement,
                "code": fixed_code,
                "timestamp": time.time()
            })
            
            # 限制成功案例数量
            if len(success_cases) > 100:
                success_cases = success_cases[-100:]
            
            # 保存回 Redis
            self.qwen_db.set(success_cases_key, json.dumps(success_cases))
            self.logger.info("手动修复的成功案例已保存到 Redis")
            return True
        except Exception as e:
            self.logger.error(f"保存手动修复的成功案例失败: {e}")
            return False


# 工厂函数，返回QwenLearning实例
def get_qwen_learning() -> QwenLearning:
    """获取QwenLearning实例
    
    Returns:
        QwenLearning实例
    """
    return QwenLearning()
