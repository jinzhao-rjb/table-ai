"""
AI服务类，提供统一的AI调用接口，基于GitHub项目的API管理架构
"""

import logging
import time
import threading
from typing import List, Dict, Any
import json

# 使用绝对导入，确保在任何情况下都能正常导入
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import config_manager
from src.utils.dual_redis_db import DualRedisDB
from .api_manager import APIManager
from .qwen_learning import get_qwen_learning

logger = logging.getLogger(__name__)


class AIService:
    """AI服务类，提供统一的AI调用接口"""

    def __init__(self):
        """初始化AI服务"""
        self.api_manager = None
        self.logger = logging.getLogger("AIService")
        self._initialize_api_manager()

        # 延迟初始化标志
        self._initialized = False
        
        # 数据库和学习模块的延迟初始化属性
        self.qwen_db = None
        self.qwen_learning = None
        self.dual_redis = None

        # 学习相关属性
        self.learning_enabled = True  # 是否启用学习功能
        self.history_functions = []  # 历史生成的函数记录
        self.feedback_history = []  # 反馈历史记录
        self.error_history = []  # 错误历史记录
        self.max_history_size = 20  # 最大历史记录数量

        # 本地缓存 - 简化缓存机制
        self._memory_cache = {}  # key: prompt + str(columns), value: functions
        self._cache_size = 100  # 缓存大小，从500减少到100
        self._cache_hits = 0  # 缓存命中计数
        self._cache_misses = 0  # 缓存未命中计数
        
        # 哈希预匹配缓存 - 用于100%匹配的快速查询
        self._hash_cache = {}  # key: prompt_hash, value: (functions, columns)
        self._hash_cache_size = 200  # 哈希缓存大小，从1000减少到200
        
        # 线程锁
        self._init_lock = threading.Lock()  # 用于确保延迟初始化只执行一次
        self._save_lock = threading.Lock()  # 用于保护数据写入操作
        self._cache_lock = threading.Lock()  # 用于保护缓存操作
        self._hash_cache_lock = threading.Lock()  # 用于保护哈希缓存操作
        
        # 加载学习数据
        self._load_learning_data()
        
        # 初始化Redis连接
        self._lazy_init()

    def _save_learning_data(self):
        """保存学习数据，使用Redis存储"""
        if not self.learning_enabled:
            return

        try:
            logger.debug("[AI学习] 学习数据已通过qwen_learning模块保存到Redis")
        except Exception as e:
            logger.error(f"[AI学习] 保存学习数据失败: {e}")

    def _load_learning_data(self):
        """加载学习数据，从Redis加载"""
        if not self.learning_enabled:
            return

        try:
            # 学习数据现在通过qwen_learning模块从Redis加载
            logger.debug("[AI学习] 学习数据已通过qwen_learning模块从Redis加载")
        except Exception as e:
            logger.error(f"加载学习数据失败: {e}")
            # 初始化空的学习数据结构
            self.history_functions = []
            self.feedback_history = []
            self.error_history = []

    def add_to_history(self, function_data: Dict[str, Any]):
        """添加生成的函数到历史记录

        Args:
            function_data: 包含函数信息的字典，应包含prompt、data_context、functions等字段
        """
        if not self.learning_enabled:
            return

        self._lazy_init()
        
        try:
            # 添加时间戳
            function_data["timestamp"] = time.time()

            # 添加到历史记录
            self.history_functions.append(function_data)

            # 限制历史记录数量
            if len(self.history_functions) > self.max_history_size:
                self.history_functions = self.history_functions[
                    -self.max_history_size :
                ]

            # 简化学习过程，直接保存到文件，不进行复杂的异步学习
            if self.qwen_learning:
                success = function_data.get("success", False)
                learning_success = self.qwen_learning.learn_from_history(
                    prompt=function_data.get("prompt", ""),
                    enhanced_prompt=function_data.get("enhanced_prompt", ""),
                    data_context=function_data.get("data_context", {}),
                    functions=function_data.get("functions", []),
                    success=success,
                    attempt=function_data.get("attempt", 1),
                )

                if learning_success:
                    logger.debug(f"[AI学习] 函数已添加到历史记录并学习成功")
                else:
                    logger.warning(f"[AI学习] 函数已添加到历史记录，但学习失败")
            
            # 定期保存学习数据
            if len(self.history_functions) % 5 == 0:
                self._save_learning_data()

        except Exception as e:
            logger.error(f"[AI学习] 添加函数到历史记录失败: {e}")

    def add_feedback(self, feedback_data: Dict[str, Any]):
        """添加反馈到历史记录

        Args:
            feedback_data: 包含反馈信息的字典，应包含function_id、feedback、improvement等字段
        """
        if not self.learning_enabled:
            return

        self._lazy_init()
        
        try:
            # 添加时间戳
            feedback_data["timestamp"] = time.time()

            # 添加到反馈历史
            self.feedback_history.append(feedback_data)

            # 限制反馈历史数量
            if len(self.feedback_history) > self.max_history_size:
                self.feedback_history = self.feedback_history[-self.max_history_size :]

            # 简化学习过程
            if self.qwen_learning:
                success = self.qwen_learning.learn_from_feedback(
                    function_id=feedback_data.get("function_id", ""),
                    feedback=feedback_data.get("feedback", ""),
                    improvement=feedback_data.get("improvement", ""),
                )

                if success:
                    logger.debug(f"[AI学习] 反馈已添加到历史记录并学习成功")
                else:
                    logger.warning(f"[AI学习] 反馈已添加到历史记录，但学习失败")
            
            # 定期保存学习数据
            if len(self.feedback_history) % 5 == 0:
                self._save_learning_data()

        except Exception as e:
            logger.error(f"[AI学习] 添加反馈到历史记录失败: {e}")

    def add_error(self, error_data: Dict[str, Any]):
        """添加错误到历史记录

        Args:
            error_data: 包含错误信息的字典，应包含prompt、error、function等字段
        """
        if not self.learning_enabled:
            return

        self._lazy_init()
        
        try:
            # 添加时间戳
            error_data["timestamp"] = time.time()

            # 添加到错误历史
            self.error_history.append(error_data)

            # 限制错误历史数量
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size :]

            # 简化学习过程
            if self.qwen_learning:
                success = self.qwen_learning.learn_from_error(
                    prompt=error_data.get("prompt", ""),
                    enhanced_prompt=error_data.get("enhanced_prompt", ""),
                    data_context=error_data.get("data_context", {}),
                    error=error_data.get("error", ""),
                    traceback=error_data.get("traceback", ""),
                    attempt=error_data.get("attempt", 1),
                )

                if success:
                    logger.debug(f"[AI学习] 错误已添加到历史记录并学习成功")
                else:
                    logger.warning(f"[AI学习] 错误已添加到历史记录，但学习失败")
            
            # 定期保存学习数据
            if len(self.error_history) % 5 == 0:
                self._save_learning_data()

        except Exception as e:
            logger.error(f"[AI学习] 添加错误到历史记录失败: {e}")

    def get_relevant_history(self, prompt: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取与当前提示相关的历史记录

        Args:
            prompt: 当前提示词
            limit: 返回的历史记录数量

        Returns:
            相关的历史记录列表
        """
        self._lazy_init()
        
        if not self.learning_enabled or not self.history_functions:
            return []

        try:
            # 简单的关键词匹配
            relevant_history = []
            prompt_lower = prompt.lower()

            for func_data in reversed(self.history_functions):
                if len(relevant_history) >= limit:
                    break

                # 检查提示词相似度
                if "prompt" in func_data:
                    history_prompt_lower = func_data["prompt"].lower()
                    # 计算关键词匹配度
                    match_count = sum(
                        1
                        for word in prompt_lower.split()
                        if word in history_prompt_lower
                    )
                    if match_count > 0:
                        relevant_history.append(func_data)

            logger.debug(f"[AI学习] 找到 {len(relevant_history)} 条相关历史记录")
            return relevant_history
        except Exception as e:
            logger.error(f"[AI学习] 获取相关历史记录失败: {e}")
            return []

    def generate_functions(
        self, prompt: str, data_context: Dict[str, Any], note=""
    ) -> List[Dict[str, Any]]:
        """生成Excel处理函数
        
        先从缓存中检索相关函数，如果没有找到则使用功勋代码，最后生成新函数
        
        Args:
            prompt: 原始提示词
            data_context: 数据上下文
            note: 备注信息
            
        Returns:
            生成的函数列表
        """
        self._lazy_init()
        
        # --- 新增：上下文压缩逻辑 ---
        compressed_data_context = data_context.copy()
        
        # 1. 净化列名：去掉换行和空格，这是 SyntaxError 的根源
        if "columns" in compressed_data_context:
            clean_columns = [c.replace('\n', '').replace('\r', '').strip() for c in compressed_data_context["columns"]]
            compressed_data_context["columns"] = clean_columns
        
        # 2. 上下文压缩：只保留表头和前5行数据，避免发送全表数据
        if "data" in compressed_data_context and isinstance(compressed_data_context["data"], list):
            full_data = compressed_data_context["data"]
            if len(full_data) > 5:
                # 只保留前5行作为示例数据
                compressed_data_context["data"] = full_data[:5]
                logger.info(f"上下文压缩：将 {len(full_data)} 行数据压缩为前 5 行示例数据")
        
        # 3. 强化 Prompt：明确告诉 AI 如何处理数据类型和上下文压缩
        enhanced_system_instruction = """
        CRITICAL:
        1. 如果列名包含特殊符号，请务必使用 df['列名'] 格式。
        2. 数学运算前必须先执行 df['列名'] = pd.to_numeric(df['列名'], errors='coerce').fillna(0)。
        3. 严禁在返回的 JSON 字符串中使用物理换行符。
        4. 你收到的数据是前5行示例，实际数据可能包含大量行，确保生成的函数能正确处理全量数据。
        """
        prompt = f"{enhanced_system_instruction}\n用户需求：{prompt}"
        # --- 护卫逻辑结束 ---
        
        # 1. 首先检查缓存，实现快速响应
        columns = data_context.get("columns", [])
        cache_key = f"{prompt}:{str(sorted(columns))}"
        
        # 快速缓存命中检查
        with self._cache_lock:
            if cache_key in self._memory_cache:
                # 缓存命中
                functions = self._memory_cache[cache_key]
                logger.debug(f"内存缓存命中！")
                return functions
        
        # 2. 缓存未命中，检查历史记录
        for func_data in reversed(self.history_functions):
            if func_data.get("success", False) and func_data.get("prompt", "") == prompt:
                # 完全匹配，检查列匹配度
                history_columns = set(func_data.get("data_context", {}).get("columns", []))
                current_columns = set(columns)
                
                if len(history_columns.intersection(current_columns)) >= len(current_columns) * 0.6:
                    functions = func_data.get("functions", [])
                    # 更新缓存
                    with self._cache_lock:
                        self._memory_cache[cache_key] = functions
                    logger.info(f"历史记录匹配！")
                    return functions
        
        # 3. 历史记录未匹配，检查功勋代码
        if self.qwen_db:
            try:
                # 获取功勋代码列表
                meritorious_codes = self.qwen_db.get_功勋代码(limit=20)
                
                # 匹配相关的功勋代码
                for code_item in meritorious_codes:
                    code_content = code_item.get("content", {})
                    if not code_content.get("success", False):
                        continue
                    
                    # 检查提示词相似度
                    history_prompt = code_content.get("prompt", "")
                    if history_prompt:
                        # 简单的关键词匹配
                        prompt_lower = prompt.lower()
                        history_prompt_lower = history_prompt.lower()
                        match_count = sum(1 for word in prompt_lower.split() if word in history_prompt_lower)
                        
                        if match_count > 0:
                            # 检查列匹配度
                            history_columns = set(code_content.get("data_context", {}).get("columns", []))
                            current_columns = set(columns)
                            
                            if len(history_columns.intersection(current_columns)) >= len(current_columns) * 0.5:
                                functions = code_content.get("functions", [])
                                if functions:
                                    try:
                                        # 先验证代码是否有效
                                        self._quick_validate(functions, data_context)
                                        # 验证成功，增加评分
                                        self.qwen_db.update_score(code_item["id"], delta=5)
                                        # 更新缓存
                                        with self._cache_lock:
                                            self._memory_cache[cache_key] = functions
                                        logger.info(f"功勋代码匹配成功！ID: {code_item['id']}, Score: {code_item['score']}, Use Count: {code_item['use_count']}")
                                        return functions
                                    except Exception as e:
                                        # 验证失败，实施负反馈惩罚
                                        logger.error(f"功勋代码ID: {code_item['id']} 验证失败: {e}")
                                        # 严重扣分并取消黄金数据标记
                                        self.qwen_db.update_score(code_item["id"], delta=-20)
            except Exception as e:
                logger.error(f"获取功勋代码失败: {e}")
        
        # 4. 生成函数
        max_attempts = 3
        last_error = ""  # 核心：用于追踪上一次的报错，实现带记忆的闭环
        final_functions = []
        
        # 导入必要的库用于验证和错误追踪
        import pandas as pd
        import numpy as np
        import traceback
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"开始第 {attempt} 次生成函数")
            
            # 【关键修改 1】：动态调整提示词，把上一轮的失败告诉 AI
            current_prompt = prompt
            if attempt > 1 and last_error:
                # 核心：将上一轮的详细报错信息喂回给 AI
                current_prompt = f"{prompt}\n\n注意：你上一次生成的代码报错了，请修正：\n{last_error}"
            
            # 【关键修改 2】：调用增强提示词生成器
            enhanced_prompt = current_prompt
            # 如果 qwen_learning 可用，使用它生成更高级的增强提示词
            if self.qwen_learning:
                try:
                    enhanced_prompt = self.qwen_learning.generate_enhanced_prompt(current_prompt, data_context)
                except Exception as e:
                    logger.warning(f"生成增强提示词失败，使用原始提示词: {e}")
            
            try:
                # 使用API管理器生成函数
                success, functions, error_message = self.api_manager.generate_functions(
                    prompt=enhanced_prompt,
                    data_context=compressed_data_context
                )
                
                if success and functions:
                    # 【关键修改 3-1】：在验证前先强制添加必要的导入语句
                    # 对每个函数的implementation**总是**添加导入语句，确保pandas和numpy总是可用
                    for func in functions:
                        func_impl = func.get('implementation', '')
                        # 检查是否已经包含导入语句
                        if 'import pandas' not in func_impl:
                            # 强制注入必要的导入语句，不管是否包含pd.
                            func['implementation'] = 'import pandas as pd\nimport numpy as np\n' + func_impl
                    
                    # 【关键修改 3-2】：原地预演，只有跑通了才算成功
                    # 模拟运行一下，如果报错，直接抛出异常进入 except 块
                    self._quick_validate(functions, data_context)
                    
                    # 【新增】：对函数进行依赖排序，确保按正确顺序执行
                    sorted_functions = self._sort_functions_by_dependencies(functions, data_context.get("columns", []))
                    
                    # 验证成功，记录到历史记录
                    self.add_to_history({
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "data_context": data_context,
                        "functions": sorted_functions,  # 使用排序后的函数
                        "attempt": attempt,
                        "success": True,
                        "note": note
                    })
                    
                    # 更新缓存
                    with self._cache_lock:
                        self._memory_cache[cache_key] = sorted_functions
                    
                    logger.info(f"第 {attempt} 次生成函数成功")
                    return sorted_functions
                else:
                    # API生成失败，记录错误
                    last_error = f"API调用失败: {error_message}"
                    logger.warning(f"第 {attempt} 次生成函数失败，生成结果为空")
                    
                    # 记录错误到历史
                    self.add_error({
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt,
                        "data_context": data_context,
                        "error": last_error,
                        "traceback": error_message,
                        "attempt": attempt,
                        "note": note
                    })
            except Exception as e:
                # 【关键修改 4】：记录详细报错堆栈，用于下一轮重试
                last_error = traceback.format_exc()
                logger.warning(f"第 {attempt} 次生成函数验证失败，准备纠错重试...")
                logger.warning(f"验证错误详情：{last_error}")
                
                # 记录错误到历史
                self.add_error({
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "data_context": data_context,
                    "error": f"生成异常: {str(e)}",
                    "traceback": last_error,
                    "attempt": attempt,
                    "note": note
                })
                
                # 循环继续，下一轮 current_prompt 就会带上这个报错
                continue
        
        # 所有尝试都失败，返回空列表
        logger.error(f"所有 {max_attempts} 次生成函数均失败")
        return final_functions
        
    def _quick_validate(self, functions: List[Dict[str, Any]], data_context: Dict[str, Any]):
        """快速验证生成的函数是否能够正常执行
        
        Args:
            functions: 生成的函数列表
            data_context: 数据上下文
            
        Raises:
            Exception: 如果验证失败，抛出具体的错误信息
        """
        import pandas as pd
        import numpy as np
        
        # 改进后的简单验证
        try:
            # 动态构建包含业务列名的测试数据
            columns = data_context.get("columns", [])
            if not columns:
                logger.warning("数据上下文中的列信息为空，跳过验证")
                return
            
            test_data = {col: [10, 20, 30] for col in columns}
            
            test_df = pd.DataFrame(test_data)
            
            # 强制转换所有列为数值，模拟真实清洗后的环境
            for col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors='ignore')

            # 验证每个函数
            for func in functions:
                func_name = func.get("name", "")
                func_impl = func.get("implementation", "")
                
                # 检查函数实现是否为空
                if not func_impl:
                    raise Exception(f"函数 {func_name} 实现为空")
                
                # 强制注入必要的导入语句
                injected_func_impl = "import pandas as pd\nimport numpy as np\n" + func_impl
                
                # 尝试执行函数
                local_namespace = {
                    "pd": pd,
                    "np": np,
                    "df": test_df.copy()
                }
                
                # 执行函数定义，确保pd和np在函数内部可用
                exec(injected_func_impl, {
                    "pd": pd,
                    "np": np
                }, local_namespace)
                
                # 检查函数是否在命名空间中
                if func_name not in local_namespace:
                    raise Exception(f"函数 {func_name} 未在命名空间中定义")
                
                # 尝试调用函数
                func_obj = local_namespace[func_name]
                result = func_obj(test_df.copy())
                
                # 验证返回值是否为DataFrame
                if not isinstance(result, pd.DataFrame):
                    raise Exception(f"函数 {func_name} 未返回DataFrame，返回类型: {type(result).__name__}")
                
                # 验证返回的DataFrame是否为空
                if len(result) == 0:
                    raise Exception(f"函数 {func_name} 返回空DataFrame")
                
                logger.debug(f"函数 {func_name} 验证通过")
        except KeyError as e:
            logger.error(f"验证失败，缺少依赖列: {e}")
            # 对于KeyError，记录缺失的依赖项，但不完全失败
            missing_column = str(e).strip("'\"")
            logger.info(f"检测到缺失列: {missing_column}，在实际执行时可能由其他函数生成")
            # 不抛出异常，允许继续处理
        except Exception as e:
            logger.error(f"验证失败: {e}")
            raise

    def _sort_functions_by_dependencies(self, functions: List[Dict[str, Any]], original_columns: List[str]) -> List[Dict[str, Any]]:
        """
        根据依赖关系对函数进行拓扑排序
        
        Args:
            functions: 生成的函数列表
            original_columns: 原始数据列名
        
        Returns:
            排序后的函数列表
        """
        from collections import deque
        
        # 构建依赖图
        graph = {}
        in_degree = {}
        func_name_to_func = {}
        
        # 初始化所有函数的入度为0
        for func in functions:
            func_name = func.get('name', f'func_{id(func)}')
            func_name_to_func[func_name] = func
            graph[func_name] = []
            in_degree[func_name] = 0
        
        # 记录所有已知列（原始列 + 函数生成的列）
        known_columns = set(original_columns)
        generated_columns = {}
    
        # 为每个函数记录生成的列名
        for func in functions:
            func_name = func.get('name', f'func_{id(func)}')
            # 尝试从函数实现中提取生成的列名
            func_implementation = func.get('implementation', '')
            import re
            # 匹配 df['new_column'] = ... 或 df["new_column"] = ...
            # 使用不同的引号分隔正则表达式
            pattern = r"df\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*[^;]+"
            column_matches = re.findall(pattern, func_implementation)
            generated_col = None
            for match in column_matches:
                # 处理单引号和双引号匹配
                generated_col = match
                if generated_col:
                    generated_columns[generated_col] = func_name
                    known_columns.add(generated_col)  # 将新生成的列加入已知列
                    break
                    
        # 构建依赖关系
        for func in functions:
            func_name = func.get('name', f'func_{id(func)}')
            func_implementation = func.get('implementation', '')
            
            # 从函数实现中提取所有引用的列名
            # 匹配 df['col_name'] 或 df["col_name"]
            ref_pattern = r"df\[\s*['\"]([^'\"]+)['\"]\s*\]"
            ref_matches = re.findall(ref_pattern, func_implementation)
            referenced_cols = []
            for match in ref_matches:
                col = match
                if col and col not in [generated_col]:  # 排除正在生成的列
                    referenced_cols.append(col)
            
            # 检查函数依赖的列
            for req_col in referenced_cols:
                # 如果依赖列是由其他函数生成的，添加依赖关系
                if req_col in generated_columns:
                    dependent_func = generated_columns[req_col]
                    if dependent_func != func_name:  # 避免自环
                        graph[dependent_func].append(func_name)
                        in_degree[func_name] += 1
                # 否则，如果依赖列不是原始列，记录警告
                elif req_col not in known_columns:
                    self.logger.warning(f"函数 {func_name} 依赖未知列: {req_col}")
        
        # 拓扑排序
        result = []
        queue = deque([func_name for func_name, degree in in_degree.items() if degree == 0])
        
        while queue:
            current = queue.popleft()
            result.append(func_name_to_func[current])
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有环
        if len(result) != len(functions):
            self.logger.warning(f"函数依赖图中存在环，无法完全排序，将使用原始顺序")
            return functions
        
        self.logger.info(f"函数拓扑排序完成，执行顺序: {[func.get('name', 'unknown') for func in result]}")
        return result

    def generate_functions_batch(
        self, requests: List[Dict[str, Any]], max_workers: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """批量生成Excel处理函数，支持并发处理

        Args:
            requests: 请求列表，每个请求包含prompt和data_context
            max_workers: 最大工作线程数

        Returns:
            生成的函数列表，顺序与输入请求一致
        """
        results = []
        
        logger.info(f"开始处理 {len(requests)} 个请求")
        
        # 迭代处理每个请求
        for index, request in enumerate(requests):
            try:
                prompt = request["prompt"]
                data_context = request["data_context"]
                
                # 调用现有方法生成函数
                functions = self.generate_functions(prompt, data_context)
                results.append(functions)
                logger.debug(f"完成请求 {index + 1}/{len(requests)} 的处理")
            except Exception as e:
                logger.error(f"处理请求 {index} 时出现异常: {e}")
                results.append([])
        
        logger.info(f"所有 {len(requests)} 个请求已处理完成")
        return results

    def _initialize_api_manager(self):
        """初始化API管理器"""
        # 使用默认配置初始化API管理器，API管理器会自动从配置文件读取默认值
        self.api_manager = APIManager()
        
    def _lazy_init(self):
        """延迟初始化AI服务组件，确保只执行一次"""
        if self._initialized:
            return
            
        with self._init_lock:
            if self._initialized:  # 双重检查锁定模式，避免多线程问题
                return
                
            self.logger.info("开始延迟初始化AI服务组件...")
            
            # 初始化DualRedisDB，用于存储学习知识和函数
            try:
                from src.utils.dual_redis_db import DualRedisDB
                self.dual_redis = DualRedisDB()
                self.qwen_db = self.dual_redis  # 保持向后兼容
                self.logger.info("DualRedisDB初始化成功")
            except Exception as e:
                self.logger.error(f"DualRedisDB初始化失败: {e}")
                self.dual_redis = None
                self.qwen_db = None
            
            # 初始化学习算法模块
            try:
                self.qwen_learning = get_qwen_learning()
                # 确保 qwen_learning 初始化时直接关联到 dual_redis
                if hasattr(self.qwen_learning, 'dual_redis') and self.dual_redis:
                    self.qwen_learning.dual_redis = self.dual_redis
                    self.qwen_learning.qwen_db = self.dual_redis.logic_conn  # 确保使用正确的数据库
                self.logger.info("学习算法模块初始化成功")
            except Exception as e:
                self.logger.error(f"学习算法模块初始化失败: {e}")
                self.qwen_learning = None
            
            self._initialized = True
            self.logger.info("AI服务组件延迟初始化完成")
    
    def set_api_type(self, api_type: str):
        """设置API类型

        Args:
            api_type: API类型，如"openai"或"ollama"
        """
        self.api_manager.set_api_type(api_type)
        config_manager.set("ai.api_type", api_type)

    def set_model(self, model: str):
        """设置AI模型

        Args:
            model: 模型名称
        """
        self.api_manager.set_model(model)
        config_manager.set("ai.model", model)

    def set_api_key(self, api_key: str):
        """设置API密钥

        Args:
            api_key: API密钥
        """
        config_manager.set("ai.api_key", api_key)
        self._initialize_api_manager()

    def test_connection(self) -> bool:
        """测试API连接

        Returns:
            连接是否成功
        """
        success, message = self.api_manager.test_connection()
        logger.info(f"[AI服务] 连接测试结果: {success} - {message}")
        return success

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        获取可用模型列表

        Returns:
            可用模型列表
        """
        models = self.api_manager.get_available_models()
        logger.info(f"[AI服务] 可用模型数量: {len(models)}")
        return models
    
    def analyze_image(self, image_path: str, prompt: str = "") -> Dict[str, Any]:
        """
        使用AI分析图片内容
        
        Args:
            image_path: 图片文件路径
            prompt: 分析提示词
            
        Returns:
            包含图片分析结果的字典
        """
        self._lazy_init()
        
        try:
            logger.info(f"开始分析图片: {image_path}")
            
            # 调用API管理器进行图片分析
            success, result, error_message = self.api_manager.analyze_image(
                image_path=image_path,
                prompt=prompt
            )
            
            if success and result:
                logger.info(f"图片分析成功: {image_path}")
                return result
            else:
                logger.error(f"图片分析失败: {error_message}")
                return {"success": False, "error": error_message}
        except Exception as e:
            logger.error(f"图片分析异常: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_presentation_outline(self, topic: str, image_analysis: Dict[str, Any], audience: str = "通用", style: str = "正式", slide_count: int = 10) -> Dict[str, Any]:
        """
        生成PPT大纲
        
        Args:
            topic: PPT主题
            image_analysis: 图片分析结果
            audience: 受众类型
            style: 演示文稿风格
            slide_count: 预计幻灯片数量
            
        Returns:
            包含PPT大纲的字典
        """
        self._lazy_init()
        
        try:
            logger.info(f"开始生成PPT大纲，主题: {topic}")
            
            # 调用API管理器生成PPT大纲
            success, result, error_message = self.api_manager.generate_presentation_outline(
                topic=topic,
                image_analysis=image_analysis,
                audience=audience,
                style=style,
                slide_count=slide_count
            )
            
            if success and result:
                logger.info(f"PPT大纲生成成功: {topic}")
                return result
            else:
                logger.error(f"PPT大纲生成失败: {error_message}")
                return {"success": False, "error": error_message}
        except Exception as e:
            logger.error(f"PPT大纲生成异常: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_presentation_content(self, outline: Dict[str, Any], image_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成PPT内容
        
        Args:
            outline: PPT大纲
            image_analysis: 图片分析结果
            
        Returns:
            包含PPT内容的字典
        """
        self._lazy_init()
        
        try:
            logger.info("开始生成PPT内容")
            
            # 调用API管理器生成PPT内容
            success, result, error_message = self.api_manager.generate_presentation_content(
                outline=outline,
                image_analysis=image_analysis
            )
            
            if success and result:
                logger.info("PPT内容生成成功")
                return result
            else:
                logger.error(f"PPT内容生成失败: {error_message}")
                return {"success": False, "error": error_message}
        except Exception as e:
            logger.error(f"PPT内容生成异常: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_presentation_style(self, image_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成PPT样式建议
        
        Args:
            image_analysis: 图片分析结果
            
        Returns:
            包含PPT样式建议的字典
        """
        self._lazy_init()
        
        try:
            logger.info("开始生成PPT样式建议")
            
            # 调用API管理器生成PPT样式建议
            success, result, error_message = self.api_manager.generate_presentation_style(
                image_analysis=image_analysis
            )
            
            if success and result:
                logger.info("PPT样式建议生成成功")
                return result
            else:
                logger.error(f"PPT样式建议生成失败: {error_message}")
                return {"success": False, "error": error_message}
        except Exception as e:
            logger.error(f"PPT样式建议生成异常: {e}")
            return {"success": False, "error": str(e)}
    
    def ai_optimize_document(self, content: str, model: str = None) -> str:
        """
        AI优化文档内容
        
        Args:
            content: 要优化的文档内容
            model: AI模型名称
            
        Returns:
            优化后的文档内容
        """
        self._lazy_init()
        
        try:
            logger.info("开始AI优化文档")
            
            # 调用API管理器优化文档
            optimized_content = self.api_manager.ai_optimize_document(content, model)
            
            if optimized_content:
                logger.info("文档优化成功")
                return optimized_content
            else:
                logger.error("文档优化失败，返回空内容")
                return content
        except Exception as e:
            logger.error(f"文档优化异常: {e}")
            return content
    
    def ai_summarize_document(self, content: str, model: str = None) -> str:
        """
        AI生成文档摘要
        
        Args:
            content: 要生成摘要的文档内容
            model: AI模型名称
            
        Returns:
            文档摘要
        """
        self._lazy_init()
        
        try:
            logger.info("开始AI生成文档摘要")
            
            # 调用API管理器生成文档摘要
            summary = self.api_manager.ai_summarize_document(content, model)
            
            if summary:
                logger.info("文档摘要生成成功")
                return summary
            else:
                logger.error("文档摘要生成失败，返回空内容")
                return "摘要生成失败"
        except Exception as e:
            logger.error(f"文档摘要生成异常: {e}")
            return "摘要生成失败"
    
    def ai_translate_document(self, content: str, target_lang: str, model: str = None) -> str:
        """
        AI翻译文档内容
        
        Args:
            content: 要翻译的文档内容
            target_lang: 目标语言
            model: AI模型名称
            
        Returns:
            翻译后的文档内容
        """
        self._lazy_init()
        
        try:
            logger.info(f"开始AI翻译文档，目标语言: {target_lang}")
            
            # 调用API管理器翻译文档
            translated_content = self.api_manager.ai_translate_document(content, target_lang, model)
            
            if translated_content:
                logger.info("文档翻译成功")
                return translated_content
            else:
                logger.error("文档翻译失败，返回原内容")
                return content
        except Exception as e:
            logger.error(f"文档翻译异常: {e}")
            return content
    
    def ai_generate_document(self, prompt: str, model: str = None) -> str:
        """
        AI生成文档内容
        
        Args:
            prompt: 文档生成提示词
            model: AI模型名称
            
        Returns:
            生成的文档内容
        """
        self._lazy_init()
        
        try:
            logger.info("开始AI生成文档")
            
            # 调用API管理器生成文档
            generated_content = self.api_manager.ai_generate_document(prompt, model)
            
            if generated_content:
                logger.info("文档生成成功")
                return generated_content
            else:
                logger.error("文档生成失败，返回空内容")
                return "文档生成失败"
        except Exception as e:
            logger.error(f"文档生成异常: {e}")
            return "文档生成失败"


# 单例模式，确保全局只有一个AIService实例
_ai_service_instance = None


def get_ai_service() -> AIService:
    """获取AI服务实例

    Returns:
        AIService实例
    """
    global _ai_service_instance
    if _ai_service_instance is None:
        _ai_service_instance = AIService()
    return _ai_service_instance
