#!/usr/bin/env python3
"""
AI函数生成与调用的核心入口
负责动态提示词生成、AI服务调用、函数解析与应用
"""

import os
import sys
import pandas as pd
import logging
import json
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.ai_service import get_ai_service
from src.modules.prompt_generator import PromptGenerator
from src.modules.vectorized_function_converter import VectorizedFunctionConverter
from src.utils.dual_redis_db import DualRedisDB

logger = logging.getLogger("MultiColumnProcessor")

class MultiColumnProcessor:
    """
    AI函数生成与调用的核心处理器
    负责根据需求和数据上下文生成AI函数，并应用于数据处理
    """
    
    def __init__(self):
        """
        初始化多列处理器
        """
        self.ai_service = None
        self.prompt_generator = PromptGenerator()
        self.vectorized_converter = VectorizedFunctionConverter()
        self.logger = logger
        # 使用 DualRedisDB 替代原来的 QwenDB
        self.dual_redis = DualRedisDB()
        self.qwen_db = self.dual_redis  # 保持向后兼容
        # 记录最后一次失败的代码
        self.last_failed_code = []
    
    def set_ai_service(self, ai_service):
        """
        设置AI服务实例
        
        Args:
            ai_service: AI服务实例
        """
        self.ai_service = ai_service
    
    def _get_ai_service(self):
        """
        获取AI服务实例
        
        Returns:
            AI服务实例
        """
        if not self.ai_service:
            self.ai_service = get_ai_service()
        return self.ai_service
    
    def _load_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        加载Excel数据
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            加载后的DataFrame
        """
        try:
            df = pd.read_excel(file_path)
            self.logger.info(f"成功加载Excel文件: {file_path}")
            self.logger.info(f"数据形状: {df.shape}")
            self.logger.info(f"数据列: {list(df.columns)}")
            return df
        except Exception as e:
            self.logger.error(f"加载Excel文件失败: {e}")
            raise
    
    def _generate_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成数据上下文
        
        Args:
            df: 数据DataFrame
            
        Returns:
            数据上下文字典
        """
        # 识别列类型
        numeric_columns = list(df.select_dtypes(include=['number']).columns)
        date_columns = list(df.select_dtypes(include=['datetime64']).columns)
        categorical_columns = list(df.select_dtypes(include=['object']).columns)
        
        # 获取数据类型信息
        data_types = {col: str(df[col].dtype) for col in df.columns}
        
        # 生成数据上下文 - 减少sample_data的干扰，只保留必要信息
        data_context = {
            "columns": list(df.columns),
            "data_types": data_types,
            "data_shape": df.shape,
            "numeric_columns": numeric_columns,
            "date_columns": date_columns,
            "categorical_columns": categorical_columns,
            # 只保留少量样本数据，减少干扰
            "sample_data": [],
            "data_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_count": len(numeric_columns),
                "date_count": len(date_columns),
                "categorical_count": len(categorical_columns)
            }
        }
        
        self.logger.info(f"生成数据上下文: {data_context}")
        return data_context
    
    def generate_multi_column_functions(self, requirement: str, data_context: Any, last_error: str = "", iteration: int = 1, max_iterations: int = 3) -> List[Dict[str, Any]]:
        """
        生成多列处理函数
        
        Args:
            requirement: 用户需求
            data_context: 数据上下文，可以是DataFrame或上下文字典
            last_error: 上一次的错误信息，用于优化提示词
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数
        
        Returns:
            生成的函数列表
        """
        try:
            # 如果传入的是DataFrame，转换为数据上下文
            if isinstance(data_context, pd.DataFrame):
                data_context = self._generate_data_context(data_context)
            
            # 获取AI服务
            ai_service = self._get_ai_service()
            
            # 从 Redis 获取历史错误记忆
            historical_error = ""
            if ai_service.qwen_learning:
                historical_error = ai_service.qwen_learning.get_error_memory(requirement)
            
            # 生成提示词
            prompt = self.prompt_generator.generate_prompt(
                requirement=requirement,
                data_context=data_context
            )
            
            # 合并所有错误信息
            all_errors = []
            if last_error:
                all_errors.append(f"上一次迭代错误: {last_error}")
            if historical_error:
                all_errors.append(f"历史错误: {historical_error}")
            
            # 如果有错误信息，添加到提示词中
            if all_errors:
                prompt += "\n\n" + "\n\n".join(all_errors)
                prompt += "\n\n请确保生成的函数能够："
                prompt += "\n1. 生成新的计算列，例如：毛利率 = (销售额 - 成本) / 销售额 * 100"
                prompt += "\n2. 使用有意义的列名，例如：毛利率、年销售额总和、同比增长等"
                prompt += "\n3. 直接执行，无需额外修改"
                prompt += "\n4. 包含完整的异常处理"
                prompt += "\n5. 特别注意避免上述历史错误和上一次迭代错误"
            
            # 当接近最大迭代次数时，从Redis获取类似成功案例作为参考
            if iteration >= max_iterations - 1:
                ai_service = self._get_ai_service()
                if ai_service.qwen_learning:
                    # 从Redis获取类似成功案例
                    similar_cases = self.dual_redis.db_conn.get(f"success_cases:{requirement[:50]}")
                    if similar_cases:
                        try:
                            similar_cases = json.loads(similar_cases)
                            if similar_cases:
                                prompt += "\n\n参考成功案例："
                                for i, case in enumerate(similar_cases[:3]):
                                    prompt += f"\n案例 {i+1}: {case[:200]}..."
                        except json.JSONDecodeError:
                            self.logger.warning("解析类似成功案例失败")
            
            # 获取AI服务
            ai_service = self._get_ai_service()
            
            # 调用AI生成函数
            functions = ai_service.generate_functions(prompt, data_context)
            
            # 验证生成的函数是否符合要求
            valid_functions = []
            for func in functions:
                func_implementation = func.get('implementation', '')
                # 检查函数是否包含生成新列的代码
                if 'df[' in func_implementation or 'df["' in func_implementation:
                    valid_functions.append(func)
                else:
                    self.logger.warning(f"函数 {func.get('name', 'unknown')} 未生成新列，跳过")
            
            if valid_functions:
                self.logger.info(f"成功生成 {len(valid_functions)} 个有效的函数")
                return valid_functions
            else:
                self.logger.warning("没有生成有效的函数，返回原始函数列表")
                return functions
        except json.JSONDecodeError as e:
            self.logger.error(f"生成函数失败: AI返回的不是有效的JSON格式")
            # 增强容错性：尝试从AI响应中直接提取Python代码
            import re
            ai_service = self._get_ai_service()
            # 从AI响应中尝试提取Python代码
            last_response = getattr(ai_service, 'last_response', '')
            if last_response:
                extracted_functions = self._extract_functions_from_response(last_response, data_context)
                if extracted_functions:
                    self.logger.info(f"从AI响应中提取到 {len(extracted_functions)} 个函数")
                    return extracted_functions
            return []
        except Exception as e:
            self.logger.error(f"生成函数失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def _extract_functions_from_response(self, response: str, data_context: Dict) -> List[Dict[str, Any]]:
        """
        从AI响应中提取函数定义
            
        Args:
            response: AI的原始响应
            data_context: 数据上下文
            
        Returns:
            提取到的函数列表
        """
        import re
        import json
            
        # 首先尝试从响应中提取JSON部分
        json_pattern = r'\[\s*\{.*?\}\s*\]'  # 匹配JSON数组
        json_match = re.search(json_pattern, response, re.DOTALL)
            
        if json_match:
            try:
                json_str = json_match.group(0)
                # 尝试修复JSON格式
                json_str = self._fix_json_format(json_str)
                functions = json.loads(json_str)
                if isinstance(functions, list):
                    return functions
            except json.JSONDecodeError:
                pass
            
        # 如果JSON解析失败，尝试直接提取Python代码
        # 匹配函数定义模式
        code_pattern = r'def\s+\w+\s*\([^)]*\):[^{]+?return[^}]*df[^}]*'
        code_matches = re.findall(code_pattern, response, re.DOTALL)
            
        # 提取Python代码块
        code_block_pattern = r'```python\s*(.*?)\s*```'
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
            
        if code_blocks:
            functions = []
            for i, code_block in enumerate(code_blocks):
                # 尝试从代码块中提取函数名
                func_name_match = re.search(r'def\s+(\w+)', code_block)
                func_name = func_name_match.group(1) if func_name_match else f'extracted_func_{i}'
                    
                functions.append({
                    'name': func_name,
                    'description': f'从AI响应中提取的函数: {func_name}',
                    'implementation': code_block,
                    'required_columns': []
                })
            return functions
            
        # 如果以上都失败，尝试提取列计算代码
        df_assignment_pattern = r'df\s*\[\s*["\']([^"\']+)["\']\s*\]\s*=\s*[^\n;]+(?:\n|;)' 
        df_assignments = re.findall(df_assignment_pattern, response)
            
        if df_assignments:
            # 创建一个包含所有列计算的函数
            function_code = 'def dynamic_calculation(df):\n'
            for line in response.split('\n'):
                if 'df[' in line and '=' in line:
                    function_code += f'    {line.strip()}\n'
            function_code += '    return df\n'
                
            return [{
                'name': 'dynamic_calculation',
                'description': '从AI响应中提取的动态计算函数',
                'implementation': function_code,
                'required_columns': []
            }]
            
        return []
        
    def _fix_json_format(self, json_str: str) -> str:
        """
        尝试修复JSON格式问题
            
        Args:
            json_str: 原始JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        # 修复常见的JSON格式问题
        # 1. 修复未转义的引号
        import re
        # 替换可能在字符串值中未转义的引号
        json_str = re.sub(r'([^\\])"([^\\])', r'\1\\"\2', json_str)
            
        # 2. 修复缺少逗号的问题
        json_str = re.sub(r'\}\s*\{', r'}, {', json_str)
            
        # 3. 确保字符串值被正确包围
        lines = json_str.split('\n')
        fixed_lines = []
        for line in lines:
            # 简单修复一些常见的格式问题
            fixed_lines.append(line.strip())
            
        return '\n'.join(fixed_lines)
    
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
            pattern = r"df\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*[^\n;]+"
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
    
    def _execute_direct_calculation(self, df: pd.DataFrame, func: Dict[str, Any]) -> pd.DataFrame:
        """
        直接执行核心计算逻辑，避免函数定义的缩进问题
    
        Args:
            df: 输入数据DataFrame
            func: 函数字典，包含函数实现和元数据
            
        Returns:
            处理后的DataFrame
        """
        import pandas as pd
        import numpy as np
        
        try:
            temp_df = df.copy()
            func_implementation = func.get('implementation', '')
            
            # 如果实现中包含具体的计算逻辑，直接执行
            if func_implementation:
                # 预处理函数实现，确保导入语句存在
                processed_func = func_implementation
                if 'import pandas' not in processed_func:
                    processed_func = 'import pandas as pd\nimport numpy as np\nimport datetime\n' + processed_func
                
                # 修复可能的语法问题
                processed_func = processed_func.replace('pd.np.', 'np.')
                processed_func = processed_func.replace('pd.np', 'np')
                
                # 自动注入日期转换提示（用于解决.dt访问器问题）
                if '.dt.' in processed_func and 'pd.to_datetime' not in processed_func:
                    # 为使用.dt访问器的列添加日期转换
                    import re
                    # 查找所有使用.dt访问器的列
                    dt_matches = re.findall(r"df\[['\"](\w+)['\"]\]\.dt\.", processed_func)
                    for col in set(dt_matches):
                        # 在函数开始处添加日期转换代码
                        conversion_code = f"    df['{col}'] = pd.to_datetime(df['{col}'])\n"
                        # 找到函数定义行并插入日期转换
                        lines = processed_func.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip().startswith('def '):
                                lines.insert(i + 1, conversion_code)
                                break
                        processed_func = '\n'.join(lines)
                
                # 创建本地命名空间
                import numpy as np
                local_namespace = {
                    'pd': pd,
                    'np': np,
                    'df': temp_df
                }
                
                # 执行函数定义
                exec(processed_func, {'pd': pd, 'np': np}, local_namespace)
                
                # 获取函数对象并执行
                func_name = func.get('name', '')
                if func_name in local_namespace:
                    func_obj = local_namespace[func_name]
                    result_df = func_obj(df.copy())
                    
                    # 保持原始日期列的格式一致（将转换后的日期列恢复为原始格式）
                    for col in df.columns:
                        if df[col].dtype == 'object':  # 原始为字符串格式
                            # 检查该列是否包含日期格式（YYYY-MM-DD）
                            sample_vals = df[col].dropna().head(5)
                            if len(sample_vals) > 0:
                                is_date_str = all(isinstance(val, str) and 
                                                len(val) == 10 and 
                                                val.count('-') == 2 and
                                                val.replace('-', '').isdigit() 
                                                for val in sample_vals if pd.notna(val))
                                if is_date_str and col in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df[col]):
                                    # 将日期时间列转换回字符串格式
                                    result_df[col] = result_df[col].dt.strftime('%Y-%m-%d').replace('NaT', None)
                    
                    new_columns = list(set(result_df.columns) - set(df.columns))
                    if new_columns:
                        self.logger.info(f"直接计算成功生成新列: {new_columns}")
                        self.logger.info(f"直接计算执行后数据形状: {result_df.shape}")
                    else:
                        self.logger.info(f"直接计算未生成新列")
                    
                    return result_df
            
            # 如果没有实现或执行失败，返回原始df
            return df
        except KeyError as e:
            self.logger.error(f"直接计算执行失败，缺少依赖列: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # 对于KeyError，记录缺失的依赖项，但继续处理
            missing_column = str(e).strip("'\"")
            self.logger.info(f"检测到缺失列: {missing_column}，将尝试在后续函数中生成")
            return df
        except Exception as e:
            self.logger.error(f"直接计算执行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def _execute_simplified_function(self, df: pd.DataFrame, func: Dict[str, Any]) -> pd.DataFrame:
        """
        执行简化后的函数，包含核心计算逻辑
    
        Args:
            df: 输入数据DataFrame
            func: 函数字典，包含函数实现和元数据
            
        Returns:
            处理后的DataFrame
        """
        func_name = func.get('name', 'unknown')
        func_implementation = func.get('implementation', '')
        
        try:
            self.logger.info(f"尝试修复并执行AI生成的函数")
            
            # 预处理函数实现，确保导入语句存在
            processed_func = func_implementation
            if 'import pandas' not in processed_func:
                processed_func = 'import pandas as pd\nimport numpy as np\nimport datetime\n' + processed_func
            
            # 修复可能的语法问题
            processed_func = processed_func.replace('pd.np.', 'np.')
            processed_func = processed_func.replace('pd.np', 'np')
            
            # 自动注入日期转换提示（用于解决.dt访问器问题）
            if '.dt.' in processed_func and 'pd.to_datetime' not in processed_func:
                # 为使用.dt访问器的列添加日期转换
                import re
                # 查找所有使用.dt访问器的列
                dt_matches = re.findall(r"df\[['\"](\w+)['\"]\]\.dt\.", processed_func)
                for col in set(dt_matches):
                    # 在函数开始处添加日期转换代码
                    conversion_code = f"    df['{col}'] = pd.to_datetime(df['{col}'])\n"
                    # 找到函数定义行并插入日期转换
                    lines = processed_func.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            lines.insert(i + 1, conversion_code)
                            break
                    processed_func = '\n'.join(lines)
            
            self.logger.info(f"处理后的函数: {processed_func[:200]}...")
            
            # 执行函数
            import numpy as np
            local_namespace = {'pd': pd, 'np': np, 'df': df.copy()}
            exec(processed_func, {'pd': pd, 'np': np}, local_namespace)
            
            if func_name in local_namespace:
                func_obj = local_namespace[func_name]
                result_df = func_obj(df.copy())  # 使用df的副本以避免修改原始数据)
                
                # 保持原始日期列的格式一致（将转换后的日期列恢复为原始格式）
                for col in df.columns:
                    if df[col].dtype == 'object':  # 原始为字符串格式
                        # 检查该列是否包含日期格式（YYYY-MM-DD）
                        sample_vals = df[col].dropna().head(5)
                        if len(sample_vals) > 0:
                            is_date_str = all(isinstance(val, str) and 
                                            len(val) == 10 and 
                                            val.count('-') == 2 and
                                            val.replace('-', '').isdigit() 
                                            for val in sample_vals if pd.notna(val))
                            if is_date_str and col in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df[col]):
                                # 将日期时间列转换回字符串格式
                                result_df[col] = result_df[col].dt.strftime('%Y-%m-%d').replace('NaT', None)
                
                if isinstance(result_df, pd.DataFrame):
                    new_columns = list(set(result_df.columns) - set(df.columns))
                    if new_columns:
                        self.logger.info(f"简化函数成功生成新列: {new_columns}")
                    else:
                        self.logger.info(f"简化函数未生成新列")
                    return result_df
                else:
                    self.logger.warning(f"函数 {func_name} 未返回DataFrame，返回类型: {type(result_df)}")
                    return df
            else:
                self.logger.warning(f"函数 {func_name} 未在命名空间中找到")
                return df
        except KeyError as e:
            self.logger.error(f"函数执行失败，缺少依赖列: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # 对于KeyError，记录缺失的依赖项，以便后续处理
            missing_column = str(e).strip("'\"")
            self.logger.info(f"检测到缺失列: {missing_column}，将尝试在后续函数中生成")
            return df
        except Exception as e:
            self.logger.error(f"简化函数执行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return df
    
    def _report_error(self, requirement: str, func: Dict[str, Any], error: Exception, traceback_str: str, attempt: int = 1, df: pd.DataFrame = None):
        """
        统一报告错误给AI服务

        Args:
            requirement: 用户需求
            func: 函数字典
            error: 错误异常
            traceback_str: 堆栈跟踪字符串
            attempt: 尝试次数
            df: 当前数据帧，用于获取列名信息
        """
        # 错误分类
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 针对KeyError进行特殊处理
        if error_type == "KeyError":
            missing_column = error_msg.strip("'\"")
            available_columns = list(df.columns) if df is not None else []
            error_msg = f"列名错误！缺少列 '{missing_column}'。当前可用列：{available_columns}"
        
        # 针对ValueError进行特殊处理
        elif error_type == "ValueError":
            error_msg = f"值错误！请检查数据类型和计算逻辑。原始错误：{error_msg}"
        
        # 针对TypeError进行特殊处理
        elif error_type == "TypeError":
            error_msg = f"类型错误！请检查函数调用和参数类型。原始错误：{error_msg}"
        
        ai_service = self._get_ai_service()
        ai_service.add_error({
            "prompt": requirement,
            "error": error_msg,
            "error_type": error_type,
            "implementation": func.get('implementation', ''),
            "function_name": func.get('name', 'unknown'),
            "traceback": traceback_str,
            "attempt": attempt,
            "available_columns": list(df.columns) if df is not None else []
        })
        
        # 核心：将失败现场存入 Redis 供下次 Prompt 参考
        if ai_service.qwen_learning:
            ai_service.qwen_learning.learn_from_error(
                prompt=requirement,
                enhanced_prompt=requirement,
                data_context={"columns": list(df.columns) if df is not None else []},
                error=error_msg,
                error_type=error_type,
                traceback=traceback_str,
                attempt=attempt
            )
    
    def process_data(self, df: pd.DataFrame, functions: List[Dict[str, Any]], requirement: str = "") -> pd.DataFrame:
        """
        应用生成的函数处理数据
    
        Args:
            df: 原始数据DataFrame
            functions: 生成的函数列表
            requirement: 用户需求，用于错误反馈
            
        Returns:
            处理后的DataFrame
        """
        try:
            processed_df = df.copy()
            original_columns = list(df.columns)
            
            # 根据依赖关系对函数进行拓扑排序
            sorted_functions = self._sort_functions_by_dependencies(functions, original_columns)
            
            self.logger.info(f"开始处理数据，原始列: {original_columns}")
            self.logger.info(f"原始数据形状: {processed_df.shape}")
            
            for func in sorted_functions:
                func_name = func.get('name', 'unknown')
                func_implementation = func.get('implementation', '')
                func_description = func.get('description', '')
                required_columns = func.get('required_columns', [])
                
                self.logger.info(f"\n=== 应用函数: {func_name} - {func_description} ===")
                self.logger.info(f"函数依赖列: {required_columns}")
                
                # 检查依赖列是否存在
                missing_columns = []
                for req_col in required_columns:
                    if req_col not in processed_df.columns and req_col not in original_columns:
                        missing_columns.append(req_col)
                
                if missing_columns:
                    self.logger.warning(f"函数 {func_name} 缺少依赖列: {missing_columns}，尝试执行...")
                else:
                    self.logger.info(f"函数 {func_name} 所需依赖列均存在")
                
                # 执行函数: 尝试多种策略
                try:
                    # 策略1: 执行简化后的函数（优先使用AI生成的函数）
                    self.logger.info(f"执行策略1: 简化函数（优先使用AI生成的函数）")
                    processed_df = self._execute_simplified_function(processed_df, func)
                    
                    # 检查是否生成了新列
                    new_columns_after_simplified = list(set(processed_df.columns) - set(df.columns))
                    if not new_columns_after_simplified:
                        # 策略2: 直接执行核心计算逻辑（仅在简化函数失败时尝试）
                        self.logger.info(f"简化函数未生成新列，尝试策略2: 直接计算核心逻辑")
                        processed_df = self._execute_direct_calculation(processed_df, func)
                except KeyError as e:
                    # 处理依赖列缺失的错误
                    missing_column = str(e)
                    self.logger.warning(f"函数 {func_name} 执行失败，缺少依赖列: {missing_column}")
                    
                    # 记录缺少的依赖项，供后续迭代使用
                    error_msg = f"缺少依赖列: {missing_column}"
                    import traceback
                    tb_str = traceback.format_exc()
                    self.logger.error(tb_str)
                    
                    # 报告错误给AI服务
                    self._report_error(requirement, func, e, tb_str, df=processed_df)
                    
                    # 跳过当前函数，继续下一个（依赖项可能在其他函数中生成）
                    continue
                except Exception as e:
                    self.logger.error(f"函数 {func_name} 执行失败: {e}")
                    import traceback
                    tb_str = traceback.format_exc()
                    self.logger.error(tb_str)
                    
                    # 报告错误给AI服务
                    self._report_error(requirement, func, e, tb_str, df=processed_df)
            
            final_columns = list(processed_df.columns)
            new_columns = list(set(final_columns) - set(original_columns))
            self.logger.info(f"\n=== 数据处理完成 ===")
            self.logger.info(f"原始列数: {len(original_columns)}, 处理后列数: {len(final_columns)}")
            self.logger.info(f"新增列: {new_columns}")
            self.logger.info(f"最终数据形状: {processed_df.shape}")
            
            return processed_df
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def _validate_result(self, df: pd.DataFrame, processed_df: pd.DataFrame, requirement: str) -> tuple[bool, str]:
        """
        验证处理结果是否符合逻辑
        
        Args:
            df: 原始DataFrame
            processed_df: 处理后的DataFrame
            requirement: 用户需求
            
        Returns:
            (success, result_msg) 元组，success为布尔值，result_msg为结果描述
        """
        try:
            # 基本验证：处理后的DataFrame不能为空
            if processed_df.empty:
                return False, "处理后的数据为空"
            
            # 基本验证：行数不应减少
            if len(processed_df) < len(df):
                return False, f"处理后行数减少: {len(processed_df)} < {len(df)}"
            
            # 根据需求类型进行特定验证
            if "总和" in requirement or "平均值" in requirement or "最大值" in requirement or "最小值" in requirement:
                # 对于计算类需求，应该生成新列
                new_columns = list(set(processed_df.columns) - set(df.columns))
                if not new_columns:
                    return False, f"计算类需求未生成新列"
            
            # 验证数值列的计算结果是否合理
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if f"{col}_总和" in processed_df.columns:
                    # 总和列的所有值应该相同
                    sum_values = processed_df[f"{col}_总和"].unique()
                    if len(sum_values) > 1:
                        return False, f"总和列 {col}_总和 包含多个不同值"
            
            return True, "验证通过"
        except Exception as e:
            return False, f"验证失败: {str(e)}"
    
    def process_multi_columns(self, file_path, requirement, max_iterations=3):
        """
        多列处理主函数，包含迭代验证逻辑
        
        Args:
            file_path: Excel文件路径
            requirement: 用户需求
            max_iterations: 最大迭代次数
            
        Returns:
            处理结果字典
        """
        try:
            # 加载数据
            df = pd.read_excel(file_path)
            current_iter = 0
            last_feedback = ""
            
            # 记录原始列名，用于错误分类
            original_columns = list(df.columns)
            
            while current_iter < max_iterations:
                self.logger.info(f"\n=== 迭代 {current_iter+1}/{max_iterations} ===")
                current_iter += 1
                
                # 1. 获取增强后的提示词（包含 Redis 中的历史避坑指南）
                data_context = self._generate_data_context(df)
                
                # 2. AI 生成代码
                functions = self.generate_multi_column_functions(requirement, data_context, last_feedback, current_iter, max_iterations)
                
                if not functions:
                    last_feedback = "AI未生成有效的函数"
                    self.logger.warning(last_feedback)
                    continue
                
                # 3. 应用函数处理数据
                processed_df = self.process_data(df, functions, requirement)
                
                # 4. 验证阶段
                success, result_msg = self._validate_result(df, processed_df, requirement)
                
                # 5. 学习与反馈
                ai_service = self._get_ai_service()
                if success:
                    # 迭代成功：记录经验并退出
                    for func in functions:
                        if ai_service.qwen_learning:
                            ai_service.qwen_learning.learn_from_iteration(
                                requirement=requirement,
                                code=func.get('implementation', ''),
                                success=True
                            )
                    
                    # 将成功案例存入Redis
                    success_case = {
                        "requirement": requirement,
                        "columns": original_columns,
                        "code": [func.get('implementation', '') for func in functions],
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    try:
                        self.dual_redis.db_conn.set(f"success_cases:{requirement[:50]}", json.dumps([func.get('implementation', '') for func in functions]))
                        self.logger.info("成功案例已存入Redis")
                    except Exception as e:
                        self.logger.warning(f"存入成功案例失败: {e}")
                    
                    # 清空最后失败代码记录
                    self.last_failed_code = []
                    
                    # 保存处理后的文件到临时位置
                    import tempfile
                    import os
                    temp_dir = tempfile.gettempdir()
                    temp_file = os.path.join(temp_dir, f"processed_{os.path.basename(file_path)}")
                    processed_df.to_excel(temp_file, index=False)
                    
                    return {
                        "success": True,
                        "file_path": temp_file,
                        "data": processed_df,
                        "message": f"成功，迭代次数: {current_iter}",
                        "new_columns": list(set(processed_df.columns) - set(df.columns))
                    }
                else:
                    # 迭代失败：记录错误，反馈给下一轮 AI
                    last_feedback = result_msg
                    
                    # 错误分类：针对不同类型的错误提供更具体的反馈
                    if "KeyError" in last_feedback:
                        # 列名错误：提供明确的列名清单
                        last_feedback = f"列名错误！当前可用列有：{original_columns}，请重新匹配。\n原始错误：{result_msg}"
                    elif "ValueError" in last_feedback:
                        # 值错误：提示数据类型问题
                        last_feedback = f"值错误！请检查数据类型和计算逻辑是否匹配。\n原始错误：{result_msg}"
                    elif "TypeError" in last_feedback:
                        # 类型错误：提示函数调用和参数问题
                        last_feedback = f"类型错误！请检查函数调用和参数类型是否正确。\n原始错误：{result_msg}"
                    elif "MultiIndex" in last_feedback and "index=False" in last_feedback:
                        # MultiIndex 导出问题
                        last_feedback = f"MultiIndex 导出错误！当存在 MultiIndex 时，必须设置 index=True，或平铺表头。\n原始错误：{result_msg}"
                    elif "Timedelta" in last_feedback:
                        # 时间差运算问题
                        last_feedback = f"时间差运算错误！对时间差进行运算前，必须使用 .dt.total_seconds() 转换为数值。\n原始错误：{result_msg}"
                    elif "UFuncNoLoopError" in last_feedback or "dtype('<U4')" in last_feedback:
                        # 类型混合问题
                        last_feedback = f"类型混合错误！检测到字符串与数字混合，请先执行 df.fillna(0) 并强制转换类型。\n原始错误：{result_msg}"
                    
                    # 记录迭代错误
                    for func in functions:
                        if ai_service.qwen_learning:
                            ai_service.qwen_learning.learn_from_iteration(
                                requirement=requirement,
                                code=func.get('implementation', ''),
                                error_msg=last_feedback,
                                success=False
                            )
                    
                    # 核心：将失败现场存入 Redis 供下次 Prompt 参考
                    if ai_service.qwen_learning:
                        ai_service.qwen_learning.learn_from_failure(
                            requirement=requirement,
                            error_msg=last_feedback
                        )
                    
                    # 记录最后一次失败的代码
                    self.last_failed_code = [func.get('implementation', '') for func in functions]
                    
                    self.logger.warning(f"迭代失败: {last_feedback}")
                    
            return {
                "success": False,
                "message": f"达到最大迭代次数 {max_iterations}，处理失败",
                "last_feedback": last_feedback
            }
        except Exception as e:
            self.logger.error(f"process_multi_columns 失败: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            self.logger.error(traceback_str)
            
            # 错误分类：针对不同类型的错误提供更具体的反馈
            error_msg = str(e)
            if "KeyError" in error_msg:
                # 列名错误：提供明确的列名清单
                error_msg = f"列名错误！当前可用列有：{list(df.columns) if 'df' in locals() else []}，请重新匹配。\n原始错误：{error_msg}"
            elif "ValueError" in error_msg:
                # 值错误：提示数据类型问题
                error_msg = f"值错误！请检查数据类型和计算逻辑是否匹配。\n原始错误：{error_msg}"
            elif "TypeError" in error_msg:
                # 类型错误：提示函数调用和参数问题
                error_msg = f"类型错误！请检查函数调用和参数类型是否正确。\n原始错误：{error_msg}"
            
            return {
                "success": False,
                "message": f"处理失败: {error_msg}",
                "traceback": traceback_str
            }
    
    def _analyze_dependencies_phase(self, data_context: Dict, requirement: str):
        """
        第一阶段：分析依赖关系 - 让AI只输出依赖关系表
        """
        # 首先尝试使用英文提示，避免中文字符导致的解析问题
        analysis_prompt = self._create_dependency_analysis_prompt(data_context, requirement)
        
        print("🔄 执行第一阶段：依赖关系分析...")
        
        # 获取AI服务实例
        ai_service = self._get_ai_service()
        
        # 使用process_single_cell方法直接获取AI文本响应
        success, response_text, error_message = ai_service.api_manager.process_single_cell(
            cell_content=analysis_prompt,
            system_prompt="你是一个数据分析专家，专门分析数据处理需求中的依赖关系。请使用英文列名，不要使用中文列名。",
            user_prompt="请分析以下数据处理需求的依赖关系，并以JSON格式返回。",
            max_tokens=1000
        )
        
        if success and response_text:
            # 尝试从响应中提取JSON
            dependency_analysis = self._extract_dependency_analysis_from_response(response_text)
            if dependency_analysis:
                print("✅ 依赖关系分析完成:")
                for item in dependency_analysis.get('dependency_analysis', []):
                    print(f"   - {item['column_name']}: 依赖于 {item['depends_on']}")
                return dependency_analysis
        
        print(f"❌ AI分析失败或返回无效格式: {error_message if not success else '解析失败'}")
        return None
    
    def _create_dependency_analysis_prompt(self, data_context: Dict, requirement: str) -> str:
        """
        创建依赖关系分析提示词
        """
        # 构建分析阶段的提示
        analysis_prompt = f"""请分析以下数据处理需求的依赖关系，并以JSON格式返回每列的依赖项。

数据列: {data_context['columns']}
数据类型: {data_context['data_types']}

处理需求: {requirement}

重要：必须使用原始数据中的中文列名，不要创建新的英文列名。使用数据中存在的列名：{data_context['columns']}。

请按照以下格式返回JSON:
{{
    "dependency_analysis": [
        {{
            "column_name": "需要生成的列名（使用中文或需求中指定的英文名）",
            "description": "列的计算描述",
            "depends_on": ["依赖的列名列表（必须是原始数据中存在的列名）"],
            "calculation_type": "计算类型，如条件计算、数学运算等"
        }}
    ]
}}

重要：只返回JSON，不要有任何其他文字。"""
        return analysis_prompt
    
    def _extract_dependency_analysis_from_response(self, response_text: str):
        """
        从AI响应中提取依赖关系分析
        """
        import re
        import json
        
        # 尝试直接解析完整JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 查找JSON部分
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试修复常见的JSON格式问题
                json_str = self._fix_json_format(json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # 如果直接JSON解析失败，尝试查找数组部分
        array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if array_match:
            array_str = array_match.group()
            try:
                # 将数组包装成完整的JSON对象
                wrapped_json = f'{{"dependency_analysis": {array_str}}}'
                return json.loads(wrapped_json)
            except json.JSONDecodeError:
                pass
        
        print(f"❌ 无法从AI响应中提取JSON: {response_text[:200]}...")
        return None
        # 构建分析阶段的提示
        analysis_prompt = f"""
请分析以下数据处理需求的依赖关系，并以JSON格式返回每列的依赖项。

数据列: {data_context['columns']}
数据类型: {data_context['data_types']}

处理需求: {requirement}

请按照以下格式返回JSON:
{{
    "dependency_analysis": [
        {{
            "column_name": "列名",
            "description": "列的计算描述",
            "depends_on": ["依赖的列名列表", "可以包含原始数据列或需要新生成的列"],
            "calculation_type": "计算类型，如条件计算、数学运算等"
        }}
    ]
}}

重要：只返回JSON，不要有任何其他文字。
"""
        
        print("🔄 执行第一阶段：依赖关系分析...")
        
        # 获取AI服务实例
        ai_service = self._get_ai_service()
        
        # 使用process_single_cell方法直接获取AI文本响应
        success, response_text, error_message = ai_service.api_manager.process_single_cell(
            cell_content=analysis_prompt,
            system_prompt="你是一个数据分析专家，专门分析数据处理需求中的依赖关系。",
            user_prompt="请分析以下数据处理需求的依赖关系，并以JSON格式返回。",
            max_tokens=1000
        )
        
        if success and response_text:
            # 尝试从响应中提取JSON
            try:
                # 查找JSON部分
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    dependency_analysis = json.loads(json_str)
                    print("✅ 依赖关系分析完成:")
                    for item in dependency_analysis.get('dependency_analysis', []):
                        print(f"   - {item['column_name']}: 依赖于 {item['depends_on']}")
                    return dependency_analysis
                else:
                    print(f"❌ 无法从AI响应中提取JSON: {response_text}")
                    return None
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误: {e}")
                print(f"AI响应: {response_text}")
                return None
        else:
            print(f"❌ AI分析失败: {error_message}")
            return None

    def _sort_dependencies_with_original_columns(self, dependency_analysis: Dict, original_columns: List[str]):
        """
        改进的拓扑排序 - 建立"原始列白名单"
        """
        print("🔄 执行第二阶段：依赖关系排序（考虑原始列）...")
        
        # 区分原始列和需要生成的新列
        original_set = set(original_columns)
        
        # 构建依赖图，但只对新列进行排序
        nodes = {}
        dependencies = {}
        
        for item in dependency_analysis.get('dependency_analysis', []):
            col_name = item['column_name']
            depends_on = item.get('depends_on', [])
            
            # 只对需要生成的列（非原始列）建立依赖图
            if col_name not in original_set:
                nodes[col_name] = item
                dependencies[col_name] = depends_on
        
        # 拓扑排序 - 只对需要生成的列进行排序
        sorted_new_columns = []
        visited = set()
        temp_visited = set()
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"循环依赖: {node}")
            if node in visited:
                return
                
            temp_visited.add(node)
            
            # 访问所有依赖项
            for dep in dependencies.get(node, []):
                # 只访问需要生成的列（非原始列）
                if dep not in original_set and dep not in visited:
                    visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            sorted_new_columns.append(node)
        
        # 对所有需要生成的节点进行访问
        for node in dependencies.keys():
            if node not in visited:
                try:
                    visit(node)
                except ValueError as e:
                    print(f"❌ 检测到循环依赖: {e}")
                    return None
        
        print("✅ 拓扑排序完成:")
        print("   灰色（原始列）:", list(original_set))
        for i, col in enumerate(sorted_new_columns):
            print(f"   蓝色（生成列）Level {i}: {col}")
        
        return {
            'original_columns': list(original_set),
            'new_columns': sorted_new_columns
        }

    def get_last_failed_code(self):
        """
        获取最后一次失败的代码，用于 Gradio 前端展示和人工修正
        
        Returns:
            最后一次失败的代码列表
        """
        return self.last_failed_code
    
    def _closed_loop_execute_and_validate(self, df: pd.DataFrame, requirement: str, sorted_result: Dict, dependency_analysis: Dict):
        """
        闭环训练：原子化执行 + 中间状态传递 + 硬性约束
        """
        print("🔄 执行第三阶段：闭环训练原子化顺序执行与验证...")
        
        # 获取AI服务实例
        ai_service = self._get_ai_service()
        
        # 初始化执行状态
        executed_columns = set(df.columns)  # 初始可用列 = 原始列
        executed_functions = []
        
        # 记录每一步的中间状态
        intermediate_states = {}
        
        print(f"   🟨 灰色：原始列已就绪 - {list(df.columns)}")
        
        # 逐个执行需要生成的新列（蓝色）
        for level, col_name in enumerate(sorted_result['new_columns']):
            print(f"\n--- Level {level}: 处理蓝色列 {col_name} ---")
            
            # 获取列信息
            col_info = None
            for item in dependency_analysis.get('dependency_analysis', []):
                if item['column_name'] == col_name:
                    col_info = item
                    break
            
            if not col_info:
                print(f"   ❌ 未找到列信息: {col_name}")
                continue
            
            # 检查依赖是否都已满足（只考虑需要生成的列）
            required_deps = set(col_info.get('depends_on', []))
            missing_deps = required_deps - executed_columns
            
            if missing_deps:
                print(f"   ❌ 依赖未满足: {col_name} 需要 {missing_deps}，但只有 {executed_columns}")
                
                # 记录依赖顺序错误到数据库（包含拓扑图信息）
                self._record_dependency_error_with_topology(ai_service, requirement, col_name, missing_deps, executed_columns, sorted_result)
                continue
            
            # 生成针对单个列的函数
            import re
            safe_col_name = re.sub(r'[^\w]', '_', col_name.lower())
            
            # 强制传递中间快照：包含已执行的列信息
            executed_state_info = ""
            if intermediate_states:
                executed_state_info = f"\n已执行的列状态:\n"
                for executed_col, state in intermediate_states.items():
                    executed_state_info += f"- {executed_col}: {state}\n"
            
            function_prompt = f"""
为以下数据列生成处理函数：

数据上下文:
- 当前可用列: {list(executed_columns)}
- 需要生成的列: {col_name}
- 列描述: {col_info.get('description', '')}
- 计算类型: {col_info.get('calculation_type', '')}
- 依赖列: {col_info.get('depends_on', [])}
{executed_state_info}

原始需求: {requirement}

重要约束条件:
1. 严禁在函数内检查列是否存在，因为系统已在预检阶段确认依赖列存在
2. 直接使用依赖列，不要添加 if '列名' in df 等检查逻辑
3. 使用标准的pandas和numpy语法，不要使用pd.np
4. 必须使用需求中指定的列名，保持与依赖分析阶段的列名一致

请生成一个Python函数，实现以下功能:
1. 函数名: calculate_{safe_col_name}
2. 输入: pandas DataFrame
3. 输出: 处理后的DataFrame（包含新列）
4. 确保使用正确的数据类型和条件判断

返回格式:
{{
    "name": "函数名",
    "description": "函数描述",
    "implementation": "函数实现代码",
    "required_columns": ["需要的列"],
    "new_columns": ["新生成的列"]
}}"""
            
            print(f"   生成函数: {col_name}")
            
            # 获取AI生成的函数
            success, response_text, error_message = ai_service.api_manager.process_single_cell(
                cell_content=function_prompt,
                system_prompt="你是一个专业的pandas函数生成器，生成高质量的数据处理函数。重要：不要检查列是否存在，直接使用已确认存在的列。",
                user_prompt="请为指定列生成处理函数，不要包含列存在性检查。",
                max_tokens=800
            )
            
            if success and response_text:
                try:
                    # 提取JSON
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        func_data = json.loads(json_match.group())
                        
                        # 验证函数结构
                        if all(key in func_data for key in ['name', 'implementation']):
                            # 执行函数并验证
                            success, result_df = self._execute_and_validate_with_state_tracking(df, func_data, col_name)
                            
                            if success and result_df is not None:
                                # 更新DataFrame状态
                                df = result_df
                                
                                # 添加到已执行列
                                new_cols = func_data.get('new_columns', [col_name])
                                executed_columns.update(new_cols)
                                executed_functions.append(func_data)
                                
                                # 记录中间状态
                                for new_col in new_cols:
                                    if new_col in df.columns:
                                        intermediate_states[new_col] = str(df[new_col].head(2).tolist())
                                
                                print(f"     ✅ 函数执行成功: {func_data['name']} -> 🟢 绿色列")
                                print(f"     📊 中间状态: {col_name} = {intermediate_states.get(col_name, 'N/A')}")
                            else:
                                print(f"     ❌ 函数执行验证失败: {func_data['name']}")
                                
                                # 记录执行错误（包含中间状态）
                                self._record_execution_error_with_state(ai_service, requirement, func_data, col_name, intermediate_states)
                        else:
                            print(f"     ❌ 函数结构不完整: {func_data}")
                            
                            # 记录结构错误
                            self._record_structure_error(ai_service, requirement, func_data, col_name)
                    else:
                        print(f"     ❌ 无法提取函数JSON: {response_text}")
                        
                        # 记录解析错误
                        self._record_parsing_error(ai_service, requirement, response_text, col_name)
                except json.JSONDecodeError as e:
                    print(f"     ❌ 函数JSON解析错误: {e}")
                    
                    # 记录解析错误
                    self._record_parsing_error(ai_service, requirement, response_text, col_name)
            else:
                print(f"     ❌ 函数生成失败: {error_message}")
                
                # 记录生成错误
                self._record_generation_error(ai_service, requirement, col_name)
        
        print(f"\n✅ 闭环训练原子化执行完成，成功执行 {len(executed_functions)} 个函数")
        return executed_functions, df, intermediate_states

    def _execute_and_validate_with_state_tracking(self, df: pd.DataFrame, func_data: Dict, col_name: str):
        """
        执行函数并验证，同时记录状态
        """
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("函数执行超时")
        
        try:
            import pandas as pd
            import numpy as np
            func_impl = func_data.get('implementation', '')
            
            # 强制环境初始化：预置正确的导入
            if 'import pandas' not in func_impl:
                func_impl = 'import pandas as pd\nimport numpy as np\nimport datetime\n' + func_impl
            
            # 替换可能的错误语法
            func_impl = func_impl.replace('pd.np.', 'np.')
            func_impl = func_impl.replace('pd.np', 'np')
            
            # 自动注入日期转换提示（用于解决.dt访问器问题）
            if '.dt.' in func_impl and 'pd.to_datetime' not in func_impl:
                # 为使用.dt访问器的列添加日期转换
                import re
                # 查找所有使用.dt访问器的列
                dt_matches = re.findall(r"df\[['\"](\w+)['\"]\]\.dt\.", func_impl)
                for col in set(dt_matches):
                    # 在函数开始处添加日期转换代码
                    conversion_code = f"    df['{col}'] = pd.to_datetime(df['{col}'])\n"
                    # 找到函数定义行并插入日期转换
                    lines = func_impl.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            lines.insert(i + 1, conversion_code)
                            break
                    func_impl = '\n'.join(lines)
            
            # 创建本地命名空间（预置环境补丁）
            import numpy as np
            local_namespace = {
                'pd': pd,
                'np': np,
                'df': df.copy(),
                'datetime': pd.Timestamp,  # 添加datetime支持
                'timedelta': pd.Timedelta  # 添加timedelta支持
            }
            
            # 执行函数定义
            exec(func_impl, {'pd': pd, 'np': np}, local_namespace)
            
            # 获取函数对象
            func_name = func_data.get('name', '')
            if func_name not in local_namespace:
                print(f"       验证失败: 函数 {func_name} 未定义")
                return False, None
            
            func_obj = local_namespace[func_name]
            
            # 设置超时机制（仅在支持的系统上）
            timeout_set = False
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30秒超时
                timeout_set = True
            except AttributeError:
                # Windows系统不支持SIGALRM
                import time
                start_time = time.time()
            
            try:
                # 执行函数
                result_df = func_obj(df.copy())
                
                # 数据类型修复：确保数据类型兼容性
                for col in result_df.columns:
                    if col not in df.columns:  # 只处理新生成的列
                        try:
                            # 尝试修复常见的数据类型不兼容问题
                            if result_df[col].dtype == 'object':
                                # 尝试将object类型的列转换为更具体的类型，但保持兼容性
                                temp_series = result_df[col]
                                # 确保数据类型一致性
                                if temp_series.apply(lambda x: pd.isna(x) or isinstance(x, (int, float, str, pd.Timestamp))).all():
                                    pass  # 数据类型已兼容
                                else:
                                    # 如果有混合类型，转换为object类型
                                    result_df[col] = result_df[col].astype('object')
                        except Exception:
                            # 如果类型转换失败，保持原样
                            pass
                
                # 保持原始日期列的格式一致（将转换后的日期列恢复为原始格式）
                for col in df.columns:
                    if df[col].dtype == 'object':  # 原始为字符串格式
                        # 检查该列是否包含日期格式（YYYY-MM-DD）
                        sample_vals = df[col].dropna().head(5)
                        if len(sample_vals) > 0:
                            is_date_str = all(isinstance(val, str) and 
                                            len(val) == 10 and 
                                            val.count('-') == 2 and
                                            val.replace('-', '').isdigit() 
                                            for val in sample_vals if pd.notna(val))
                            if is_date_str and col in result_df.columns and pd.api.types.is_datetime64_any_dtype(result_df[col]):
                                # 将日期时间列转换回字符串格式
                                result_df[col] = result_df[col].dt.strftime('%Y-%m-%d').replace('NaT', None)
                
                # 取消超时（如果设置了的话）
                if timeout_set:
                    signal.alarm(0)
                else:
                    # Windows系统检查执行时间
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 30:
                        print(f"       验证失败: 函数 {func_name} 执行超时")
                        return False, None
                
                # 验证结果
                if not isinstance(result_df, pd.DataFrame):
                    print(f"       验证失败: 函数 {func_name} 未返回DataFrame")
                    return False, None
                
                if len(result_df) != len(df):
                    print(f"       验证失败: 函数 {func_name} 返回的行数不匹配")
                    return False, None
                
                # 检查是否生成了预期的新列
                expected_new_cols = func_data.get('new_columns', [col_name])
                for new_col in expected_new_cols:
                    if new_col not in result_df.columns:
                        print(f"       验证失败: 函数 {func_name} 未生成预期列 {new_col}")
                        return False, None
                
                print(f"       验证成功: 函数 {func_name} 执行并验证通过")
                return True, result_df
                
            except TimeoutError:
                print(f"       验证失败: 函数 {func_name} 执行超时")
                return False, None
            except Exception as e:
                # 取消超时（如果设置了的话）
                if timeout_set:
                    signal.alarm(0)
                raise e
            
        except KeyError as e:
            print(f"       验证失败: 函数 {func_data.get('name', 'unknown')} 缺少依赖列: {e}")
            # 对于KeyError，记录缺失的依赖项，但不完全失败
            missing_column = str(e).strip("'\"")
            print(f"       检测到缺失列: {missing_column}，在实际执行时可能由其他函数生成")
            return False, None
        except Exception as e:
            print(f"       验证失败: 执行函数 {func_data.get('name', 'unknown')} 时出错: {e}")
            return False, None

    def _record_dependency_error_with_topology(self, ai_service, requirement, col_name, missing_deps, available_columns, topology_info):
        """
        记录依赖顺序错误到数据库（包含拓扑图信息）
        """
        error_record = {
            "prompt": requirement,
            "error_type": "dependency_order_error",
            "error_detail": {
                "target_column": col_name,
                "missing_dependencies": list(missing_deps),
                "available_columns": list(available_columns),
                "error_message": f"列 {col_name} 依赖于 {list(missing_deps)}，但这些列尚未生成",
                "topology_info": topology_info  # 包含拓扑图信息
            },
            "priority": "high",  # 高优先级
            "logic_tag": "dependency_order",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存到数据库
        self.qwen_db.save_data("error_history", {
            "error_type": "dependency_order_error",
            "content": error_record,
            "is_golden": 0,
            "logic_tag": "dependency_order",
            "score": 0
        })
        
        print(f"     📝 已记录依赖顺序错误到数据库（含拓扑信息）: {col_name}")

    def _record_execution_error_with_state(self, ai_service, requirement, func_data, col_name, intermediate_states):
        """
        记录执行错误（包含中间状态）
        """
        error_record = {
            "prompt": requirement,
            "error_type": "execution_error",
            "error_detail": {
                "function_name": func_data.get('name'),
                "target_column": col_name,
                "implementation": func_data.get('implementation', ''),
                "intermediate_states": intermediate_states,  # 包含中间状态
                "error_message": "函数执行失败"
            },
            "priority": "high",  # 高优先级
            "logic_tag": "execution_error",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存到数据库
        self.qwen_db.save_data("error_history", {
            "error_type": "execution_error",
            "content": error_record,
            "is_golden": 0,
            "logic_tag": "execution_error",
            "score": 0
        })

    def _record_structure_error(self, ai_service, requirement, func_data, col_name):
        """
        记录结构错误
        """
        error_record = {
            "prompt": requirement,
            "error_type": "structure_error",
            "error_detail": {
                "function_name": func_data.get('name'),
                "target_column": col_name,
                "provided_data": func_data,
                "error_message": "函数结构不完整"
            },
            "priority": "low",
            "logic_tag": "structure_error",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存到数据库
        self.qwen_db.save_data("error_history", {
            "error_type": "structure_error",
            "content": error_record,
            "is_golden": 0,
            "logic_tag": "structure_error",
            "score": 0
        })

    def _record_parsing_error(self, ai_service, requirement, response_text, col_name):
        """
        记录解析错误
        """
        error_record = {
            "prompt": requirement,
            "error_type": "parsing_error",
            "error_detail": {
                "target_column": col_name,
                "response_text": response_text,
                "error_message": "无法解析AI响应为JSON"
            },
            "priority": "medium",
            "logic_tag": "parsing_error",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存到数据库
        self.qwen_db.save_data("error_history", {
            "error_type": "parsing_error",
            "content": error_record,
            "is_golden": 0,
            "logic_tag": "parsing_error",
            "score": 0
        })

    def _record_generation_error(self, ai_service, requirement, col_name):
        """
        记录生成错误
        """
        error_record = {
            "prompt": requirement,
            "error_type": "generation_error",
            "error_detail": {
                "target_column": col_name,
                "error_message": "AI函数生成失败"
            },
            "priority": "medium",
            "logic_tag": "generation_error",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存到数据库
        self.qwen_db.save_data("error_history", {
            "error_type": "generation_error",
            "content": error_record,
            "is_golden": 0,
            "logic_tag": "generation_error",
            "score": 0
        })

    def process_data_enhanced(self, df: pd.DataFrame, functions: List[Dict[str, Any]], requirement: str, data_context: Dict) -> tuple:
        """
        增强版数据处理方法 - 使用闭环训练方法
        """
        try:
            print("🚀 启动增强版闭环训练解决方案")
            print("="*60)
            
            # 第一阶段：依赖分析
            dependency_analysis = self._analyze_dependencies_phase(data_context, requirement)
            if not dependency_analysis:
                print("❌ 第一阶段失败，回退到传统方法")
                # 回退到传统处理方法
                return self.process_data(df, functions, requirement), False
            
            # 第二阶段：改进的依赖排序（考虑原始列）
            sorted_result = self._sort_dependencies_with_original_columns(dependency_analysis, list(df.columns))
            if not sorted_result:
                print("❌ 第二阶段失败，回退到传统方法")
                # 回退到传统处理方法
                return self.process_data(df, functions, requirement), False
            
            # 第三阶段：闭环训练原子化顺序执行与验证
            executed_functions, final_df, intermediate_states = self._closed_loop_execute_and_validate(
                df, requirement, sorted_result, dependency_analysis
            )
            
            if executed_functions:
                print(f"\n✅ 增强版闭环训练解决方案成功完成！")
                print(f"   - 识别列数: {len(dependency_analysis.get('dependency_analysis', []))}")
                print(f"   - 原始列数: {len(sorted_result['original_columns'])}")
                print(f"   - 需生成列数: {len(sorted_result['new_columns'])}")
                print(f"   - 成功执行函数数: {len(executed_functions)}")
                print(f"   - 最终DataFrame列数: {len(final_df.columns)}")
                print(f"   - 中间状态记录数: {len(intermediate_states)}")
                
                return final_df, True
            else:
                print(f"\n❌ 增强版执行阶段失败，回退到传统方法")
                # 回退到传统处理方法
                return self.process_data(df, functions, requirement), False
        except Exception as e:
            print(f"❌ 增强版处理失败: {e}")
            import traceback
            print(traceback.format_exc())
            # 回退到传统处理方法
            return self.process_data(df, functions, requirement), False


        try:
            # 1. 加载数据
            self.logger.info(f"开始处理: {file_path}")
            self.logger.info(f"处理需求: {requirement}")
            
            df = self._load_excel_data(file_path)
            original_df = df.copy()
            
            # 初始化迭代变量
            best_processed_df = df.copy()
            best_functions = []
            best_result = {
                "success": False,
                "message": "未生成有效的处理函数",
                "iteration": 0
            }
            
            # 2. 迭代执行，直到成功或达到最大迭代次数
            # 记录所有生成的临时文件路径
            temp_files = []
            final_result_iteration = 0
            
            for iteration in range(1, max_iterations + 1):
                self.logger.info(f"\n=== 开始第 {iteration}/{max_iterations} 次迭代 ===")
                
                try:
                    # 3. 生成数据上下文（每次迭代都重新生成，可能会有动态变化）
                    data_context = self._generate_data_context(best_processed_df)
                    
                    # 4. 生成函数
                    self.logger.info(f"第 {iteration} 次迭代：开始生成处理函数...")
                    
                    # 准备上一次的错误信息
                    last_error = ""
                    if iteration > 1:
                        # 获取上一次的错误信息，如果是JSONDecodeError，则使用特定提示
                        last_error = best_result.get("message", "")
                        
                        # 如果上一次是JSON格式错误，使用特定的修复提示
                        if "JSON" in last_error or "json" in last_error:
                            last_error = "你上一次返回的不是有效的JSON格式，请检查括号对齐并确保没有多余文字。"
                        elif "函数" in last_error or "语法" in last_error:
                            # 如果是函数语法错误，使用更具体的提示
                            last_error = "你上一次生成的函数有语法错误，请确保函数定义正确，使用def关键字开头，包含正确的缩进和返回语句。"
                        elif "缺少依赖列" in last_error:
                            # 如果是缺少依赖项，特别提醒AI需要先计算依赖项
                            last_error = f"你上一次的处理失败了，因为缺少必要的依赖列。请分析需求并首先生成缺少的依赖列，如环境修正指数、实时健康值等。错误信息：{last_error}。请按步骤分解需求并先计算基础依赖项。"
                        else:
                            # 其他类型的错误，使用通用提示
                            last_error = f"你上一次生成的内容存在问题：{last_error}，请重新生成有效的函数。"
                    
                    functions = self.generate_multi_column_functions(requirement, data_context, last_error)
                    
                    if not functions:
                        self.logger.warning(f"第 {iteration} 次迭代：未能生成有效的处理函数，尝试下一次迭代")
                        continue
                    
                    # 5. 使用增强版的闭环训练方法处理数据
                    self.logger.info(f"第 {iteration} 次迭代：开始应用函数处理数据...")
                    
                    # 使用增强版的闭环训练方法
                    processed_df, execution_success = self.process_data_enhanced(best_processed_df, functions, requirement, data_context)
                    
                    # 6. 检查处理结果
                    new_columns = list(set(processed_df.columns) - set(original_df.columns))
                    if new_columns or execution_success:
                        self.logger.info(f"第 {iteration} 次迭代：处理成功，新增了 {len(new_columns)} 列")
                        
                        # 更新最佳结果
                        best_processed_df = processed_df.copy()
                        best_functions = functions
                        
                        # 构建当前迭代的成功结果
                        current_result = {
                            "success": True,
                            "iteration": iteration,
                            "generated_functions": len(functions),
                            "new_columns": new_columns,
                            "original_columns": list(original_df.columns),
                            "processed_columns": list(processed_df.columns),
                            "rows_processed": len(processed_df)
                        }
                        
                        # 保存当前结果
                        output_file = os.path.splitext(file_path)[0] + f"_processed_iteration_{iteration}.xlsx"
                        processed_df.to_excel(output_file, index=False)
                        temp_files.append(output_file)  # 记录临时文件
                        current_result["output_path"] = output_file
                        current_result["message"] = f"第 {iteration} 次迭代处理成功"
                        
                        # 更新最佳结果
                        best_result = current_result
                        final_result_iteration = iteration  # 记录最终结果的迭代次数
                        
                        # 如果生成了期望的列，可以提前结束迭代
                        if len(new_columns) >= 1:  # 可以根据实际需求调整终止条件
                            self.logger.info(f"第 {iteration} 次迭代已生成有效结果，提前结束迭代")
                            break
                    else:
                        self.logger.warning(f"第 {iteration} 次迭代：未生成新列，尝试下一次迭代")
                        
                except json.JSONDecodeError:
                    self.logger.error(f"第 {iteration} 次迭代失败: AI返回的不是有效的JSON格式")
                    
                    # 记录错误并准备下一次迭代的提示
                    self._get_ai_service().add_error({
                        "prompt": requirement,
                        "error": "你上一次返回的不是有效的JSON格式，请检查括号对齐并确保没有多余文字。",
                        "iteration": iteration,
                        "traceback": "JSONDecodeError"
                    })
                    
                except KeyError as e:
                    # 特别处理缺少依赖列的错误
                    missing_column = str(e)
                    self.logger.error(f"第 {iteration} 次迭代失败: 缺少依赖列 {missing_column}")
                    
                    # 将缺少依赖列的信息反馈给AI服务，要求AI先计算依赖项
                    self._get_ai_service().add_error({
                        "prompt": requirement,
                        "error": f"缺少依赖列: {missing_column}。请按需求分解步骤，先计算基础依赖列。",
                        "iteration": iteration,
                        "traceback": f"KeyError: {missing_column}"
                    })
                    
                except Exception as e:
                    self.logger.error(f"第 {iteration} 次迭代失败: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
                    # 将错误信息反馈给AI服务，用于后续迭代优化
                    self._get_ai_service().add_error({
                        "prompt": requirement,
                        "error": str(e),
                        "iteration": iteration,
                        "traceback": traceback.format_exc()
                    })
            
            # 最终保存最佳结果
            if best_result["success"]:
                final_output_file = os.path.splitext(file_path)[0] + "_processed.xlsx"
                best_processed_df.to_excel(final_output_file, index=False)
                best_result["final_output_path"] = final_output_file
                best_result["message"] += f"，最终结果已保存到: {final_output_file}"
                
                # 清理所有迭代过程中的临时文件
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                return best_result
            else:
                # 所有迭代都失败
                return {
                    "success": False,
                    "error": "所有迭代均失败",
                    "message": f"经过 {max_iterations} 次迭代，未能生成有效的处理结果",
                    "max_iterations": max_iterations
                }
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "message": "处理失败"
            }
    
    def batch_process(self, file_paths: List[str], requirement: str) -> List[Dict[str, Any]]:
        """
        批量处理多个Excel文件
    
        Args:
            file_paths: Excel文件路径列表
            requirement: 用户需求
            
        Returns:
            处理结果列表
        """
        results = []
        for file_path in file_paths:
            result = self.process_multi_columns(file_path, requirement)
            result["file_path"] = file_path
            results.append(result)
        return results

# 创建全局多列处理器实例
multi_column_processor = MultiColumnProcessor()