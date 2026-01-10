"""
API Manager for AI service, supporting multiple API types including Qwen model.
"""

import json
import logging
import os
import time
import re
from typing import Tuple, Optional, Generator, Dict, Any, List

from openai import OpenAI
import pandas as pd

logger = logging.getLogger("APIManager")


# PaddleOCR单例类，优化初始化性能
class PaddleOCRSingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PaddleOCRSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            try:
                from paddleocr import PaddleOCR
                # 初始化PaddleOCR，使用默认参数
                self.ocr = PaddleOCR(
                    lang='ch'
                )
                # 模型预热，减少首次调用延迟
                self._warm_up()
                logger.info("PaddleOCR单例初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {e}")
                self.ocr = None
            self._initialized = True
    
    def _warm_up(self):
        """模型预热，减少首次调用延迟"""
        try:
            # 使用空白图片进行预热
            from PIL import Image
            import tempfile
            # 创建空白图片
            warm_up_image = Image.new('RGB', (100, 100), color='white')
            # 保存为临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                warm_up_image.save(temp_file, format='PNG')
                temp_file_path = temp_file.name
            
            # 进行一次OCR调用，新版本PaddleOCR不再支持cls参数
            if self.ocr:
                self.ocr.ocr(temp_file_path)
            
            # 删除临时文件
            import os
            os.unlink(temp_file_path)
            logger.info("PaddleOCR模型预热完成")
        except Exception as e:
            logger.error(f"PaddleOCR模型预热失败: {e}")
            # 确保临时文件被删除
            try:
                import os
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except:
                pass
    
    def get_ocr_instance(self):
        return self.ocr


def smart_to_numeric(s):
    """
    从混合文本中提取数值
    
    Args:
        s: 输入值
        
    Returns:
        提取到的数值，如果无法提取则返回0
    """
    if pd.isna(s) or s == '': 
        return 0
    # 提取字符串中的第一个数字（含小数点）
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(s).replace(',', ''))
    return float(match.group()) if match else 0


class APIManager:
    """
    API Manager that supports multiple API types (openai, qwen, ollama)
    for the AI service.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        api_type: str = None,
        ollama_url: str = None,
    ):
        """
        Initialize the API Manager

        Args:
            api_key: API key for the AI service
            model: Model name to use
            api_type: API type (openai, qwen, ollama)
            ollama_url: URL for Ollama API (if using ollama)
        """
        # Use default values from config if not provided
        import sys
        import os

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        )
        from src.utils.config import config_manager

        self.api_key = api_key or config_manager.get("ai.api_key", "")
        # 使用Dashscope兼容模式支持的模型
        self.model = model or config_manager.get("ai.model", "qwen-plus")
        self.api_type = api_type or config_manager.get("ai.api_type", "qwen")
        self.ollama_url = ollama_url or config_manager.get(
            "ai.ollama_url", "http://localhost:11434"
        )
        self.logger = logging.getLogger("APIManager")

        # Rate limiting
        self.request_count = 0
        self.request_start_time = time.time()
        self.max_requests_per_minute = 30

        # Initialize OpenAI client for Qwen and other OpenAI-compatible APIs
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[OpenAI]:
        """
        Initialize the OpenAI client based on API type

        Returns:
            Initialized OpenAI client or None if initialization failed
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                # For Qwen, use OpenAI compatible client with Dashscope endpoint
                base_url = (
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    if self.api_type == "qwen"
                    else None
                )
                return OpenAI(api_key=self.api_key, base_url=base_url)
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize client: {e}")
            return None

    def set_api_type(self, api_type: str):
        """
        Set the API type and reinitialize client

        Args:
            api_type: New API type to use
        """
        self.api_type = api_type
        self.client = self._initialize_client()

    def set_model(self, model: str):
        """
        Set the model to use

        Args:
            model: Model name
        """
        self.model = model

    def set_api_key(self, api_key: str):
        """
        Set the API key and reinitialize client

        Args:
            api_key: New API key
        """
        self.api_key = api_key
        self.client = self._initialize_client()

    def set_ollama_url(self, url: str):
        """
        Set the Ollama URL

        Args:
            url: Ollama API URL
        """
        self.ollama_url = url

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the API connection

        Returns:
            Tuple of (success, message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, "Failed to initialize client"

                # Send a simple test request
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, test connection."},
                    ],
                    max_tokens=10,
                )
                return True, f"Connection successful. Model: {completion.model}"
            else:
                return (
                    False,
                    f"API type {self.api_type} not supported for connection test",
                )
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False, f"Connection failed: {str(e)}"

    def process_single_cell(
        self,
        cell_content: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.01,  # 降低随机性，提升逻辑确定性
        max_tokens: int = 150,
        context_data: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process a single cell using AI

        Args:
            cell_content: Content of the cell
            system_prompt: System prompt for AI
            user_prompt: User prompt for AI
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            context_data: Optional context data

        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client"

                # Format the prompt
                formatted_prompt = f"{user_prompt}\n\nCell content: {cell_content}"

                # Add context if available
                if context_data and len(context_data) > 0:
                    context_text = "\n\nContext information:\n"
                    for key, value in context_data.items():
                        context_text += f"- {key}: {value}\n"
                    formatted_prompt += context_text

                # Create messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt},
                ]

                # Send request
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                result = completion.choices[0].message.content.strip()
                return True, result, None
            else:
                return (
                    False,
                    None,
                    f"API type {self.api_type} not supported for cell processing",
                )
        except Exception as e:
            self.logger.error(f"Error processing cell: {e}")
            return False, None, f"Error: {str(e)}"

    def generate_functions(
        self,
        prompt: str,
        data_context: Dict[str, Any],
        system_prompt: str = None,
        temperature: float = 0.5,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Generate Excel processing functions using AI

        Args:
            prompt: User prompt for function generation
            data_context: Contextual data about the Excel file
            system_prompt: System prompt for AI
            temperature: Temperature for generation (lower = more deterministic)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter (lower = more focused)
            frequency_penalty: Penalizes repeated tokens
            presence_penalty: Penalizes new tokens based on presence in conversation

        Returns:
            Tuple of (success, generated_functions, error_message)
        """
        if system_prompt is None:
            system_prompt = """你是Office Lazy Tool项目的Excel处理函数生成专家。根据用户需求和数据上下文，生成准确、高效的Python函数来处理Excel数据。

# 核心要求
1. 函数必须是完整的Python函数定义，使用小写字母和下划线命名（snake_case）
2. **强制规范**：函数名必须仅使用英文小写字母和下划线（snake_case），严禁包含中文、空格或特殊符号。例如：使用 `calculate_gross_profit` 而不是 `calculate毛利`
3. 函数仅接受一个DataFrame参数`df`，并返回处理后的DataFrame
4. **重要：请直接输出逻辑代码，不要包含复杂的错误处理封装（如try-except块）**
5. 使用pandas库进行数据处理，高效执行
6. 能处理不同列名，生成有意义的新列名
7. 处理缺失值，避免除以零，验证数据类型
8. **CRITICAL: 如果列名包含换行符（如 '成本\n(元)'）或空格（如 '单 价(元)'），请在代码中使用完整的列名，并确保JSON格式严格闭合**

# 输出格式
请严格按照以下JSON格式输出，不要包含任何额外的文本、解释或代码块标记：

[
    {
        "name": "function_name",
        "description": "function_description",
        "implementation": "def function_name(df):\n    df['new_column'] = df['col1'] + df['col2']\n    return df"
    }
]

# 输出注意事项
1. **必须严格按照上述格式输出，只包含JSON内容，不包含其他任何内容**
2. 函数代码中的换行符必须使用`\n`转义，不要使用实际换行符
3. 函数代码中如果包含引号，请使用单引号，避免使用双引号
4. **CRITICAL: 确保JSON中的字符串值被正确转义，尤其是列名中的特殊字符**
5. 不要在JSON中添加任何注释
6. 确保JSON格式严格符合RFC 8259规范
7. 每个字段名必须使用双引号包围
8. 不要包含多余的空格或换行符
9. **严禁在JSON响应中输出物理换行符**
10. **如果列名中包含空格或换行符，请在JSON字符串中使用正确的转义格式，例如：**
    - 对于列名 '单 价(元)'，代码中应写为 df['单 价(元)']
    - 对于列名 '成本\n(元)'，代码中应写为 df['成本\\n(元)']
11. **绝对禁止在JSON中出现未经转义的换行符或制表符**
12. **确保函数实现中的所有字符串引号都正确配对**
        """

        # Serialize data context
        serialized_data_context = self._serialize_data(data_context)
        
        # 优化上下文：精简样本数据，减少AI处理时间
        # 将sample_data设置为空列表，减少上下文长度
        if isinstance(serialized_data_context, dict):
            serialized_data_context['sample_data'] = []

        # Format the full prompt
        full_prompt = f"{prompt}\n\nData Context: {json.dumps(serialized_data_context, ensure_ascii=False)}"

        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return (
                            False,
                            None,
                            "Failed to initialize client: Please check your API key and network connection",
                        )

                # Create messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ]

                # Send request with enhanced parameters
                try:
                    # 设置较长的超时时间，针对高负载任务
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        response_format={"type": "json_object"},  # 请求JSON格式响应
                        timeout=120,  # 设置120秒超时，防止长时间等待
                    )
                except Exception as api_error:
                    # 处理API调用错误
                    error_message = f"API call failed: {str(api_error)}"
                    if "API key" in error_message:
                        error_message = "API call failed: Invalid API key. Please check your API key configuration."
                    elif "network" in error_message.lower():
                        error_message = "API call failed: Network error. Please check your network connection."
                    elif "model" in error_message.lower():
                        error_message = f"API call failed: Invalid model '{self.model}'. Please check your model configuration."

                    self.logger.error(f"API call failed: {api_error}")
                    return False, None, error_message

                result = completion.choices[0].message.content.strip()
                self.logger.debug(f"AI response: {result}")

                # Parse the JSON result with enhanced robust error handling
                try:
                    # 1. 清理响应，移除可能的代码块标记
                    clean_result = result
                    if result.startswith("```json"):
                        clean_result = result[7:].rstrip("```")
                    elif result.startswith("```"):
                        clean_result = result[3:].rstrip("```")
                    elif "```json" in result:
                        # 处理中间有代码块的情况
                        clean_result = result.split("```json")[1].split("```")[0]
                    elif "```" in result:
                        # 处理其他代码块标记
                        clean_result = result.split("```")[1].split("```")[0]

                    # 移除多余的空格和换行
                    clean_result = clean_result.strip()
                    
                    # 2. 强力清洗JSON，确保只保留JSON结构
                    clean_result = self._clean_json_response(clean_result)
                    
                    # 3. 尝试解析JSON
                    functions = json.loads(clean_result)
                    if isinstance(functions, dict):
                        functions = [functions]

                    # 3. 增强的函数格式验证 - 支持多种函数格式
                    valid_functions = []
                    for i, func in enumerate(functions):
                        func_valid = True

                        # 支持多种函数格式
                        # 格式1: 标准格式（name, description, implementation）
                        if all(
                            field in func
                            for field in ["name", "description", "implementation"]
                        ):
                            # 标准格式，直接使用
                            valid_func = func.copy()
                        # 格式2: AI返回的特殊格式（new_column_name, function_text, description）
                        elif all(
                            field in func
                            for field in [
                                "new_column_name",
                                "function_text",
                                "description",
                            ]
                        ):
                            # 转换为标准格式
                            column_name = func["new_column_name"]
                            function_text = func["function_text"]

                            # 检查function_text是否已经是一个完整的函数定义
                            if "def " in function_text and "return " in function_text:
                                # 如果已经是完整的函数定义，直接使用
                                valid_func = {
                                    "name": column_name,
                                    "description": func["description"],
                                    "implementation": function_text,
                                }
                            else:
                                # 清理function_text，移除可能的函数定义和重复赋值
                                function_text = function_text.replace(
                                    "def calculate_gross_margin(df):", ""
                                )
                                function_text = function_text.replace(
                                    "def calculate_growth_rate(df):", ""
                                )
                                function_text = function_text.replace(
                                    "def calculate_qualified_rate(df):", ""
                                )
                                # 移除可能的重复赋值
                                function_text = function_text.replace(
                                    f'df["{column_name}"] = df["{column_name}"] = ',
                                    f'df["{column_name}"] = ',
                                )
                                function_text = function_text.replace(
                                    f"df['{column_name}'] = df['{column_name}'] = ",
                                    f"df['{column_name}'] = ",
                                )
                                # 移除可能的多余空格和换行
                                function_text = function_text.strip()

                                # 增强函数的健壮性，处理空值情况
                                valid_func = {
                                    "name": column_name,
                                    "description": func["description"],
                                    "implementation": "import pandas as pd\nimport numpy as np\n\ndef "
                                    + column_name
                                    + "(df):\n    try:\n        df_copy = df.copy()  # 避免修改原始数据\n        # 处理空值和异常情况\n        "
                                    + function_text
                                    + "\n        return df_copy\n    except Exception as e:\n        print(f'处理数据时出错: {{e}}')\n        import traceback\n        traceback.print_exc()\n        return df",
                                }
                        # 格式3: 只有function_text的简化格式
                        elif "function_text" in func:
                            func_text = func["function_text"]

                            # 检查function_text是否已经是一个完整的函数定义
                            if "def " in func_text and "return " in func_text:
                                # 提取函数名
                                import re

                                func_name_match = re.match(
                                    r"def\s+([a-zA-Z_]\w*)\s*\(", func_text
                                )
                                func_name = (
                                    func_name_match.group(1)
                                    if func_name_match
                                    else f"function_{i}"
                                )
                                valid_func = {
                                    "name": func_name,
                                    "description": f"自动生成的函数: {func_name}",
                                    "implementation": func_text,
                                }
                            else:
                                # 转换为标准格式
                                func_name = f"function_{i}"
                                result_col = f"result_{i}"
                                valid_func = {
                                    "name": func_name,
                                    "description": f"自动生成的函数 {i}",
                                    "implementation": "import pandas as pd\nimport numpy as np\n\ndef "
                                    + func_name
                                    + "(df):\n    try:\n        df_copy = df.copy()  # 避免修改原始数据\n        # 处理空值和异常情况\n        "
                                    + func_text
                                    + "\n        return df_copy\n    except Exception as e:\n        print(f'处理数据时出错: {{e}}')\n        import traceback\n        traceback.print_exc()\n        return df",
                                }
                        else:
                            func_valid = False

                        # 修复生成的函数实现
                        if func_valid:
                            implementation = valid_func["implementation"]
                            if "def " not in implementation:
                                func_valid = False
                            else:
                                # 添加语法检查和修复
                                implementation = self._fix_function_syntax(
                                    implementation
                                )
                                
                                # 双重验证：进行语法检查
                                if not self._check_function_syntax(implementation):
                                    self.logger.warning(f"函数 {valid_func['name']} 语法检查失败，跳过")
                                    func_valid = False
                                else:
                                    valid_func["implementation"] = implementation
                                    
                                    # 确保函数名与实际生成的函数名一致
                                    import re

                                    func_name_match = re.match(
                                        r"def\s+([a-zA-Z_]\w*)\s*\(", implementation
                                    )
                                    if func_name_match:
                                        actual_func_name = func_name_match.group(1)
                                        # 更新函数名，确保与实际生成的函数名一致
                                        valid_func["name"] = actual_func_name

                        if func_valid:
                            valid_functions.append(valid_func)

                    if valid_functions:
                        self.logger.info(
                            f"成功解析 {len(valid_functions)} 个有效函数，跳过 {len(functions) - len(valid_functions)} 个无效函数"
                        )
                        return True, valid_functions, None
                    else:
                        self.logger.warning(f"AI返回的函数均无效")
                        # 继续尝试其他解析方式
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse AI response as JSON: {e}")
                    self.logger.debug(f"Raw AI response: {result}")
                    self.logger.debug(f"Cleaned AI response: {clean_result}")
                except Exception as general_error:
                    self.logger.error(
                        f"Unexpected error when parsing AI response: {general_error}"
                    )
                    import traceback

                    self.logger.error(traceback.format_exc())

                # 4. 战神版：针对性解析修复（处理带换行符和空格的列名）
                try:
                    import re
                    import ast
                    
                    def power_parse_json(raw_text):
                        """
                        强力解析JSON，专门处理带换行符和空格的列名
                        """
                        import re
                        import json
                        import ast
                        
                        # 1. 清理原始响应，移除可能的标记
                        clean_text = raw_text.strip()
                        if clean_text.startswith('```json'):
                            clean_text = clean_text[7:].rstrip('```')
                        elif clean_text.startswith('```'):
                            clean_text = clean_text[3:].rstrip('```')
                        
                        self.logger.debug(f"清理后的原始响应: {clean_text[:150]}...")
                        
                        # 2. 提取最外层的JSON结构 (支持数组和对象)
                        json_pattern = r'(\[\s*\{.*?\}\s*\]|\{\s*"[^"]+".*?\}\s*)'  # 匹配数组或对象
                        match = re.search(json_pattern, clean_text, re.DOTALL)
                        if not match:
                            # 尝试更宽松的匹配
                            json_pattern_relaxed = r'(\[.*?\]|\{.*?\})'  # 匹配任何数组或对象
                            match = re.search(json_pattern_relaxed, clean_text, re.DOTALL)
                            if not match:
                                self.logger.error(f"无法提取JSON结构: {clean_text[:100]}...")
                                return None
                        
                        json_str = match.group(0)
                        self.logger.debug(f"提取的JSON字符串: {json_str[:100]}...")
                        
                        # 3. 【核心修复】处理字符串内部的非法物理换行
                        # 我们只针对 JSON 值（引号包围的部分）内部的换行进行转义
                        def escape_internal_newlines(m):
                            content = m.group(2)
                            # 保护代码：如果看起来是 Python 代码块，保留 \n
                            # 如果是普通列名或描述，将物理换行替换为 \n 字符串
                            processed_content = content.replace("\n", "\\n")
                            return f'{m.group(1)}"{processed_content}"'

                        # 正则匹配 JSON 的值部分
                        json_str = re.sub(r'(\s*:\s*)"([\s\S]*?)"', escape_internal_newlines, json_str)
                        
                        # 4. 修复常见的JSON格式问题
                        def fix_common_json_errors(json_str):
                            # 1. 移除行首行尾的空格和多余的换行
                            json_str = json_str.strip()
                            # 2. 修复单引号问题，但只处理键名和值部分，不处理代码块
                            # 只替换不在引号内的单引号
                            def safe_replace_quotes(match):
                                return match.group(0).replace("'", '"') if match.group(1) not in ['implementation', 'description'] else match.group(0)
                            json_str = re.sub(r'"(implementation|description|name)":\s*"([^"]*)"', safe_replace_quotes, json_str)
                            # 3. 修复键值对格式问题，如: 'key': 'value'
                            json_str = re.sub(r"'([^']+)'\s*:\s*'([^']+)'", r'"\1": "\2"', json_str)
                            # 4. 移除尾部逗号
                            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                            return json_str
                        
                        json_str = fix_common_json_errors(json_str)
                        
                        # 5. 多种解析方法依次尝试
                        parsing_methods = [
                            # 方法1: 直接JSON解析
                            lambda x: json.loads(x),
                            # 方法2: ast.literal_eval (更宽容)
                            lambda x: ast.literal_eval(x),
                            # 方法3: 移除所有空格和换行后解析
                            lambda x: json.loads(re.sub(r'\s+', '', x)),
                        ]
                        
                        for i, parse_method in enumerate(parsing_methods):
                            try:
                                result = parse_method(json_str)
                                self.logger.debug(f"解析方法 {i+1} 成功")
                                return result
                            except Exception as e:
                                self.logger.debug(f"解析方法 {i+1} 失败: {e}")
                                continue
                        
                        # 6. 最后尝试: 分段解析，提取所有可能的函数定义
                        self.logger.debug("尝试分段解析函数定义")
                        try:
                            # 提取所有函数定义
                            function_pattern = re.compile(r'def\s+(\w+)\s*\(df\):[^}]+return\s+df', re.DOTALL)
                            function_matches = function_pattern.findall(json_str)
                            
                            if function_matches:
                                # 构建简单的函数列表
                                functions = []
                                for func_code in function_matches:
                                    # 提取函数名
                                    func_name_match = re.match(r'def\s+(\w+)', func_code)
                                    if func_name_match:
                                        func_name = func_name_match.group(1)
                                        functions.append({
                                            "name": func_name,
                                            "description": f"自动提取的函数: {func_name}",
                                            "implementation": func_code
                                        })
                                if functions:
                                    return functions
                        except Exception as e:
                            self.logger.error(f"分段解析失败: {e}")
                        
                        self.logger.error(f"所有解析方法均失败，原始JSON: {json_str[:200]}...")
                        return None
                    
                    # 使用强力解析函数
                    functions = power_parse_json(result)
                    if functions:
                        if isinstance(functions, dict):
                            functions = [functions]
                        
                        # 验证函数
                        valid_functions = [
                            f for f in functions 
                            if all(k in f for k in ["name", "description", "implementation"])
                            and "def " in f["implementation"]
                        ]
                        
                        if valid_functions:
                            self.logger.info(f"通过战神版解析恢复，成功解析 {len(valid_functions)} 个函数")
                            return True, valid_functions, None

                    # 方式2: 尝试修复常见的JSON格式问题
                    import re

                    # 预处理: 清理可能的多余内容
                    json_candidate = result
                    # 只保留JSON相关内容
                    if "[" in json_candidate and "]" in json_candidate:
                        json_candidate = json_candidate[
                            json_candidate.find("[") : json_candidate.rfind("]") + 1
                        ]

                    # 修复单引号问题
                    json_candidate = json_candidate.replace("'", '"')
                    # 处理格式如: 'key': 'value' 的单引号
                    json_candidate = re.sub(r"'([^']+)':", r'"\1":', json_candidate)
                    # 修复值部分的单引号为双引号，如: "key": 'value' -> "key": "value"
                    json_candidate = re.sub(r":\s*'([^']+)':", r': "\1"', json_candidate)
                    # 修复尾部逗号
                    json_candidate = re.sub(r',\s*\n\s*}', '\n}', json_candidate)
                    json_candidate = re.sub(r',\s*\n\s*]', '\n]', json_candidate)

                    functions = json.loads(json_candidate)
                    if isinstance(functions, dict):
                        functions = [functions]

                    # 再次验证
                    valid_functions = [
                        f
                        for f in functions
                        if all(
                            k in f for k in ["name", "description", "implementation"]
                        )
                        and "def " in f["implementation"]
                    ]
                    if valid_functions:
                        self.logger.info(
                            f"通过方式2恢复，成功解析 {len(valid_functions)} 个函数"
                        )
                        return True, valid_functions, None

                    # 方式3: 尝试从AI响应中提取单个函数
                    self.logger.info("尝试从AI响应中提取单个函数")

                    # 提取函数定义
                    import re

                    function_pattern = re.compile(
                        r"def\s+([a-zA-Z_]\w*)\s*\(.*?\):.*?(?=def\s+|$)", re.DOTALL
                    )
                    function_matches = function_pattern.findall(result)

                    if function_matches:
                        # 构建简单的函数对象
                        extracted_functions = []
                        for match in function_matches:
                            # 提取函数名
                            func_name_match = re.match(
                                r"def\s+([a-zA-Z_]\w*)\s*\(", match
                            )
                            if func_name_match:
                                func_name = func_name_match.group(1)
                                extracted_functions.append(
                                    {
                                        "name": func_name,
                                        "description": f"从AI响应中提取的函数: {func_name}",
                                        "implementation": match,
                                    }
                                )

                        if extracted_functions:
                            self.logger.info(
                                f"通过方式3恢复，成功提取 {len(extracted_functions)} 个函数"
                            )
                            return True, extracted_functions, None
                except Exception as extract_error:
                    self.logger.error(
                        f"Failed to extract JSON from response: {extract_error}"
                    )
                    import traceback

                    self.logger.error(traceback.format_exc())

                # 5. 所有解析方式都失败，返回更友好的错误信息
                self.logger.error(f"所有解析方式均失败，原始响应: {result}")
                return (
                    False,
                    None,
                    f"Failed to parse AI response as valid functions. Please try again with a simpler requirement.",
                )
            else:
                return (
                    False,
                    None,
                    f"API type {self.api_type} not supported for function generation. Supported types: openai, qwen",
                )
        except Exception as e:
            self.logger.error(f"Error generating functions: {e}", exc_info=True)
            error_message = f"Unexpected error: {str(e)}"
            if "timeout" in error_message.lower():
                error_message = "Error: Request timeout. Please check your network connection and try again."
            elif "permission" in error_message.lower():
                error_message = (
                    "Error: Permission denied. Please check your API key permissions."
                )

            return False, None, error_message

    def _fix_function_syntax(self, implementation: str) -> str:
        """
        修复生成的函数语法错误

        Args:
            implementation: 生成的函数实现代码

        Returns:
            修复后的函数实现代码
        """
        import re

        # 修复常见的语法错误
        fixed_implementation = implementation

        # 修复1: 移除`result = def`这样的错误语法
        # 使用更直接的替换方式
        fixed_implementation = re.sub(r"result\s*=\s*def", "def", fixed_implementation)

        # 修复2: 移除嵌套函数定义中的result赋值
        lines = fixed_implementation.split("\n")
        fixed_lines = []
        in_function = False
        function_indent = 0

        for line in lines:
            stripped_line = line.strip()

            # 跳过空行
            if not stripped_line:
                fixed_lines.append(line)
                continue

            # 检查是否在函数内部
            if stripped_line.startswith("def "):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= function_indent:
                    # 函数结束
                    in_function = False
                    fixed_lines.append(line)
                else:
                    # 函数内部行
                    if stripped_line.startswith("result = "):
                        # 跳过result赋值行
                        continue
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        fixed_implementation = "\n".join(fixed_lines)

        # 修复3: 确保函数内部包含必要的导入语句
        # 分离import语句和函数内容
        import_lines = []
        function_lines = []
        has_def = False

        for line in fixed_implementation.split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith("import ") or stripped_line.startswith("from "):
                import_lines.append(line)
            else:
                function_lines.append(line)
                if stripped_line.startswith("def "):
                    has_def = True

        # 确保有必要的导入
        has_pandas = any("pandas" in line or "pd" in line for line in import_lines)
        has_numpy = any("numpy" in line or "np" in line for line in import_lines)

        # 如果没有pandas导入，添加
        if not has_pandas:
            import_lines.append("import pandas as pd")
        # 如果没有numpy导入，添加
        if not has_numpy:
            import_lines.append("import numpy as np")

        # 重新组合代码
        if has_def:
            # 如果有函数定义，将import语句放在函数内部
            # 查找函数定义后的第一个非空行，插入import语句
            new_function_lines = []
            for i, line in enumerate(function_lines):
                new_function_lines.append(line)
                stripped_line = line.strip()
                if stripped_line.startswith("def "):
                    # 在函数定义后添加导入语句
                    indent = " " * (len(line) - len(line.lstrip()))
                    for imp_line in import_lines:
                        new_function_lines.insert(i + 1, indent + imp_line)
                    break
            fixed_implementation = "\n".join(
                new_function_lines + function_lines[i + 1 :]
            )
        else:
            # 如果没有函数定义，直接组合
            fixed_implementation = "\n".join(import_lines + function_lines)

        # 修复4: 移除重复的import语句
        lines = fixed_implementation.split("\n")
        seen_imports = set()
        unique_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                    unique_lines.append(line)
            else:
                unique_lines.append(line)
        fixed_implementation = "\n".join(unique_lines)

        # 修复5: 确保函数定义以def开头
        if not fixed_implementation.strip().startswith("def "):
            # 查找函数定义的开始位置
            def_start = re.search(r"def\s+\w+\s*\(", fixed_implementation)
            if def_start:
                # 移除函数定义前的所有内容
                fixed_implementation = fixed_implementation[def_start.start() :]

        # 修复6: 修复函数缩进问题，特别是try语句块
        lines = fixed_implementation.split("\n")
        fixed_lines = []
        in_function = False
        indent_level = 0  # 当前缩进层级
        indent_size = 4  # 每个缩进层级的空格数

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                fixed_lines.append(" " * indent_level + stripped_line)
                continue

            if stripped_line.startswith("def "):
                # 函数定义行，重置缩进
                fixed_lines.append(stripped_line)
                in_function = True
                indent_level = indent_size
            elif in_function:
                # 处理控制流语句的缩进
                if stripped_line.endswith(":"):
                    # 控制流语句（if, for, while, try, except等）
                    fixed_lines.append(" " * indent_level + stripped_line)
                    # 增加缩进层级
                    if stripped_line.startswith(
                        (
                            "if ",
                            "for ",
                            "while ",
                            "with ",
                            "try",
                            "except ",
                            "elif ",
                            "else:",
                        )
                    ):
                        indent_level += indent_size
                elif stripped_line.startswith(("return ", "break", "continue", "pass")):
                    # 简单语句，不需要增加缩进
                    fixed_lines.append(" " * indent_level + stripped_line)
                elif stripped_line.startswith(("except ", "finally:")):
                    # 异常处理，需要减少缩进
                    indent_level = max(indent_level - indent_size, indent_size)
                    fixed_lines.append(" " * indent_level + stripped_line)
                    # 异常处理后需要增加缩进
                    indent_level += indent_size
                else:
                    # 普通语句，使用当前缩进层级
                    fixed_lines.append(" " * indent_level + stripped_line)
            else:
                # 不在函数内部，保持原样
                fixed_lines.append(line)

        fixed_implementation = "\n".join(fixed_lines)

        # 修复7: 专门处理try-except语句块的缩进
        lines = fixed_implementation.split("\n")
        fixed_lines = []
        in_try_block = False
        try_indent = 0

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                fixed_lines.append(line)
                continue

            # 查找try语句
            if stripped_line.startswith("try:"):
                in_try_block = True
                # 记录try语句的缩进
                try_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
            elif in_try_block:
                # 检查是否是except或finally
                if stripped_line.startswith("except ") and stripped_line.endswith(":"):
                    # except语句应该和try语句同一缩进层级
                    fixed_lines.append(" " * try_indent + stripped_line)
                elif stripped_line.startswith("finally:"):
                    # finally语句应该和try语句同一缩进层级
                    fixed_lines.append(" " * try_indent + stripped_line)
                    in_try_block = False
                else:
                    # try或except块内的语句，应该比try语句多一级缩进
                    fixed_lines.append(" " * (try_indent + 4) + stripped_line)
            else:
                fixed_lines.append(line)

        fixed_implementation = "\n".join(fixed_lines)

        return fixed_implementation

    def _clean_json_response(self, text: str) -> str:
        """
        强力提取 JSON 部分，精准区分代码换行和JSON非法换行
        """
        import re
        # 1. 提取最外层结构
        match = re.search(r'\[\s*\{.*\}\s*\]|\{\s*".*"\s*\}', text, re.DOTALL)
        if not match:
            return text
        
        clean_text = match.group(0)

        # 2. 【核心修复】处理字符串内部的非法物理换行
        # 我们只针对 JSON 值（引号包围的部分）内部的换行进行转义
        def escape_internal_newlines(m):
            content = m.group(2)
            # 保护代码：如果看起来是 Python 代码块，保留 \n
            # 如果是普通列名或描述，将物理换行替换为 \n 字符串
            processed_content = content.replace("\n", "\\n")
            return f'{m.group(1)}"{processed_content}"'

        # 正则匹配 JSON 的值部分
        clean_text = re.sub(r'(\s*:\s*)"([\s\S]*?)"', escape_internal_newlines, clean_text)

        # 3. 修复常见的单引号和逗号问题
        clean_text = clean_text.replace("'", '"')
        clean_text = re.sub(r',\s*([\]}])', r'\1', clean_text)
        
        return clean_text

    def _check_function_syntax(self, implementation: str) -> bool:
        """
        快速检查函数实现的语法是否正确

        Args:
            implementation: 函数实现代码

        Returns:
            语法正确返回True，否则返回False
        """
        try:
            # 尝试编译函数代码，检查语法
            compile(implementation, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
        except Exception:
            # 其他异常不影响，因为可能是运行时异常，不是语法错误
            return True

    def _serialize_data(self, data: Any) -> Any:
        """
        Serialize data to JSON compatible format

        Args:
            data: Data to serialize

        Returns:
            Serialized data
        """
        if isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        elif hasattr(data, "strftime"):
            # Handle datetime objects
            return data.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(data, (int, float, bool, str, type(None))):
            return data
        else:
            return str(data)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models based on API type

        Returns:
            List of available models
        """
        models = []

        if self.api_type in ["openai", "qwen"]:
            # Common models that work with OpenAI compatible APIs
            models.extend(
                [
                    {"name": "qwen-plus-v1", "description": "Qwen Plus model"},
                    {"name": "qwen-max-v1", "description": "Qwen Max model"},
                    {"name": "qwen-turbo-v1", "description": "Qwen Turbo model"},
                ]
            )

        return models
    
    def analyze_image(self, image_path: str, prompt: str = "") -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Analyze image content using AI
        
        Args:
            image_path: Path to the image file
            prompt: Analysis prompt
            
        Returns:
            Tuple of (success, result, error_message)
        """
    
    def analyze_table_structure(self, image_path: str, use_vl_model: bool = False) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Analyze table structure in image using AI
        
        Args:
            image_path: Path to the table image file
            use_vl_model: Whether to use Qwen-VL-Max model for table analysis
            
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                # 如果指定使用VL模型或当前模型是qwen-vl-max，则使用QwenVLManager
                if use_vl_model or self.model == "qwen-vl-max":
                    try:
                        from .qwen_vl_manager import QwenVLManager
                        vl_manager = QwenVLManager(api_key=self.api_key, api_type=self.api_type)
                        return vl_manager.analyze_table_structure(image_path)
                    except Exception as vl_error:
                        self.logger.warning(f"Failed to use Qwen-VL-Max manager: {vl_error}, falling back to regular table analysis")
                
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client: Please check your API key and network connection"

                # 检查文件是否存在
                if not os.path.exists(image_path):
                    return False, None, f"Image file not found: {image_path}"

                # 检查文件大小和格式
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:  # 10MB限制
                    return False, None, f"Image file is too large. Maximum size is 10MB."

                # 读取图像文件并转换为base64
                import base64
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")

                # 构建专门用于表格结构和内容识别的消息
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的表格分析助手，能够准确识别图片中的表格结构和内容。请按照以下格式输出表格的结构化信息：\n\n表格结构分析结果：\n- 总行数：<行数>\n- 总列数：<列数>\n- 表头行：<表头所在行号，从1开始>\n- 合并单元格：<合并单元格列表，格式为(行1, 列1, 行2, 列2)，表示从第1行第1列到第2行第2列合并>\n- 行数据：<每行的单元格内容列表，格式为['内容1', '内容2', ...]>\n\n请确保输出的行数和列数准确，表头行号正确，合并单元格信息完整，单元格内容准确。只输出表格的结构化信息，不要输出其他无关内容。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请详细分析这张图片中的表格结构和内容，包括行数、列数、表头位置、合并单元格信息以及每个单元格的文本内容。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]

                # 调用AI API进行图像分析
                model_to_use = self.model
                self.logger.info(f"对于表格结构分析，使用模型: {model_to_use}")

                completion = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=2000
                )

                # 处理结果
                result = completion.choices[0].message.content.strip()
                return True, {"content": result}, None
            else:
                return False, None, f"API type {self.api_type} not supported for table structure analysis. Please use openai or qwen API."
        except Exception as e:
            self.logger.error(f"Error in analyze_table_structure: {e}")
            # 提供更详细的错误信息
            error_msg = f"表格结构分析失败: {str(e)}"
            if "API key" in str(e) or "authentication" in str(e):
                error_msg = "表格结构分析失败: API密钥无效或认证失败，请检查您的API配置"
            elif "model_not_found" in str(e).lower():
                error_msg = f"表格结构分析失败: 模型 {self.model} 不存在或没有访问权限"
            elif "model" in str(e).lower() or "not supported" in str(e).lower():
                error_msg = f"表格结构分析失败: 当前模型 {self.model} 不支持图像分析功能"
            elif "404" in str(e):
                error_msg = "表格结构分析失败: API端点不存在或访问受限"
            elif "network" in str(e).lower() or "timeout" in str(e).lower():
                error_msg = "表格结构分析失败: 网络连接超时，请检查您的网络连接"
            elif "image" in str(e).lower() or "vision" in str(e).lower():
                error_msg = "表格结构分析失败: 当前API或模型不支持图像分析功能"
            return False, None, error_msg
    
    def analyze_image(self, image_path: str, prompt: str = "", use_vl_model: bool = False) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Analyze image content using AI
        
        Args:
            image_path: Path to the image file
            prompt: Analysis prompt
            use_vl_model: Whether to use Qwen-VL-Max model for image analysis
            
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                # 如果指定使用VL模型或当前模型是qwen-vl-max，则使用QwenVLManager
                if use_vl_model or self.model == "qwen-vl-max":
                    try:
                        from .qwen_vl_manager import QwenVLManager
                        vl_manager = QwenVLManager(api_key=self.api_key, api_type=self.api_type)
                        return vl_manager.analyze_image(image_path, prompt)
                    except Exception as vl_error:
                        self.logger.warning(f"Failed to use Qwen-VL-Max manager: {vl_error}, falling back to regular image analysis")
                
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client: Please check your API key and network connection"

                # 检查文件是否存在
                if not os.path.exists(image_path):
                    return False, None, f"Image file not found: {image_path}"

                # 检查文件大小和格式
                file_size = os.path.getsize(image_path)
                if file_size > 10 * 1024 * 1024:  # 10MB限制
                    return False, None, f"Image file is too large. Maximum size is 10MB."

                # 读取图像文件并转换为base64
                import base64
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")

                # 构建消息
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can analyze images. Describe the image content in detail, including any text present."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or "请详细分析这张图片的内容，包括所有文字信息。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]

                # 调用AI API进行图像分析
                model_to_use = self.model
                self.logger.info(f"对于图像分析，使用模型: {model_to_use}")

                completion = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=2000
                )

                # 处理结果
                result = completion.choices[0].message.content.strip()
                return True, {"content": result}, None
            else:
                return False, None, f"API type {self.api_type} not supported for image analysis. Please use openai or qwen API."
        except Exception as e:
            self.logger.error(f"Error in analyze_image: {e}")
            # 提供更详细的错误信息
            error_msg = f"AI分析失败: {str(e)}"
            if "API key" in str(e) or "authentication" in str(e):
                error_msg = "AI分析失败: API密钥无效或认证失败，请检查您的API配置"
            elif "model_not_found" in str(e).lower():
                error_msg = f"AI分析失败: 模型 {self.model} 不存在或没有访问权限"
            elif "model" in str(e).lower() or "not supported" in str(e).lower():
                error_msg = f"AI分析失败: 当前模型 {self.model} 不支持图像分析功能"
            elif "404" in str(e):
                error_msg = "AI分析失败: API端点不存在或访问受限"
            elif "network" in str(e).lower() or "timeout" in str(e).lower():
                error_msg = "AI分析失败: 网络连接超时，请检查您的网络连接"
            elif "image" in str(e).lower() or "vision" in str(e).lower():
                error_msg = "AI分析失败: 当前API或模型不支持图像分析功能"
            return False, None, error_msg
    
    def ocr_image(self, image_path: str, lang: str = "chi_sim+eng", use_vl_model: bool = False, engine: str = "ai") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Perform OCR on an image using AI
        
        Args:
            image_path: Path to the image file
            lang: Language(s) to use for OCR (comma-separated)
            use_vl_model: Whether to use Qwen-VL-Max model for OCR
            engine: OCR engine to use (ai, paddle, tesseract, auto)
            
        Returns:
            Tuple of (success, ocr_text, error_message)
        """
        try:
            import os
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False, None, f"Image file not found: {image_path}"
            
            # 如果指定使用VL模型或当前模型是qwen-vl-max，则使用QwenVLManager
            if use_vl_model or self.model == "qwen-vl-max":
                try:
                    from .qwen_vl_manager import QwenVLManager
                    vl_manager = QwenVLManager(api_key=self.api_key, api_type=self.api_type)
                    return vl_manager.ocr_image(image_path, lang)
                except Exception as vl_error:
                    self.logger.warning(f"Failed to use Qwen-VL-Max manager: {vl_error}, falling back to regular OCR")
            
            # 引擎选择逻辑
            engines_to_try = []
            if engine == "ai":
                engines_to_try = ["ai", "paddle", "tesseract"]
            elif engine == "paddle":
                engines_to_try = ["paddle", "tesseract", "ai"]
            elif engine == "tesseract":
                engines_to_try = ["tesseract", "paddle", "ai"]
            elif engine == "auto":
                engines_to_try = ["ai", "paddle", "tesseract"]
            
            # 依次尝试各引擎
            for current_engine in engines_to_try:
                try:
                    if current_engine == "ai":
                        if self.api_type in ["openai", "qwen"]:
                            if not self.client:
                                self.client = self._initialize_client()
                                if not self.client:
                                    raise Exception("Failed to initialize client")
                            
                            # 读取图像文件并转换为base64
                            import base64
                            with open(image_path, "rb") as f:
                                base64_image = base64.b64encode(f.read()).decode("utf-8")
                            
                            # 构建OCR提示词，根据语言选择合适的提示
                            # 首先检测语言
                            if "eng" in lang.lower():
                                # 英文提示词
                                system_prompt = "You are a professional OCR assistant. Extract all text from the image with high accuracy. Preserve the original formatting, line breaks, and punctuation exactly as they appear in the image. Only output the recognized text, no additional explanations or comments."
                                ocr_prompt = "Please extract all text from this image. Preserve the original formatting, line breaks, and punctuation. Output only the recognized text, with no additional descriptions or explanations."
                            elif "jpn" in lang.lower():
                                # 日文提示词
                                system_prompt = "あなたは専門的なOCRアシスタントです。画像からすべてのテキストを高精度で抽出してください。元のフォーマット、改行、句読点を正確に維持してください。認識したテキストのみを出力し、追加の説明やコメントは含めないでください。"
                                ocr_prompt = "この画像からすべてのテキストを抽出してください。元のフォーマット、改行、句読点を維持してください。認識したテキストのみを出力し、追加の説明や解釈は含めないでください。"
                            elif "kor" in lang.lower():
                                # 韩文提示词
                                system_prompt = "당신은 전문적인 OCR 어시스턴트입니다. 이미지에서 모든 텍스트를 고정밀도로 추출하세요. 원래 형식, 줄 바꿈, 구두점을 정확히 유지하세요. 인식된 텍스트만 출력하고, 추가 설명이나 댓글은 포함하지 마세요."
                                ocr_prompt = "이 이미지에서 모든 텍스트를 추출하세요. 원래 형식, 줄 바꿈, 구두점을 유지하세요. 인식된 텍스트만 출력하고, 추가 설명이나 해석은 포함하지 마세요."
                            elif "fra" in lang.lower() or "deu" in lang.lower() or "rus" in lang.lower():
                                # 其他欧洲语言提示词
                                system_prompt = "You are a professional OCR assistant. Extract all text from the image with high accuracy. Preserve the original formatting, line breaks, and punctuation exactly as they appear in the image. Only output the recognized text, no additional explanations or comments."
                                ocr_prompt = "Please extract all text from this image. Preserve the original formatting, line breaks, and punctuation. Output only the recognized text, with no additional descriptions or explanations."
                            else:
                                # 默认中文提示词
                                system_prompt = "你是一个专业的OCR助手，能够高精度地从图像中提取所有文字。请严格保持原文的格式、排版、换行和标点符号，确保识别结果的准确性和完整性。只输出识别到的文字内容，不要添加任何额外的描述或解释。"
                                ocr_prompt = "请高精度地提取这张图片中的所有文字，严格保持原文的格式、排版、换行和标点符号。只输出识别到的文字内容，不要添加任何额外的描述或解释。"
                            
                            # 构建消息
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_prompt
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": ocr_prompt
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ]
                            
                            # 调用AI API进行OCR
                            completion = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                max_tokens=2000
                            )
                            
                            # 处理结果
                            ocr_text = completion.choices[0].message.content.strip()
                            logger.info(f"AI OCR识别成功，使用引擎: {current_engine}")
                            return True, ocr_text, None
                        else:
                            raise Exception(f"API type {self.api_type} not supported for AI OCR")
                    elif current_engine == "paddle":
                        # 使用PaddleOCR进行识别
                        return self._paddle_ocr(image_path, lang)
                    elif current_engine == "tesseract":
                        # 使用Tesseract进行识别
                        return self._tesseract_ocr(image_path, lang)
                except Exception as e:
                    self.logger.error(f"{current_engine} OCR失败: {e}")
                    # 继续尝试下一个引擎
                    continue
            
            # 所有引擎都失败
            return False, None, "所有OCR引擎都失败"
        except Exception as e:
            self.logger.error(f"Error in ocr_image: {e}")
            return False, None, str(e)
    
    def _paddle_ocr(self, image_path: str, lang: str = "chi_sim+eng") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        使用PaddleOCR进行OCR识别
        
        Args:
            image_path: Path to the image file
            lang: Language(s) to use for OCR
            
        Returns:
            Tuple of (success, ocr_text, error_message)
        """
        try:
            # 使用单例模式获取PaddleOCR实例
            paddle_ocr_singleton = PaddleOCRSingleton()
            ocr = paddle_ocr_singleton.get_ocr_instance()
            
            if not ocr:
                raise Exception("PaddleOCR未初始化成功")
            
            # 执行OCR识别，新版本PaddleOCR不再支持cls参数
            result = ocr.ocr(image_path)
            
            # 提取识别文本
            ocr_text = ""
            for line in result:
                if line:
                    for word_info in line:
                        if word_info and len(word_info) > 1 and len(word_info[1]) > 0:
                            ocr_text += word_info[1][0] + "\n"
            
            if ocr_text.strip():
                logger.info("PaddleOCR识别成功")
                return True, ocr_text.strip(), None
            else:
                return False, None, "PaddleOCR未识别到任何文本"
        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return False, None, str(e)
    
    def _tesseract_ocr(self, image_path: str, lang: str = "chi_sim+eng") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        使用Tesseract进行OCR识别
        
        Args:
            image_path: Path to the image file
            lang: Language(s) to use for OCR
            
        Returns:
            Tuple of (success, ocr_text, error_message)
        """
        try:
            import pytesseract
            from PIL import Image
            
            # 使用系统默认的Tesseract路径，不硬编码
            
            # 打开图像并执行OCR
            img = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(img, lang=lang)
            
            if ocr_text.strip():
                logger.info("Tesseract识别成功")
                return True, ocr_text.strip(), None
            else:
                return False, None, "Tesseract未识别到任何文本"
        except Exception as e:
            logger.error(f"Tesseract识别失败: {e}")
            return False, None, str(e)
    
    def image_to_table(self, image_path: str, use_vl_model: bool) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        将图片中的表格转化为可编辑表格
        
        Args:
            image_path: Path to the image file containing table
            use_vl_model: Whether to use Qwen-VL-Max model for table recognition
            
        Returns:
            Tuple of (success, table_data, error_message)
        """
        try:
            import os
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False, None, f"Image file not found: {image_path}"
            
            # 如果指定使用VL模型或当前模型是qwen-vl-max，则使用QwenVLManager
            if use_vl_model or self.model == "qwen-vl-max":
                try:
                    from .qwen_vl_manager import QwenVLManager
                    vl_manager = QwenVLManager(api_key=self.api_key, model="qwen-vl-max", api_type=self.api_type)
                    success, table_data, error = vl_manager.analyze_table_structure(image_path)
                    
                    # 添加父级和子级关系识别
                    if success and table_data:
                        print("\n=== 父级和子级关系识别 ===")
                        
                        # 1. 获取合并单元格信息
                        merged_cells = table_data.get("merged_cells", [])
                        if not merged_cells:
                            print("未识别到合并单元格")
                        else:
                            # 2. 获取数据行，因为AI可能将表头行识别为数据行
                            rows = table_data.get("rows", [])
                            
                            # 3. 从第一行数据中提取子级表头
                            if rows:
                                first_row = rows[0]
                                if len(first_row) >= 2:
                                    # 提取可能的子级表头
                                    possible_headers = [cell.strip() for cell in first_row if cell.strip()]
                                    
                                    # 4. 遍历合并单元格，寻找父级表头
                                    for merge in merged_cells:
                                        if len(merge) == 4:
                                            start_row, start_col, end_row, end_col = merge
                                            
                                            # 只处理表头区域的合并单元格
                                            if start_row < 2:
                                                # 父级表头文本
                                                headers = table_data.get("headers", [])
                                                parent_text = headers[start_col] if start_col < len(headers) else ""
                                                
                                                # 5. 识别子级表头
                                                # 从第一行数据中提取被合并单元格覆盖的列作为子级
                                                child_cells = []
                                                # 遍历合并单元格覆盖的所有列
                                                for c in range(start_col, end_col + 1):
                                                    if c < len(first_row):
                                                        child_text = first_row[c].strip()
                                                        if child_text:
                                                            child_cells.append((c, child_text))
                                                
                                                # 6. 检查是否找到了正确的子级表头
                                                if len(child_cells) > 1:
                                                    print(f"\n父级表头: '{parent_text}'")
                                                    print(f"  位置: 行{start_row+1}-{end_row+1}, 列{start_col+1}-{end_col+1}")
                                                    print(f"  子级表头: ")
                                                    for child_col, child_text in child_cells:
                                                        print(f"    - '{child_text}' (列{child_col+1})")
                        
                        print("\n=== 关系识别完成 ===")
                    
                    return success, table_data, error
                except Exception as vl_error:
                    self.logger.warning(f"Failed to use Qwen-VL-Max manager: {vl_error}, falling back to regular table analysis")
            
            # 读取图像文件并转换为base64
            import base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
            # 构建表格识别提示词 - 平衡反模板干扰和识别指导
            system_prompt = "你是一个专业的表格分析助手，能够准确识别图片中的表格结构和内容。请你严格按照图片中的实际内容进行识别，只提取你在图片中实际看到的内容，不要添加任何不存在的信息，不要使用任何预设的表格模板。你需要仔细观察图片中的表格结构，识别出表头、行数据、总行数和总列数。"
            table_prompt = "请你识别这张图片中的表格，严格按照以下要求输出：\n1. 只提取图片中实际可见的内容，不要添加任何不存在的信息\n2. 不要使用任何预设的表格模板，只基于图片中的实际内容进行识别\n3. 仔细观察表格结构，识别出表头、行数据、总行数和总列数\n4. 输出JSON格式，包含：\n   - headers: 表头列表\n   - rows: 行数据列表，每行是一个单元格内容列表\n   - total_rows: 总行数\n   - total_cols: 总列数\n\n只输出JSON，不要添加任何额外的描述或解释。"
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": table_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用AI API进行表格识别
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            
            # 处理结果
            table_data = completion.choices[0].message.content.strip()
            import json
            table_json = json.loads(table_data)
            logger.info(f"表格识别成功，行数: {table_json.get('total_rows', 0)}, 列数: {table_json.get('total_cols', 0)}")
            return True, table_json, None
        except Exception as e:
            logger.error(f"表格识别失败: {e}")
            return False, None, str(e)
    
    def semantic_correction(self, ocr_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        利用AI大模型对OCR识别结果进行语义纠错和逻辑核查
        
        Args:
            ocr_text: OCR识别结果文本
            
        Returns:
            Tuple of (success, corrected_text, error_message)
        """
        try:
            if not ocr_text.strip():
                return False, None, "OCR文本为空"
            
            # 构建语义纠错提示词
            system_prompt = "你是一个专业的文本纠错助手，能够根据上下文语义对OCR识别结果进行纠错和优化。请保持原文的格式和结构，只修正识别错误的字符和词语。"
            correction_prompt = f"请对以下OCR识别结果进行语义纠错和逻辑核查，修正识别错误的字符、词语和语法错误，保持原文格式和结构不变：\n\n{ocr_text}\n\n只输出纠错后的文本，不要添加任何额外的描述或解释。"
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": correction_prompt
                }
            ]
            
            # 调用AI API进行语义纠错
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000
            )
            
            # 处理结果
            corrected_text = completion.choices[0].message.content.strip()
            logger.info("语义纠错成功")
            return True, corrected_text, None
        except Exception as e:
            logger.error(f"语义纠错失败: {e}")
            return False, None, str(e)
    
    def chart_to_data(self, image_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        将图片中的图表转化为Excel数据
        
        Args:
            image_path: Path to the image file containing chart
            
        Returns:
            Tuple of (success, chart_data, error_message)
        """
        try:
            import os
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False, None, f"Image file not found: {image_path}"
            
            # 读取图像文件并转换为base64
            import base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            
            # 构建图表识别提示词
            system_prompt = "你是一个专业的图表识别助手，能够高精度地从图像中提取图表数据。请将图表内容转换为JSON格式，包含图表类型、数据系列、坐标轴信息等。"
            chart_prompt = "请识别这张图片中的图表，提取其完整数据，输出JSON格式，包含：\n1. chart_type: 图表类型（柱状图、折线图、饼图等）\n2. title: 图表标题\n3. categories: 横坐标分类列表\n4. series: 数据系列列表，每个系列包含name和data字段\n5. x_axis: 横坐标名称\n6. y_axis: 纵坐标名称\n\n只输出JSON，不要添加任何额外的描述或解释。"
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": chart_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用AI API进行图表识别
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            
            # 处理结果
            chart_data = completion.choices[0].message.content.strip()
            import json
            chart_json = json.loads(chart_data)
            logger.info(f"图表识别成功，类型: {chart_json.get('chart_type', '未知')}")
            return True, chart_json, None
        except Exception as e:
            logger.error(f"图表识别失败: {e}")
            return False, None, str(e)
    
    def document_qa(self, document_content: str, question: str, context: list = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        基于文档内容进行交互式问答
        
        Args:
            document_content: 文档内容文本
            question: 用户问题
            context: 对话上下文列表，每个元素是{"role": "user/assistant", "content": "内容"}格式
            
        Returns:
            Tuple of (success, answer, error_message)
        """
        try:
            if not document_content.strip():
                return False, None, "文档内容为空"
            
            if not question.strip():
                return False, None, "问题为空"
            
            # 构建文档问答提示词
            system_prompt = "你是一个专业的文档问答助手，能够基于提供的文档内容准确回答用户问题。请严格按照文档内容回答，不要添加任何文档中没有的信息。"
            
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"文档内容：\n{document_content}\n\n基于以上文档内容，请回答以下问题：\n{question}"
                }
            ]
            
            # 添加对话上下文
            if context and isinstance(context, list):
                messages.extend(context)
            
            # 调用AI API进行问答
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000
            )
            
            # 处理结果
            answer = completion.choices[0].message.content.strip()
            logger.info("文档问答成功")
            return True, answer, None
        except Exception as e:
            logger.error(f"文档问答失败: {e}")
            return False, None, str(e)
    
    def generate_presentation_outline(self, topic: str, image_analysis: Dict[str, Any], audience: str = "通用", style: str = "正式", slide_count: int = 10) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Generate PPT outline using AI
        
        Args:
            topic: PPT topic
            image_analysis: Image analysis results
            audience: Audience type
            style: Presentation style
            slide_count: Expected slide count
            
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"

                # 设计提示词模板，生成PPT大纲
                outline_prompt = f"""
                请为主题为"{topic}"的PowerPoint演示文稿生成一个详细的大纲。
                受众：{audience}
                风格：{style}
                预计幻灯片数量：{slide_count}张
                
                图片分析结果：{image_analysis}
                
                要求：
                1. 大纲结构清晰，逻辑严谨，包含封面、目录、正文各部分和结尾
                2. 每个幻灯片要有明确的标题和主要内容要点
                3. 输出格式为JSON，包含title（PPT主标题）、subtitle（PPT副标题）、slides（幻灯片列表）
                4. slides列表中的每个元素包含slide_title（幻灯片标题）和content（内容要点列表）
                5. 确保生成的内容专业、全面、符合主题和图片风格
                """

                # 发送请求
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的PPT大纲生成专家，能够根据主题和图片分析结果生成高质量的PPT大纲。"},
                        {"role": "user", "content": outline_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=2000,
                )

                import json
                result = json.loads(completion.choices[0].message.content.strip())
                return True, result, None
            else:
                return False, None, f"API type {self.api_type} not supported for presentation outline generation"
        except Exception as e:
            self.logger.error(f"Error in generate_presentation_outline: {e}")
            # 移除降级处理，必须由AI处理，发生异常时返回错误
            return False, None, f"AI处理失败: {str(e)}"
    
    def generate_presentation_content(self, outline: Dict[str, Any], image_analysis: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Generate PPT content using AI
        
        Args:
            outline: PPT outline
            image_analysis: Image analysis results
            
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"

                # 优化提示词，确保生成更丰富的内容
                content_prompt = f"""
                请根据以下PPT大纲和图片分析结果，为每张幻灯片生成详细、专业、丰富的内容：
                
                PPT大纲：{outline}
                图片分析结果：{image_analysis}
                
                要求：
                1. 内容必须与大纲主题一致，结合图片分析结果
                2. 每张幻灯片内容要详细，包含多个要点
                3. 内容风格要专业，符合科技感主题
                4. 输出格式为严格的JSON，包含slides数组
                5. 每个slide包含slide_title和detailed_content
                6. detailed_content是字符串数组，每个元素是一个完整的内容要点
                7. 每个内容要点要包含足够的详细信息，至少3-4行
                8. 避免使用过长的段落，保持内容清晰易读
                9. 确保JSON格式正确，没有语法错误
                
                示例输出格式：
                {{"slides": [{{"slide_title": "封面", "detailed_content": ["人工智能的发展趋势", "探索AI的未来可能性"]}}, {{"slide_title": "目录", "detailed_content": ["1. 主题介绍", "2. 发展背景", "3. 核心内容", "4. 应用场景", "5. 未来展望", "6. 总结与致谢"]}}]}}
                """

                # 发送请求
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的PPT内容创作专家，能够根据大纲和图片分析结果生成高质量、详细的PPT内容。"},
                        {"role": "user", "content": content_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=4000,  # 增加token限制，生成更丰富的内容
                )

                import json
                # 增强的JSON解析，处理可能的格式问题
                content = completion.choices[0].message.content.strip()
                
                # 尝试修复可能的JSON格式问题
                if not content.startswith('{'):
                    # 寻找JSON开始位置
                    json_start = content.find('{')
                    if json_start != -1:
                        content = content[json_start:]
                    else:
                        return False, None, "AI返回结果格式错误，无法找到JSON开始位置"
                if not content.endswith('}'):
                    # 寻找JSON结束位置
                    json_end = content.rfind('}')
                    if json_end != -1:
                        content = content[:json_end+1]
                    else:
                        return False, None, "AI返回结果格式错误，无法找到JSON结束位置"
                
                result = json.loads(content)
                
                # 检查结果格式，确保包含slides数组
                if "slides" not in result or not isinstance(result["slides"], list):
                    return False, None, "AI返回结果格式错误，缺少slides数组"
                
                # 检查是否有内容
                if len(result["slides"]) < 3:
                    return False, None, "AI返回内容不足，需要至少3张幻灯片"
                
                return True, result, None
            else:
                return False, None, f"API type {self.api_type} not supported for presentation content generation"
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            return False, None, f"AI返回结果JSON解析失败: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error in generate_presentation_content: {e}")
            return False, None, f"AI处理失败: {str(e)}"
    
    def generate_presentation_style(self, image_analysis: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Generate PPT style suggestions using AI
        
        Args:
            image_analysis: Image analysis results
            
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"

                # 设计提示词模板，生成PPT样式建议
                style_prompt = f"""
                请根据以下图片分析结果，生成PPT样式建议：
                
                图片分析结果：{image_analysis}
                
                要求：
                1. 样式要与图片风格相匹配
                2. 包含文字样式、形状设计、颜色搭配和布局建议
                3. 输出格式为JSON，包含：
                   - text_styles: 文字样式建议
                   - shape_suggestions: 形状设计建议
                   - color_scheme: 颜色搭配方案
                   - layout_suggestions: 布局建议
                4. 样式建议要具体、可执行
                """

                # 发送请求
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的PPT样式设计专家，能够根据图片分析结果生成高质量的PPT样式建议。"},
                        {"role": "user", "content": style_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=2000,
                )

                import json
                result = json.loads(completion.choices[0].message.content.strip())
                # 确保返回结果包含success字段
                if isinstance(result, dict) and "success" not in result:
                    result["success"] = True
                return True, result, None
            else:
                return False, None, f"API type {self.api_type} not supported for presentation style generation"
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            return False, None, f"AI返回结果JSON解析失败: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error in generate_presentation_style: {e}")
            return False, None, f"AI处理失败: {str(e)}"

    def set_api_key(self, api_key: str):
        """
        Set the API key
        """
        self.api_key = api_key
        self.client = self._initialize_client()

    def set_model(self, model: str):
        """
        Set the model to use
        """
        self.model = model

    def set_api_type(self, api_type: str):
        """
        Set the API type
        """
        self.api_type = api_type
        self.client = self._initialize_client()
    
    def ai_optimize_document(self, content: str, model: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Optimize document content using AI
        
        Args:
            content: Document content to optimize
            model: AI model name
            
        Returns:
            Tuple of (success, optimized_content, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"
            else:
                return False, None, f"API type {self.api_type} not supported for document optimization"
            
            # 使用千问API优化文档
            prompt = f"""
            请优化以下文档内容，使其更加清晰、流畅、专业：
            
            {content}
            
            要求：
            1. 保持原文意思不变
            2. 提高文章的逻辑性和可读性
            3. 修正语法错误和用词不当
            4. 优化段落结构和排版
            5. 直接返回优化后的完整内容，不要添加任何解释
            """
            
            completion = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档优化专家，能够优化各种类型的文档内容。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
            )
            
            return True, completion.choices[0].message.content.strip(), None
        except Exception as e:
            self.logger.error(f"Error in ai_optimize_document: {e}")
            return False, None, f"AI处理失败: {str(e)}"
    
    def ai_summarize_document(self, content: str, model: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Summarize document content using AI
        
        Args:
            content: Document content to summarize
            model: AI model name
            
        Returns:
            Tuple of (success, summary, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"
            else:
                return False, None, f"API type {self.api_type} not supported for document summarization"
            
            # 使用千问API生成文档摘要
            prompt = f"""
            请为以下文档内容生成一个简洁、全面的摘要：
            
            {content}
            
            要求：
            1. 涵盖文档的主要内容和核心观点
            2. 保持逻辑清晰，结构完整
            3. 语言简洁，表达流畅
            4. 直接返回摘要内容，不要添加任何解释
            """
            
            completion = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档摘要专家，能够生成准确、简洁的文档摘要。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
            )
            
            return True, completion.choices[0].message.content.strip(), None
        except Exception as e:
            self.logger.error(f"Error in ai_summarize_document: {e}")
            return False, None, f"AI处理失败: {str(e)}"
    
    def ai_translate_document(self, content: str, target_lang: str, model: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Translate document content using AI
        
        Args:
            content: Document content to translate
            target_lang: Target language
            model: AI model name
            
        Returns:
            Tuple of (success, translated_content, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"
            else:
                return False, None, f"API type {self.api_type} not supported for document translation"
            
            # 使用千问API翻译文档
            prompt = f"""
            请将以下文档内容翻译成{target_lang}：
            
            {content}
            
            要求：
            1. 保持原文意思准确无误
            2. 译文语言自然流畅，符合目标语言的表达习惯
            3. 保留原文的格式和结构
            4. 直接返回译文内容，不要添加任何解释
            """
            
            completion = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的翻译专家，能够准确翻译各种类型的文档内容。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
            )
            
            return True, completion.choices[0].message.content.strip(), None
        except Exception as e:
            self.logger.error(f"Error in ai_translate_document: {e}")
            return False, None, f"AI处理失败: {str(e)}"
    
    def ai_generate_document(self, prompt: str, model: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Generate document content using AI
        
        Args:
            prompt: Prompt for document generation
            model: AI model name
            
        Returns:
            Tuple of (success, generated_content, error_message)
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize client, please check your API configuration"
            else:
                return False, None, f"API type {self.api_type} not supported for document generation"
            
            # 使用千问API生成文档
            gen_prompt = f"""
            请根据以下提示生成一篇完整的文档：
            
            {prompt}
            
            要求：
            1. 内容全面，结构清晰
            2. 语言专业，表达流畅
            3. 符合逻辑，层次分明
            4. 直接返回生成的文档内容，不要添加任何解释
            """
            
            completion = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档生成专家，能够根据提示生成高质量的文档内容。"},
                    {"role": "user", "content": gen_prompt},
                ],
                max_tokens=4000,
            )
            
            return True, completion.choices[0].message.content.strip(), None
        except Exception as e:
            self.logger.error(f"Error in ai_generate_document: {e}")
            return False, None, f"AI处理失败: {str(e)}"