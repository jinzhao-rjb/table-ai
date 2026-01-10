"""
Qwen-VL-Max Model Manager for OCR functionality.
This module provides specialized handling for Qwen-VL-Max model calls for OCR tasks.
"""

import json
import logging
import os
import time
from typing import Tuple, Optional, Dict, Any

from openai import OpenAI

logger = logging.getLogger("QwenVLManager")


class QwenVLManager:
    """
    Qwen-VL-Max Model Manager for OCR and image analysis tasks.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        api_type: str = None,
        redis_client=None,
    ):
        """
        Initialize the Qwen-VL-Max Model Manager

        Args:
            api_key: API key for the AI service (从配置获取默认值)
            model: Model name to use (从配置获取默认值)
            api_type: API type (从配置获取默认值)
            redis_client: Redis客户端实例，用于速率限制
        """
        # 从配置文件获取默认值，避免硬编码
        import sys
        import os

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        )
        from src.utils.config import config_manager

        self.api_key = api_key if api_key is not None else config_manager.get("ai.api_key", "")
        self.model = model if model is not None else "qwen-vl-max"
        self.api_type = api_type if api_type is not None else config_manager.get("ai.api_type", "qwen")
        self.logger = logging.getLogger("QwenVLManager")
        self.redis_client = redis_client

        # 从配置获取速率限制设置
        self.max_requests_per_minute = config_manager.get("rate_limiting.max_requests_per_minute", 30)
        self.rate_limit_key = "qwen_vl_requests_count"

        # Initialize OpenAI client for Qwen-VL-Max
        self.client = self._initialize_client()
        self.max_retries = 3

    def _initialize_client(self) -> Optional[OpenAI]:
        """
        Initialize the OpenAI client for Qwen-VL-Max

        Returns:
            Initialized OpenAI client or None if initialization failed
        """
        try:
            if self.api_type in ["openai", "qwen"]:
                # For Qwen-VL-Max, use OpenAI compatible client with Dashscope endpoint
                base_url = (
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    if self.api_type == "qwen"
                    else None
                )
                return OpenAI(api_key=self.api_key, base_url=base_url)
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize Qwen-VL-Max client: {e}")
            return None

    def ocr_image(self, image_path: str, lang: str = "chi_sim+eng") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Perform OCR on an image using Qwen-VL-Max
        
        Args:
            image_path: Path to the image file
            lang: Language(s) to use for OCR (comma-separated)
            
        Returns:
            Tuple of (success, ocr_text, error_message)
        """
        try:
            if not self.client:
                self.client = self._initialize_client()
                if not self.client:
                    return False, None, "Failed to initialize Qwen-VL-Max client"

            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False, None, f"Image file not found: {image_path}"

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

            # 调用AI API进行OCR，使用qwen-vl-max模型
            completion = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
                max_tokens=2000
            )

            # 处理结果
            ocr_text = completion.choices[0].message.content.strip()
            return True, ocr_text, None
        except Exception as e:
            self.logger.error(f"Error in Qwen-VL-Max OCR: {e}")
            error_msg = f"Qwen-VL-Max OCR failed: {str(e)}"
            if "API key" in str(e) or "authentication" in str(e):
                error_msg = "Qwen-VL-Max OCR failed: Invalid API key or authentication error"
            elif "model_not_found" in str(e).lower():
                error_msg = "Qwen-VL-Max OCR failed: Model qwen-vl-max not found or no access permission"
            elif "network" in str(e).lower() or "timeout" in str(e).lower():
                error_msg = "Qwen-VL-Max OCR failed: Network timeout error"
            return False, None, error_msg

    def analyze_image(self, image_path: str, prompt: str = "") -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Analyze image content using Qwen-VL-Max
        
        Args:
            image_path: Path to the image file
            prompt: Analysis prompt
            
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            if not self.client:
                self.client = self._initialize_client()
                if not self.client:
                    return False, None, "Failed to initialize Qwen-VL-Max client"

            # 检查文件是否存在
            if not os.path.exists(image_path):
                return False, None, f"Image file not found: {image_path}"

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

            # 调用AI API进行图像分析，使用qwen-vl-max模型
            completion = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
                max_tokens=2000
            )

            # 处理结果
            result = completion.choices[0].message.content.strip()
            self.logger.info(f"Qwen-VL-Max原始识别结果: {result}")
            
            # 提取JSON内容的简单方法：找到第一个'{'和最后一个'}'
            try:
                # 找到JSON开始和结束位置
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    # 提取JSON字符串
                    json_str = result[json_start:json_end]
                    self.logger.info(f"提取到的JSON字符串: {json_str}")
                    
                    # 解析JSON格式结果
                    table_json = json.loads(json_str)
                    
                    # 验证表格数据完整性
                    if not isinstance(table_json, dict):
                        raise ValueError("识别结果不是有效的JSON对象")
                        
                    # 确保返回正确的表格结构，防止单列模式
                    if 'headers' not in table_json or 'rows' not in table_json:
                        # 如果缺少必要字段，尝试从内容中提取
                        self.logger.warning("表格识别结果缺少headers或rows字段，正在尝试重新处理...")
                        # 简单处理：假设第一行是表头，其余是数据
                        # 这里可以添加更复杂的处理逻辑
                        return False, None, "表格识别结果格式不正确"
                    
                    # 计算总行数和总列数
                    headers = table_json['headers']
                    rows = table_json['rows']
                    
                    # 确保headers是列表
                    if not isinstance(headers, list):
                        headers = [headers]
                    
                    # 确保rows是列表
                    if not isinstance(rows, list):
                        rows = [rows]
                    
                    # 计算总列数
                    total_cols = len(headers)
                    if total_cols == 0:
                        # 如果表头为空，尝试从第一行数据获取列数
                        if rows:
                            first_row = rows[0]
                            if isinstance(first_row, list):
                                total_cols = len(first_row)
                            else:
                                total_cols = 1
                        else:
                            total_cols = 0
                    
                    # 计算总行数
                    total_rows = len(rows)
                    
                    # 更新表格JSON
                    table_json['total_rows'] = total_rows
                    table_json['total_cols'] = total_cols
                    
                    # 确保返回正确的表格结构，防止单列模式
                    if total_cols < 2:
                        # 对于可能被误识别为单列的表格，再次处理以确保正确的多列结构
                        self.logger.warning(f"表格被识别为单列（{total_cols}列），正在重新处理...")
                        # 这里可以添加更复杂的处理逻辑，或者直接返回原始结果
                    
                    return True, table_json, None
                else:
                    self.logger.error("无法在结果中找到有效的JSON内容")
                    return False, None, "无法在结果中找到有效的JSON内容"
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Qwen-VL-Max table result as JSON: {e}")
                self.logger.error(f"Raw result: {result}")
                # 如果JSON解析失败，尝试提取表格数据
                return False, None, f"Failed to parse table result: {e}"
            except Exception as e:
                self.logger.error(f"Error processing Qwen-VL-Max table result: {e}")
                self.logger.error(f"Raw result: {result}")
                return False, None, f"Error processing table result: {e}"
        except Exception as e:
            self.logger.error(f"Error in Qwen-VL-Max image analysis: {e}")
            return False, None, f"Qwen-VL-Max image analysis failed: {str(e)}"

    def _check_rate_limit(self):
        """
        检查API速率限制，使用Redis实现跨线程速率控制
        """
        import time
        current_time = int(time.time())
        current_minute = current_time // 60
        
        # 构造速率限制键
        key = f"{self.rate_limit_key}:{current_minute}"
        
        # 初始化Redis客户端（如果未提供）
        if not self.redis_client:
            import redis
            from src.utils.config import config_manager
            redis_host = config_manager.get("redis.host", "localhost")
            redis_port = config_manager.get("redis.port", 6379)
            redis_db = config_manager.get("redis.db", 0)
            self.logger.info(f"初始化Redis客户端：host={redis_host}, port={redis_port}, db={redis_db}")
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
        
        try:
            # 使用Redis的原子操作实现速率限制
            request_count = self.redis_client.get(key)
            request_count = int(request_count) if request_count else 0
            
            self.logger.debug(f"检查速率限制：key={key}, 当前请求数={request_count}, 限制={self.max_requests_per_minute}")
            
            if request_count >= self.max_requests_per_minute:
                # 超过速率限制，等待到下一分钟
                wait_time = (current_minute + 1) * 60 - current_time
                self.logger.warning(f"API速率限制：已达到每分钟{self.max_requests_per_minute}次请求，等待{wait_time}秒")
                time.sleep(wait_time)
                # 重置key和请求计数
                current_time = int(time.time())
                current_minute = current_time // 60
                key = f"{self.rate_limit_key}:{current_minute}"
                request_count = 0
                self.logger.info(f"重置速率限制：新key={key}")
            
            # 增加请求计数，设置过期时间为1分钟
            new_count = self.redis_client.incr(key)
            self.redis_client.expire(key, 60)
            self.logger.debug(f"速率限制更新：key={key}, 新请求数={new_count}, 过期时间=60秒")
        except redis.RedisError as e:
            self.logger.error(f"Redis速率限制检查失败：{e}")
            # Redis错误时，使用内存中的简单速率限制作为降级方案
            self.logger.warning("Redis不可用，使用内存速率限制降级方案")
            # 简单的内存速率限制实现
            self._memory_rate_limit()
    
    def _memory_rate_limit(self):
        """
        内存中的简单速率限制，作为Redis不可用时的降级方案
        """
        import time
        
        # 初始化内存中的速率限制计数器
        if not hasattr(self, '_memory_rate_limit_data'):
            self._memory_rate_limit_data = {
                'current_minute': int(time.time()) // 60,
                'request_count': 0
            }
        
        current_time = int(time.time())
        current_minute = current_time // 60
        
        # 检查是否需要重置计数器
        if current_minute != self._memory_rate_limit_data['current_minute']:
            self._memory_rate_limit_data['current_minute'] = current_minute
            self._memory_rate_limit_data['request_count'] = 0
        
        # 检查速率限制
        if self._memory_rate_limit_data['request_count'] >= self.max_requests_per_minute:
            wait_time = (current_minute + 1) * 60 - current_time
            self.logger.warning(f"内存速率限制：已达到每分钟{self.max_requests_per_minute}次请求，等待{wait_time}秒")
            time.sleep(wait_time)
            # 重置计数器
            current_time = int(time.time())
            current_minute = current_time // 60
            self._memory_rate_limit_data['current_minute'] = current_minute
            self._memory_rate_limit_data['request_count'] = 0
        
        # 增加请求计数
        self._memory_rate_limit_data['request_count'] += 1
        self.logger.debug(f"内存速率限制：当前请求数={self._memory_rate_limit_data['request_count']}")
    
    def extract_table_to_html(self, image_input):
        """
        强制 AI 输出极简 HTML，确保包含物理合并属性
        """
        import time
        max_retries = self.max_retries
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.client:
                    self.client = self._initialize_client()
                    if not self.client:
                        return False, None, "Failed to initialize Qwen-VL-Max client"

                # 检查API速率限制
                self._check_rate_limit()

                # 处理图像输入，可以是路径或字节流
                import base64
                if isinstance(image_input, str):
                    # 检查文件是否存在
                    if not os.path.exists(image_input):
                        return False, None, f"Image file not found: {image_input}"

                    # 检查文件大小
                    file_size = os.path.getsize(image_input)
                    if file_size > 10 * 1024 * 1024:  # 10MB限制
                        return False, None, f"Image file is too large. Maximum size is 10MB."

                    # 读取图像文件并转换为base64
                    with open(image_input, "rb") as f:
                        base64_image = base64.b64encode(f.read()).decode("utf-8")
                else:
                    # 直接使用字节流转换为base64
                    base64_image = base64.b64encode(image_input).decode("utf-8")

                # 升级防御型提示词，强化合并单元格逻辑和嵌套表格处理
                prompt = """ 
你是一个精通复杂网页前端和表格解析的专家。请将图片中的表格转换为高质量、结构严谨的 HTML 代码。

### 特别任务：处理嵌套表格（Nested Tables）
1. **识别嵌套结构**：如果某个单元格内包含另一个子表格，必须在 <td> 标签内部嵌套一个完整的 <table> 结构。
2. **保持层级清晰**：子表格必须拥有完整的 <tr> 和 <td>，严禁将子表格的内容强行平铺到主表格中。

### 核心规则：
1. **合并单元格**：精准计算 rowspan 和 colspan。如果一个格子横跨或纵跨，必须严格标注。
2. **文字完整性**：提取表格内所有文字，保持原始格式和换行。
3. **过滤干扰**：
   - 忽略表格框线外的所有标题、页码、批注文字。
   - 过滤掉单元格内部的打勾符号（如 √、x 等选择性标记）。
4. **列数守恒**：确保每一行在考虑了 colspan 后的总列数是完全一致的，不要出现错位。

### 输出要求：
- 只输出 HTML <table> 标签及其内容。
- 不要有任何 Markdown 标识符（如 ```html）、解释文字或额外提示。
"""

                # 构建消息
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个像素级的表格识别助手，专注于财务报表还原。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt
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

                # 调用AI API进行表格分析，使用qwen-vl-max模型
                completion = self.client.chat.completions.create(
                    model="qwen-vl-max",
                    messages=messages,
                    max_tokens=4096,  # 关键修改：从默认值调大到 4096，确保支持 100 行以上的表格
                    temperature=0.1,  # 降低随机性，保证结构准确
                )

                # 处理结果
                result = completion.choices[0].message.content.strip()
                self.logger.info(f"Qwen-VL-Max原始识别结果: {result}")
                self.logger.info(f"结果长度: {len(result)} 字符")
                
                # 构建返回结果，包含原始content和类型标识
                table_data = {
                    'content': result,
                    'type': 'html' if '<table' in result.lower() else 'markdown',
                    'headers': [],  # 暂时为空，后续由table_processor填充
                    'rows': [],     # 暂时为空，后续由table_processor填充
                    'merged_cells': [],  # 暂时为空，后续由table_processor填充
                    'total_rows': 0,     # 暂时为空，后续由table_processor填充
                    'total_cols': 0      # 暂时为空，后续由table_processor填充
                }
                
                return True, table_data, None
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                if "network" in error_msg.lower() or "timeout" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # 指数退避
                        self.logger.warning(f"网络超时或速率限制，{wait_time}秒后重试 ({retry_count}/{max_retries}): {e}")
                        time.sleep(wait_time)
                        continue
                self.logger.error(f"Error in Qwen-VL-Max table analysis: {e}")
                return False, None, f"Qwen-VL-Max table analysis failed: {str(e)}"
    
    def get_table_html(self, image_input: str or bytes) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        获取图片中的表格HTML，针对电子版图片优化
        已统一为HTML格式，不再支持Markdown
        
        Args:
            image_input: Path to the table image file or image bytes
            
        Returns:
            Tuple of (success, html_content, error_message)
        """
        # 直接调用统一的extract_table_to_html方法
        success, result, error = self.extract_table_to_html(image_input)
        if success and result:
            return success, result.get('content', ''), error
        return success, None, error

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the connection to Qwen-VL-Max API
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.client:
                self.client = self._initialize_client()
                if not self.client:
                    return False, "Failed to initialize Qwen-VL-Max client"

            # 创建一个简单的测试消息
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, test connection with Qwen-VL-Max. Please respond with 'Connection successful'."
                        }
                    ]
                },
            ]

            # Send a simple test request
            completion = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
                max_tokens=20,
            )
            return True, f"Connection successful. Model: {completion.model}"
        except Exception as e:
            self.logger.error(f"Qwen-VL-Max connection test failed: {e}")
            return False, f"Connection failed: {str(e)}"
