"""数据解析模块

负责从HTML中提取结构化数据
"""

from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DataParser:
    """数据解析器"""
    
    @staticmethod
    def parse_html_table(html_content: str, table_selector: str = "table") -> List[pd.DataFrame]:
        """解析HTML表格为DataFrame列表
        
        Args:
            html_content: HTML内容
            table_selector: 表格CSS选择器
            
        Returns:
            DataFrame列表
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            tables = soup.select(table_selector)
            
            if not tables:
                logger.warning(f"未找到匹配的表格: {table_selector}")
                return []
            
            dataframes = []
            for i, table in enumerate(tables):
                df = DataParser._parse_single_table(table)
                if df is not None:
                    dataframes.append(df)
                    logger.info(f"成功解析表格 {i+1}/{len(tables)}, 数据形状: {df.shape}")
            
            return dataframes
            
        except Exception as e:
            logger.error(f"解析HTML表格失败: {e}")
            return []
    
    @staticmethod
    def _parse_single_table(table) -> Optional[pd.DataFrame]:
        """解析单个HTML表格为DataFrame
        
        Args:
            table: BeautifulSoup表格对象
            
        Returns:
            DataFrame或None
        """
        try:
            # 提取所有行
            all_rows = table.find_all('tr')
            if not all_rows:
                return None
            
            # 提取表头
            headers = []
            header_cells = table.find_all('th')
            
            if header_cells:
                # 有<th>标签，使用这些作为表头
                headers = [th.text.strip() for th in header_cells]
            else:
                # 没有<th>标签，检查第一行是否可以作为表头
                first_row_cells = all_rows[0].find_all(['td', 'th'])
                if first_row_cells:
                    headers = [cell.text.strip() for cell in first_row_cells]
            
            # 如果没有有效表头，使用默认列名
            if not headers:
                first_row_cells = all_rows[0].find_all(['td', 'th'])
                headers = [f"列{i+1}" for i in range(len(first_row_cells))]
            
            # 确定数据开始行
            start_row = 0
            if header_cells:  # 如果有<th>标签，第一行是表头，数据从第二行开始
                start_row = 1
            elif headers:  # 如果第一行被用作表头，数据从第二行开始
                first_row_content = [cell.text.strip() for cell in all_rows[0].find_all(['td', 'th'])]
                if len(first_row_content) == len(headers) and all(f == h for f, h in zip(first_row_content, headers)):
                    start_row = 1
            
            # 提取数据行
            rows = []
            for tr in all_rows[start_row:]:
                cells = tr.find_all(['td', 'th'])
                row_data = [cell.text.strip() for cell in cells]
                
                if row_data:
                    # 确保每行数据与表头数量一致
                    if len(row_data) < len(headers):
                        # 不足的列用空字符串填充
                        row_data.extend([''] * (len(headers) - len(row_data)))
                    elif len(row_data) > len(headers):
                        # 多余的列截断
                        row_data = row_data[:len(headers)]
                    
                    rows.append(row_data)
            
            if not rows:
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(rows, columns=headers)
            return df
            
        except Exception as e:
            logger.error(f"解析单个表格失败: {e}")
            return None
    
    @staticmethod
    def parse_json_from_html(html_content: str, json_selector: str = "script[type='application/json']") -> List[Dict]:
        """从HTML中提取JSON数据
        
        Args:
            html_content: HTML内容
            json_selector: JSON脚本的CSS选择器
            
        Returns:
            JSON数据列表
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            json_scripts = soup.select(json_selector)
            
            json_data_list = []
            for script in json_scripts:
                try:
                    json_data = json.loads(script.string)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试提取特定的JSON字符串
                    import re
                    script_content = script.string
                    if script_content:
                        # 寻找可能的JSON数据结构
                        json_matches = re.findall(r'\{.*?\}|\[.*?\]', script_content, re.DOTALL)
                        for match in json_matches:
                            try:
                                json_data = json.loads(match)
                                json_data_list.append(json_data)
                            except json.JSONDecodeError:
                                continue
            
            logger.info(f"成功提取 {len(json_data_list)} 个JSON数据")
            return json_data_list
            
        except Exception as e:
            logger.error(f"提取JSON数据失败: {e}")
            return []
    
    @staticmethod
    def parse_text_from_html(html_content: str, selector: str) -> List[str]:
        """从HTML中提取指定选择器的文本内容
        
        Args:
            html_content: HTML内容
            selector: CSS选择器
            
        Returns:
            文本内容列表
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            elements = soup.select(selector)
            
            texts = [element.text.strip() for element in elements]
            logger.info(f"成功提取 {len(texts)} 个文本元素")
            return texts
            
        except Exception as e:
            logger.error(f"提取文本内容失败: {e}")
            return []
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """清洗DataFrame数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        try:
            cleaned_df = df.copy()
            
            # 删除完全重复的行
            cleaned_df = cleaned_df.drop_duplicates()
            
            # 重置索引
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            # 填充空值
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['float64', 'int64']:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna('')
            
            # 去除字符串列的空格
            for col in cleaned_df.select_dtypes(include=['object']).columns:
                cleaned_df[col] = cleaned_df[col].str.strip()
            
            logger.info(f"数据清洗完成，清洗前后形状: {df.shape} → {cleaned_df.shape}")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"清洗DataFrame失败: {e}")
            return df
    
    @staticmethod
    def convert_to_json(df: pd.DataFrame) -> str:
        """将DataFrame转换为JSON字符串
        
        Args:
            df: DataFrame
            
        Returns:
            JSON字符串
        """
        try:
            json_str = df.to_json(orient='records', force_ascii=False, indent=2)
            return json_str
        except Exception as e:
            logger.error(f"转换DataFrame为JSON失败: {e}")
            return "[]"
    
    @staticmethod
    def convert_to_csv(df: pd.DataFrame) -> str:
        """将DataFrame转换为CSV字符串
        
        Args:
            df: DataFrame
            
        Returns:
            CSV字符串
        """
        try:
            csv_str = df.to_csv(index=False, encoding='utf-8')
            return csv_str
        except Exception as e:
            logger.error(f"转换DataFrame为CSV失败: {e}")
            return ""
    
    @staticmethod
    def parse_list_data(html_content: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从HTML中解析列表数据
        
        Args:
            html_content: HTML内容
            config: 解析配置，格式如下：
                {
                    "row_selector": "CSS选择器，用于定位列表中的每一行",
                    "fields": {
                        "字段名1": {
                            "selector": "CSS选择器，相对于row_selector",
                            "extract_type": "text"或"attribute",
                            "attribute": "如果extract_type是attribute，则指定要提取的属性名"
                        },
                        "字段名2": { ... }
                    }
                }
        
        Returns:
            列表数据，每个元素是一个字典
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            row_selector = config.get("row_selector")
            fields_config = config.get("fields", {})
            
            if not row_selector:
                logger.error("缺少row_selector配置")
                return []
            
            # 定位所有行
            rows = soup.select(row_selector)
            if not rows:
                logger.warning(f"未找到匹配的行: {row_selector}")
                return []
            
            results = []
            for row in rows:
                item = {}
                for field_name, field_config in fields_config.items():
                    selector = field_config.get("selector")
                    extract_type = field_config.get("extract_type", "text")
                    attribute = field_config.get("attribute", "href")
                    
                    if not selector:
                        continue
                    
                    try:
                        # 在当前行内查找元素
                        elements = row.select(selector)
                        if elements:
                            element = elements[0]
                            if extract_type == "text":
                                item[field_name] = element.text.strip()
                            elif extract_type == "attribute":
                                item[field_name] = element.get_attribute(attribute) if hasattr(element, 'get_attribute') else element.get(attribute, "")
                    except Exception as e:
                        logger.warning(f"提取字段 {field_name} 失败: {e}")
                        item[field_name] = ""
                
                # 只有当至少有一个字段有值时才添加到结果中
                if any(value for value in item.values()):
                    results.append(item)
            
            logger.info(f"成功解析 {len(results)} 条列表数据")
            return results
            
        except Exception as e:
            logger.error(f"解析列表数据失败: {e}")
            return []
    
    @staticmethod
    def list_data_to_dataframe(list_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """将列表数据转换为DataFrame
        
        Args:
            list_data: 列表数据，每个元素是一个字典
            
        Returns:
            DataFrame
        """
        try:
            if not list_data:
                logger.warning("列表数据为空，无法转换为DataFrame")
                return pd.DataFrame()
            
            # 获取所有字段名
            all_fields = set()
            for item in list_data:
                all_fields.update(item.keys())
            
            # 创建DataFrame
            df = pd.DataFrame(list_data, columns=sorted(all_fields))
            logger.info(f"成功将列表数据转换为DataFrame，数据形状: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"将列表数据转换为DataFrame失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def parse_xpath_list(html_content: str, xpath: str, extract_type: str = "text", attribute: str = "href") -> List[str]:
        """使用XPath从HTML中提取列表数据
        
        Args:
            html_content: HTML内容
            xpath: XPath表达式
            extract_type: 提取类型，"text"或"attribute"
            attribute: 如果extract_type是"attribute"，则指定要提取的属性名
            
        Returns:
            提取的数据列表
        """
        try:
            from lxml import etree
            
            tree = etree.HTML(html_content)
            elements = tree.xpath(xpath)
            
            results = []
            for element in elements:
                if isinstance(element, str):
                    # 直接返回的文本
                    results.append(element.strip())
                elif extract_type == "text":
                    # 元素的文本内容
                    results.append(element.xpath("string()").strip())
                elif extract_type == "attribute":
                    # 元素的属性
                    attr_value = element.get(attribute, "")
                    results.append(attr_value.strip())
            
            logger.info(f"成功使用XPath提取 {len(results)} 个元素")
            return results
            
        except Exception as e:
            logger.error(f"使用XPath提取数据失败: {e}")
            return []
