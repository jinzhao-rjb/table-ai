#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量函数转换层
将AI生成的函数转换为pandas向量化操作，提高处理速度
"""

import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

class VectorizedFunctionConverter:
    """向量函数转换类
    将AI生成的单元格函数转换为pandas向量化操作
    """
    
    def __init__(self):
        """初始化向量函数转换层
        """
        # 函数转换规则映射
        self.function_mappings = {
            # 文本处理函数
            'LOWER': self._convert_lower,
            'UPPER': self._convert_upper,
            'TRIM': self._convert_trim,
            'SUBSTITUTE': self._convert_substitute,
            'LEN': self._convert_len,
            'LEFT': self._convert_left,
            'RIGHT': self._convert_right,
            'MID': self._convert_mid,
            'FIND': self._convert_find,
            
            # 数值处理函数
            'ROUND': self._convert_round,
            'INT': self._convert_int,
            'ABS': self._convert_abs,
            
            # 日期处理函数
            'TEXT': self._convert_text,
            
            # 条件处理函数
            'IF': self._convert_if,
            
            # 其他函数
            'CONCATENATE': self._convert_concatenate,
            '&': self._convert_concat_operator
        }
        
        logger.info("向量函数转换层已初始化")
    
    def convert_function(self, function_text: str, columns: list) -> Callable:
        """将AI生成的函数转换为向量化函数
        
        Args:
            function_text: AI生成的函数文本（如=LOWER(A2)）
            columns: 数据框的列名列表
            
        Returns:
            向量化函数，接收数据框，返回处理结果
        """
        logger.debug(f"转换函数: {function_text}")
        
        # 移除可能的等号
        if function_text.startswith('='):
            function_text = function_text[1:]
        
        # 解析函数，提取主函数名
        main_function_match = re.match(r'([A-Za-z_]+)\(', function_text)
        if main_function_match:
            main_function = main_function_match.group(1).upper()
            
            if main_function in self.function_mappings:
                # 使用映射的转换函数
                convert_func = self.function_mappings[main_function]
                return convert_func(function_text, columns)
            else:
                # 无法识别的函数，使用通用转换
                logger.warning(f"无法识别的函数: {main_function}")
                return self._convert_generic(function_text, columns)
        else:
            # 简单表达式（如A2*2）
            return self._convert_expression(function_text, columns)
    
    def _replace_cell_references(self, expression: str, columns: list) -> str:
        """替换表达式中的单元格引用
        
        Args:
            expression: 包含单元格引用的表达式（如A2、B3:C5）
            columns: 数据框的列名列表
            
        Returns:
            替换后的表达式
        """
        # 匹配单元格引用（如A2、B3:C5）
        cell_ref_pattern = r'([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?'
        
        def replace_match(match):
            col_letter = match.group(1)
            row_num = match.group(2)
            end_col = match.group(3)
            end_row = match.group(4)
            
            if end_col and end_row:
                # 范围引用（如A2:B5），返回列名
                start_col_idx = ord(col_letter.upper()) - ord('A')
                end_col_idx = ord(end_col.upper()) - ord('A')
                
                if start_col_idx < len(columns) and end_col_idx < len(columns):
                    start_col = columns[start_col_idx]
                    end_col = columns[end_col_idx]
                    return f"df['{start_col}']"  # 简化处理，只返回起始列
            else:
                # 单个单元格引用（如A2），返回列名
                col_idx = ord(col_letter.upper()) - ord('A')
                if col_idx < len(columns):
                    return f"df['{columns[col_idx]}']"
            
            # 无法解析的引用，返回原字符串
            return match.group(0)
        
        # 替换所有单元格引用
        replaced_expression = re.sub(cell_ref_pattern, replace_match, expression)
        
        logger.debug(f"替换引用前: {expression}, 替换后: {replaced_expression}")
        
        return replaced_expression
    
    def _convert_generic(self, function_text: str, columns: list) -> Callable:
        """通用转换方法
        
        Args:
            function_text: AI生成的函数文本
            columns: 数据框的列名列表
            
        Returns:
            向量化函数
        """
        # 替换单元格引用
        vectorized_expr = self._replace_cell_references(function_text, columns)
        
        def vectorized_func(df):
            # 使用eval执行向量化表达式
            return eval(vectorized_expr, {}, {'df': df, 'np': np, 'pd': pd})
        
        return vectorized_func
    
    def _convert_expression(self, expression: str, columns: list) -> Callable:
        """转换简单表达式
        
        Args:
            expression: 简单表达式（如A2*2、B3+C3）
            columns: 数据框的列名列表
            
        Returns:
            向量化函数
        """
        # 替换单元格引用
        vectorized_expr = self._replace_cell_references(expression, columns)
        
        def vectorized_func(df):
            return eval(vectorized_expr, {}, {'df': df, 'np': np, 'pd': pd})
        
        return vectorized_func
    
    # 文本处理函数转换
    def _convert_lower(self, function_text: str, columns: list) -> Callable:
        """转换LOWER函数
        
        如: =LOWER(A2) -> df['A'].str.lower()
        """
        # 提取参数
        arg_match = re.search(r'LOWER\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(1)
            vectorized_arg = self._replace_cell_references(arg, columns)
            
            def vectorized_func(df):
                return eval(f"{vectorized_arg}.str.lower()", {}, {'df': df, 'np': np, 'pd': pd})
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_upper(self, function_text: str, columns: list) -> Callable:
        """转换UPPER函数
        
        如: =UPPER(A2) -> df['A'].str.upper()
        """
        arg_match = re.search(r'UPPER\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(1)
            vectorized_arg = self._replace_cell_references(arg, columns)
            
            def vectorized_func(df):
                return eval(f"{vectorized_arg}.str.upper()", {}, {'df': df, 'np': np, 'pd': pd})
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_trim(self, function_text: str, columns: list) -> Callable:
        """转换TRIM函数
        
        如: =TRIM(A2) -> df['A'].str.strip()
        """
        arg_match = re.search(r'TRIM\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(1)
            vectorized_arg = self._replace_cell_references(arg, columns)
            
            def vectorized_func(df):
                return eval(f"{vectorized_arg}.str.strip()", {}, {'df': df, 'np': np, 'pd': pd})
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_substitute(self, function_text: str, columns: list) -> Callable:
        """转换SUBSTITUTE函数
        
        如: =SUBSTITUTE(B2, " ", "") -> df['B'].str.replace(" ", "")
        """
        arg_match = re.search(r'SUBSTITUTE\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 3:
                text_arg = args[0].strip()
                old_text = args[1].strip().strip('"\'')
                new_text = args[2].strip().strip('"\'')
                
                vectorized_text = self._replace_cell_references(text_arg, columns)
                
                def vectorized_func(df):
                    return eval(f"{vectorized_text}.str.replace('{old_text}', '{new_text}')", {}, {'df': df, 'np': np, 'pd': pd})
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_len(self, function_text: str, columns: list) -> Callable:
        """转换LEN函数
        
        如: =LEN(A2) -> df['A'].str.len()
        """
        arg_match = re.search(r'LEN\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(1)
            vectorized_arg = self._replace_cell_references(arg, columns)
            
            def vectorized_func(df):
                return eval(f"{vectorized_arg}.str.len()", {}, {'df': df, 'np': np, 'pd': pd})
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    # 数值处理函数转换
    def _convert_round(self, function_text: str, columns: list) -> Callable:
        """转换ROUND函数
        
        如: =ROUND(A2, 2) -> df['A'].round(2)
        """
        arg_match = re.search(r'ROUND\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 2:
                num_arg = args[0].strip()
                decimals = args[1].strip()
                
                vectorized_num = self._replace_cell_references(num_arg, columns)
                
                def vectorized_func(df):
                    return eval(f"{vectorized_num}.round({decimals})")
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_int(self, function_text: str, columns: list) -> Callable:
        """转换INT函数
        
        如: =INT(A2) -> df['A'].astype(int)
        """
        arg_match = re.search(r'INT\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(1)
            vectorized_arg = self._replace_cell_references(arg, columns)
            
            def vectorized_func(df):
                return eval(f"{vectorized_arg}.astype(int)", {}, {'df': df, 'np': np, 'pd': pd})
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_abs(self, function_text: str, columns: list) -> Callable:
        """转换ABS函数
        
        如: =ABS(A2) -> df['A'].abs()
        """
        arg_match = re.search(r'ABS\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            arg = arg_match.group(1)
            vectorized_arg = self._replace_cell_references(arg, columns)
            
            def vectorized_func(df):
                return eval(f"{vectorized_arg}.abs()", {}, {'df': df, 'np': np, 'pd': pd})
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    # 日期处理函数转换
    def _convert_text(self, function_text: str, columns: list) -> Callable:
        """转换TEXT函数
        
        如: =TEXT(C2, "mm/dd/yyyy") -> df['C'].dt.strftime("%m/%d/%Y")
        """
        arg_match = re.search(r'TEXT\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 2:
                date_arg = args[0].strip()
                format_str = args[1].strip().strip('"\'')
                
                # 转换Excel格式到Python格式
                python_format = self._excel_date_format_to_python(format_str)
                vectorized_date = self._replace_cell_references(date_arg, columns)
                
                def vectorized_func(df):
                    import re
                    # 使用正则表达式提取列名，支持df['A']或df.A格式
                    col_match = re.search(r"['\"](\w+)['\"]|\.([a-zA-Z_]\w*)", vectorized_date)
                    if col_match:
                        col_name = col_match.group(1) or col_match.group(2)
                        
                        # 检查列是否为数值类型（Excel中时间以小数存储）
                        if pd.api.types.is_numeric_dtype(df[col_name]):
                            # 将数值转换为时间：Excel中1代表24小时，所以乘以24*3600得到秒数
                            return pd.to_datetime(df[col_name] * 24 * 3600, unit='s', origin='1899-12-30').dt.strftime(python_format)
                        else:
                            # 已经是datetime类型，直接使用strftime
                            return df[col_name].dt.strftime(python_format)
                    else:
                        # 如果无法提取列名，使用通用转换
                        return eval(f"{vectorized_date}.dt.strftime('{python_format}')", {}, {'df': df, 'np': np, 'pd': pd})
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _excel_date_format_to_python(self, excel_format: str) -> str:
        """将Excel日期格式转换为Python日期格式
        
        Args:
            excel_format: Excel日期格式字符串
            
        Returns:
            Python日期格式字符串
        """
        # 简单直接的格式映射，只处理明确的时间格式
        # 注意：Excel中使用mm表示分钟，MM表示月份
        # 我们只处理明确的时间格式，避免与日期格式冲突
        python_format = excel_format
        
        # 直接替换常见的时间格式
        if 'hh:mm:ss' in excel_format:
            python_format = python_format.replace('hh:mm:ss', '%H:%M:%S')
        elif 'h:mm:ss' in excel_format:
            python_format = python_format.replace('h:mm:ss', '%H:%M:%S')
        elif 'hh:mm' in excel_format:
            python_format = python_format.replace('hh:mm', '%H:%M')
        elif 'h:mm' in excel_format:
            python_format = python_format.replace('h:mm', '%H:%M')
        
        # 单独替换时间组件
        # 注意：只替换时间相关的格式，不处理日期格式
        python_format = python_format.replace('hh', '%H')  # 小时，带前导零
        python_format = python_format.replace('h', '%H')   # 小时，不带前导零
        python_format = python_format.replace('mm', '%M')  # 分钟，带前导零
        python_format = python_format.replace('m', '%M')   # 分钟，不带前导零
        python_format = python_format.replace('ss', '%S')  # 秒，带前导零
        python_format = python_format.replace('s', '%S')   # 秒，不带前导零
        
        return python_format
    
    # 条件处理函数转换
    def _convert_if(self, function_text: str, columns: list) -> Callable:
        """转换IF函数
        
        如: =IF(E2="不适用", "", E2) -> df['E'].where(df['E'] != "不适用", "")
        """
        arg_match = re.search(r'IF\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 3:
                condition = args[0].strip()
                true_value = args[1].strip()
                false_value = args[2].strip()
                
                # 替换单元格引用
                vectorized_condition = self._replace_cell_references(condition, columns)
                
                # 处理字符串值，去除引号
                true_value = true_value.strip('"\'')
                false_value = false_value.strip('"\'')
                
                def vectorized_func(df):
                    # 计算条件结果
                    cond = df.eval(vectorized_condition)
                    # 使用numpy的where函数实现条件逻辑
                    return np.where(cond, true_value, false_value)
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    # 其他函数转换
    def _convert_concatenate(self, function_text: str, columns: list) -> Callable:
        """转换CONCATENATE函数
        
        如: =CONCATENATE(A2, "_", B2) -> df['A'] + "_" + df['B']
        """
        arg_match = re.search(r'CONCATENATE\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            vectorized_args = [self._replace_cell_references(arg.strip(), columns) for arg in args]
            
            def vectorized_func(df):
                result = None
                for arg in vectorized_args:
                    if result is None:
                        result = eval(arg, {}, {'df': df, 'np': np, 'pd': pd})
                    else:
                        result += eval(arg, {}, {'df': df, 'np': np, 'pd': pd})
                return result
            
            return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_concat_operator(self, function_text: str, columns: list) -> Callable:
        """转换连接运算符
        
        如: =A2&B2 -> df['A'] + df['B']
        """
        # 替换单元格引用
        vectorized_expr = self._replace_cell_references(function_text, columns)
        # 将&替换为+
        vectorized_expr = vectorized_expr.replace('&', '+')
        
        def vectorized_func(df):
            return eval(vectorized_expr, {}, {'df': df, 'np': np, 'pd': pd})
        
        return vectorized_func
    
    def _convert_left(self, function_text: str, columns: list) -> Callable:
        """转换LEFT函数
        
        如: =LEFT(A2,5) -> df['A'].str[:5]
        如: =LEFT(A2,FIND("@",A2)-1) -> df['A'].str.split('@').str[0]
        """
        arg_match = re.search(r'LEFT\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 2:
                text_arg = args[0].strip()
                num_chars_arg = args[1].strip()
                
                vectorized_text = self._replace_cell_references(text_arg, columns)
                
                # 特殊处理: 如果是FIND函数的情况，直接使用split方法
                if 'FIND(' in num_chars_arg and '@' in num_chars_arg:
                    # 处理 =LEFT(A2,FIND("@",A2)-1) -> 使用split方法更简单
                    def vectorized_func(df):
                        return eval(f"{vectorized_text}.str.split('@').str[0]", {}, {'df': df, 'np': np, 'pd': pd})
                    return vectorized_func
                else:
                    # 普通情况: =LEFT(A2,5)
                    num_chars = num_chars_arg
                    def vectorized_func(df):
                        return eval(f"{vectorized_text}.str[:{num_chars}]", {}, {'df': df, 'np': np, 'pd': pd})
                    return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_right(self, function_text: str, columns: list) -> Callable:
        """转换RIGHT函数
        
        如: =RIGHT(A2,5) -> df['A'].str[-5:]
        """
        arg_match = re.search(r'RIGHT\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 2:
                text_arg = args[0].strip()
                num_chars = args[1].strip()
                
                vectorized_text = self._replace_cell_references(text_arg, columns)
                
                def vectorized_func(df):
                    return eval(f"{vectorized_text}.str[-{num_chars}:]", {}, {'df': df, 'np': np, 'pd': pd})
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_mid(self, function_text: str, columns: list) -> Callable:
        """转换MID函数
        
        如: =MID(A2,2,3) -> df['A'].str[1:4]
        """
        arg_match = re.search(r'MID\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 3:
                text_arg = args[0].strip()
                start_num = str(int(args[1].strip()) - 1)  # 转换为Python的0-based索引
                num_chars = args[2].strip()
                
                vectorized_text = self._replace_cell_references(text_arg, columns)
                end_num = f"int({start_num})+int({num_chars})"
                
                def vectorized_func(df):
                    return eval(f"{vectorized_text}.str[{start_num}:{end_num}]", {}, {'df': df, 'np': np, 'pd': pd})
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_find(self, function_text: str, columns: list) -> Callable:
        """转换FIND函数
        
        如: =FIND("@",A2) -> df['A'].str.find("@") + 1
        """
        arg_match = re.search(r'FIND\(([^)]+)\)', function_text, re.IGNORECASE)
        if arg_match:
            args = arg_match.group(1).split(',')
            if len(args) >= 2:
                find_text = args[0].strip().strip('"\'')
                text_arg = args[1].strip()
                
                vectorized_text = self._replace_cell_references(text_arg, columns)
                
                def vectorized_func(df):
                    # Python的str.find返回索引，Excel的FIND返回位置（索引+1）
                    return eval(f"{vectorized_text}.str.find('{find_text}') + 1", {}, {'df': df, 'np': np, 'pd': pd})
                
                return vectorized_func
        return self._convert_generic(function_text, columns)
    
    def _convert_generic(self, function_text: str, columns: list) -> Callable:
        """通用转换函数
        
        对于无法识别的函数，尝试直接使用eval进行向量化处理
        
        Args:
            function_text: 函数文本
            columns: 列名列表
            
        Returns:
            向量化函数
        """
        # 替换单元格引用
        vectorized_expr = self._replace_cell_references(function_text, columns)
        
        def vectorized_func(df):
            return eval(vectorized_expr, {}, {'df': df, 'np': np, 'pd': pd})
        
        return vectorized_func
    
    def _convert_expression(self, function_text: str, columns: list) -> Callable:
        """转换简单表达式
        
        如: =A2*2+1 -> df['A']*2+1
        
        Args:
            function_text: 表达式文本
            columns: 列名列表
            
        Returns:
            向量化函数
        """
        # 替换单元格引用
        vectorized_expr = self._replace_cell_references(function_text, columns)
        
        def vectorized_func(df):
            return eval(vectorized_expr, {}, {'df': df, 'np': np, 'pd': pd})
        
        return vectorized_func


# 创建全局转换实例
vectorized_converter = VectorizedFunctionConverter()
