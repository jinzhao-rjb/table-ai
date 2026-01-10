"""格式匹配器

该文件实现了格式匹配功能，负责：
1. 分析用户输入的格式需求
2. 匹配最佳的可用格式选项
3. 处理格式冲突
4. 返回结构化的格式参数
"""

import logging
from typing import Dict, Any, Optional
from modules.format_options import (
    SUPPORTED_FONTS,
    SUPPORTED_FONT_SIZES,
    SUPPORTED_ALIGNMENTS,
    SUPPORTED_BORDER_STYLES
)

logger = logging.getLogger(__name__)

# 常用行间距值
COMMON_LINE_SPACING = {
    "single": 1.0,
    "one_half": 1.5,
    "double": 2.0
}

# 常用缩进值（英寸）
COMMON_INDENTS = {
    "small": 0.25,
    "normal": 0.5,
    "large": 0.75
}


class FormatMatcher:
    """格式匹配器类，负责分析和匹配格式需求"""

    @staticmethod
    def match_font(font_name: str) -> str:
        """匹配最佳字体

        Args:
            font_name: 期望的字体名称

        Returns:
            匹配到的字体名称

        Raises:
            ValueError: 如果没有匹配到字体"""
        if not font_name:
            raise ValueError("字体名称不能为空")

        # 精确匹配支持的字体列表
        if font_name in SUPPORTED_FONTS:
            return font_name

        # 转换为小写进行匹配
        font_name_lower = font_name.lower()

        # 模糊匹配支持的字体列表
        for supported_font in SUPPORTED_FONTS:
            if font_name_lower in supported_font.lower():
                return supported_font

        # 没有匹配到字体，抛出异常
        raise ValueError(
            f"无法匹配字体: {font_name}，请从支持的字体列表中选择: {SUPPORTED_FONTS}")

    @staticmethod
    def match_font_size(font_size: Any) -> float:
        """匹配最佳字号

        Args:
            font_size: 期望的字号，可以是数字、字符串

        Returns:
            匹配到的字号（磅值）

        Raises:
            ValueError: 如果没有匹配到字号"""
        if not font_size:
            raise ValueError("字号不能为空")

        # 尝试转换为数字
        try:
            if isinstance(font_size, str):
                # 处理带单位的字号字符串
                if "pt" in font_size.lower():
                    font_size = float(
                        font_size.lower().replace("pt", "").strip())
                elif "号" in font_size:
                    # 处理中文字号
                    size_map = {
                        "八号": 8,
                        "七号": 9,
                        "小五号": 10.5,
                        "五号": 12,
                        "四号": 14,
                        "三号": 16,
                        "二号": 18,
                        "小一号": 22,
                        "一号": 24,
                        "小初号": 26,
                        "初号": 36
                    }
                    result = size_map.get(font_size)
                    if result is None:
                        raise ValueError(f"无法匹配字号: {font_size}")
                    # 检查转换后的字号是否在支持的字号列表中
                    if result in SUPPORTED_FONT_SIZES:
                        return result
                    raise ValueError(f"转换后的字号不在支持列表中: {result}")
                else:
                    font_size = float(font_size)
            else:
                font_size = float(font_size)
        except (ValueError, TypeError):
            raise ValueError(f"无法匹配字号: {font_size}")

        # 精确匹配支持的字号列表
        if font_size in SUPPORTED_FONT_SIZES:
            return font_size

        # 没有匹配到字号，抛出异常
        raise ValueError(
            f"无法匹配字号: {font_size}，请从支持的字号列表中选择: {SUPPORTED_FONT_SIZES}")

    @staticmethod
    def match_alignment(alignment: str) -> str:
        """匹配最佳对齐方式

        Args:
            alignment: 期望的对齐方式

        Returns:
            匹配到的对齐方式，如：left, center, right, justify

        Raises:
            ValueError: 如果没有匹配到对齐方式
        """
        if not alignment:
            raise ValueError("对齐方式不能为空")

        # 转换为小写进行匹配
        alignment_lower = alignment.lower()

        # 精确匹配支持的对齐方式列表
        if alignment_lower in SUPPORTED_ALIGNMENTS:
            return alignment_lower

        # 检查SUPPORTED_ALIGNMENTS的值，因为它是一个字典
        for align_key, align_value in SUPPORTED_ALIGNMENTS.items():
            if alignment_lower == align_key:
                return align_key

        # 没有匹配到对齐方式，抛出异常
        raise ValueError(
            f"无法匹配对齐方式: {alignment}，请从支持的对齐方式列表中选择: {list(SUPPORTED_ALIGNMENTS.keys())}")

    @staticmethod
    def match_line_spacing(line_spacing: Any) -> float:
        """匹配最佳行间距

        Args:
            line_spacing: 期望的行间距，可以是数字、字符串

        Returns:
            匹配到的行间距值

        Raises:
            ValueError: 如果没有匹配到行间距
        """
        if not line_spacing:
            raise ValueError("行间距不能为空")

        # 尝试转换为数字或匹配常用行间距
        try:
            if isinstance(line_spacing, str):
                # 处理中文行间距描述
                if "单倍" in line_spacing or "1倍" in line_spacing:
                    return COMMON_LINE_SPACING["single"]
                elif "1.5倍" in line_spacing or "一倍半" in line_spacing:
                    return COMMON_LINE_SPACING["one_half"]
                elif "双倍" in line_spacing or "2倍" in line_spacing:
                    return COMMON_LINE_SPACING["double"]
                else:
                    line_spacing = float(line_spacing)
            else:
                line_spacing = float(line_spacing)
        except (ValueError, TypeError):
            raise ValueError(f"无法匹配行间距: {line_spacing}")

        # 精确匹配常用行间距值
        for spacing_value in COMMON_LINE_SPACING.values():
            if line_spacing == spacing_value:
                return line_spacing

        # 没有匹配到行间距，抛出异常
        raise ValueError(
            f"无法匹配行间距: {line_spacing}，请从支持的行间距列表中选择: {list(COMMON_LINE_SPACING.values())}")

    @staticmethod
    def match_indent(indent: Any) -> float:
        """匹配最佳缩进值

        Args:
            indent: 期望的缩进值，可以是数字、字符串

        Returns:
            匹配到的缩进值（英寸）

        Raises:
            ValueError: 如果缩进值无效或无法匹配
        """
        if indent is None:
            raise ValueError("缩进值不能为空")

        # 尝试转换为数字或匹配常用缩进值
        try:
            if isinstance(indent, str):
                # 处理中文缩进描述
                if "无" in indent or "0" in indent:
                    return 0
                elif "小" in indent:
                    return COMMON_INDENTS["small"]
                elif "正常" in indent or "标准" in indent:
                    return COMMON_INDENTS["normal"]
                elif "大" in indent:
                    return COMMON_INDENTS["large"]
                else:
                    indent = float(indent)
            else:
                indent = float(indent)
        except (ValueError, TypeError):
            raise ValueError(f"无法匹配缩进值: {indent}")

        # 精确匹配常用缩进值
        for indent_value in COMMON_INDENTS.values():
            if indent == indent_value:
                return indent

        # 没有匹配到缩进值，抛出异常
        raise ValueError(
            f"无法匹配缩进值: {indent}，请从支持的缩进值列表中选择: {list(COMMON_INDENTS.values())}")

    @staticmethod
    def merge_format_params(base_params: Dict[str, Any], new_params: Dict[str, Any]) -> Dict[str, Any]:
        """合并格式参数，处理格式冲突

        Args:
            base_params: 基础格式参数
            new_params: 新的格式参数

        Returns:
            合并后的格式参数
        """
        if not new_params:
            return base_params

        # 中文键名到英文键名的映射
        CHINESE_TO_ENGLISH_KEYS = {
            "字体名称": "font_name",
            "字体大小": "font_size",
            "加粗": "bold",
            "斜体": "italic",
            "下划线": "underline",
            "字体颜色": "color",
            "高亮颜色": "highlight_color",
            "首行缩进": "first_line_indent",
            "悬挂缩进": "hanging_indent",
            "左缩进": "left_indent",
            "右缩进": "right_indent",
            "段前间距": "space_before",
            "段后间距": "space_after",
            "行间距": "line_spacing",
            "对齐方式": "alignment",
            "段落保持在一起": "keep_together",
            "与下一段保持在一起": "keep_with_next",
            "左对齐": "left",
            "居中对齐": "center",
            "右对齐": "right",
            "两端对齐": "justify"
        }

        # 将中文键名转换为英文键名
        converted_params = {}
        for key, value in new_params.items():
            # 如果是中文键名，转换为英文键名
            if key in CHINESE_TO_ENGLISH_KEYS:
                converted_key = CHINESE_TO_ENGLISH_KEYS[key]
                # 如果值是中文对齐方式，也转换为英文
                if converted_key == "alignment" and value in CHINESE_TO_ENGLISH_KEYS:
                    converted_params[converted_key] = CHINESE_TO_ENGLISH_KEYS[value]
                else:
                    converted_params[converted_key] = value
            # 否则保留原键名
            else:
                converted_params[key] = value

        merged = base_params.copy()

        # 合并参数，新参数优先级更高
        for key, value in converted_params.items():
            if value is not None:
                # 递归合并嵌套字典
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = FormatMatcher.merge_nested_dicts(
                        merged[key], value)
                else:
                    merged[key] = value

        return merged

    @staticmethod
    def merge_nested_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并嵌套字典

        Args:
            base: 基础字典
            update: 要合并的新字典

        Returns:
            合并后的字典
        """
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = FormatMatcher.merge_nested_dicts(
                    result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def analyze_format_requirement(description: str) -> Dict[str, Any]:
        """分析格式需求描述

        Args:
            description: 格式需求描述

        Returns:
            结构化的格式参数字典

        Raises:
            ValueError: 如果无法从描述中提取有效的格式参数
        """
        logger.info(f"分析格式需求：{description}")

        # 直接从用户描述中提取格式需求，不使用默认值
        format_params = {}

        # 处理常见格式需求
        description_lower = description.lower()

        # 处理字体需求
        for font in SUPPORTED_FONTS:
            if font in description_lower:
                format_params["font_name"] = font
                break

        # 处理字号需求
        for size in SUPPORTED_FONT_SIZES:
            size_str = str(size)
            if size_str in description_lower:
                format_params["font_size"] = size
                break

        # 处理对齐方式需求
        for align_key in SUPPORTED_ALIGNMENTS.keys():
            if align_key in description_lower:
                format_params["alignment"] = align_key
                break

        # 处理行间距需求
        for spacing_key, spacing_value in COMMON_LINE_SPACING.items():
            if spacing_key in description_lower:
                format_params["line_spacing"] = spacing_value
                break

        logger.info(f"分析结果：{format_params}")

        # 如果没有提取到任何有效的格式参数，抛出异常
        if not format_params:
            raise ValueError(f"无法从描述中提取有效的格式参数：{description}")

        return format_params

    @staticmethod
    def validate_format_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """验证和规范化格式参数

        Args:
            params: 原始格式参数字典

        Returns:
            验证和规范化后的格式参数字典

        Raises:
            ValueError: 如果格式参数为空或无效
        """
        if not params:
            raise ValueError("格式参数不能为空")

        # 中文键名到英文键名的映射
        CHINESE_TO_ENGLISH_KEYS = {
            "字体名称": "font_name",
            "字体大小": "font_size",
            "加粗": "bold",
            "斜体": "italic",
            "下划线": "underline",
            "字体颜色": "color",
            "高亮颜色": "highlight_color",
            "首行缩进": "first_line_indent",
            "悬挂缩进": "hanging_indent",
            "左缩进": "left_indent",
            "右缩进": "right_indent",
            "段前间距": "space_before",
            "段后间距": "space_after",
            "行间距": "line_spacing",
            "对齐方式": "alignment",
            "段落保持在一起": "keep_together",
            "与下一段保持在一起": "keep_with_next",
            "左对齐": "left",
            "居中对齐": "center",
            "右对齐": "right",
            "两端对齐": "justify"
        }

        # 将中文键名转换为英文键名
        converted_params = {}
        for key, value in params.items():
            # 如果是中文键名，转换为英文键名
            if key in CHINESE_TO_ENGLISH_KEYS:
                converted_key = CHINESE_TO_ENGLISH_KEYS[key]
                # 如果值是中文对齐方式，也转换为英文
                if converted_key == "alignment" and value in CHINESE_TO_ENGLISH_KEYS:
                    converted_params[converted_key] = CHINESE_TO_ENGLISH_KEYS[value]
                else:
                    converted_params[converted_key] = value
            # 否则保留原键名
            else:
                converted_params[key] = value

        # 只验证和应用存在的格式参数，不使用默认值
        validated = {}

        # 验证和规范化每个存在的参数
        if "font_name" in converted_params:
            validated["font_name"] = FormatMatcher.match_font(
                converted_params["font_name"])
        if "font_size" in converted_params:
            validated["font_size"] = FormatMatcher.match_font_size(
                converted_params["font_size"])
        if "alignment" in converted_params:
            validated["alignment"] = FormatMatcher.match_alignment(
                converted_params["alignment"])
        if "line_spacing" in converted_params:
            validated["line_spacing"] = FormatMatcher.match_line_spacing(
                converted_params["line_spacing"])
        if "bold" in converted_params:
            validated["bold"] = bool(converted_params["bold"])
        if "italic" in converted_params:
            validated["italic"] = bool(converted_params["italic"])
        if "underline" in converted_params:
            validated["underline"] = bool(converted_params["underline"])
        if "first_line_indent" in converted_params:
            validated["first_line_indent"] = FormatMatcher.match_indent(
                converted_params["first_line_indent"])
        if "hanging_indent" in converted_params:
            validated["hanging_indent"] = FormatMatcher.match_indent(
                converted_params["hanging_indent"])
        if "left_indent" in converted_params:
            validated["left_indent"] = FormatMatcher.match_indent(
                converted_params["left_indent"])
        if "right_indent" in converted_params:
            validated["right_indent"] = FormatMatcher.match_indent(
                converted_params["right_indent"])
        if "space_before" in converted_params:
            validated["space_before"] = max(
                0, min(72, float(converted_params["space_before"])))
        if "space_after" in converted_params:
            validated["space_after"] = max(
                0, min(72, float(converted_params["space_after"])))
        if "keep_together" in converted_params:
            validated["keep_together"] = bool(
                converted_params["keep_together"])
        if "keep_with_next" in converted_params:
            validated["keep_with_next"] = bool(
                converted_params["keep_with_next"])

        # 如果没有验证到任何有效的格式参数，抛出异常
        if not validated:
            raise ValueError(f"无法验证任何有效的格式参数: {params}")

        return validated

    @staticmethod
    def get_format_params(description: Optional[str], ai_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """获取最终的格式参数

        Args:
            description: 格式需求描述
            ai_result: AI分析结果

        Returns:
            最终的格式参数字典

        Raises:
            ValueError: 如果没有有效的格式参数
        """
        # 如果有AI结果，优先使用AI结果
        if ai_result and ai_result.get("is_valid", False):
            logger.info("使用AI分析结果作为格式参数")

            # 验证和规范化AI结果，只处理存在的参数
            processed_result = {
                "is_valid": True
            }

            if "body_format" in ai_result:
                processed_result["body_format"] = FormatMatcher.validate_format_params(
                    ai_result["body_format"])

            if "heading_formats" in ai_result:
                processed_heading_formats = {}
                for level, heading_format in ai_result["heading_formats"].items():
                    try:
                        processed_heading_formats[level] = FormatMatcher.validate_format_params(
                            heading_format)
                    except ValueError as e:
                        logger.warning(f"跳过无效的标题格式: {level}, 错误: {e}")
                if processed_heading_formats:
                    processed_result["heading_formats"] = processed_heading_formats

            if "title_format" in ai_result:
                processed_result["title_format"] = ai_result["title_format"]

            if "subtitle_format" in ai_result:
                processed_result["subtitle_format"] = ai_result["subtitle_format"]

            return processed_result

        # 如果没有AI结果，但有用户描述，调用AI辅助分析
        if description:
            logger.info("使用AI辅助分析用户描述")

            # 调用AI服务，将自然语言需求转换为结构化格式参数
            try:
                from modules.ai_service import AIService

                # 构建AI提示词
                ai_prompt = "请将以下Word文档格式需求转换为结构化的JSON格式，只返回JSON，不要添加任何解释：\n" + description + "\n\nJSON格式要求：\n{\n  \"is_valid\": true,\n  \"body_format\": {\n    \"font_name\": \"字体名称\",\n    \"font_size\": 字号（数字）,\n    \"bold\": 布尔值,\n    \"italic\": 布尔值,\n    \"underline\": 布尔值,\n    \"alignment\": \"left/center/right/justify\",\n    \"first_line_indent\": 首行缩进（数字）,\n    \"line_spacing\": 行间距（1.0/1.5/2.0等）,\n    \"space_before\": 段前间距（数字）,\n    \"space_after\": 段后间距（数字）\n  },\n  \"heading_formats\": {\n    \"heading_1\": {\n      \"font_name\": \"字体名称\",\n      \"font_size\": 字号（数字）,\n      \"bold\": 布尔值,\n      \"italic\": 布尔值,\n      \"underline\": 布尔值,\n      \"alignment\": \"left/center/right/justify\",\n      \"first_line_indent\": 首行缩进（数字）,\n      \"line_spacing\": 行间距（1.0/1.5/2.0等）,\n      \"space_before\": 段前间距（数字）,\n      \"space_after\": 段后间距（数字）\n    },\n    \"heading_2\": {\n      \"font_name\": \"字体名称\",\n      \"font_size\": 字号（数字）,\n      \"bold\": 布尔值,\n      \"italic\": 布尔值,\n      \"underline\": 布尔值,\n      \"alignment\": \"left/center/right/justify\",\n      \"first_line_indent\": 首行缩进（数字）,\n      \"line_spacing\": 行间距（1.0/1.5/2.0等）,\n      \"space_before\": 段前间距（数字）,\n      \"space_after\": 段后间距（数字）\n    }\n  }\n}\n\n只返回JSON，不要添加任何解释或说明。"

                # 调用AI服务
                ai_response = AIService.call_doubao(ai_prompt)
                logger.info(f"AI返回的结果: {ai_response}")

                # 解析AI返回的JSON，增加容错处理
                import json
                try:
                    # 尝试提取JSON部分，处理AI可能返回的额外文本
                    import re
                    # 寻找JSON字符串的开始和结束
                    json_match = re.search(r'\{[\s\S]*\}', ai_response)
                    if json_match:
                        ai_response = json_match.group(0)
                    ai_result = json.loads(ai_response)
                    ai_result["is_valid"] = True
                except json.JSONDecodeError:
                    logger.error("AI返回的结果不是有效的JSON格式，使用默认格式参数")
                    # 如果解析失败，使用默认的有效格式参数
                    ai_result = {
                        "is_valid": True,
                        "body_format": {
                            "font_name": "微软雅黑",
                            "font_size": 12,
                            "alignment": "left",
                            "line_spacing": 1.5
                        }
                    }
                except Exception as e:
                    logger.error(f"解析AI结果失败: {e}")
                    # 使用默认格式参数
                    ai_result = {
                        "is_valid": True,
                        "body_format": {
                            "font_name": "微软雅黑",
                            "font_size": 12,
                            "alignment": "left",
                            "line_spacing": 1.5
                        }
                    }

                # 递归验证和规范化AI结果，增加容错处理
                try:
                    if "body_format" in ai_result:
                        ai_result["body_format"] = FormatMatcher.validate_format_params(
                            ai_result["body_format"])
                except Exception as e:
                    logger.error(f"验证body_format失败: {e}")
                    # 使用默认body_format
                    ai_result["body_format"] = {
                        "font_name": "微软雅黑",
                        "font_size": 12,
                        "alignment": "left",
                        "line_spacing": 1.5
                    }

                try:
                    if "heading_formats" in ai_result:
                        for level, heading_format in ai_result["heading_formats"].items():
                            try:
                                ai_result["heading_formats"][level] = FormatMatcher.validate_format_params(
                                    heading_format)
                            except Exception as e:
                                logger.warning(f"验证标题格式{level}失败，跳过: {e}")
                except Exception as e:
                    logger.error(f"处理heading_formats失败: {e}")
                    # 移除无效的heading_formats
                    if "heading_formats" in ai_result:
                        del ai_result["heading_formats"]

                logger.info(f"AI分析结果: {ai_result}")
                return ai_result

            except Exception as e:
                logger.error(f"AI辅助分析失败: {e}")
                # 如果AI调用失败，尝试直接从用户描述中提取
                logger.info("AI调用失败，尝试直接从用户描述中提取格式参数")
                try:
                    analyzed_params = FormatMatcher.analyze_format_requirement(
                        description)

                    # 验证和规范化最终参数
                    validated_params = {
                        "body_format": FormatMatcher.validate_format_params(analyzed_params),
                        "is_valid": True
                    }

                    logger.info(f"最终格式参数：{validated_params}")
                    return validated_params
                except Exception as e:
                    logger.error(f"直接提取格式参数失败: {e}")
                    # 如果直接提取也失败，使用默认格式参数
                    logger.info("直接提取格式参数失败，使用默认格式参数")
                    return {
                        "is_valid": True,
                        "body_format": {
                            "font_name": "微软雅黑",
                            "font_size": 12,
                            "alignment": "left",
                            "line_spacing": 1.5
                        }
                    }

        # 如果没有AI结果也没有用户描述，返回默认格式参数
        logger.warning("没有提供有效的格式参数，使用默认格式")
        return {
            "is_valid": True,
            "body_format": {
                "font_name": "微软雅黑",
                "font_size": 12,
                "alignment": "left",
                "line_spacing": 1.5
            }
        }
