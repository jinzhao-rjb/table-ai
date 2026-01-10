"""格式选项常量定义

该文件定义了所有支持的Word文档格式选项，包括字体、字号、对齐方式、边框样式等。
这些常量用于统一应用程序中的格式处理，确保格式的一致性和可用性。
"""

# 支持的中文字体列表
SUPPORTED_FONTS = [
    "宋体",
    "黑体",
    "微软雅黑",
    "楷体",
    "仿宋",
    "隶书",
    "幼圆",
    "等线",
    "华文宋体",
    "华文楷体",
    "华文黑体",
    "华文仿宋",
    "华文隶书",
    "华文幼圆",
    "方正书宋",
    "方正黑体",
    "方正楷体",
    "方正仿宋",
    "Times New Roman",
    "Arial",
    "Calibri",
    "Georgia",
    "Verdana",
    "Courier New"
]

# 支持的字号列表（磅值）
SUPPORTED_FONT_SIZES = [
    8,    # 八号
    9,    # 七号
    10.5,  # 小五号
    12,   # 五号
    14,   # 四号
    16,   # 三号
    18,   # 二号
    22,   # 小一号
    24,   # 一号
    26,   # 小初号
    36,   # 初号
    42,   # 大初号
    48,   # 特大号
    72    # 特特大号
]

# 字号名称映射
FONT_SIZE_NAMES = {
    8: "八号",
    9: "七号",
    10.5: "小五号",
    12: "五号",
    14: "四号",
    16: "三号",
    18: "二号",
    22: "小一号",
    24: "一号",
    26: "小初号",
    36: "初号",
    42: "大初号",
    48: "特大号",
    72: "特特大号"
}

# 支持的对齐方式
supported_alignment = {
    "left": "左对齐",
    "center": "居中对齐",
    "right": "右对齐",
    "justify": "两端对齐"
}

SUPPORTED_ALIGNMENTS = supported_alignment

# 支持的边框样式
SUPPORTED_BORDER_STYLES = [
    "single",    # 单线
    "double",    # 双线
    "dotted",    # 点线
    "dashed",    # 虚线
    "dashDot",   # 点划线
    "dashDotDot"  # 双点划线
]

# 边框样式名称映射
BORDER_STYLE_NAMES = {
    "single": "单线",
    "double": "双线",
    "dotted": "点线",
    "dashed": "虚线",
    "dashDot": "点划线",
    "dashDotDot": "双点划线"
}

# 支持的字体属性列表（英文键名，系统内部使用）
SUPPORTED_FONT_PROPERTIES_EN = [
    "font_name",     # 字体名称
    "font_size",     # 字体大小（磅值）
    "bold",          # 是否加粗
    "italic",        # 是否斜体
    "underline",     # 是否下划线
    "color",         # 字体颜色
    "highlight_color"  # 高亮颜色
]

# 字体属性中文映射
FONT_PROPERTY_NAMES = {
    "font_name": "字体名称",
    "font_size": "字体大小",
    "bold": "加粗",
    "italic": "斜体",
    "underline": "下划线",
    "color": "字体颜色",
    "highlight_color": "高亮颜色"
}

# 支持的字体属性列表（中文显示）
SUPPORTED_FONT_PROPERTIES = [
    FONT_PROPERTY_NAMES[prop] for prop in SUPPORTED_FONT_PROPERTIES_EN
]

# 支持的段落格式选项（英文键名，系统内部使用）
SUPPORTED_PARAGRAPH_FORMATS_EN = [
    "first_line_indent",  # 首行缩进
    "hanging_indent",      # 悬挂缩进
    "left_indent",         # 左缩进
    "right_indent",        # 右缩进
    "space_before",        # 段前间距
    "space_after",         # 段后间距
    "line_spacing",        # 行间距
    "alignment",           # 对齐方式
    "keep_together",       # 段落保持在一起
    "keep_with_next"       # 与下一段保持在一起
]

# 段落格式中文映射
PARAGRAPH_FORMAT_NAMES = {
    "first_line_indent": "首行缩进",
    "hanging_indent": "悬挂缩进",
    "left_indent": "左缩进",
    "right_indent": "右缩进",
    "space_before": "段前间距",
    "space_after": "段后间距",
    "line_spacing": "行间距",
    "alignment": "对齐方式",
    "keep_together": "段落保持在一起",
    "keep_with_next": "与下一段保持在一起"
}

# 支持的段落格式选项（中文显示）
SUPPORTED_PARAGRAPH_FORMATS = [
    PARAGRAPH_FORMAT_NAMES[prop] for prop in SUPPORTED_PARAGRAPH_FORMATS_EN
]

# 支持的所有格式属性列表（英文键名，系统内部使用）
SUPPORTED_ALL_FORMATS_EN = SUPPORTED_FONT_PROPERTIES_EN + \
    SUPPORTED_PARAGRAPH_FORMATS_EN

# 支持的所有格式属性列表（中文显示）
SUPPORTED_ALL_FORMATS = [
    FONT_PROPERTY_NAMES.get(
        prop, prop) if prop in SUPPORTED_FONT_PROPERTIES_EN else PARAGRAPH_FORMAT_NAMES.get(prop, prop)
    for prop in SUPPORTED_ALL_FORMATS_EN
]

# 常用段落缩进值（英寸）
COMMON_INDENTS = {
    "small": 0.25,   # 小四分之一英寸
    "normal": 0.5,   # 小半英寸（常用首行缩进）
    "large": 1.0     # 一英寸
}

# 常用行间距值
COMMON_LINE_SPACING = {
    "single": 1.0,      # 单倍行距
    "one_half": 1.5,    # 1.5倍行距
    "double": 2.0,       # 双倍行距
    "at_least": 1.0,     # 最小行距
    "exactly": 1.0       # 固定行距
}

# 支持的标题级别
SUPPORTED_HEADING_LEVELS = [
    "heading_1",
    "heading_2",
    "heading_3",
    "heading_4",
    "heading_5",
    "heading_6"
]

# 标题级别映射
HEADING_LEVEL_NAMES = {
    "heading_1": "一级标题",
    "heading_2": "二级标题",
    "heading_3": "三级标题",
    "heading_4": "四级标题",
    "heading_5": "五级标题",
    "heading_6": "六级标题"
}

# 移除了所有预设格式列表，所有格式参数将来自外部输入
# 支持的文档类型
SUPPORTED_DOCUMENT_TYPES = []
