# 清理计划

## 核心功能
1. 表格提取功能
2. 函数调用功能
3. MultiColumnProcessor核心功能（用户特别要求保留）

## 需要保留的文件

### 表格提取功能
- `src/modules/table_processor.py` - 表格处理器，负责将HTML转换为Excel
- `src/modules/qwen_vl_manager.py` - Qwen VL管理器，负责将图片转换为HTML表格
- `src/modules/image_tools.py` - 图像处理工具
- `yolo11n.pt` - YOLO模型文件（根目录）
- `runs/a4_table_lora_finetune2/weights/best.pt` - 训练好的YOLO模型

### 函数调用功能
- `src/modules/multi_column_processor.py` - AI函数生成与调用的核心入口
- `src/modules/ai_service.py` - AI服务类，提供统一的AI调用接口
- `src/modules/api_manager.py` - API管理器，支持多种API类型
- `src/modules/prompt_generator.py` - 提示词生成器
- `src/modules/column_matcher.py` - 列匹配器
- `src/modules/vectorized_function_converter.py` - 向量化函数转换器

### 通用工具
- `src/utils/config.py` - 配置管理器
- `src/utils/logger.py` - 日志管理器
- `src/utils/qwen_db.py` - Qwen数据库
- `src/utils/qwen_db_sqlite.py` - SQLite数据库实现
- `src/modules/file_manager.py` - 文件管理器
- `src/modules/data_parser.py` - 数据解析器

## 需要删除的文件

### UI相关
- `src/main.py` - 主程序入口（UI相关）
- `src/ui/` - UI目录

### PPT相关
- `src/modules/ppt_tools.py` - PPT工具
- `src/modules/ppt_merge_engine.py` - PPT合并引擎
- `src/modules/slide_feature_extractor.py` - 幻灯片特征提取器

### 其他工具
- `src/modules/meeting_tools.py` - 会议工具
- `src/modules/password_tools.py` - 密码工具
- `src/modules/translation_tools.py` - 翻译工具
- `src/modules/speech_tools.py` - 语音工具
- `src/modules/excel_tools.py` - Excel工具
- `src/modules/form_tools.py` - 表单工具
- `src/modules/function_template_library.py` - 函数模板库
- `src/modules/report_generator.py` - 报告生成器
- `src/modules/run_batch.py` - 批量运行工具
- `src/modules/scheduler.py` - 调度器
- `src/modules/screenshot.py` - 截图工具
- `src/modules/ollama_api_manager.py` - Ollama API管理器
- `src/modules/qwen_learning.py` - Qwen学习模块
- `src/modules/qwen_analytics.py` - Qwen分析模块
- `src/modules/cross_table_processor.py` - 跨表格处理器
- `src/modules/data_manager.py` - 数据管理器
- `src/modules/playwright_engine.py` - Playwright引擎

### 测试和缓存文件
- `src/__pycache__/` - Python缓存目录
- `src/modules/__pycache__/` - 模块缓存目录
- `src/utils/__pycache__/` - 工具缓存目录
- `.mypy_cache/` - Mypy缓存目录
- `src/cache/` - 缓存目录

## 清理步骤
1. 删除UI相关文件
2. 删除PPT相关文件
3. 删除其他无关工具文件
4. 删除测试和缓存文件
5. 验证清理结果

## 验证计划
1. 运行表格提取测试脚本
2. 运行函数调用测试脚本
3. 验证核心功能仍能正常工作