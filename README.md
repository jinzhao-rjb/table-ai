# Table AI - 智能表格处理系统

Table AI是一个功能强大的智能表格处理系统，提供从图像提取表格数据和AI计算两大核心功能，支持FastAPI后端服务和Gradio Web界面两种使用方式。

## 📁 项目结构

项目核心代码位于 `src/` 目录，包含功能模块和工具函数：

```
table-ai/
├── src/                  # 核心源代码目录
│   ├── modules/          # 功能模块
│   │   ├── ai_service.py            # AI服务管理
│   │   ├── api_manager.py           # API请求管理
│   │   ├── data_parser.py           # 数据解析处理
│   │   ├── file_manager.py          # 文件管理与批量操作
│   │   ├── image_tools.py           # 图像处理工具
│   │   ├── multi_column_processor.py # 多列表格处理
│   │   ├── prompt_generator.py      # AI提示词生成
│   │   ├── qwen_vl_manager.py       # Qwen-VL API客户端
│   │   ├── table_processor.py       # 表格提取核心处理
│   │   ├── vectorized_function_converter.py # 向量化函数转换（AI计算）
│   │   ├── voice_service.py         # 语音识别服务
│   ├── utils/            # 工具函数
│   │   ├── ast_security_checker.py  # AST代码安全检查
│   │   ├── async_task_manager.py    # 异步任务管理
│   │   ├── config.py                # 配置管理
│   │   ├── dual_redis_db.py         # Redis数据库管理
│   │   ├── format_matcher.py        # 格式匹配
│   │   ├── format_options.py        # 格式选项
│   │   ├── logger.py                # 日志管理
│   │   ├── qwen_db.py               # Qwen数据库
│   │   ├── qwen_db_sqlite.py        # SQLite数据库
│   ├── check_redis_status.py        # Redis状态检查
│   ├── enqueue_tasks.py             # 任务入队
│   ├── redis_integrated_main.py     # Redis集成主程序
│   ├── run_full_html_pipeline.py    # HTML处理流水线
│   ├── table.yaml                   # 表格配置
├── main.py               # FastAPI后端服务入口
├── gradio_app.py         # Gradio Web界面入口
├── config.json           # 配置文件
├── requirements.txt      # 依赖列表
└── README.md             # 项目说明
```

## 🔗 两个主要入口

### 1. FastAPI后端服务 (`main.py`)
- **功能**：提供RESTful API接口，支持文件上传、任务管理和状态查询
- **用途**：适合集成到其他系统、自动化脚本或作为微服务使用
- **主要端点**：
  - `POST /api/upload`: 批量上传文件并添加到任务队列
  - `GET /api/task/{task_id}`: 获取特定任务状态
- **技术栈**：FastAPI + Uvicorn + Redis

### 2. Gradio Web界面 (`gradio_app.py`)
- **功能**：提供可视化交互界面，适合直接使用
- **用途**：无需编程，适合普通用户快速使用
- **主要功能**：
  - 图像上传与预览
  - 语音输入支持
  - 实时进度显示
  - 结果预览与下载
  - 批量处理支持
  - 自动ZIP打包结果
- **技术栈**：Gradio + Python

## 🚀 两大核心功能

### 1. 智能表格提取系统

**功能描述**：从各种图像格式中提取表格数据，支持复杂布局和多列表格

**实现模块**：`src/modules/table_processor.py`、`src/modules/qwen_vl_manager.py`、`src/modules/multi_column_processor.py`

**技术栈**：
- **目标检测**：YOLO模型（表格区域检测）
- **AI识别**：Qwen-VL API（表格内容提取）
- **图像处理**：OpenCV（图像增强、尺寸优化）
- **HTML解析**：lxml（快速表格结构解析）
- **结果生成**：openpyxl（Excel文件生成）

**处理流程**：
1. **图像预处理**：
   - 使用YOLO模型裁剪表格核心区域（去除页码和白边）
   - 图像增强：对比度增强，使边框更明显
   - 图像优化：自动调整尺寸（最大1536px），提高AI推理速度
   - 置信度评估：记录低置信度案例，用于模型优化

2. **表格提取**：
   - 调用Qwen-VL API进行表格识别
   - 生成HTML格式的表格结构
   - 支持复杂表格布局
   - 支持多列表格处理

3. **结果处理**：
   - HTML转Excel：使用lxml解析器快速处理
   - 动态行高调整
   - 边框样式设置
   - 单元格内容对齐

**核心特性**：
- 支持多种图像格式
- 处理复杂表格布局
- 支持多列表格
- 高精度识别
- 批量处理支持
- 自动ZIP打包结果

### 2. AI计算系统

**功能描述**：使用AI进行数据处理和计算，支持多种Excel函数转换和向量化操作

**实现模块**：`src/modules/vectorized_function_converter.py`、`src/modules/ai_service.py`

**技术栈**：
- **数据处理**：pandas（向量化操作优化）
- **AI服务**：Qwen API（智能计算）
- **函数转换**：自定义算法（Excel函数转pandas操作）

**支持的函数类型**：
- **文本处理**：LOWER, UPPER, TRIM, SUBSTITUTE, LEN, LEFT, RIGHT, MID, FIND
- **数值处理**：ROUND, INT, ABS
- **日期处理**：TEXT
- **条件处理**：IF
- **连接函数**：CONCATENATE, &

**处理流程**：
1. 解析AI生成的函数
2. 转换为pandas向量化操作
3. 执行优化后的操作
4. 返回处理结果

**核心特性**：
- 高性能：向量化操作，比循环快10-100倍
- 支持多种Excel函数
- 自动优化：智能选择最优处理方式
- 安全可靠：AST代码安全检查
- 易于扩展：支持自定义函数

## 🛠️ 辅助功能

### 1. 语音输入支持 (`src/modules/voice_service.py`)
- **功能**：通过语音控制应用，支持中文语音命令
- **技术**：Whisper模型(本地) + Qwen API(云端备份)
- **支持**：中文语音命令、可靠的降级机制

### 2. 高性能处理 (`src/redis_integrated_main.py`)
- **功能**：快速处理大量图像，提高系统吞吐量
- **技术**：8线程并行处理 + Redis分布式队列
- **优化**：图像大小限制、lxml解析器、API参数优化

### 3. 代码安全检查 (`src/utils/ast_security_checker.py`)
- **功能**：检测Python代码中的潜在安全问题
- **技术**：AST语法树分析
- **支持**：危险函数检测、风险等级评估

### 4. 灵活配置管理 (`src/utils/config.py`)
- **功能**：动态调整应用参数，支持多种配置方式
- **技术**：环境变量 + 配置文件
- **优先级**：环境变量 > 配置文件 > 默认值

## 🛠️ 技术栈详细说明

### 核心框架
| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| Web框架 | FastAPI | - | 后端RESTful API服务 |
| 可视化框架 | Gradio | - | Web界面交互 |
| GUI框架 | PyQt5 | >=5.15.7 | 桌面应用界面 |

### 文档处理
| 技术 | 版本 | 用途 |
|------|------|------|
| openpyxl | >=3.1.2 | Excel文件创建和修改 |
| pandas | >=1.5.3 | 数据分析和处理（AI计算核心） |
| python-docx | >=0.8.11 | Word文档处理 |
| python-pptx | >=0.6.21 | PowerPoint处理 |

### 图像处理（表格提取核心）
| 技术 | 版本 | 用途 |
|------|------|------|
| Pillow | >=10.0.1 | 图像处理基础库 |
| pytesseract | >=0.3.10 | OCR文字识别 |
| opencv-python | >=4.8.1.78 | 计算机视觉处理 |
| ultralytics | - | YOLO模型集成 |
| paddlepaddle | >=3.2.2 | 深度学习框架 |
| paddleocr | >=3.3.2 | 高精度OCR识别 |

### AI服务（两大核心功能共享）
| 技术 | 版本 | 用途 |
|------|------|------|
| Qwen-VL API | - | 表格提取AI模型 |
| Qwen API | - | 智能计算AI模型 |
| Whisper | - | 本地语音识别模型 |

### 网络和数据
| 技术 | 版本 | 用途 |
|------|------|------|
| requests | >=2.31.0 | HTTP请求处理 |
| beautifulsoup4 | >=4.12.2 | HTML解析 |
| lxml | >=4.9.3 | 高性能XML/HTML解析 |
| redis | - | 分布式队列和缓存 |

### 文本处理
| 技术 | 版本 | 用途 |
|------|------|------|
| nltk | >=3.8.1 | 自然语言处理 |
| jieba | >=0.42.1 | 中文分词 |
| markdown | >=3.4.4 | Markdown处理 |
| pyyaml | >=6.0.1 | YAML配置处理 |

### 数据可视化
| 技术 | 版本 | 用途 |
|------|------|------|
| matplotlib | >=3.7.5 | 数据可视化 |
| seaborn | >=0.12.2 | 统计数据可视化 |
| numpy | >=1.24.3 | 数值计算 |

## 📋 系统要求

- **操作系统**：Windows 10/11，Linux
- **Python版本**：Python 3.10+
- **内存**：推荐8GB及以上
- **存储空间**：500MB可用空间
- **可选**：Redis服务器（用于分布式处理）

## 🛠️ 安装与部署

### 1. 克隆项目
```bash
git clone https://github.com/jinzhao-rjb/table-ai.git
cd table-ai
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境

#### 环境变量配置
```env
# AI API配置
AI_API_KEY=your_api_key_here
AI_MODEL=qwen-vl-plus
AI_API_TYPE=qwen

# Redis配置（可选）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

#### 配置文件
编辑 `config.json` 文件，设置相关参数：
```json
{
  "ai_api_key": "your_api_key",
  "ai_model": "qwen-vl-plus",
  "max_threads": 8,
  "max_image_size": 1536,
  "max_tokens": 2000
}
```

## 🚀 快速开始

### 方式1：启动FastAPI后端服务
```bash
python main.py
```
服务将运行在 `http://localhost:8000`

### 方式2：启动Gradio Web界面
```bash
python gradio_app.py
```
界面将运行在 `http://localhost:7860`

## 📊 性能优化

### 优化措施
1. **图像大小限制**：最大1536px，减少AI推理时间
2. **多线程处理**：8线程并行，提高处理速度
3. **lxml解析器**：比BeautifulSoup更快的HTML解析
4. **API参数优化**：max_tokens=2000，平衡速度和准确性
5. **Redis队列**：支持分布式扩展
6. **向量化操作**：将函数转换为pandas向量化操作，提高数据处理速度
7. **模型优化**：YOLO模型fuse()优化，提高推理速度
8. **连接池管理**：优化API连接，减少连接建立时间

### 性能对比
- **优化前**：约3分钟/图像
- **优化后**：约30秒/图像（10倍提升）

## 🧪 测试

### 运行单元测试
```bash
python -m pytest
```

### 测试特定模块
```bash
# 测试表格提取
python -m pytest test_table_extraction.py

# 测试Redis连接
python -m pytest test_redis_connection.py

# 测试API配置
python -m pytest test_api_config.py

# 测试多列处理器
python -m pytest test_multi_column_processor.py
```

## 🔄 更新日志

### v1.0.0 (2026-01-10)
- ✅ 智能表格提取功能
- ✅ AI计算系统
- ✅ 语音输入支持（Whisper+Qwen API）
- ✅ 8线程并行处理
- ✅ 批量结果ZIP打包
- ✅ 文件批量重命名功能
- ✅ 向量化函数转换
- ✅ AST代码安全检查
- ✅ Redis分布式处理
- ✅ FastAPI后端服务
- ✅ Gradio Web界面
- ✅ 环境变量配置
- ✅ 性能优化（10倍提升）
- ✅ YOLO模型集成
- ✅ 多列表格处理

## 🤝 贡献

欢迎对项目进行贡献！贡献流程：

1. Fork项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "Add your feature"`
4. 推送到分支：`git push origin feature/your-feature`
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

- **GitHub**：[https://github.com/jinzhao-rjb/table-ai](https://github.com/jinzhao-rjb/table-ai)
- **Issues**：[https://github.com/jinzhao-rjb/table-ai/issues](https://github.com/jinzhao-rjb/table-ai/issues)

## 🙏 致谢

- **Whisper模型**：OpenAI的语音识别模型
- **Qwen-VL**：阿里云的视觉语言模型
- **Gradio**：交互式Web界面库
- **FastAPI**：现代Web框架
- **Redis**：内存数据结构存储
- **lxml**：高性能XML/HTML解析库
- **Pandas**：数据分析和处理库
- **YOLO**：目标检测模型

---

**Table AI** - 让表格处理变得简单、快速、准确！
