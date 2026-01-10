# Table AI - 智能表格处理系统

Table AI是一个功能强大的智能表格处理系统，提供从图像提取表格数据、批量文件处理、语音交互等多种功能，支持FastAPI后端服务和Gradio Web界面两种使用方式。

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
│   │   ├── vectorized_function_converter.py # 向量化函数转换
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
- **核心特性**：
  - 批量文件上传与任务队列
  - Trace ID跟踪机制
  - Redis分布式处理
  - 任务状态查询
  - 高并发支持
- **主要端点**：
  - `POST /api/upload`: 批量上传文件并添加到任务队列
  - `GET /api/task/{task_id}`: 获取特定任务状态
- **技术栈**：FastAPI + Uvicorn + Redis

### 2. Gradio Web界面 (`gradio_app.py`)
- **功能**：提供可视化交互界面，适合直接使用
- **用途**：无需编程，适合普通用户快速使用
- **核心特性**：
  - 直观的图像上传界面
  - 语音输入支持
  - 实时进度显示
  - 结果预览与下载
  - 批量处理支持
  - 自动ZIP打包结果
- **技术栈**：Gradio + Python

## 🚀 核心功能详细说明

### 1. 智能表格提取 (`src/modules/table_processor.py`)

**功能描述**：从各种图像格式中提取表格数据，支持复杂布局和多列表格

**实现细节**：
- **图像处理流程**：
  1. 使用YOLO模型裁剪表格核心区域（去除页码和白边）
  2. 图像增强：对比度增强，使边框更明显
  3. 图像优化：自动调整尺寸（最大1536px），提高AI推理速度
  4. 置信度评估：记录低置信度案例，用于模型优化

- **表格提取**：
  - 调用Qwen-VL API进行表格识别
  - 生成HTML格式的表格结构
  - 支持复杂表格布局
  - 支持多列表格处理

- **结果处理**：
  - HTML转Excel：使用lxml解析器快速处理
  - 动态行高调整
  - 边框样式设置
  - 单元格内容对齐

**技术栈**：OpenCV + YOLO + Qwen-VL API + lxml + openpyxl

### 2. 语音输入支持 (`src/modules/voice_service.py`)

**功能描述**：通过语音控制应用，支持中文语音命令

**实现细节**：
- **双模型支持**：
  - 本地Whisper模型：低延迟，离线可用
  - Qwen API云端备份：提高识别准确性
  - 自动降级机制：本地模型失败时自动切换到云端

- **语音处理流程**：
  1. 录音：使用PyAudio捕获音频
  2. 语音转文本：优先使用本地Whisper模型
  3. 命令解析：识别用户意图
  4. 执行操作：根据命令执行相应功能

**技术栈**：Whisper + Qwen API + SpeechRecognition + PyAudio

### 3. 高性能处理 (`src/redis_integrated_main.py`)

**功能描述**：快速处理大量图像，提高系统吞吐量

**实现细节**：
- **8线程并行处理**：
  - 使用ThreadPoolExecutor实现
  - 同时处理多个图像
  - 自动负载均衡

- **Redis分布式队列**：
  - 任务优先级管理
  - 支持分布式部署
  - 任务状态监控
  - 故障恢复机制

- **性能优化**：
  - 图像大小限制（最大1536px）
  - lxml解析器替代html.parser
  - API参数优化（max_tokens=2000）
  - 连接池管理

**技术栈**：ThreadPoolExecutor + Redis + lxml

### 4. 批量处理与结果管理 (`src/modules/file_manager.py`)

**功能描述**：处理多个图像并管理结果

**实现细节**：
- **批量上传**：
  - 支持多文件上传
  - 自动验证文件格式
  - 进度跟踪

- **结果管理**：
  - 自动生成ZIP压缩包
  - 支持多种输出格式（JSON、Excel）
  - 结果预览
  - 历史记录管理

**技术栈**：Python ZIP库 + 文件操作

### 5. 文件管理功能 (`src/modules/file_manager.py`)

**功能描述**：提供文件和文件夹管理功能

**实现细节**：
- **批量重命名**：
  - 支持自定义命名规则
  - 编号格式定制
  - 正则表达式支持
  - 预览功能

- **文件格式转换**：
  - 支持多种格式转换
  - 批量转换功能
  - 转换质量设置

- **文件夹管理**：
  - 文件夹结构创建
  - 文件分类管理
  - 重复文件查找

**技术栈**：Python文件操作 + 正则表达式

### 6. 向量化函数转换 (`src/modules/vectorized_function_converter.py`)

**功能描述**：将AI生成的函数转换为pandas向量化操作，提高数据处理速度

**实现细节**：
- **支持的函数类型**：
  - 文本处理：LOWER, UPPER, TRIM, SUBSTITUTE等
  - 数值处理：ROUND, INT, ABS等
  - 日期处理：TEXT等
  - 条件处理：IF等
  - 连接函数：CONCATENATE, &等

- **转换流程**：
  1. 解析AI生成的函数
  2. 转换为pandas向量化操作
  3. 执行优化后的操作
  4. 返回处理结果

**技术栈**：pandas + 正则表达式

### 7. 代码安全检查 (`src/utils/ast_security_checker.py`)

**功能描述**：检测Python代码中的潜在安全问题

**实现细节**：
- **AST语法树分析**：
  - 解析Python代码生成AST
  - 遍历AST检测危险模式
  - 支持多种安全规则

- **危险函数检测**：
  - 文件系统操作（open, os.remove等）
  - 命令执行（os.system, subprocess等）
  - 代码注入（exec, eval等）
  - 网络操作（requests, socket等）

- **风险评估**：
  - 高、中、低风险等级
  - 详细的安全报告
  - 修复建议

**技术栈**：Python AST模块

### 8. 多列表格处理 (`src/modules/multi_column_processor.py`)

**功能描述**：处理复杂的多列表格布局

**实现细节**：
- **多列检测**：
  - 识别表格中的多列结构
  - 处理跨列单元格
  - 支持不规则表格

- **内容重组**：
  - 将多列数据重组为标准表格
  - 保持数据关联性
  - 处理合并单元格

**技术栈**：OpenCV + 图像处理算法

### 9. 灵活配置管理 (`src/utils/config.py`)

**功能描述**：动态调整应用参数，支持多种配置方式

**实现细节**：
- **配置优先级**：
  1. 环境变量（最高优先级）
  2. 配置文件
  3. 默认值

- **支持的配置项**：
  - AI_API_KEY：API密钥
  - AI_MODEL：AI模型选择
  - AI_API_TYPE：API提供商
  - max_threads：并行线程数
  - max_image_size：图像最大尺寸
  - max_tokens：AI模型最大token数

**技术栈**：Python配置管理

## 🛠️ 技术栈详细说明

### 核心框架
| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| Web框架 | FastAPI | - | 后端RESTful API服务 |
| 可视化框架 | Gradio | - | Web界面交互 |
| GUI框架 | PyQt5 | >=5.15.7 | 桌面应用界面 |
| 异步框架 | - | - | 异步任务处理 |

### 文档处理
| 技术 | 版本 | 用途 |
|------|------|------|
| openpyxl | >=3.1.2 | Excel文件创建和修改 |
| pandas | >=1.5.3 | 数据分析和处理 |
| python-docx | >=0.8.11 | Word文档处理 |
| python-pptx | >=0.6.21 | PowerPoint处理 |
| PyPDF2 | >=3.0.1 | PDF基本处理 |
| PyMuPDF | >=1.23.4 | 高级PDF和图像处理 |
| reportlab | >=3.6.12 | PDF生成 |

### 图像处理
| 技术 | 版本 | 用途 |
|------|------|------|
| Pillow | >=10.0.1 | 图像处理基础库 |
| pytesseract | >=0.3.10 | OCR文字识别 |
| opencv-python | >=4.8.1.78 | 计算机视觉处理 |
| qrcode | >=7.4.2 | 二维码生成 |
| paddlepaddle | >=3.2.2 | 深度学习框架 |
| paddleocr | >=3.3.2 | 高精度OCR识别 |
| ultralytics | - | YOLO模型集成 |

### 语音识别
| 技术 | 版本 | 用途 |
|------|------|------|
| speechrecognition | >=3.10.0 | 语音识别基础库 |
| pyaudio | >=0.2.13 | 音频输入处理 |
| whisper | - | 本地语音识别模型 |

### 网络和数据
| 技术 | 版本 | 用途 |
|------|------|------|
| schedule | >=1.2.0 | 任务调度 |
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

### 开发工具
| 技术 | 版本 | 用途 |
|------|------|------|
| pytest | >=7.3.2 | 单元测试 |
| flake8 | >=6.0.0 | 代码质量检查 |
| black | >=23.3.0 | 代码格式化 |
| mypy | >=1.3.0 | 类型检查 |
| pyinstaller | >=5.13.0 | 应用打包 |

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
