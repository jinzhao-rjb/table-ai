import gradio as gr
import pandas as pd
import os
import sys
import zipfile
import shutil
import uuid # 关键：用于生成唯一请求 ID
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
# 新增：添加语音识别功能
from faster_whisper import WhisperModel

# 确保导入路径正确
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.table_processor import TableProcessor
from src.modules.qwen_vl_manager import QwenVLManager
from src.modules.multi_column_processor import MultiColumnProcessor
from src.modules.ai_service import get_ai_service
from src.modules.voice_service import get_voice_service

# 初始化语音服务
try:
    voice_service = get_voice_service()
except Exception as e:
    print(f"初始化语音服务失败: {e}")
    voice_service = None

# --- 1. 后台单例初始化 ---
def init_all_modules():
    # Try different YOLO model paths to ensure compatibility
    possible_yolo_paths = [
        os.path.join(project_root, "runs/a4_table_lora_finetune2/weights/best.pt"),
        os.path.join(project_root, "weights/best.pt"),
        os.path.join(os.getcwd(), "runs/a4_table_lora_finetune2/weights/best.pt"),
        os.path.join(os.getcwd(), "weights/best.pt")
    ]
    
    yolo_path = None
    for path in possible_yolo_paths:
        if os.path.exists(path):
            yolo_path = path
            break
    
    # If no YOLO model found, skip TableProcessor initialization for now
    table_proc = None
    if yolo_path:
        table_proc = TableProcessor(yolo_path)
    
    vl_manager = QwenVLManager()
    column_proc = MultiColumnProcessor()
    ai_service = get_ai_service()
    column_proc.set_ai_service(ai_service)
    return table_proc, vl_manager, column_proc, ai_service

table_proc, vl_manager, column_proc, ai_service = init_all_modules()

# --- 1. 语音转文字函数 (ASR) ---
def transcribe_audio(audio_path):
    """
    处理语音输入：将麦克风录音转为文本需求
    使用独立的voice_service进行语音识别
    """
    if audio_path is None:
        return ""
    try:
        if voice_service is None:
            return "语音服务未初始化"
        # 调用独立的voice_service进行语音识别
        text = voice_service.transcribe(audio_path)
        return text
    except Exception as e:
        return f"语音调用异常: {str(e)}"

# --- 2. 逻辑处理函数 (增加类型保护，防止 JS 报错) ---

def ocr_only_step(image):
    """仅执行 OCR 提取逻辑"""
    # 核心修改：DataFrame 组件绝不能接收 None
    empty_df = pd.DataFrame(columns=["提示"], data=[["等待上传图片..."]])
    
    if image is None:
        return None, empty_df, "⚠️ 请先上传图片"
    
    output_excel = "ocr_extracted_result.xlsx"
    try:
        # 直接使用 QwenVLManager 提取表格 HTML
        success, table_html, error = vl_manager.get_table_html(image)
        
        if success and table_html:
            if table_proc is None:
                # 创建临时 TableProcessor 实例如果主实例未初始化
                from src.modules.table_processor import TableProcessor
                temp_table_proc = TableProcessor(None)  # Pass None to bypass YOLO model requirement
                save_success = temp_table_proc.save_html_to_excel(table_html, output_excel)
            else:
                # 使用主实例
                save_success = table_proc.save_html_to_excel(table_html, output_excel)
                
            if save_success and os.path.exists(output_excel):
                df = pd.read_excel(output_excel)
                # 关键：清除 NaN，否则 JS 渲染会崩溃
                df = df.fillna("").astype(str) 
                return output_excel, df.head(20), "✅ 提取成功！"
        return None, empty_df, f"❌ 提取失败：{error if error else '未知错误'}"
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return None, empty_df, f"🚨 运行时错误: {str(e)}"

def ai_logic_only_step(file, requirement):
    """仅执行 AI 逻辑函数处理 (针对 20/20 成功案例)"""
    empty_df = pd.DataFrame(columns=["状态"], data=[["等待执行..."]])
    
    if file is None or not requirement:
        return None, empty_df, "⚠️ 请上传 Excel 并输入需求", gr.update(visible=False, value="")
    
    try:
        result = column_proc.process_multi_columns(file.name, requirement, max_iterations=3)
        
        if result.get("success"):
            out_file = result.get("file_path")
            df_preview = pd.read_excel(out_file)
            df_preview = df_preview.fillna("").astype(str).head(20)
            return out_file, df_preview, "✅ AI 处理完成", gr.update(visible=False, value="")
        else:
            failed_codes = getattr(column_proc, 'last_failed_code', [])
            last_code = failed_codes[-1] if failed_codes else "# AI 迭代中未记录到代码"
            
            error_df = pd.DataFrame(columns=["错误信息"], data=[[result.get("error")]])
            return None, error_df, f"❌ 迭代失败: {result.get('error')}", gr.update(visible=True, value=last_code)
            
    except Exception as e:
        return None, empty_df, f"🚨 系统故障: {str(e)}", gr.update(visible=False, value="")

# --- 批量处理功能 ---

def process_single_image(image_path, output_path, table_proc, vl_manager):
    """单个图片处理函数，用于线程池调用"""
    try:
        # 直接使用 QwenVLManager 提取表格 HTML
        success, table_html, error = vl_manager.get_table_html(image_path)
        
        if success and table_html:
            if table_proc is None:
                # 创建临时 TableProcessor 实例
                from src.modules.table_processor import TableProcessor
                temp_table_proc = TableProcessor(None)
                save_success = temp_table_proc.save_html_to_excel(table_html, output_path)
            else:
                # 使用主实例
                save_success = table_proc.save_html_to_excel(table_html, output_path)
            
            return save_success and os.path.exists(output_path)
        return False
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return False

# --- 2. 批量提取函数（升级 8 线程 + ZIP） ---
def batch_ocr_handler(files, progress=gr.Progress()):
    if not files: return None, "请上传图片", None, pd.DataFrame()
    
    request_id = str(uuid.uuid4())[:8]
    # 统一存放在 outputs 目录下，方便授权预览
    out_dir = os.path.join(os.getcwd(), "outputs", f"ocr_{request_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    results_paths = []
    total = len(files)
    
    # 估算 Token 成本（展示产品意识）
    estimated_tokens = total * 1800
    
    # 升级为 8 线程提速
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_path = {}
        for f_obj in files:
            # 【名字对应】保持原始文件名
            orig_name = os.path.basename(f_obj.name).rsplit('.', 1)[0]
            out_path = os.path.join(out_dir, f"{orig_name}.xlsx")
            # 提交任务 - 使用 process_single_image 辅助函数
            future = executor.submit(
                process_single_image,
                f_obj.name, 
                out_path,
                table_proc,
                vl_manager
            )
            future_to_path[future] = out_path
        
        count = 0
        for future in as_completed(future_to_path):
            count += 1
            progress(count/total, desc=f"正在并行提取 {count}/{total}")
            if future.result():
                results_paths.append(future_to_path[future])
                
    if not results_paths:
        return None, "❌ 处理失败", None, pd.DataFrame()

    # 自动打包 ZIP
    zip_path = os.path.join(os.getcwd(), f"OCR_Result_{request_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in results_paths:
            zipf.write(f, arcname=os.path.basename(f))
            
    status_msg = f"✅ 提取完成 (ID: {request_id}) | 预估消耗: {estimated_tokens} Tokens"
    return results_paths, status_msg, zip_path, pd.DataFrame()

# --- 3. 批量生成函数（增加 ZIP 打包） ---
def batch_logic_handler(files, requirement, progress=gr.Progress()):
    """批量AI计算：增加 ZIP 打包支持"""
    if not files or not requirement:
        return None, "请上传Excel文件并输入处理需求", None, pd.DataFrame()
    
    request_id = str(uuid.uuid4())[:8]
    # 统一存放在 outputs 目录下，方便授权预览
    out_dir = os.path.join(os.getcwd(), "outputs", f"logic_{request_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    processed_paths = []
    total = len(files)
    
    # 限制并发为 3，防止 AI 迭代导致 Qwen API 报错
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_info = {}
        for file_obj in files:
            # 获取原始文件名 (名字对应)
            orig_name = os.path.basename(file_obj.name).rsplit('.', 1)[0]
            
            # 提交任务
            future = executor.submit(
                process_single_logic,
                file_obj.name, 
                requirement
            )
            future_to_info[future] = orig_name
        
        # 收集结果并更新进度
        completed = 0
        for future in as_completed(future_to_info):
            completed += 1
            orig_name = future_to_info[future]
            progress(completed/total, desc=f"AI 并行处理... ({completed}/{total})")
            try:
                result = future.result()
                if result and result.get("success"):
                    # 名字对应：RESULT_原名.xlsx，并保存到临时目录
                    final_path = os.path.join(out_dir, f"RESULT_{orig_name}.xlsx")
                    if os.path.exists(result["file_path"]):
                        import shutil
                        shutil.copy2(result["file_path"], final_path)
                        processed_paths.append(final_path)
            except Exception as e:
                print(f"处理 {orig_name} 出错: {e}")
                continue
    
    if not processed_paths:
        return None, "❌ 所有文件处理失败", None, pd.DataFrame()

    # 自动打包生成的结果
    zip_path = os.path.join(os.getcwd(), f"Logic_Result_{request_id}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in processed_paths:
            zipf.write(f, arcname=os.path.basename(f))
            
    return processed_paths, "✅ 批量 AI 处理并打包完成", zip_path, pd.DataFrame()

def process_single_logic(file_path, requirement):
    """单个 Excel 逻辑处理函数，用于线程池调用"""
    try:
        # 调用 20/20 通过的 process_multi_columns 函数
        result = column_proc.process_multi_columns(file_path, requirement, max_iterations=3)
        return result
    except Exception as e:
        print(f"处理 {os.path.basename(file_path)} 时出错: {e}")
        return None

# --- 1. 修复预览崩溃的函数 (关键修复) ---
def quick_preview_file(file_data: gr.SelectData):
    """
    加固后的预览函数：解决 UUID 隔离下的路径找不到问题
    """
    if not file_data or file_data.value is None:
        return pd.DataFrame(columns=["提示"], data=[["请选择有效文件预览"]])
    
    try:
        # 获取 Gradio 传入的文件元数据
        file_info = file_data.value
        
        # 核心：优先从字典中获取真实磁盘路径 'orig_name' 或 'name'
        if isinstance(file_info, dict):
            # 在某些 Gradio 版本中，name 是临时路径，需要确认它是否存在
            file_path = file_info.get('name')
        else:
            file_path = file_info
        
        # 调试诊断：如果找不到文件，尝试在当前目录下搜索
        if not file_path or not os.path.exists(file_path):
            # 这里的报错就是你看到的：提示已被清理
            return pd.DataFrame(columns=["提示"], data=[["文件加载失败，请重新点击或检查路径"]])

        # 读取时限制行列，减轻前端 JS 压力，防止 ERR_ABORTED
        df = pd.read_excel(file_path).iloc[:20, :15]
        
        # 彻底清洗：转为字符串防止 JS 渲染崩溃
        return df.fillna("").astype(str)
        
    except Exception as e:
        return pd.DataFrame({"预览失败": [f"错误原因: {str(e)}"]})
        

# --- 3. Gradio 界面设计 (解耦布局) ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI 表格专家系统") as demo:
    gr.Markdown("# 📊 AI 表格全能工作站")
    
    with gr.Tabs():
        # --- 标签页 1: 独立提取功能 ---
        with gr.TabItem("🔍 场景一：表格图片提取"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="filepath", label="上传图片")
                    btn_ocr = gr.Button("开始 OCR 提取", variant="primary")
                with gr.Column(scale=2):
                    ocr_status = gr.Markdown("等待任务...")
                    ocr_file = gr.File(label="下载提取结果")
                    ocr_preview = gr.DataFrame(label="提取预览 (前20行)")
            
            btn_ocr.click(ocr_only_step, inputs=[img_input], outputs=[ocr_file, ocr_preview, ocr_status])

        # --- 标签页 2: 独立逻辑处理 ---
        with gr.TabItem("🤖 场景二：AI 逻辑计算"):
            gr.Markdown("## 🎤 支持语音输入")
            gr.Markdown("**使用说明**：点击麦克风图标，在浏览器弹窗中允许麦克风访问权限，然后说出您的需求。录制完成后，系统会自动将语音转为文字。")
            gr.Markdown("""**权限问题排查**：
            1. 确保浏览器已获得系统麦克风权限
            2. 在浏览器地址栏左侧的锁图标中，检查并允许麦克风访问
            3. 如果使用的是 HTTPS，请确认证书有效
            4. 尝试刷新页面后重新授权""")
            with gr.Row():
                with gr.Column(scale=1):
                    excel_input = gr.File(label="上传 Excel")
                    
                    # --- 新增语音输入 ---
                    with gr.Row():
                        single_audio_input = gr.Audio(label="🎤 语音说出需求", sources=["microphone"], type="filepath", show_label=True)
                    
                    logic_req = gr.Textbox(label="您的需求 (语音自动转写)", placeholder="例如：计算 17/20 案例中的时间戳...", lines=4)
                    
                    # 语音录制完自动填入文本框
                    single_audio_input.change(fn=transcribe_audio, inputs=[single_audio_input], outputs=[logic_req])
                    
                    btn_ai = gr.Button("调用专家函数", variant="primary")
                with gr.Column(scale=2):
                    ai_status = gr.Markdown("就绪")
                    ai_file = gr.File(label="下载处理结果")
                    ai_preview = gr.DataFrame(label="结果预览 (前20行)")

            # 调试区 (仅在失败时弹出)
            with gr.Column(visible=False) as debug_section:
                gr.Markdown("### 🛠️ AI 逻辑微调 (人机协作)")
                code_editor = gr.Code(language="python", label="AI 生成的源代码")
                btn_save = gr.Button("修正并存入 Redis 黄金库")
                
            btn_ai.click(
                ai_logic_only_step, 
                inputs=[excel_input, logic_req], 
                outputs=[ai_file, ai_preview, ai_status, debug_section]
            )
            
            btn_save.click(
                lambda r, c: ai_service.qwen_learning.save_success_case(r, c),
                inputs=[logic_req, code_editor],
                outputs=[ai_status]
            )

        # --- 标签页 3: 批量处理功能 ---
        with gr.TabItem("🚀 场景三：批量处理工厂"):
            gr.Markdown("## 🚀 批量表格处理专家系统")
            gr.Markdown("支持批量 OCR 提取和批量 AI 计算，图片名与 Excel 名严格对应")
            
            with gr.Tabs():
                gr.Markdown("## 🎤 支持语音输入")
                gr.Markdown("**使用说明**：点击麦克风图标，在浏览器弹窗中允许麦克风访问权限，然后说出您的需求。录制完成后，系统会自动将语音转为文字。")
                gr.Markdown("""**权限问题排查**：
                1. 确保浏览器已获得系统麦克风权限
                2. 在浏览器地址栏左侧的锁图标中，检查并允许麦克风访问
                3. 如果使用的是 HTTPS，请确认证书有效
                4. 尝试刷新页面后重新授权""")
                # 子标签页 1: 批量 OCR
                with gr.TabItem("📂 批量图片提取"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_ocr_files = gr.File(
                                label="上传图片 (支持多选)", 
                                file_count="multiple", 
                                file_types=["image"]
                            )
                            batch_ocr_btn = gr.Button("🚀 启动多线程提取", variant="primary")
                            batch_ocr_status = gr.Markdown("状态：就绪")
                            # --- 新增 ZIP 下载组件 ---
                            zip_download_box = gr.File(label="🎁 点击下载一键打包结果 (.zip)")
                        with gr.Column(scale=2):
                            batch_ocr_results = gr.File(
                                label="生成的 Excel 列表 (点击下方文件可预览)", 
                                file_count="multiple"
                            )
                            batch_ocr_preview = gr.DataFrame(label="选中表格内容预览", wrap=True, interactive=False)
                    
                    batch_ocr_btn.click(
                        batch_ocr_handler,
                        inputs=[batch_ocr_files],
                        outputs=[batch_ocr_results, batch_ocr_status, zip_download_box, batch_ocr_preview]
                    )
                    
                    # 文件预览功能
                    batch_ocr_results.select(
                        quick_preview_file,
                        outputs=[batch_ocr_preview]
                    )

                # 子标签页 2: 批量 AI 计算
                with gr.TabItem("⚙️ 批量 AI 函数处理"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_logic_files = gr.File(
                                label="上传 Excel (支持多选)", 
                                file_count="multiple", 
                                file_types=[".xlsx"]
                            )
                            
                            # --- 核心改动：增加语音输入 ---
                            with gr.Row():
                                audio_input = gr.Audio(label="🎤 语音说出需求", sources=["microphone"], type="filepath")
                            
                            batch_logic_req = gr.Textbox(
                                label="统一处理需求 (语音自动转写)", 
                                placeholder="例如：计算所有文件的税后金额...", 
                                lines=4
                            )
                            
                            # 语音录制完自动填入文本框
                            audio_input.change(fn=transcribe_audio, inputs=[audio_input], outputs=[batch_logic_req])
                            
                            batch_logic_btn = gr.Button("启动批量 AI 处理", variant="primary")
                            batch_logic_status = gr.Markdown("就绪")
                            # --- 新增 ZIP 下载组件 ---
                            batch_logic_zip = gr.File(label="🎁 下载批量 AI 计算打包结果 (.zip)")
                        with gr.Column(scale=2):
                            batch_logic_results = gr.File(
                                label="处理结果 (点击文件名预览)", 
                                file_count="multiple"
                            )
                            batch_logic_preview = gr.DataFrame(label="结果预览", wrap=True)
                    
                    batch_logic_btn.click(
                        batch_logic_handler,
                        inputs=[batch_logic_files, batch_logic_req],
                        outputs=[batch_logic_results, batch_logic_status, batch_logic_zip, batch_logic_preview]
                    )
                    
                    # 文件预览功能
                    batch_logic_results.select(
                        quick_preview_file,
                        outputs=[batch_logic_preview]
                    )

if __name__ == "__main__":
    demo.queue(max_size=30) # 开启队列，支持高并发
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        max_threads=200, # 调高线程，解决语音+任务冲突导致的 Broken Connection
        allowed_paths=[os.getcwd(), os.path.join(os.getcwd(), "outputs")]
    )
