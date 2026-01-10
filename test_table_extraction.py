import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.modules.table_processor import TableProcessor
from src.modules.qwen_vl_manager import QwenVLManager
from src.utils.config import config_manager

def test_table_extraction():
    # 配置路径
    test_image_path = "D:\\office\\office-lazy-tool\\database_a4\\augmented_images\\IMG_20260102_142821_add_noise.jpg"
    yolo_model_path = "D:\\office\\office-lazy-tool\\runs\\a4_table_lora_finetune2\\weights\\best.pt"
    output_excel_path = "output_test.xlsx"
    
    # 检查测试图片是否存在
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return False
    
    # 检查YOLO模型是否存在
    if not os.path.exists(yolo_model_path):
        print(f"YOLO模型不存在: {yolo_model_path}")
        return False
    
    print("初始化表格处理器...")
    # 初始化TableProcessor
    table_processor = TableProcessor(yolo_model_path)
    
    print("初始化Qwen VL管理器...")
    # 初始化QwenVLManager
    qwen_manager = QwenVLManager(
        api_key=config_manager.get("ai.api_key", ""),
        model="qwen-vl-max",
        api_type="qwen"
    )
    
    print(f"开始处理图片: {test_image_path}")
    # 步骤1: 使用YOLO裁剪表格区域
    crop_img, confidence = table_processor.process_image(test_image_path)
    print(f"YOLO检测置信度: {confidence:.2f}")
    
    # 保存裁剪后的图片用于调试
    crop_img_path = "crop_test.jpg"
    import cv2
    cv2.imwrite(crop_img_path, crop_img)
    print(f"裁剪后的图片已保存: {crop_img_path}")
    
    # 步骤2: 使用Qwen-VL将图片转换为HTML表格
    print("使用Qwen-VL进行表格识别...")
    success, table_data, error = qwen_manager.extract_table_to_html(crop_img_path)
    
    if not success:
        print(f"表格识别失败: {error}")
        return False
    
    html_content = table_data.get('content', '')
    print(f"识别到的HTML内容长度: {len(html_content)} 字符")
    print("HTML内容预览:")
    print(html_content[:500] + "..." if len(html_content) > 500 else html_content)
    
    # 步骤3: 将HTML转换为Excel
    print("将HTML转换为Excel...")
    success = table_processor.save_html_to_excel(html_content, output_excel_path)
    
    if success:
        print(f"Excel文件已成功保存: {output_excel_path}")
        return True
    else:
        print("HTML转换为Excel失败")
        return False

if __name__ == "__main__":
    print("开始测试表格提取功能...")
    success = test_table_extraction()
    if success:
        print("测试成功!")
    else:
        print("测试失败!")
        sys.exit(1)