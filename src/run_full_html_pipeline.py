#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的全HTML物理还原流水线
用于处理300张A4纸表格，尤其是带合并单元格的纸质版或电子版
"""

import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import redis

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入所需模块
from src.modules.table_processor import TableProcessor
from src.modules.qwen_vl_manager import QwenVLManager

# 连接 Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def process_single_image(img_path, output_folder, processor, vl_manager):
    """
    处理单张图片的完整流程：
    1. YOLO裁剪（切除干扰，提高识别速度）
    2. AI生成HTML（带合并属性）
    3. 物理还原到Excel（使用occupied矩阵）
    
    Args:
        img_path: 图片路径
        output_folder: 输出文件夹
        processor: TableProcessor实例
        vl_manager: QwenVLManager实例
    """
    try:
        logger.info(f"开始处理图片: {img_path}")
        
        # 1. YOLO裁剪（切除干扰）
        crop_img, yolo_confidence = processor.process_image(img_path)
        if crop_img is None:
            logger.error(f"YOLO裁剪失败: {img_path}")
            return False, f"YOLO裁剪失败"
        
        logger.info(f"YOLO裁剪成功，置信度: {yolo_confidence:.2f}: {img_path}")
        
        # 2. AI生成HTML
        import cv2
        _, img_bytes = cv2.imencode('.jpg', crop_img)
        img_bytes = img_bytes.tobytes()
        
        success, table_data, error = vl_manager.extract_table_to_html(img_bytes)
        if not success or not table_data:
            logger.error(f"AI生成HTML失败: {error}")
            return False, f"AI生成HTML失败: {error}"
        
        html_content = table_data.get('content', '')
        if not html_content:
            logger.error(f"AI未返回HTML内容")
            return False, f"AI未返回HTML内容"
        
        logger.info(f"AI生成HTML成功，内容长度: {len(html_content)}字符")
        
        # 3. 风险预警检查
        risk_warning = False
        risk_reasons = []
        
        # 检查YOLO置信度
        if yolo_confidence < 0.7:
            risk_warning = True
            risk_reasons.append(f"YOLO置信度过低 ({yolo_confidence:.2f})")
        
        # 检查HTML完整性
        if '</table>' not in html_content.lower():
            risk_warning = True
            risk_reasons.append("HTML不完整（缺少</table>结尾）")
        
        # 4. 物理还原到Excel
        base_name = os.path.basename(img_path)
        file_name = os.path.splitext(base_name)[0] + "_table.xlsx"
        if risk_warning:
            file_name = os.path.splitext(base_name)[0] + "_table_RISK.xlsx"
            logger.warning(f"风险预警: {img_path} - {'; '.join(risk_reasons)}")
        
        save_path = os.path.join(output_folder, file_name)
        
        if processor.save_html_to_excel(html_content, save_path):
            logger.info(f"成功导出Excel: {save_path}")
            return True, f"成功导出Excel: {save_path}"
        else:
            logger.error(f"导出Excel失败")
            return False, f"导出Excel失败"
            
    except Exception as e:
        logger.error(f"处理图片时发生异常: {e}")
        return False, f"处理异常: {str(e)}"

def process_single_image_task(task_data, output_folder, processor, vl_manager):
    """
    处理单个 Redis 任务
    
    Args:
        task_data: 任务数据，格式为 "img_path#img_hash"
        output_folder: 输出文件夹
        processor: TableProcessor 实例
        vl_manager: QwenVLManager 实例
    """
    try:
        # 解析任务数据
        img_path, img_hash = task_data.split('#')
        
        # 检查 Redis 里的 result_cache 是否已经有这张图的结果
        if r.hexists('result_cache', img_hash):
            logger.info(f"结果已存在，跳过处理: {img_path} - {img_hash}")
            return True, f"结果已存在"
        
        # 执行原始处理逻辑
        success, message = process_single_image(img_path, output_folder, processor, vl_manager)
        
        # 更新 Redis 状态
        if success:
            # 存入结果缓存
            r.hset('result_cache', img_hash, 'success')
            # 更新任务状态
            r.hincrby('task_status', 'success', 1)
            r.hincrby('task_status', 'pending', -1)
            logger.info(f"任务完成，状态已更新: {img_path}")
        else:
            # 存入失败结果
            r.hset('result_cache', img_hash, f'failed: {message}')
            # 更新任务状态
            r.hincrby('task_status', 'failed', 1)
            r.hincrby('task_status', 'pending', -1)
            logger.error(f"任务失败，状态已更新: {img_path} - {message}")
        
        return success, message
    except Exception as e:
        logger.error(f"处理任务时发生异常: {task_data}, 错误: {e}")
        # 更新失败状态
        r.hincrby('task_status', 'failed', 1)
        r.hincrby('task_status', 'pending', -1)
        return False, f"处理任务异常: {str(e)}"

def main():
    """
    主函数：从 Redis 批量获取任务并处理
    """
    # 设置输出路径
    output_folder = "output_excel"  # 替换为你的输出文件夹路径
    yolo_model_path = r"D:\办公 - 副本\office-lazy-tool\runs\a4_table_lora_finetune2\weights\best.pt"  # 替换为你的YOLO模型路径
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 初始化处理器
    processor = TableProcessor(yolo_model_path=yolo_model_path)
    vl_manager = QwenVLManager()
    
    # 获取总任务数
    total = r.hget('task_status', 'total')
    if total:
        logger.info(f"Redis 中的总任务数: {total}")
    
    # 设置并发数
    max_workers = 8
    
    logger.info(f"使用 {max_workers} 个并发从 Redis 获取任务处理")
    
    # 使用 ThreadPoolExecutor 进行多线程处理
    from concurrent.futures import ThreadPoolExecutor
    
    # 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 持续从 Redis 获取任务，直到队列为空
        while True:
            # 从 Redis 弹出一个任务（阻塞方式，超时2秒）
            task_data = r.blpop('image_tasks', timeout=2)
            
            if not task_data:
                # 队列为空，退出循环
                logger.info("Redis 任务队列为空，等待所有任务完成...")
                break
            
            # 解析任务数据
            _, task_data = task_data
            
            # 提交任务到线程池
            executor.submit(process_single_image_task, task_data, output_folder, processor, vl_manager)
    
    # 获取最终状态
    status = r.hgetall('task_status')
    logger.info(f"最终处理状态: 总计: {status.get('total', 0)} | 成功: {status.get('success', 0)} | 失败: {status.get('failed', 0)} | 待处理: {status.get('pending', 0)}")
    return int(status.get('success', 0))

if __name__ == "__main__":
    main()
