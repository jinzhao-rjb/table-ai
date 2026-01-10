#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成Redis的主入口文件
负责管理Redis任务队列和表格提取功能
"""

import os
import sys
import logging
import time
import redis

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志，添加更详细的信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(message)s'
)
logger = logging.getLogger("RedisIntegratedMain")

# 导入核心模块
from src.modules.table_processor import TableProcessor
from src.modules.qwen_vl_manager import QwenVLManager
from src.utils.config import config_manager
from src.utils.dual_redis_db import DualRedisDB

class RedisIntegratedProcessor:
    """
    集成Redis的表格提取处理器
    负责管理Redis任务队列和表格提取功能
    """
    
    def __init__(self):
        """
        初始化集成处理器
        """
        # 初始化双Redis数据库管理器
        self.dual_redis = DualRedisDB()
        # 使用任务库连接作为主要的Redis客户端
        self.redis_client = self.dual_redis.task_conn
        
        # 初始化表格处理器和Qwen VL管理器
        self.yolo_model_path = os.path.join(project_root, "runs/a4_table_lora_finetune2/weights/best.pt")
        self.table_processor = TableProcessor(self.yolo_model_path)
        self.qwen_manager = QwenVLManager(
            api_key=config_manager.get("ai.api_key", ""),
            model="qwen-vl-max",
            api_type="qwen",
            redis_client=self.dual_redis.logic_conn  # 使用逻辑库存储函数和学习知识
        )
        
        logger.info("Redis集成处理器初始化完成")
    
    def add_image_to_queue(self, image_path):
        """
        将图片添加到Redis任务队列
        
        Args:
            image_path: 图片路径
            
        Returns:
            bool: 添加成功返回True，否则返回False
        """
        try:
            # 检查图片是否存在
            if not os.path.exists(image_path):
                logger.error(f"图片不存在: {image_path}")
                return False
            
            # 直接添加到table_extraction_tasks队列，格式为img_path#img_hash
            import hashlib
            img_hash = hashlib.md5(image_path.encode('utf-8')).hexdigest()
            task_data = f"{image_path}#{img_hash}"
            self.redis_client.lpush('table_extraction_tasks', task_data)
            
            # 更新状态统计
            self.redis_client.hincrby("table_extraction_status", "total", 1)
            self.redis_client.hincrby("table_extraction_status", "pending", 1)
            
            logger.info(f"任务已添加到Redis任务队列: {image_path}")
            return True
        except Exception as e:
            logger.error(f"添加任务到队列失败: {e}")
            return False
    
    def add_image_folder_to_queue(self, folder_path):
        """
        将文件夹中的所有图片添加到Redis任务队列
        
        Args:
            folder_path: 图片文件夹路径
            
        Returns:
            int: 添加成功的图片数量
        """
        try:
            success_count = 0
            
            # 获取图片列表
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, file)
                    if self.add_image_to_queue(img_path):
                        success_count += 1
            
            logger.info(f"成功添加 {success_count} 张图片到队列")
            return success_count
        except Exception as e:
            logger.error(f"添加文件夹图片到队列失败: {e}")
            return 0
    
    def _process_task_with_reliability(self, task_data):
        """
        可靠地处理单个Redis任务，处理完成后从处理中队列移除
        
        Args:
            task_data: 任务数据，格式为 "img_path#img_hash"
        """
        try:
            # 处理任务
            success = self.process_single_task(task_data)
            
            # 从处理中队列移除任务
            self.redis_client.lrem('table_extraction_processing', 0, task_data)
            
            return success
        except Exception as e:
            logger.error(f"任务处理异常，从处理中队列移除任务: {e}")
            # 确保任务从处理中队列移除，避免任务泄露
            self.redis_client.lrem('table_extraction_processing', 0, task_data)
            return False
    
    def process_single_task(self, task_data):
        """
        处理单个Redis任务
        
        Args:
            task_data: 任务数据，格式为 "img_path#img_hash" 或 "img_path#img_hash#trace_id"
            
        Returns:
            bool: 处理成功返回True，否则返回False
        """
        try:
            # 解析任务数据
            parts = task_data.split('#')
            if len(parts) == 3:
                img_path, img_hash, trace_id = parts
            else:
                img_path, img_hash = parts
                trace_id = "unknown"  # 兼容旧格式，没有trace_id时使用unknown
            
            logger.info(f"处理任务: {img_path} (trace_id: {trace_id})")
            
            # 初始化命中统计
            hit_rate_key = 'table_hit_rate'
            total_requests = self.redis_client.hget(hit_rate_key, 'total_requests')
            total_requests = int(total_requests) if total_requests else 0
            
            hit_count = self.redis_client.hget(hit_rate_key, 'hit_count')
            hit_count = int(hit_count) if hit_count else 0
            
            # 计算命中率
            hit_rate = (hit_count / total_requests * 100) if total_requests > 0 else 0
            
            logger.info(f"当前命中率: {hit_rate:.2f}% (总请求: {total_requests}, 命中: {hit_count})")
            
            # 检查结果缓存
            result_exists = self.redis_client.hexists('table_extraction_results', img_hash)
            expire_key_exists = self.redis_client.exists(f'result_expire:{img_hash}')
            
            # 只有当命中率达到100%时才返回缓存数据
            if result_exists and expire_key_exists and hit_rate >= 100:
                logger.info(f"缓存命中，跳过处理: {img_path} - {img_hash}")
                # 更新命中统计
                self.redis_client.hincrby(hit_rate_key, 'hit_count', 1)
                self.redis_client.hincrby(hit_rate_key, 'total_requests', 1)
                return True
            elif result_exists:
                # 结果存在但不符合条件，清理结果
                logger.info(f"清理不符合条件的结果: {img_path} - {img_hash} (命中率: {hit_rate:.2f}%)")
                self.redis_client.hdel('table_extraction_results', img_hash)
                self.redis_client.delete(f'result_expire:{img_hash}')
            
            # 更新总请求数
            self.redis_client.hincrby(hit_rate_key, 'total_requests', 1)
            # 否则继续处理
            
            # 获取重试次数
            retry_key = f"retry_count:{img_hash}"
            retry_count = self.redis_client.get(retry_key)
            retry_count = int(retry_count) if retry_count else 0
            
            max_retries = 3
            if retry_count >= max_retries:
                logger.warning(f"任务超过最大重试次数({max_retries})，标记为失败: {img_path}")
                self._update_task_status('failed')
                self.redis_client.hset('table_extraction_results', img_hash, f'failed:超过最大重试次数({max_retries})')
                self.redis_client.delete(retry_key)  # 清理重试计数
                return False
            
            logger.info(f"开始处理图片: {img_path} (重试次数: {retry_count})")
            
            # 添加时间锚点日志，定位耗时大头
            import time
            t0 = time.time()
            
            # 1. YOLO裁剪表格区域
            crop_img, yolo_confidence = self.table_processor.process_image(img_path)
            t1 = time.time()
            
            if crop_img is None:
                logger.error(f"YOLO裁剪失败: {img_path}")
                return self._handle_task_failure(task_data, img_hash, img_path, 'YOLO裁剪失败', retry_count)
            
            logger.info(f"YOLO裁剪成功，置信度: {yolo_confidence:.2f}: {img_path}")
            
            # 2. AI生成HTML
            import cv2
            # 在转Base64前进行质量压缩
            _, img_bytes = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_bytes = img_bytes.tobytes()
            
            success, table_data, error = self.qwen_manager.extract_table_to_html(img_bytes)
            t2 = time.time()
            
            if not success or not table_data:
                logger.error(f"AI生成HTML失败: {error}")
                return self._handle_task_failure(task_data, img_hash, img_path, f'AI生成HTML失败: {error}', retry_count)
            
            html_content = table_data.get('content', '')
            if not html_content:
                logger.error(f"AI未返回HTML内容")
                return self._handle_task_failure(task_data, img_hash, img_path, 'AI未返回HTML内容', retry_count)
            
            logger.info(f"AI生成HTML成功，内容长度: {len(html_content)}字符")
            
            # 3. 物理还原到Excel
            base_name = os.path.basename(img_path)
            file_name = os.path.splitext(base_name)[0] + "_table.xlsx"
            output_folder = os.path.join(project_root, "output")
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, file_name)
            
            if self.table_processor.save_html_to_excel(html_content, save_path):
                t3 = time.time()
                
                logger.info(f"成功导出Excel: {save_path} (trace_id: {trace_id})")
                self._update_task_status('success')
                
                # 保存结果并设置过期时间（30天）
                # 结果格式: success:{save_path}#{trace_id}
                self.redis_client.hset('table_extraction_results', img_hash, f'success:{save_path}#{trace_id}')
                # 使用单独的键来跟踪过期时间
                self.redis_client.setex(f'result_expire:{img_hash}', 30 * 24 * 60 * 60, 1)  # 30天过期
                
                self.redis_client.delete(retry_key)  # 清理重试计数
                
                # 打印耗时日志
                logger.info(f"处理耗时: 预处理: {t1-t0:.1f}s, API响应: {t2-t1:.1f}s, 解析保存: {t3-t2:.1f}s, 总耗时: {t3-t0:.1f}s")
                return True
            else:
                logger.error(f"导出Excel失败")
                return self._handle_task_failure(task_data, img_hash, img_path, '导出Excel失败', retry_count)
                
        except Exception as e:
            logger.error(f"处理图片时发生异常: {e}")
            try:
                img_path, img_hash = task_data.split('#')
                return self._handle_task_failure(task_data, img_hash, img_path, f'处理异常: {str(e)}', retry_count)
            except:
                logger.error(f"任务数据格式错误: {task_data}")
                return False
    
    def _handle_task_failure(self, task_data, img_hash, img_path, error_msg, current_retry):
        """
        处理任务失败，实现指数退避重试
        
        Args:
            task_data: 任务数据
            img_hash: 图片哈希值
            img_path: 图片路径
            error_msg: 错误信息
            current_retry: 当前重试次数
            
        Returns:
            bool: 始终返回False，表示任务失败
        """
        # 错误分类：区分可重试错误和不可重试错误
        # 可重试错误
        retryable_errors = [
            "timeout", "Timeout", "503", "rate limit", "Rate limit", 
            "网络", "Network", "connection", "Connection"
        ]
        
        # 不可重试错误
        non_retryable_errors = [
            "找不到", "不存在", "损坏", "invalid", "Invalid", 
            "API key", "API Key", "apikey", "权限", "Permission",
            "违规", "Violation", "prompt", "Prompt", "YOLO裁剪失败"
        ]
        
        # 检查是否为不可重试错误
        is_non_retryable = any(error in error_msg for error in non_retryable_errors)
        # 检查是否为可重试错误
        is_retryable = any(error in error_msg for error in retryable_errors)
        
        max_retries = 3
        next_retry = current_retry + 1
        
        # 获取trace_id
        parts = task_data.split('#')
        trace_id = parts[2] if len(parts) == 3 else "unknown"
        
        # 只有可重试错误且未超过最大重试次数时才重试
        if is_retryable and next_retry <= max_retries:
            # 计算指数退避时间（秒）
            wait_time = 2 ** next_retry
            
            # 保存重试计数
            retry_key = f"retry_count:{img_hash}"
            self.redis_client.setex(retry_key, wait_time * 2, next_retry)
            
            # 将任务重新添加到队列，带延迟
            logger.warning(f"任务失败，{wait_time}秒后重试 ({next_retry}/{max_retries}): {img_path} - {error_msg} (trace_id: {trace_id})")
            
            # 使用优先级队列重新添加任务
            self.dual_redis.add_task_to_queue(task_data, priority=5)  # 重试任务使用中等优先级
        else:
            # 不可重试错误或超过最大重试次数
            if is_non_retryable:
                logger.error(f"不可重试错误，直接标记为失败: {img_path} - {error_msg} (trace_id: {trace_id})")
            else:
                logger.error(f"任务失败，超过最大重试次数 ({max_retries}): {img_path} - {error_msg} (trace_id: {trace_id})")
            
            # 标记为失败
            self._update_task_status('failed')
            self.redis_client.hset('table_extraction_results', img_hash, f'failed:{error_msg}')
            self.redis_client.delete(f"retry_count:{img_hash}")  # 清理重试计数
        
        return False
    
    def _update_task_status(self, status):
        """
        更新任务状态
        
        Args:
            status: 任务状态，'success'或'failed'
        """
        self.redis_client.hincrby('table_extraction_status', status, 1)
        self.redis_client.hincrby('table_extraction_status', 'pending', -1)
    
    def run_batch_processing(self, max_workers=4):
        """
        批量处理Redis队列中的任务
        
        Args:
            max_workers: 最大并发数
        """
        logger.info(f"开始批量处理Redis队列中的任务，最大并发数: {max_workers}")
        
        import signal
        from concurrent.futures import ThreadPoolExecutor
        
        # 优雅停机标志
        self._shutdown_flag = False
        
        # 信号处理函数
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，准备优雅停机")
            self._shutdown_flag = True
        
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 持续从Redis获取任务，直到队列为空或收到停机信号
            while not self._shutdown_flag:
                # 先从优先级队列获取任务
                task_data = self.dual_redis.get_task_from_priority_queue()
                
                if not task_data:
                    # 如果优先级队列为空，尝试从旧的列表队列获取（兼容旧版本）
                    task_data = self.redis_client.brpoplpush('table_extraction_tasks', 'table_extraction_processing', timeout=1)
                    
                    if not task_data:
                        # 队列为空，继续检查停机标志
                        continue
                else:
                    # 将优先级队列的任务添加到处理中队列
                    self.redis_client.lpush('table_extraction_processing', task_data)
                
                # 提交任务到线程池
                executor.submit(self._process_task_with_reliability, task_data)
        
        logger.info("优雅停机完成，所有任务已处理")
    
    def get_status(self):
        """
        获取当前任务状态
        
        Returns:
            dict: 任务状态字典
        """
        status = self.redis_client.hgetall('table_extraction_status')
        
        # 确保所有状态字段都存在
        if 'total' not in status:
            status['total'] = '0'
        if 'success' not in status:
            status['success'] = '0'
        if 'failed' not in status:
            status['failed'] = '0'
        if 'pending' not in status:
            status['pending'] = '0'
        
        return status
    
    def clear_queue(self):
        """
        清空所有相关的Redis数据
        """
        # 1. 清空任务库 (task_conn)
        self.dual_redis.clear_task_queue()
        
        # 2. 清空存储库 (logic_conn) 中的学习记忆 (可选，防止错误的记忆固化)
        self.dual_redis.logic_conn.delete("qwen_learning_memory")
        
        # 3. 同时清除其他相关Redis键
        other_keys = self.redis_client.keys('retry_count:*')
        other_keys += self.redis_client.keys('result_expire:*')
        other_keys += self.redis_client.keys('qwen_vl_requests_count:*')
        other_keys += self.redis_client.keys('table_hit_rate:*')
        
        if other_keys:
            self.redis_client.delete(*other_keys)
            logger.info(f"已清除 {len(other_keys)} 个其他Redis键")
        
        logger.info("任务队列及 AI 学习记忆已彻底重置")
    
    def start_worker(self, worker_id=1, max_workers=8):
        """
        启动工作线程
        
        Args:
            worker_id: 工作线程ID
            max_workers: 最大并发数
        """
        logger.info(f"启动工作线程 {worker_id}")
        self.run_batch_processing(max_workers)

def main():
    """
    主函数
    """
    # 初始化处理器
    processor = RedisIntegratedProcessor()
    
    # === 关键：修改代码后运行第一次时，取消下面这一行的注释来清空旧的错误缓存 ===
    processor.clear_queue()
    
    # 示例：将测试图片添加到队列
    test_image_path = r"D:\office\office-lazy-tool\database_a4\augmented_images\IMG_20260102_142821_add_noise.jpg"
    processor.add_image_to_queue(test_image_path)
    
    # 示例：启动批量处理
    processor.run_batch_processing(max_workers=4)
    
    # 打印最终状态
    status = processor.get_status()
    logger.info(f"最终处理状态: 总计: {status['total']} | 成功: {status['success']} | 失败: {status['failed']} | 待处理: {status['pending']}")

if __name__ == "__main__":
    main()
