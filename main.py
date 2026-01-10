#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI后端服务
负责处理文件上传、任务管理和AI服务接口
"""

from fastapi import FastAPI, UploadFile, File, Request
from typing import List, Dict
import uvicorn
import os
import hashlib
import uuid
from src.redis_integrated_main import RedisIntegratedProcessor

# 创建FastAPI应用
app = FastAPI(title="智能表格处理系统", version="1.0")

# 添加Trace ID中间件
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """
    为每个请求添加唯一的Trace ID
    """
    # 生成Trace ID
    trace_id = str(uuid.uuid4())
    
    # 将Trace ID存储在请求状态中
    request.state.trace_id = trace_id
    
    # 将Trace ID添加到响应头中
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    
    return response

# 初始化处理器
processor = RedisIntegratedProcessor()

# 创建上传目录
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def batch_upload(request: Request, files: List[UploadFile] = File(...)):
    """
    批量上传文件并添加到任务队列
    
    Args:
        request: 请求对象，用于获取trace ID
        files: 上传的文件列表
        
    Returns:
        dict: 包含上传状态、任务ID列表和trace ID
    """
    task_ids = []
    
    # 获取当前请求的trace ID
    trace_id = request.state.trace_id
    
    for file in files:
        # 1. 保存文件到本地
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 2. 生成文件哈希作为任务ID
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        task_id = file_hash
        
        # 3. 调用RedisIntegratedProcessor添加任务到队列
        # 任务数据格式: file_path#file_hash#trace_id
        task_data = f"{file_path}#{file_hash}#{trace_id}"
        
        # 为当前正在查看的文件设置高优先级（示例：优先级1）
        # 这里可以根据实际业务逻辑动态调整优先级
        priority = 1  # 默认高优先级
        
        # 使用优先级队列添加任务
        processor.dual_redis.add_task_to_queue(task_data, priority)
        
        # 4. 初始化任务状态
        processor.redis_client.hincrby('table_extraction_status', 'total', 1)
        processor.redis_client.hincrby('table_extraction_status', 'pending', 1)
        
        task_ids.append({"file": file.filename, "task_id": task_id})
    
    return {"status": "queued", "tasks": task_ids, "trace_id": trace_id}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        dict: 包含任务ID和状态
    """
    # 查询任务结果
    result = processor.redis_client.hget('table_extraction_results', task_id)
    
    if result:
        result_str = result.decode('utf-8') if isinstance(result, bytes) else result
        if result_str.startswith('success:'):
            return {"task_id": task_id, "status": "success", "result": result_str.split(':', 1)[1]}
        elif result_str.startswith('failed:'):
            return {"task_id": task_id, "status": "failed", "error": result_str.split(':', 1)[1]}
    
    # 检查是否在处理中队列
    processing_tasks = processor.redis_client.lrange('table_extraction_processing', 0, -1)
    for task in processing_tasks:
        task_str = task.decode('utf-8') if isinstance(task, bytes) else task
        if task_id in task_str:
            return {"task_id": task_id, "status": "processing"}
    
    # 检查是否在待处理队列
    pending_count = processor.redis_client.llen('table_extraction_tasks')
    if pending_count > 0:
        pending_tasks = processor.redis_client.lrange('table_extraction_tasks', 0, -1)
        for task in pending_tasks:
            task_str = task.decode('utf-8') if isinstance(task, bytes) else task
            if task_id in task_str:
                return {"task_id": task_id, "status": "pending"}
    
    return {"task_id": task_id, "status": "not_found"}

@app.get("/api/status")
async def get_system_status():
    """
    获取系统整体状态
    
    Returns:
        dict: 包含系统状态信息
    """
    status = processor.get_status()
    
    # 转换为整数类型
    return {
        "total": int(status.get('total', 0)),
        "success": int(status.get('success', 0)),
        "failed": int(status.get('failed', 0)),
        "pending": int(status.get('pending', 0)),
        "queue_length": processor.redis_client.llen('table_extraction_tasks')
    }

@app.get("/api/functions")
async def get_available_functions():
    """
    获取可用的AI函数
    
    Returns:
        dict: 包含可用函数列表
    """
    # 从Redis逻辑库中获取所有函数
    functions = processor.dual_redis.get_all_functions()
    return {"functions": functions}

@app.post("/api/function/{function_id}/weight")
async def update_function_weight(function_id: str, weight: float):
    """
    更新函数权重，用于RLHF
    
    Args:
        function_id: 函数ID
        weight: 新的权重值
        
    Returns:
        dict: 更新结果
    """
    success = processor.dual_redis.update_function_weight(function_id, weight)
    return {"success": success, "function_id": function_id, "weight": weight}

if __name__ == "__main__":
    import threading
    
    # 启动任务处理线程
    def start_task_processor():
        """
        启动任务处理线程
        """
        processor.run_batch_processing(max_workers=8)
    
    # 创建并启动任务处理线程
    task_thread = threading.Thread(target=start_task_processor, daemon=True)
    task_thread.start()
    
    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)
