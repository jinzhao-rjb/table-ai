from ultralytics import YOLO
import os

def evaluate_model(image_dir='datasets/images/val', predict_count=5):
    # 使用训练好的最佳模型
    model_paths = [
        'runs/detect/train/weights/best.pt',  # 当前最佳模型
        'models/best_epochs260.pt',  # 之前训练的模型
        'models/yolov12s.pt'  # 新的基础模型
    ]
    
    # 选择存在的最佳模型
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("没有找到可用的模型文件")
        return
    
    print(f"使用模型: {model_path}")
    model = YOLO(model_path)
    
    # 1. 快速验证模型性能 - 只使用少量图像
    print("\n=== 快速验证模型性能 ===")
    # 使用优化后的推理参数
    optimized_params = {
        'imgsz': 800,      # 更大的输入尺寸，提高小目标检测能力
        'conf': 0.3,       # 调整后的置信度阈值
        'iou': 0.55,       # 调整后的NMS阈值
        'augment': True,   # 启用增强推理，提高鲁棒性
        'max_det': 20      # 最大检测数量，根据实际场景调整
    }
    
    metrics = model.val(
        data='dishu.yaml', 
        fraction=0.3,  # 只使用30%的数据进行验证
        save=False,  # 不保存结果
        plots=False,  # 禁用绘图
        **optimized_params
    )
    
    # 打印关键指标
    print(f"\n关键指标:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    # 计算平均F1分数
    f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-8)
    print(f"F1 Score: {f1_score:.4f}")
    
    # 2. 对指定目录的图像进行预测
    print(f"\n=== 对 {image_dir} 目录的图像进行预测 ===")
    # 获取指定目录下的所有图像
    import glob
    image_paths = glob.glob(f'{image_dir}/*')
    
    if not image_paths:
        print(f"在目录 {image_dir} 中没有找到图像文件")
        return
    
    # 如果predict_count为0或负数，则处理所有图像
    if predict_count <= 0:
        total_images = len(image_paths)
    else:
        total_images = min(predict_count, len(image_paths))
    
    # 只处理指定数量的图像
    image_paths = image_paths[:total_images]
    
    print(f"共找到 {len(image_paths)} 张图像，开始处理...")
    
    # 分批处理图片，每批处理10张，避免内存溢出
    batch_size = 10
    processed_count = 0
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_start = i + 1
        batch_end = min(i + batch_size, len(image_paths))
        
        print(f"\n处理批次 {batch_start}-{batch_end}/{len(image_paths)}...")
        
        # 处理当前批次
        results = model.predict(
            batch_paths,
            save=True,  # 保存预测结果
            save_txt=True,  # 保存标签文件
            save_conf=True,  # 保存置信度
            show_labels=True,
            show_conf=True,
            **optimized_params  # 使用优化后的推理参数
        )
        
        processed_count += len(batch_paths)
        print(f"已处理 {processed_count}/{len(image_paths)} 张图像")
    
    print(f"\n已完成对 {len(image_paths)} 张图像的预测")
    print("预测结果已保存到'runs/detect/predict'目录")
    
    # 3. 输出模型优化建议
    print("\n=== 当前使用的优化参数 ===")
    for key, value in optimized_params.items():
        print(f"{key}: {value}")
    
    print("\n=== 模型优化建议 ===")
    print("当前模型性能已经相当不错，可以考虑以下进一步优化:")
    print("1. 尝试不同的置信度阈值和NMS阈值组合，找到最适合当前场景的参数")
    print("2. 尝试使用更大的输入图像尺寸：1024")
    print("3. 考虑使用更复杂的模型（如yolov12s或yolov12m）进行微调")
    print("4. 优化训练数据，增加更多样化的样本")
    print("5. 尝试模型集成，使用多个模型进行预测")
    print("6. 考虑使用ONNX或TensorRT进行模型优化，提高推理速度")

if __name__ == '__main__':
    import sys
    
    # 解析命令行参数
    # 用法：python 识别.py [image_dir] [predict_count]
    # 示例：python 识别.py datasets/images/train 10
    
    image_dir = 'datasets/images/val'  # 默认目录
    predict_count = 5  # 默认预测数量
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            predict_count = int(sys.argv[2])
        except ValueError:
            print(f"无效的预测数量：{sys.argv[2]}，将使用默认值 5")
    
    # 调用评估函数
    evaluate_model(image_dir, predict_count)
