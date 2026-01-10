import multiprocessing
import os
from ultralytics import YOLO

# Windows多进程支持
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置文件路径
    yaml_path = os.path.join(current_dir, 'table.yaml')
    
    # 使用YOLO11n-seg分割模型（适合表格检测任务）
    model = YOLO('yolo11n-seg.pt')

    # 开始训练
    print("开始训练YOLO11n-seg表格检测模型...")
    model.train(
        data=yaml_path,        # 数据集配置文件
        epochs=100,             # 训练轮数
        batch=4,                # 减小批次大小，降低显存占用和训练不稳定性
        imgsz=640,              # 输入图像尺寸
        device=0,               # 使用GPU 0
        amp=False,              # 关闭AMP，避免数值不稳定导致的nan
        workers=0,              # Windows系统设置为0避免多进程问题
        patience=10,            # 早停机制，如果10个epoch没有改进就停止
        name='table_detector',  # 训练结果保存名称
        project=os.path.join(current_dir, '../runs'),  # 结果保存路径
        save=True,              # 保存训练结果
        save_period=10,         # 每10个epoch保存一次模型
        lr0=0.0001,             # 减小初始学习率，避免梯度爆炸
        momentum=0.9,           # 调整动量参数
        weight_decay=0.0001,    # 调整权重衰减
        single_cls=True,        # 单类别检测，适合表格检测任务
        dropout=0.1,            # 添加 dropout 防止过拟合
    )

    print("训练完成！")
    print(f"训练结果保存在 {os.path.join(current_dir, '../runs/segment/table_detector')} 目录下")

    # 模型训练后的评估
    print("\n开始评估训练好的模型...")
    metrics = model.val(data=yaml_path, imgsz=640)
    print(f"模型评估结果：")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"分割mAP50: {metrics.seg.map50:.4f}")
    print(f"分割mAP50-95: {metrics.seg.map:.4f}")
