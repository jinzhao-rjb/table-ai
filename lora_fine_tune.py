from ultralytics import YOLO
import torch
import torch.nn as nn
import os

# 1. 加载预训练模型
model_path = r"D:\办公 - 副本\office-lazy-tool\runs\table_detector2\weights\best.pt"
model = YOLO(model_path)

# 2. 查看模型结构
print("模型结构:")
print(model.model)

# 3. 准备数据集配置
# 使用更新后的table.yaml配置文件
config_path = r"D:\办公 - 副本\office-lazy-tool\src\table.yaml"

# 4. 实现LoRA微调的训练脚本
# 由于Ultralytics YOLO官方暂不支持LoRA，我们使用冻结部分层的方式实现类似效果
# 冻结模型的大部分层，只训练检测头部分
print("\n开始冻结模型层...")

# 获取模型的网络层
model_ = model.model

# 冻结backbone和neck层，只训练head层
for param in model_.parameters():
    param.requires_grad = False

# 解冻检测头层
if hasattr(model_, 'model'):
    # YOLO11模型结构
    if hasattr(model_.model, '23'):  # 检测头通常在最后几层
        # 解冻检测头相关层
        for i in range(20, len(model_.model)):
            for param in model_.model[i].parameters():
                param.requires_grad = True

print("模型层冻结完成，只训练检测头部分")

# 5. 开始微调训练
print("\n开始微调训练...")
results = model.train(
    data=config_path,
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    amp=False,
    workers=0,
    patience=10,
    name='a4_table_lora_finetune',
    project=r"D:\办公 - 副本\office-lazy-tool\runs",
    save=True,
    save_period=10,
    lr0=0.0001,  # 使用更小的学习率进行微调
    weight_decay=0.0001
)

print("微调训练完成！")
print(f"训练结果保存在: D:\办公 - 副本\office-lazy-tool\runs\a4_table_lora_finetune")
