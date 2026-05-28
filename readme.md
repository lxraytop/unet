# 图像分割项目

## 项目概述

本项目实现了一个完整的图像分割流程，包括数据标注、加载、模型训练、结果评估和可视化。项目采用Python语言开发，基于PyTorch深度学习框架，使用UNet架构实现二分类图像分割任务。

## 功能特点

1. **数据标注**：支持使用LabelMe工具进行图像分割数据标注
2. **数据加载**：提供高效的数据加载和预处理模块
3. **模型训练**：基于UNet架构实现的二分类图像分割模型
4. **结果评估**：计算多种评估指标，包括IoU、Dice系数和准确率
5. **可视化界面**：提供直观的图形界面，展示原始图像、真实掩码和预测结果

## 安装依赖
pip install -r requirements.txt
## 项目结构
image-segmentation/
├── data/                   # 数据目录
│   ├── train/              # 训练数据
│   │   ├── images/         # 训练图像
│   │   └── labels/         # 训练标签(LabelMe JSON格式)
│   └── val/                # 验证数据
│       ├── images/         # 验证图像
│       └── labels/         # 验证标签
├── output/                 # 输出目录
│   ├── model_best.pth      # 最佳模型
│   ├── metrics_final.json  # 最终评估指标
│   └── predictions/        # 预测结果
├── data_loader.py          # 数据加载模块
├── model.py                # 模型定义
├── trainer.py              # 训练和评估模块
├── storage.py              # 结果存储模块
├── gui.py                  # 图形界面
├── main.py                 # 主程序入口
├── README.md               # 项目说明
└── requirements.txt        # 依赖列表
## 使用方法

### 1. 数据标注

使用LabelMe工具对图像进行分割标注：

1. 安装LabelMe：`pip install labelme`
2. 启动LabelMe：`labelme`
3. 打开图像并创建多边形标注
4. 将前景区域标注为"cat"
5. 保存为JSON格式到`data/train/labels`或`data/val/labels`目录

### 2. 训练模型
python main.py --mode train --data_dir data --output_dir output --epochs 10 --batch_size 4
### 3. 运行预测
python main.py --mode predict --model_path output/model_best.pth
### 4. 使用图形界面
python main.py --mode gui --model_path output/model_best.pth
## 评估指标

项目计算以下评估指标：

- IoU (Intersection over Union)
- Dice系数
- 准确率
- 真阳性(TP)
- 真阴性(TN)
- 假阳性(FP)
- 假阴性(FN)

## 可视化界面

图形界面提供以下功能：

- 加载和显示原始图像
- 加载和显示真实掩码
- 运行模型预测并显示结果
- 计算和显示评估指标
- 批量评估多个图像

## 扩展和改进

1. 支持多类别分割
2. 添加数据增强功能
3. 实现更复杂的分割模型
4. 添加模型优化和超参数调整
5. 增强可视化界面功能  