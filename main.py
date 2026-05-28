import os
import argparse
from data_loader import get_dataloader
from model import SimpleUNet
from trainer import SegmentationTrainer
from storage import ResultSaver
from gui import SegmentationGUI
from PyQt5.QtWidgets import QApplication

def main():
    parser = argparse.ArgumentParser(description='图像分割项目')
    parser.add_argument('--mode', choices=['train', 'predict', 'gui'], default='gui', help='运行模式')
    parser.add_argument('--data_dir', default='data', help='数据目录')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--model_path', default='output/model_best.pth', help='模型路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        # 训练模型
        train(args)
    elif args.mode == 'predict':
        # 运行预测
        predict(args)
    elif args.mode == 'gui':
        # 启动图形界面
        launch_gui(args)

def train(args):
    # 加载数据
    train_loader = get_dataloader(
        image_dir=os.path.join(args.data_dir, 'train/images'),
        label_dir=os.path.join(args.data_dir, 'train/labels'),
        batch_size=args.batch_size
    )
    
    val_loader = get_dataloader(
        image_dir=os.path.join(args.data_dir, 'val/images'),
        label_dir=os.path.join(args.data_dir, 'val/labels'),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 初始化模型
    model = SimpleUNet()
    
    # 初始化训练器
    trainer = SegmentationTrainer(model, train_loader, val_loader)
    
    # 训练模型
    metrics = trainer.train(epochs=args.epochs)
    
    # 保存模型和指标
    saver = ResultSaver(args.output_dir)
    saver.save_model(model)
    saver.save_metrics(metrics)
    
    # 保存预测结果
    trainer.save_predictions(os.path.join(args.output_dir, 'predictions'))
    
    print(f"训练完成，模型和结果已保存到 {args.output_dir}")

def predict(args):
    # 加载模型
    model = SimpleUNet()
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 加载单张图像进行预测
    from PIL import Image
    import torchvision.transforms as transforms
    
    image_path = input("请输入要预测的图像路径: ")
    if not os.path.exists(image_path):
        print(f"错误: 图像 {image_path} 不存在")
        return
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # 运行预测
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output > 0.5).float().squeeze().cpu().numpy()
    
    # 保存预测结果
    from storage import ResultSaver
    saver = ResultSaver(args.output_dir)
    saver.save_prediction(image, None, prediction, 0, original_size=image.size)
    
    print(f"预测完成，结果已保存到 {os.path.join(args.output_dir, 'predictions')}")

def launch_gui(args):
    app = QApplication([])
    window = SegmentationGUI(args.model_path if os.path.exists(args.model_path) else None)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()  