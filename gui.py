import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, 
                            QListWidget, QSplitter, QGroupBox, QGridLayout, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
import torchvision.transforms as transforms
from PIL import Image

class SegmentationGUI(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.setWindowTitle("图像分割可视化工具")
        self.resize(1200, 800)
        
        self.model = None
        self.model_path = model_path
        self.current_image_path = None
        self.current_prediction = None
        self.metrics = {}
        
        self.init_ui()
        if self.model_path:
            self.load_model(self.model_path)
    
    def init_ui(self):
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 创建标签页
        self.tabs = QTabWidget()
        
        # 第一个标签页：数据可视化
        self.visualization_tab = QWidget()
        self.init_visualization_tab()
        self.tabs.addTab(self.visualization_tab, "数据可视化")
        
        # 第二个标签页：模型预测
        self.prediction_tab = QWidget()
        self.init_prediction_tab()
        self.tabs.addTab(self.prediction_tab, "模型预测")
        
        # 第三个标签页：评估指标
        self.metrics_tab = QWidget()
        self.init_metrics_tab()
        self.tabs.addTab(self.metrics_tab, "评估指标")
        
        main_layout.addWidget(self.tabs)
        
        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def init_visualization_tab(self):
        layout = QVBoxLayout()
        
        # 图像选择区域
        select_layout = QHBoxLayout()
        self.select_image_btn = QPushButton("选择图像")
        self.select_image_btn.clicked.connect(self.select_image)
        select_layout.addWidget(self.select_image_btn)
        
        # 图像显示区域
        self.image_display = QLabel("请选择一张图像")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 400)
        
        # 结果显示区域
        result_layout = QHBoxLayout()
        
        self.original_image_label = QLabel("原始图像")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        
        self.ground_truth_label = QLabel("真实掩码")
        self.ground_truth_label.setAlignment(Qt.AlignCenter)
        self.ground_truth_label.setMinimumSize(300, 300)
        
        self.prediction_label = QLabel("预测结果")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setMinimumSize(300, 300)
        
        result_layout.addWidget(self.original_image_label)
        result_layout.addWidget(self.ground_truth_label)
        result_layout.addWidget(self.prediction_label)
        
        layout.addLayout(select_layout)
        layout.addWidget(self.image_display)
        layout.addLayout(result_layout)
        
        self.visualization_tab.setLayout(layout)
    
    def init_prediction_tab(self):
        layout = QVBoxLayout()
        
        # 模型加载区域
        model_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.select_model)
        self.model_status = QLabel("未加载模型")
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_status)
        
        # 预测区域
        predict_layout = QHBoxLayout()
        self.predict_btn = QPushButton("运行预测")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        predict_layout.addWidget(self.predict_btn)
        
        # 预测结果显示
        self.prediction_display = QLabel("请先加载模型和图像")
        self.prediction_display.setAlignment(Qt.AlignCenter)
        self.prediction_display.setMinimumSize(600, 400)
        
        layout.addLayout(model_layout)
        layout.addLayout(predict_layout)
        layout.addWidget(self.prediction_display)
        
        self.prediction_tab.setLayout(layout)
    
    def init_metrics_tab(self):
        layout = QVBoxLayout()
        
        # 评估指标显示
        metrics_layout = QGridLayout()
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setPlaceholderText("运行预测后显示评估指标...")
        
        # 可视化区域
        self.metrics_display = QLabel("评估指标可视化")
        self.metrics_display.setAlignment(Qt.AlignCenter)
        self.metrics_display.setMinimumSize(600, 300)
        
        metrics_layout.addWidget(QLabel("评估指标"), 0, 0)
        metrics_layout.addWidget(self.metrics_text, 1, 0)
        metrics_layout.addWidget(self.metrics_display, 1, 1)
        
        # 批量评估按钮
        self.batch_eval_btn = QPushButton("批量评估")
        self.batch_eval_btn.clicked.connect(self.batch_evaluation)
        
        layout.addLayout(metrics_layout)
        layout.addWidget(self.batch_eval_btn)
        
        self.metrics_tab.setLayout(layout)
    
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.setEnabled(bool(self.model))
    
    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)
        
        # 尝试查找对应的掩码文件
        base_name = os.path.splitext(image_path)[0]
        mask_path = base_name + '_mask.png'
        
        if os.path.exists(mask_path):
            self.display_mask(mask_path)
        else:
            self.ground_truth_label.setText("未找到掩码文件")
    
    def display_mask(self, mask_path):
        pixmap = QPixmap(mask_path)
        scaled_pixmap = pixmap.scaled(self.ground_truth_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ground_truth_label.setPixmap(scaled_pixmap)
    
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "模型文件 (*.pth *.pt)")
        if file_path:
            self.model_path = file_path
            self.load_model(file_path)
    
    def load_model(self, model_path):
        try:
            from model import SimpleUNet
            self.model = SimpleUNet()
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.model_status.setText(f"已加载模型: {os.path.basename(model_path)}")
            self.predict_btn.setEnabled(bool(self.current_image_path))
            QMessageBox.information(self, "成功", "模型加载成功")
        except Exception as e:
            self.model = None
            self.model_status.setText("模型加载失败")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
    
    def run_prediction(self):
        if not self.model or not self.current_image_path:
            return
        
        try:
            # 加载并预处理图像
            image = Image.open(self.current_image_path).convert('RGB')
            original_size = image.size
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            # 运行模型预测
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = (output > 0.5).float().squeeze().cpu().numpy()
            
            self.current_prediction = prediction
            
            # 显示原始图像
            original_pixmap = QPixmap(self.current_image_path)
            scaled_original = original_pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_image_label.setPixmap(scaled_original)
            
            # 显示预测结果
            prediction_image = self.create_prediction_image(prediction, original_size)
            self.prediction_label.setPixmap(prediction_image)
            self.prediction_display.setPixmap(prediction_image)
            
            # 计算并显示评估指标
            self.calculate_metrics(prediction)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")
    
    def create_prediction_image(self, prediction, original_size=None):
        # 将预测结果转换为图像
        if original_size:
            prediction = cv2.resize(prediction.astype(np.uint8), original_size)
        
        height, width = prediction.shape
        qimage = QImage(prediction * 255, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.prediction_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return scaled_pixmap
    
    def calculate_metrics(self, prediction):
        # 尝试加载真实掩码以计算指标
        base_name = os.path.splitext(self.current_image_path)[0]
        mask_path = base_name + '_mask.png'
        
        if os.path.exists(mask_path):
            # 加载真实掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (prediction.shape[1], prediction.shape[0]))
            mask = (mask > 127).astype(np.uint8)
            
            # 计算评估指标
            tp = np.sum((prediction == 1) & (mask == 1))
            tn = np.sum((prediction == 0) & (mask == 0))
            fp = np.sum((prediction == 1) & (mask == 0))
            fn = np.sum((prediction == 0) & (mask == 1))
            
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            self.metrics = {
                'IoU': iou,
                'Dice系数': dice,
                '准确率': accuracy,
                '真阳性': tp,
                '真阴性': tn,
                '假阳性': fp,
                '假阴性': fn
            }
            
            # 显示指标
            metrics_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in self.metrics.items()])
            self.metrics_text.setText(metrics_text)
            
            # TODO: 创建指标可视化
            self.create_metrics_visualization()
    
    def create_metrics_visualization(self):
        # 简单的文本可视化
        metrics_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in self.metrics.items()])
        self.metrics_display.setText(metrics_text)
    
    def batch_evaluation(self):
        # 批量评估功能
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if not folder_path:
            return
        
        try:
            # 查找所有图像文件
            image_extensions = ['png', 'jpg', 'jpeg']
            image_files = [f for f in os.listdir(folder_path) if f.split('.')[-1].lower() in image_extensions]
            
            if not image_files:
                QMessageBox.information(self, "提示", "未找到图像文件")
                return
            
            # 对每个图像运行评估
            all_metrics = []
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                self.current_image_path = img_path
                
                # 加载并预处理图像
                image = Image.open(img_path).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0)
                
                # 运行模型预测
                with torch.no_grad():
                    output = self.model(input_tensor)
                    prediction = (output > 0.5).float().squeeze().cpu().numpy()
                
                # 尝试加载真实掩码以计算指标
                base_name = os.path.splitext(img_path)[0]
                mask_path = base_name + '_mask.png'
                
                if os.path.exists(mask_path):
                    # 加载真实掩码
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (prediction.shape[1], prediction.shape[0]))
                    mask = (mask > 127).astype(np.uint8)
                    
                    # 计算评估指标
                    tp = np.sum((prediction == 1) & (mask == 1))
                    tn = np.sum((prediction == 0) & (mask == 0))
                    fp = np.sum((prediction == 1) & (mask == 0))
                    fn = np.sum((prediction == 0) & (mask == 0))
                    
                    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                    
                    all_metrics.append({
                        'image': img_file,
                        'IoU': iou,
                        'Dice': dice,
                        'Accuracy': accuracy
                    })
            
            # 计算平均指标
            if all_metrics:
                avg_metrics = {
                    '平均IoU': sum(m['IoU'] for m in all_metrics) / len(all_metrics),
                    '平均Dice': sum(m['Dice'] for m in all_metrics) / len(all_metrics),
                    '平均Accuracy': sum(m['Accuracy'] for m in all_metrics) / len(all_metrics)
                }
                
                # 显示结果
                metrics_text = "批量评估结果:\n\n"
                for m in all_metrics:
                    metrics_text += f"{m['image']} - IoU: {m['IoU']:.4f}, Dice: {m['Dice']:.4f}, Accuracy: {m['Accuracy']:.4f}\n"
                
                metrics_text += "\n" + "\n".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
                self.metrics_text.setText(metrics_text)
                
                QMessageBox.information(self, "成功", f"批量评估完成，共评估 {len(all_metrics)} 张图像")
            else:
                QMessageBox.information(self, "提示", "未找到带掩码的图像")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量评估失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationGUI()
    window.show()
    sys.exit(app.exec_())  