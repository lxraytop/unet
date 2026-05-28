import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import cv2

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * images.size(0)
            
            # 验证
            val_loss, metrics = self.evaluate()
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss/len(self.train_loader.dataset):.4f}')
            print(f'Val Loss: {val_loss/len(self.val_loader.dataset):.4f}')
            print(f'IoU: {metrics["iou"]:.4f}, Dice: {metrics["dice"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}')
            print('-' * 50)
        
        return metrics
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device).unsqueeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # 转换为numpy数组进行评估
                preds = outputs.cpu().numpy() > 0.5
                masks = masks.cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_masks.extend(masks.flatten())
        
        # 计算评估指标
        metrics = self._calculate_metrics(np.array(all_preds), np.array(all_masks))
        return val_loss, metrics
    
    def _calculate_metrics(self, preds, masks):
        tn, fp, fn, tp = confusion_matrix(masks, preds).ravel()
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def save_predictions(self, output_dir, num_samples=10):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(self.val_loader):
                if i >= num_samples:
                    break
                    
                images = images.to(self.device)
                outputs = self.model(images)
                
                for j in range(images.size(0)):
                    # 保存原始图像
                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    img_np = img_np.astype(np.uint8)
                    cv2.imwrite(os.path.join(output_dir, f'sample_{i*self.val_loader.batch_size+j}_orig.jpg'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    
                    # 保存真实掩码
                    mask_np = masks[j].cpu().numpy() * 255
                    cv2.imwrite(os.path.join(output_dir, f'sample_{i*self.val_loader.batch_size+j}_mask.jpg'), mask_np)
                    
                    # 保存预测掩码
                    pred_np = (outputs[j].cpu().numpy()[0] > 0.5) * 255
                    cv2.imwrite(os.path.join(output_dir, f'sample_{i*self.val_loader.batch_size+j}_pred.jpg'), pred_np)  