import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ResultSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    def save_model(self, model, epoch=None):
        if epoch is not None:
            model_path = os.path.join(self.output_dir, f'model_epoch_{epoch}.pth')
        else:
            model_path = os.path.join(self.output_dir, 'model_best.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def save_metrics(self, metrics, epoch=None):
        if epoch is not None:
            metrics_path = os.path.join(self.output_dir, f'metrics_epoch_{epoch}.json')
        else:
            metrics_path = os.path.join(self.output_dir, 'metrics_final.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    
    def save_prediction(self, image, mask, pred, index, original_size=None):
        # 保存原始图像
        img_path = os.path.join(self.output_dir, 'images', f'image_{index}.png')
        self._save_image(image, img_path, original_size)
        
        # 保存真实掩码
        mask_path = os.path.join(self.output_dir, 'masks', f'mask_{index}.png')
        self._save_mask(mask, mask_path, original_size)
        
        # 保存预测掩码
        pred_path = os.path.join(self.output_dir, 'predictions', f'prediction_{index}.png')
        self._save_mask(pred, pred_path, original_size)
    
    def _save_image(self, image, path, original_size=None):
        # 假设image是归一化的Tensor [C, H, W]
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1, 2, 0).numpy()
            # 反归一化
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            image = image.astype(np.uint8)
        
        if original_size:
            image = cv2.resize(image, original_size)
        
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def _save_mask(self, mask, path, original_size=None):
        # 假设mask是Tensor [H, W]或[1, H, W]
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            mask = mask.cpu().numpy()
        
        # 将概率图转换为二值图
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8) * 255
        else:
            mask = (mask > 0.5).astype(np.uint8) * 255
        
        if original_size:
            mask = cv2.resize(mask, original_size)
        
        cv2.imwrite(path, mask)
    
    def create_visualization(self, image, mask, pred, index):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原始图像
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            axes[0].imshow(img_np)
        else:
            axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 显示真实掩码
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # 显示预测掩码
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy()
        else:
            pred_np = pred
        axes[2].imshow(pred_np, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(self.output_dir, f'visualization_{index}.png')
        plt.savefig(vis_path)
        plt.close()
        return vis_path  