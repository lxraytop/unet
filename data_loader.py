import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 获取对应的LabelMe JSON文件
        label_name = os.path.splitext(img_name)[0] + '.json'
        label_path = os.path.join(self.label_dir, label_name)
        
        # 解析LabelMe标注文件生成掩码
        mask = self._parse_labelme_json(label_path, image.size)

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).squeeze(0)

        return image, mask

    def _parse_labelme_json(self, json_path, img_size):
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            for shape in data['shapes']:
                if shape['label'] == 'cat':  # 假设前景标签为'cat'
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)  # 填充前景区域
        return Image.fromarray(mask)

def get_dataloader(image_dir, label_dir, batch_size=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SegmentationDataset(image_dir, label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader  