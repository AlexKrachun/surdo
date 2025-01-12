import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import s3d, S3D_Weights
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

path = 'slovo/annotations.csv'
data = pd.read_csv(path, sep='\t')
train_data = data[data['train']]



# Создаем свой класс датасета
class GestureVideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_paths = list(Path(video_dir).glob('*.mp4'))
        self.transform = transform
        # Здесь нужно определить mapping жестов в числовые метки
        self.label_mapping = {i[1]: i[0] for i in enumerate(train_data['text'])}  # например: {'gesture1': 0, 'gesture2': 1, ...}
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = str(self.video_paths[idx])
        # Получаем метку из имени файла или другим способом
        label = ...  # зависит от вашей структуры данных
        
        # Читаем видео
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # Преобразуем в тензор
        frames = torch.FloatTensor(np.array(frames))
        # Перемещаем временное измерение
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        if self.transform:
            frames = self.transform(frames)
            
        return frames, label

# Настраиваем трансформации
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
])

# Создаем датасеты и загрузчики данных
train_dataset = GestureVideoDataset('path/to/train/videos', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Загружаем предобученную модель
model = s3d(weights=S3D_Weights.KINETICS400_V1)

# Модифицируем последний слой под наше количество классов
num_classes = len(train_dataset.label_mapping)
model.classifier[3] = nn.Linear(1024, num_classes)

# Переносим модель на GPU если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Настраиваем оптимизатор и функцию потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 9:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/10:.3f}')
            running_loss = 0.0

# Сохраняем обученную модель
torch.save(model.state_dict(), 'gesture_s3d_model.pth')