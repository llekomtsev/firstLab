import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from tqdm import tqdm
import os
from PIL import Image
import time

# Пути к данным
train_annos_file = r"D:\CarDatasets\cars_train_annos.mat"
test_annos_file = r"D:\CarDatasets\cars_test_annos_withlabels_eval.mat"
meta_file = r"D:\CarDatasets\cars_meta.mat"
train_images_dir = r"D:\CarDatasets\cars_train\cars_train"
test_images_dir = r"D:\CarDatasets\cars_test\cars_test"


# Загрузка аннотаций и метаданных
def load_annotations(annotations_file):
    data = loadmat(annotations_file)
    annotations = data['annotations'][0]
    images_labels = [
        (str(annotation['fname'][0]), int(annotation['class'][0].item()) - 1)
        for annotation in annotations
    ]
    return images_labels


def load_meta(meta_file):
    meta_data = loadmat(meta_file)
    class_names = np.array([item[0] for item in meta_data['class_names'][0]], dtype=str)
    return class_names


# Кастомный Dataset
class CarDataset(Dataset):
    def __init__(self, annotations, images_dir, transform=None):
        self.annotations = annotations
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, label = self.annotations[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(Image.fromarray(image))
        return image, torch.tensor(label, dtype=torch.long)


# Гиперпараметры
batch_size = 64
num_epochs = 10
learning_rate = 0.0001

# Загрузка данных
train_annotations = load_annotations(train_annos_file)
test_annotations = load_annotations(test_annos_file)
class_names = load_meta(meta_file)

# Подготовка данных
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CarDataset(train_annotations, train_images_dir, transform_train)
test_dataset = CarDataset(test_annotations, test_images_dir, transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Модель SqueezeNet
device = torch.device('cpu')
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1))
model.num_classes = len(class_names)
model = model.to(device)

# Функция потерь и оптимизаторы
criterion = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
optimizer_amsgrad = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)


# Функция тренировки модели с фиксированным прогресс-баром
def train_model(optimizer, name):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        print(f"\nЭпоха {epoch + 1}/{num_epochs} ({name}):")

        pbar = tqdm(train_loader, desc="Обучение", leave=True, dynamic_ncols=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()  # Закрытие прогресс-бара после завершения эпохи
        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Эпоха завершена: Потеря = {avg_loss:.4f} | Время = {elapsed_time:.2f} сек.")


# Функция для оценки модели на тестовом наборе
def evaluate_model(name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Точность на тестовом наборе ({name}): {accuracy:.2f}%")
    return accuracy


# Тренировка и замер времени для Adam и AmsGrad
if __name__ == "__main__":
    print("=" * 40)
    print(f"Классов автомобилей: {len(class_names)}")
    print(f"Пример класса: {class_names[0]}")
    print("=" * 40)

    print("Тренировка с Adam...")
    train_model(optimizer_adam, "Adam")
    evaluate_model("Adam")

    print("\nТренировка с AmsGrad...")
    train_model(optimizer_amsgrad, "AmsGrad")
    evaluate_model("AmsGrad")
