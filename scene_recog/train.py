import os
import time

# import matplotlib.pyplot as plt
# import onnx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =============== BLOCKS =============== #
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class NoShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(NoShortcutBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


# =============== RES NET =============== #
class NoResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(NoResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(NoShortcutBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(NoShortcutBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(NoShortcutBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(NoShortcutBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, classes_count):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes_count)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResNet152(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet152, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


# =============== DATASET CLASS =============== #
class ImageDataset(Dataset):
    def __init__(self, data_folder, labels_csv, image_transform):
        self.data_folder = data_folder
        self.csv_data = pd.read_csv(labels_csv)
        self.image_transform = image_transform

    def __getitem__(self, index):
        image_label = int(self.csv_data.iloc[index, 1])
        image_path = os.path.join(self.data_folder, self.csv_data.iloc[index, 0])

        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        return image, image_label

    def __len__(self):
        return len(self.csv_data)


# =============== DATA TRANSFORM AND DATASET READING =============== #
resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_main_transform = transforms.Compose([resize, to_tensor, normalize])

rotation = transforms.RandomRotation(15)
erasing = transforms.RandomErasing(p=0.5)
image_additional_transform = transforms.Compose([resize, rotation, to_tensor, erasing])

train_data_path = r'data2/train-scene classification/train/'
train_labels_path = r'data2/train-scene classification/train.csv'
main_dataset = ImageDataset(train_data_path, train_labels_path, image_main_transform)
additional_dataset = ImageDataset(train_data_path, train_labels_path, image_additional_transform)

# =============== DIVISION TO TRAIN AND TEST =============== #
total_len = len(main_dataset)
train_len = int(total_len * 0.8)
test_length = total_len - train_len

train_set, val_set = torch.utils.data.random_split(main_dataset, [train_len, test_length])
train_set = torch.utils.data.ConcatDataset([train_set, additional_dataset])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# =============== MODELS CREATION =============== #
models = [
    {'model': NoResNet18(6), 'model_name': 'no_18_resnet'},
    {'model': ResNet18(6), 'model_name': '18_resnet'},
    {'model': ResNet152(6), 'model_name': '152_resnet'},
]

for model_data in models:
    model_name = model_data['model_name']
    model = model_data['model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model_name, device)

    # =============== LEARNING =============== #
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    num_epochs = 9
    print("Starting on {} epochs".format(num_epochs))
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        learning_loss = 0.0
        correct_predict_count = 0
        total_predict_count = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            learning_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_predict_count += labels.size(0)
            correct_predict_count += (predicted == labels).sum().item()

        train_accuracy = (correct_predict_count / total_predict_count) * 100
        test_accuracy = 0
        correct_val_predictions = 0
        total_val_samples = 0
        model.eval()
        with torch.no_grad():
            for inputs_val, labels_val in test_loader:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                outputs_val = model(inputs_val)
                _, predicted_val = torch.max(outputs_val, 1)
                total_val_samples += labels_val.size(0)
                correct_val_predictions += (predicted_val == labels_val).sum().item()
        test_accuracy = (correct_val_predictions / total_val_samples) * 100

        train_losses.append(learning_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        test_loss = learning_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            "Epoch {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}".format(
                epoch + 1, learning_loss / len(train_loader), train_accuracy, test_loss, test_accuracy
            )
        )
    print(f"Time: {time.time() - start_time} sec")
    

    # =============== LOSS GRAPHIC =============== #
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    # plt.title(f'Training / Test Loss Over Epochs ({model_name})')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # =============== ACCURACY GRAPHIC =============== #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    # plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    # plt.title(f'Training / Test Accuracy Over Epochs ({model_name})')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # =============== NORMAL SAVING =============== #
    input_data = torch.randn(1, 3, 224, 224).to(device)
    path_to_save = model_name + ".onnx"
    torch.onnx.export(model, input_data, path_to_save, verbose=True, input_names=['input'], output_names=['output'])
    print("Normal: Exported to ONNX and saved into ", path_to_save)
