from collections import Counter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import torch
import torchvision.transforms as transforms
from utils.env import ZIP_DIR_PATH, EXTRACT_DIR_NAME

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = os.path.join(ZIP_DIR_PATH + EXTRACT_DIR_NAME)

train_dataset_path = os.path.join(dataset_path, "train")
test_dataset_path = os.path.join(dataset_path, "val")

# the training transforms
train_transform = transforms.Compose([
    transforms.CenterCrop((256, 256)),
    transforms.RandomRotation(degrees=(40, 75)),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# the test transforms
test_transform = transforms.Compose([
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
train_dataset = ImageFolder(train_dataset_path, transform=train_transform)
test_dataset = ImageFolder(test_dataset_path, transform=test_transform)
    
train_split, val_split = random_split(train_dataset, [0.8, 0.2])

print("\nTrain samples:\t", len(train_split))
print("Validation samples:\t", len(val_split)) 
print("Test samples:\t", len(test_dataset)) 
print("\nImage Size:\t", train_split[0][0].shape)

labels = train_dataset.classes

value_counts = Counter(train_dataset.targets)

print("\nTraining Sample Distribution")
for i in range(10):
    print(labels[i], " : ", value_counts[i])
print('\n')

train_loader = DataLoader(train_split,8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_split, 8, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, 1, num_workers=4, pin_memory=True)