import matplotlib.pyplot as plt
from nn import ConvNeuralNet
from utils.env import ZIP_DIR_PATH, ZIP_FILE_NAME, EXTRACT_DIR_NAME, DEVICE
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from dataloaders import train_loader, test_loader, val_loader

def main():
    cnn_params = [
        {
            "in_features": 3,
            "out_features": 64,
            "kernel_size": 2,
            "stride": 1,
            "padding": 1
        },
        {
            "in_features": 64,
            "out_features": 128,
            "kernel_size": 2,
            "stride": 1,
            "padding": 1
        },
        {
            "in_features": 128,
            "out_features": 256,
            "kernel_size": 2,
            "stride": 1,
            "padding": 1
        },
        {
            "in_features": 256,
            "out_features": 512,
            "kernel_size": 2,
            "stride": 1,
            "padding": 1
        },
        {
            "in_features": 512,
            "out_features": 1024,
            "kernel_size": 2,
            "stride": 1,
            "padding": 1
        }
    ]
    out_features_fc1 = 2048
    model = ConvNeuralNet(1e-2, 40, nn.ReLU(), nn.CrossEntropyLoss(), cnn_params, out_features_fc1, DEVICE).to(DEVICE)
    model.print_description()
    print(model)
    model.fit(train_loader, val_loader) 
    
if __name__ == "__main__":
    main()
