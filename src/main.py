import matplotlib.pyplot as plt
from nn import ConvNeuralNet
from utils.env import ZIP_DIR_PATH, ZIP_FILE_NAME, EXTRACT_DIR_NAME, DEVICE
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from dataloaders import train_loader, test_loader, val_loader
def main():
    model = ConvNeuralNet(DEVICE).to(DEVICE)
    model.print_description()
    print(model)
    # print(model.summary)
    model.fit(train_loader, val_loader) 
    
if __name__ == "__main__":
    main()
