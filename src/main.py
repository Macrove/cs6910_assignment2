import os
import matplotlib.pyplot as plt
from utils.env import ZIP_DIR_PATH, ZIP_FILE_NAME, EXTRACT_DIR_NAME
from utils.prepare_dataset import load_dataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

def main():
    dataset_path = os.path.join(ZIP_DIR_PATH + EXTRACT_DIR_NAME)

    train_dataset_path = os.path.join(dataset_path, "train")
    test_dataset_path = os.path.join(dataset_path, "val")

    transform = transforms.ToTensor()

    train_dataset = ImageFolder(train_dataset_path, transform=transform)
    test_dataset = ImageFolder(test_dataset_path, transform=transform)
    
    img, lbl = train_dataset[100]
    plt.imshow(img.permute(2,1,0))
    plt.show()
    print(lbl)




    
if __name__ == "__main__":
    main()





















