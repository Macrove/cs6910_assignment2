from typing import List
import os
import zipfile
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
def load_dataset(dir_path, zip_file_name, extract_dir_name, extract = False):
    if extract:
        with zipfile.ZipFile(os.path.join(dir_path, zip_file_name), 'r') as dataset:
            dataset.extractall(dir_path)
    
    dataset_path = os.path.join(dir_path, extract_dir_name)
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"))
    val_dataset = ImageFolder(os.path.join(dataset_path, "train"))
    print(train_dataset.shape)
    print(train_dataset[0:5])
    # print(os.getcwd())
#     CATEGORIES = os.listdir(os.path.join(dataset_path, "train"))

#     if CATEGORIES[0] == ".DS_Store":
#         CATEGORIES = CATEGORIES[1:]
    
#     x_train, y_train = get_data("train", CATEGORIES, dataset_path)
#     x_test, y_test = get_data("val")

#     return x_train, y_train, x_test, y_test

# def get_data(DATASET_TYPE: str, CATEGORIES: List[str], dataset_path: str):
#     x_data, y_data = [], []
#     for cat in CATEGORIES:
#         cat_dir_path = os.path.join(dataset_path, DATASET_TYPE, cat)
#         imgs = os.listdir(cat_dir_path)
#         if imgs[0] == ".DS_Store":
#             imgs = imgs[1:]
        
#         for img in imgs:
#             img_path = os.path.join(cat_dir_path, img)
#             x_data.append(read_image(img_path))
#             y_data.append(cat)
    
#     return x_data, y_data
                
        
    
        
        

