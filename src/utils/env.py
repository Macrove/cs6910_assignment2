import torch.cuda as cuda
ZIP_DIR_PATH = "./src/dataset/"
ZIP_FILE_NAME = "nature_12K.zip"
EXTRACT_DIR_NAME = "inaturalist_12K"
DEVICE = 'cuda' if cuda.is_available() else 'gpu'
BEST_MODEL = "train_34.65_test_33.26663331665833.pth"