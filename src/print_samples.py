from main import main
from dataloaders import test_loader

def print_samples(model):
    model.test(test_loader)