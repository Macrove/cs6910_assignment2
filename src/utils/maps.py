import torch.nn as nn
activation_map = {
    "ReLU": nn.ReLU(),
    "SiLU": nn.SiLU(),
    "GELU": nn.GELU(),
    "Mish": nn.Mish()
}

loss_map = {
    "cross_entropy": nn.CrossEntropyLoss()
}