from nn import ConvNeuralNet
from utils.env import DEVICE
from dataloaders import train_loader, test_loader, val_loader

def main(epochs, activation, cnn_params, out_features_fc1, dropout, loss, learning_rate, optimizer, batch_normalisation, init, use_wandb):
    model = ConvNeuralNet(cnn_params, out_features_fc1, dropout, loss, learning_rate, 
                          optimizer["name"], optimizer["default_params"], activation, 
                          epochs, batch_normalisation, init, DEVICE, use_wandb).to(DEVICE)
    model.print_description()
    print(model)
    model.fit(train_loader, val_loader) 
    