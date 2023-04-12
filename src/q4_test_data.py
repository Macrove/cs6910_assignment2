from dataloaders import test_loader
from utils.env import BEST_MODEL
import torch
from nn import ConvNeuralNet
from utils.prepare_params import get_cnn_params
from params.default_params import default_model_params


################TO BE FIXED#######################:wq


model_params = torch.load(BEST_MODEL)
cnn_params = get_cnn_params()
model = ConvNeuralNet(cnn_params, out_features_fc1, dropout, loss, learning_rate, 
                          optimizer["name"], optimizer["default_params"], activation, 
                          epochs, batch_normalisation, DEVICE, use_wandb)


loss, acc = model.test(test_loader)
print("Testing")
print("Loss: ", loss, "Accuracy: ", acc)


# model.predict()