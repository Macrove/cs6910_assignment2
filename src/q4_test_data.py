import matplotlib.pyplot as plt
from dataloaders import test_loader
from utils.env import BEST_MODEL
import torch
from nn import ConvNeuralNet
from utils.prepare_params import get_cnn_params
from params.default_params import default_model_params




model = torch.load(BEST_MODEL)

loss, acc = model.test(test_loader)
print("Testing")
print("Loss: ", loss, "Accuracy: ", acc)

prediction_samples = [[]]*10
for data in test_loader:
    
    img, lbl = data
    if len(prediction_samples[lbl]) != 3:
        pred = model.predict(img)
        prediction_samples[lbl].append((img, pred))

classes = test_loader.dataset.classes
print(classes, prediction_samples)

fig, ax = plt.subplots(10, 3, figsize=(15, 15))
for code, clss in enumerate(classes):

    ax[code].set_ylabel(clss)
    for i in range(3):
        ax[code][i].imshow(prediction_samples[code][0])
        ax[code].set_title(prediction_samples[code][1])
        ax[code][i].axis('off')
    

plt.show()