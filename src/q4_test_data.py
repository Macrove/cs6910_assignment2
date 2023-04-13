import matplotlib.pyplot as plt
import wandb
from dataloaders import test_loader
from utils.env import BEST_MODEL
import torch
import numpy as np
from params.default_params import default_credentials
model = torch.load(BEST_MODEL)

loss, acc = model.test(test_loader)
print("Testing")
print("Loss: ", loss, "Accuracy: ", acc)
wandb.init(project=default_credentials["wandb_project"], entity=default_credentials["wandb_entity"])

prediction_samples = [[], [], [], [], [], [], [], [], [], []]
classes = test_loader.dataset.classes

table = wandb.Table(columns=["Actual Label", "Images" ])


for data in test_loader:
    
    img, lbl = data
    lbl = lbl.item()
    if len(prediction_samples[lbl]) != 3:
        pred = model.predict(img)
        pred = pred.item()
        img = img[0]
        img = img.permute(1,2,0)
        img = img.numpy()
        np.clip(img, 0.001, 0.999)
        prediction_samples[lbl].append((img, pred))

for code, clss in enumerate(classes):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    for i in range(3):
        ax[i].imshow(prediction_samples[code][i][0])
        ax[i].set_title("Predicted class: {}".format(classes[prediction_samples[code][i][1]]), fontsize=9)
        ax[i].axis('off')
            
    table.add_data(clss, wandb.Image(fig))
    fig.savefig("{}.png".format(clss))

wandb.log({"Classification grid": table})