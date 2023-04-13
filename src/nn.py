import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from utils.maps import activation_map, loss_map

class ConvNeuralNet(nn.Module):
    def __init__(self, cnn_params, out_features_fc1, dropout, loss, learning_rate, 
                 optimizer, optimizer_params, activation, epochs, batch_normalisation, 
                 init, DEVICE, use_wandb):
        
        super(ConvNeuralNet, self).__init__()

        self.device = DEVICE
        print("Device in use: ", self.device)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation_map[activation]
        self.loss_fn = loss_map[loss]
        self.cnn_params = cnn_params
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.use_wandb = use_wandb
        self.out_features_fc1 = out_features_fc1
        self.init = init

        self.pool = nn.MaxPool2d(2, 2)

        cnn_layers, fc_layers = self.prepare_layers()
        self.cnn_stack = nn.Sequential(*cnn_layers)
        self.fc_stack = nn.Sequential(*fc_layers)

        self.cnn_stack.to(device=self.device)
        self.fc_stack.to(device=self.device)

        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), **optimizer_params)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), **optimizer_params)


    def prepare_layers(self):
        print(self.cnn_params)
        cnn_layers = []
        for i in range(5):
            cnn_layers.append(
                nn.Conv2d(self.cnn_params[i]["in_features"], self.cnn_params[i]["out_features"], self.cnn_params[i]["kernel_size"], 
                    self.cnn_params[i]["stride"], self.cnn_params[i]["padding"])
            )
            if self.init == "xavier_normal":
                nn.init.xavier_normal_(cnn_layers[-1].weight)
            cnn_layers.append(self.activation)
            if self.batch_normalisation:
                cnn_layers.append(nn.BatchNorm2d(self.cnn_params[i]["out_features"]))
            cnn_layers.append(self.pool)

        fc_layers = []
        fc_layers.append(
            nn.Linear(self.cnn_params[4]["out_features"] * 8 * 8, self.out_features_fc1),
        )
        if self.batch_normalisation:
            fc_layers.append(
                nn.BatchNorm1d(self.out_features_fc1),
            )
        if self.dropout:
            fc_layers.append(
                nn.Dropout(self.dropout),
            )
        fc_layers.extend([
            self.activation,
            nn.Linear(self.out_features_fc1, 10),
            nn.Softmax(-1)
        ])
        return cnn_layers, fc_layers
        
    def forward(self, x):
        x = self.cnn_stack(x)
        x = x.view(-1, self.cnn_params[4]["out_features"] * 8 * 8)
        x = self.fc_stack(x)
        return x

    def print_description(self):
        total_params = sum(p.numel() for p in self.parameters())
        print("total parameters: ", total_params)
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("training parameters: ", total_trainable_params)

    def fit(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss, train_epoch_acc = self.train_loop(train_loader)
            val_epoch_loss, val_epoch_acc = self.validate_loop(val_loader)
            print(f"Training loss: {train_epoch_loss:.4f}, training acc: {train_epoch_acc:.4f} %")
            print(f"Validation loss: {val_epoch_loss:.4f}, validation acc: {val_epoch_acc:.4f} %")
            if self.use_wandb:
                wandb.log({
                    "train_acc" : train_epoch_acc,
                    "val_acc" : val_epoch_acc,
                    "train_loss" : train_epoch_loss,
                    "val_loss" : val_epoch_loss
                })
            print('_________________________________________________')
            # if train_epoch_acc > 25 and val_epoch_acc > 25:
            #     print("Training and Validation accuracies are greater than 30%.\nSaving model parameters")
            torch.save(self, "train_{}_val_{}.pth".format(round(train_epoch_acc, 3), round(val_epoch_acc, 3)))

        print('TRAINING COMPLETE')

    def train_loop(self, trainloader):
        self.train()
        print('Training')
        net_loss = 0
        true_pos = 0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, labels = data
            image = image.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            # forward pass
            outputs = self(image)
            # calculate the loss
            loss = self.loss_fn(outputs, labels)
            net_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            true_pos += (preds == labels).sum().item()
            # backpropagation
            loss.backward()
            # update the optimizer parameters
            self.optimizer.step()
    
        # loss and accuracy for the complete epoch
        epoch_loss = net_loss / len(trainloader.dataset)
        epoch_acc = 100 * (true_pos / len(trainloader.dataset))
        return epoch_loss, epoch_acc
    # validation
    def validate_loop(self, valloader):
        self.eval()
        print('Validation')
        net_loss = 0
        true_pos = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(valloader), total=len(valloader)):
            
                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = self(image)
                # calculate the loss
                loss = self.loss_fn(outputs, labels)
                net_loss += loss.item()
                # calculate the accuracy
                _, preds = torch.max(outputs.data, 1)
                true_pos += (preds == labels).sum().item()
        
        # loss and accuracy for the complete epoch
        epoch_loss = net_loss / len(valloader.dataset)
        epoch_acc = 100 * (true_pos / len(valloader.dataset))
        return epoch_loss, epoch_acc


    def test(self, testloader):
        self.eval()
        print('Test')
        net_loss = 0
        true_pos = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            
                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                # forward pass
                outputs = self(image)
                # calculate the loss
                loss = self.loss_fn(outputs, labels)
                net_loss += loss.item()
                # calculate the accuracy
                _, preds = torch.max(outputs.data, 1)
                true_pos += (preds == labels).sum().item()
        
        # loss and accuracy for the complete epoch
        epoch_loss = net_loss / len(testloader.dataset)
        epoch_acc = 100 * (true_pos / len(testloader.dataset))
        return epoch_loss, epoch_acc

    # predicts only for an instance
    def predict(self, x):
        self.eval()
        print('Test')
        with torch.no_grad():
            image = x.to(self.device)
            # forward pass
            outputs = self(image)
            _, pred = torch.max(outputs.data, -1)
        return pred
        
