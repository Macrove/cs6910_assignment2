import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from utils.maps import activation_map, loss_map

class ConvNeuralNet(nn.Module):
    def __init__(self, cnn_params, out_features_fc1, dropout, loss, learning_rate, optimizer, activation, epochs, batch_normalisation, DEVICE, use_wandb):
        
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
        self.batch_norm = nn.BatchNorm2d(3)
        self.use_wandb = use_wandb

        self.pool = nn.MaxPool2d(2, 2)
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(cnn_params[0]["in_features"], cnn_params[0]["out_features"], cnn_params[0]["kernel_size"], 
                      cnn_params[0]["stride"], cnn_params[0]["padding"], device=self.device),
            self.activation,
            self.pool,
            nn.Conv2d(cnn_params[1]["in_features"], cnn_params[1]["out_features"], cnn_params[1]["kernel_size"], 
                      cnn_params[1]["stride"], cnn_params[1]["padding"], device=self.device),
            self.activation,
            self.pool,
            nn.Conv2d(cnn_params[2]["in_features"], cnn_params[2]["out_features"], cnn_params[2]["kernel_size"], 
                      cnn_params[2]["stride"], cnn_params[2]["padding"], device=self.device),
            self.activation,
            self.pool,
            nn.Conv2d(cnn_params[3]["in_features"], cnn_params[3]["out_features"], cnn_params[3]["kernel_size"], 
                      cnn_params[3]["stride"], cnn_params[3]["padding"], device=self.device),
            self.activation,
            self.pool,
            nn.Conv2d(cnn_params[4]["in_features"], cnn_params[4]["out_features"], cnn_params[4]["kernel_size"], 
                      cnn_params[4]["stride"], cnn_params[4]["padding"], device=self.device),
            self.activation,
            self.pool,
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(cnn_params[4]["out_features"] * 8 * 8, out_features_fc1).to(self.device),
            nn.Dropout(self.dropout),
            self.activation,
            nn.Linear(out_features_fc1, 10).to(self.device),
            nn.Softmax(-1)
        )
        self.cnn_stack.to(device=self.device)
        self.fc_stack.to(device=self.device)
        # self.optimizer = optim.Adam(self.parameters(), self.learning_rate, betas=(0.7, 0.7))
        self.optimizer = optim.SGD(self.parameters(), self.learning_rate)

        
    def forward(self, x):
        if self.batch_normalisation:
            x = self.batch_norm(x)
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
            print(f"[INFO]: Epoch {epoch+1} of {epoch}")
            train_epoch_loss, train_epoch_acc = self.train_loop(train_loader)
            val_epoch_loss, val_epoch_acc = self.validate_loop(val_loader)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f} %")
            print(f"Validation loss: {val_epoch_loss:.3f}, validation acc: {val_epoch_acc:.3f} %")
            if self.use_wandb:
                wandb.log({
                    "train_acc" : train_epoch_acc,
                    "val_acc" : val_epoch_acc,
                    "train_loss" : train_epoch_loss,
                    "val_loss" : val_epoch_loss
                })
            print('_________________________________________________')

        print('TRAINING COMPLETE')

    def train_loop(self, trainloader):
        self.train()
        print('Training')
        net_loss = 0
        true_pos = 0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader), mininterval=0.5):
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