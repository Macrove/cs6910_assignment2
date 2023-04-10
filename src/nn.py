import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class ConvNeuralNet(nn.Module):
    def __init__(self, learning_rate, epochs, activation, loss_fn, cnn_params, out_features_fc1, DEVICE):
        
        super(ConvNeuralNet, self).__init__()

        self.device = DEVICE
        print("Device in use: ", self.device)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.loss_fn = loss_fn
        self.cnn_params = cnn_params

        self.pool = nn.MaxPool2d(2, 2)
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(cnn_params[0]["in_features"], cnn_params[0]["out_features"], cnn_params[0]["kernel_size"], 
                      cnn_params[0]["stride"], cnn_params[0]["padding"], device=self.device),
            self.activation,
            nn.Dropout(0.3),
            self.pool,
            nn.Conv2d(cnn_params[1]["in_features"], cnn_params[1]["out_features"], cnn_params[1]["kernel_size"], 
                      cnn_params[1]["stride"], cnn_params[1]["padding"], device=self.device),
            self.activation,
            nn.Dropout(0.3),
            self.pool,
            nn.Conv2d(cnn_params[2]["in_features"], cnn_params[2]["out_features"], cnn_params[2]["kernel_size"], 
                      cnn_params[2]["stride"], cnn_params[2]["padding"], device=self.device),
            self.activation,
            nn.Dropout(0.3),
            self.pool,
            nn.Conv2d(cnn_params[3]["in_features"], cnn_params[3]["out_features"], cnn_params[3]["kernel_size"], 
                      cnn_params[3]["stride"], cnn_params[3]["padding"], device=self.device),
            self.activation,
            nn.Dropout(0.3),
            self.pool,
            nn.Conv2d(cnn_params[4]["in_features"], cnn_params[4]["out_features"], cnn_params[4]["kernel_size"], 
                      cnn_params[4]["stride"], cnn_params[4]["padding"], device=self.device),
            self.activation,
            nn.Dropout(0.3),
            self.pool,
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(cnn_params[4]["out_features"] * 8 * 8, out_features_fc1).to(self.device),
            nn.Dropout(0.3),
            self.activation,
            nn.Linear(out_features_fc1, 10).to(self.device),
            nn.Softmax(-1)
        )
        self.cnn_stack.to(device=self.device)
        self.fc_stack.to(device=self.device)
        # self.optimizer = optim.Adam(self.parameters(), self.learning_rate, betas=(0.7, 0.7))
        self.optimizer = optim.SGD(self.parameters(), self.learning_rate)

        
    def forward(self, x):
        x = self.cnn_stack(x)
        # batch_size = x.shape[0]
        # x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        # x = x.reshape(batch_size, -1)
        # print(x.shape)
        x = x.view(-1, self.cnn_params[4]["out_features"] * 8 * 8)
        # print(x.shape)
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
            valid_epoch_loss, valid_epoch_acc = self.validate_loop(val_loader)
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f} %")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f} %")
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