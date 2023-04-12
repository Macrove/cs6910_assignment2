import torch
import torchvision
from dataloaders import train_loader, val_loader, test_loader
from tqdm import tqdm
DEVICE = 'cuda' if torch.cuda.is_available() else 'gpu'

model = torch.hub.load('pytorch/vision', 'resnet152', pretrained = True)

for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

model.fc = torch.nn.Linear(2048, 10)

#printing model decription
total_params = sum(p.numel() for p in model.parameters())
print("total parameters: ", total_params)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("training parameters: ", total_trainable_params)
print(model)

#defining loss and optimizers
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

model.to(DEVICE)
n_epochs = 10
for epoch in range(n_epochs):
    # Train
    model.train()
    true_pos = 0
    for _, (imgs, lbls) in tqdm(enumerate(train_loader), total = len(train_loader)):
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, lbls)
        loss.backward()
        optimizer.step()

        preds = torch.max(logits, 1)[1]
        true_pos += (preds == lbls).sum().item()
    train_acc = true_pos / len(train_loader.dataset) * 100
    print('Epoch [{}/{}], Loss: {:.3f}, Train acc: {:.3f} %'
            .format(epoch+1, n_epochs, loss.item(), train_acc))

    # Val
    model.eval()
    with torch.no_grad():
        true_pos = 0
        total = 0
        for _, (imgs, lbls) in tqdm(enumerate(val_loader), total = len(val_loader)):
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)
            logits = model(imgs)
            preds = torch.max(logits, 1)[1]
            true_pos += (preds == lbls).sum().item()

        val_acc = true_pos / len(val_loader.dataset) * 100
        print('Epoch [{}/{}], Loss: {:.3f}, Val acc: {:.3f} %'
              .format(epoch+1, n_epochs, loss.item(), val_acc))
