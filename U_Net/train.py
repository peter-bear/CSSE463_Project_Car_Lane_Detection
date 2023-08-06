import numpy as np

from data.MyDataSet import MyDataSet
from unet.unet_model import UNet
from torch.utils.data import DataLoader
import torch
import visdom
import torch.optim as optim
import torch.nn as nn

def train_net(net, device, train_loader, optimizer, epochIdx, criterion, visual):
    net.train()
    total_loss = 0
    indexes = np.array([])
    epoch_loss = np.array([])
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device=device, dtype=torch.float32), targets.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        # predict the output
        output = net(images)
        # calculate the loss
        loss = criterion(output, targets)
        # back forward to update parameters
        loss.backward()
        optimizer.step()
        indexes.append(indexes, [batch_idx])
        epoch_loss.append(epoch_loss, [loss.item()])
        visual.line(X=indexes, Y=epoch_loss, opts=dict(showlegend=True))
        # print("Loss: "+loss.item())
        # total_loss += loss.item()
    # print(f"Epoch [{epochIdx}], Loss: {total_loss/len(train_loader):.4f}")



if __name__ == '__main__':
    # prepare data set
    trainSet = MyDataSet("./data/SDLane/train/", "./data/SDLane/train/train_list.txt")
    testSet = MyDataSet("./data/SDLane/test/", "./data/SDLane/test/test_list.txt")
    train_loader = DataLoader(dataset=trainSet, batch_size=20, shuffle=True)
    # test_loader = DataLoader(dataset=testSet, batch_size=20, shuffle=True)

    # prepare net parameters
    net = UNet(n_channels=1, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    # optimizer and loss function
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    # train network
    visual = visdom.Visdom()
    train_net(net, device, train_loader, optimizer, 0, criterion, visual)