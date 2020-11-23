from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from dataloader import Image_Loader
from cbam import *

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride = 1, padding = 1) 
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.cbam = CBAM(32, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1152, 6) # stride 1: 2304, 2:512
        self.fc2 = nn.Linear(512, 5)
        self.fc3 = nn.Linear(512, 6)


    def forward(self, x):
        x = self.conv1(x) #48.48.8     #32.32.8      #64.64.8
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x) #24.24.16     #16.16.16   #32.32.16
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x) #12.12.32     #8.8.32     #16.16.32
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.cbam(x) 
        # x = self.conv4(x) #6.6.64     #4.4.64     #8.8.64
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2) #3.3.64  #2.2.64
        # x = self.conv5(x)       # 3.3.128
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2) 

        x = torch.flatten(x, 1) 
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)

        # x = F.leaky_relu(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Default value
    batch_size = 16

    # Load data for training
    img_size =[48,48]
    train_data = Image_Loader(root_path='./data_train.csv',image_size= img_size, transforms_data=True)
    # Load data for testing
    test_data = Image_Loader(root_path='./data_test.csv',image_size= img_size,   transforms_data=True)

    total_train_data = len(train_data)
    total_test_data = len(test_data)

    #print(train_data, total_test_data)

    # Generate the batch in each iteration for training and testing
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    model = Network().to(device)
    print(model)
    summary(model, (3, 48, 48)) #summary(your_model, input_size=(channels, H, W))

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # if args.save_model:
    torch.save(model.state_dict(), "cnn_8.pt")


if __name__ == '__main__':
    main()
