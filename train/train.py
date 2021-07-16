import torch
from torch.autograd import Variable
from dataLoader import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from LeNet5 import LeNet
from torchvision import transforms

mytransform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    # transforms.Normalize((0.1307,), (0.3081,))
])


def train(epoch):
    log_interval = 100
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)

        output = model(data)
        test_loss += torch.nn.functional.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
          format(test_loss, correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    root = '../mnistImgs/'

    train_data = MyDataset(root, txt='train.txt', transform=mytransform)
    test_data = MyDataset(root, txt='test.txt', transform=mytransform)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64)

    model = LeNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    epochs = 10
    for epoch in range(1, epochs+1):
        train(epoch)
        test()

    torch.save(model, 'LeNet5_MNIST_parameter_and_model.pth')
    torch.save(model.state_dict(), 'LeNet5_MNIST_parameter.pth')
