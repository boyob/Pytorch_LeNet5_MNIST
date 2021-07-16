import torch


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net1 = torch.nn.Sequential(
            # 卷积块1
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(num_features=6),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积块2
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.net2 = torch.nn.Sequential(
            # 全连接层块1
            torch.nn.Linear(400, 120),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            # 全连接层块2
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            # 全连接层3
            torch.nn.Linear(84, 10)
        )

    def forward(self, data):
        temp = self.net1(data)
        temp = temp.view(temp.size(0), -1)
        return self.net2(temp)
