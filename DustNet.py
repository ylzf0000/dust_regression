import torch.nn as nn
import torch.nn.functional as F
import torchvision
'''
1.输入固定尺寸图像
2.两个输出，一个分类，一个回归
'''

class DustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 9, 3)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(225, 75)
        self.fc2 = nn.Linear(75, 25)
        self.fc3 = nn.Linear(25, 5)
        self.fc4 = nn.Linear(1000, 3)
        self.softmax = nn.Softmax(1)
        self.mode = 'classify'

    def setClassify(self):
        self.mode = 'classify'

    def setRegression(self):
        self.mode = 'regression'

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc4(x)
        x = self.softmax(x)
        # print(f'x.shape: {x.shape}')
        return x
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # # print(f'x.shape: {x.shape}')
        # # x = self.globalPool(x)
        # x = x.view(-1, 225)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        # if self.mode == 'classify':
        #     y_class = self.softmax(x)
        #     return y_class
        # else:
        #     y_reg = F.sigmoid(x)
        #     return y_reg

if __name__ == '__main__':
    net = DustNet()
    y1, y2 = net()
