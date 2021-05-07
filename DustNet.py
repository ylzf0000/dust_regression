import torch.nn as nn
import torch.nn.functional as F

'''
1.输入非固定尺寸图像
2.两个输出，一个分类，一个回归
'''

class DustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 9, 3)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(9, 3)
        self.softmax = nn.Softmax(3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.globalPool(x)
        x = x.view(-1, 9)

        y_class = F.relu(self.fc1(x))
        y_class = self.softmax(y_class)

        y_reg = F.sigmoid(x)
        return y_class, y_reg
if __name__ == '__main__':
    net = DustNet()
    y1, y2 = net()
