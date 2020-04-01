import torch
import torch.nn as nn

#old model used in the first paper (fruit)
class fruit3conv(nn.Module):
    def __init__(self):
        super(fruit3conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        #self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        #self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        #self.fc2 = nn.Linear(64, 32)
        self.fc = nn.Linear(128, 6)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        #x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) #x = x.view(-1, 16 * 5 * 5)        
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc(x)
        #x = self.softmax(x)
        return x

class fruit5conv(nn.Module):
    def __init__(self):
        super(fruit5conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) #128x128
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)#64x64
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)#32x32
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)#16x16
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)#8x8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) #x = x.view(-1, 16 * 5 * 5)        
        x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x


class fruit2conv(nn.Module):
    def __init__(self):
        super(fruit2conv, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        #self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(4, 4) #64x64
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(4, 4) #32x32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) #x = x.view(-1, 16 * 5 * 5)        
        x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False) #same as resnet except for number of filters (64) --> 128x128
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        #self.fc2 = nn.Linear(64, 32)
        self.fc = nn.Linear(128, 8)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) #x = x.view(-1, 16 * 5 * 5)        
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc(x)
        #x = self.softmax(x)
        return x


class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False) #same as resnet except for number of filters (64) --> 128x128
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(16, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        #self.fc2 = nn.Linear(64, 32)
        self.fc = nn.Linear(128, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) #x = x.view(-1, 16 * 5 * 5)        
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc(x)
        #x = self.softmax(x)
        return x
