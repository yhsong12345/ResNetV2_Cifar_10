import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), 'constant', 0))
            if option == 'B':
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )

    
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        o += self.shortcut(x)
        o = F.relu(o)

        return o



class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )
        
    
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.bn3(self.conv3(o))
        o += self.shortcut(x)
        o = F.relu(o)
        return o



class PreActBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(PreActBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), 'constant', 0))
            if option == 'B':
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )

    
    def forward(self, x):
        o = self.conv1(F.relu(self.bn1(x)))
        o = self.conv2(F.relu(self.bn2(o)))
        o += self.shortcut(x)

        return o


class PreActBottleneck(nn.Module):
    expansion=4
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )
        
    
    def forward(self, x):
        o = F.relu(self.bn1(x))
        o = F.relu(self.bn2(self.conv1(o)))
        o = self.bn3(self.conv2(o))
        o = self.conv3(o)
        o += self.shortcut(x)
        return o