import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *



def SelectModel(m):
    
    if m == 'Resnet20':
        return ResNet(BasicBlock, [3, 3, 3])
    elif m == 'Resnet32':
        return ResNet(BasicBlock, [5, 5, 5])
    elif m == 'ResnetV2-20':
        return ResNet(PreActBasicBlock, [3, 3, 3])
    elif m == 'ResnetV2-32':
        return ResNet(PreActBottleneck, [5, 5, 5])
    




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        
        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)


                
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.layer1(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = F.avg_pool2d(o, o.size()[3])
        o = o.view(o.size(0), -1)
        o = self.linear(o)

        return o
    
    


# class ResnetV2(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(self, ResnetV2).__init__()
#         self.inplanes = 16

#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[0], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[0], stride=2)
#         self.linear = nn.Linear(64, num_classes)
        

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers=[]
#         for stride in strides:
#             layers.append(block(self.inplanes, planes, stride))
#             self.inplanes = planes*block.expansion
        
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         o = F.relu(self.bn1(self.conv1(x)))
#         o = self.layer1(o)
#         o = self.layer2(o)
#         o = self.layer3(o)
#         o = F.avg_pool2d(o, o.size()[3])
#         o = o.view(o.size(0), -1)
#         o = self.linear(o)

#         return o
