import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):

        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
         
    def forward(self, x):
        x = x.mm(self.w)
        return x 
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.conv1_sigma = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_gamma = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_sigma = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_gamma = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3_sigma = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
        self.conv3_gamma = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    def forward(self, x):
        out_sigma = self.conv1_sigma(x)
        out_gamma = self.conv1_gamma(x)
        out = out_sigma + out_gamma
        out = F.relu(self.bn1(out))

        out_sigma = self.conv2_sigma(out)
        out_gamma = self.conv2_gamma(out)
        out = out_sigma + out_gamma
        out = self.bn2(out)

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            out_sigma = self.conv3_sigma(x)
            out_gamma = self.conv3_gamma(x)
            out1 = out_sigma + out_gamma
            out1 = self.bn3(out1)
            out += out1
        else:
            out += x

        out = F.relu(out)
        return out

    def predict(self, x):
        out = self.conv1_sigma(x)
        out = F.relu(self.bn1(out))

        out = self.conv2_sigma(out)
        out = self.bn2(out)

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            out1 = self.conv3_sigma(x)
            out1 = self.bn3(out1)
            out += out1
        else:
            out += x

        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.conv1_sigma = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1_gamma = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_sigma = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2_gamma = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3_sigma = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.conv3_gamma = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # short_cut
        self.conv4_sigma = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        self.conv4_gamma = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        self.bn4 = nn.BatchNorm2d(self.expansion*planes)


    def forward(self, x):

        out_sigma = self.conv1_sigma(x)
        out_gamma = self.conv1_gamma(x)
        out = out_sigma + out_gamma
        out = F.relu(self.bn1(out))

        out_sigma = self.conv2_sigma(out)
        out_gamma = self.conv2_gamma(out)
        out = out_sigma + out_gamma
        out = F.relu(self.bn2(out))

        out_sigma = self.conv3_sigma(out)
        out_gamma = self.conv3_gamma(out)
        out = out_sigma + out_gamma
        out = self.bn3(out)

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            out_sigma = self.conv4_sigma(x)
            out_gamma = self.conv4_gamma(x)
            out1 = out_sigma + out_gamma
            out1 = self.bn4(out1)
            out += out1
        else:
            out += x

        out = F.relu(out)

        return out

    def predict(self, x):

        out = self.conv1_sigma(x)
        out = F.relu(self.bn1(out))

        out = self.conv2_sigma(out)
        out = F.relu(self.bn2(out))

        out = self.conv3_sigma(out)
        out = self.bn3(out)
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            out1 = self.conv4_sigma(x)
            out1 = self.bn4(out1)
            out += out1
        else:
            out += x
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channel, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1_sigma = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv1_gamma = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_sigma = nn.Linear(512 * block.expansion, num_classes)
        self.linear_gamma = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.float()
        out_sigma = self.conv1_sigma(x)
        out_gamma = self.conv1_gamma(x)
        out = out_sigma + out_gamma
        out = F.relu(self.bn1(out))

        for i in range(len(self.layer1)):
            out = self.layer1[i](out)
        for i in range(len(self.layer2)):
            out = self.layer2[i](out)
        for i in range(len(self.layer3)):
            out = self.layer3[i](out)
        for i in range(len(self.layer4)):
            out = self.layer4[i](out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out_sigma = self.linear_sigma(out)
        out_gamma = self.linear_gamma(out)
        out = out_sigma + out_gamma
        return out

    def predict(self, x):
        x = x.float()
        out = self.conv1_sigma(x)
        out = F.relu(self.bn1(out))

        for i in range(len(self.layer1)):
            out = self.layer1[i].predict(out)
        for i in range(len(self.layer2)):
            out = self.layer2[i].predict(out)
        for i in range(len(self.layer3)):
            out = self.layer3[i].predict(out)
        for i in range(len(self.layer4)):
            out = self.layer4[i].predict(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear_sigma(out)
        return out



def ResNet18(input_channel, num_classes):
    return ResNet(BasicBlock, [2,2,2,2], input_channel, num_classes)

def ResNet34(input_channel, num_classes):
    return ResNet(BasicBlock, [3,4,6,3], input_channel, num_classes)

def ResNet50(input_channel, num_classes):
    return ResNet(Bottleneck, [3,4,6,3], input_channel, num_classes)

def ResNet101(input_channel, num_classes):
    return ResNet(Bottleneck, [3,4,23,3], input_channel, num_classes)

def ResNet152(input_channel, num_classes):
    return ResNet(Bottleneck, [3,8,36,3], input_channel, num_classes)

# if __name__ == '__main__':
