import torch
import torch.nn as nn
import torch.nn.functional as F

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """
    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)

#the sub-branch used to transform the 16x features into pred_semantic
class layer1_for_LR_Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layer1_for_LR_Branch, self).__init__()

        self.conv1 = Conv2dIBNormRelu(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dIBNormRelu(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2dIBNormRelu(out_channels, 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)
    
    def forward(self, x, inference):
        #x: features of 16x
        #lr16x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv1(x)
        lr8x = F.interpolate(lr16x, scale_factor=2.0, mode='bilinear', align_corners=False)
        lr8x = self.conv2(lr8x)

        pred_semantic = torch.tensor([])
        if not inference:
            lr = self.conv3(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x

#the sub-branch used to transform the 8x features into pred_semantic
class layer2_for_LR_Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layer2_for_LR_Branch, self).__init__()

        self.conv1 = Conv2dIBNormRelu(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dIBNormRelu(out_channels, 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)
    
    def forward(self, x, inference):
        #x: features of 8x
        #lr8x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv1(x)

        pred_semantic = torch.tensor([])
        if not inference:
            lr = self.conv2(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x
     
#the sub-branch used to transform the 4x features into pred_detail
class layer1_for_HR_Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layer1_for_HR_Branch, self).__init__()

        self.conv1 = Conv2dIBNormRelu(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dIBNormRelu(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2dIBNormRelu(out_channels+3, 1, kernel_size=3, stride=1, padding=1, with_ibn=False, with_relu=False)
    
    def forward(self, x, img, inference):
        #x: features of 4x
        #lr16x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv1(x)
        hr2x = F.interpolate(hr4x, scale_factor=2.0, mode='bilinear', align_corners=False)
        hr2x = self.conv2(hr2x)
        pred_detail = torch.tensor([])
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2.0, mode='bilinear', align_corners=False)
            hr = self.conv3(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x

#the sub-branch used to transform the 2x features into pred_detail
class layer2_for_HR_Branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layer2_for_HR_Branch, self).__init__()

        self.conv1 = Conv2dIBNormRelu(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dIBNormRelu(out_channels+3, 1, kernel_size=3, stride=1, padding=1, with_ibn=False, with_relu=False)
    
    def forward(self, x, img, inference):
        #x: features of 16x
        #lr8x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv1(x)

        pred_detail = torch.tensor([])
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2.0, mode='bilinear', align_corners=False)
            hr = self.conv2(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x
