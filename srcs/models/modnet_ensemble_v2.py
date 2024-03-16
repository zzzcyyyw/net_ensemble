import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SUPPORTED_BACKBONES
from .layers import IBNorm, Conv2dIBNormRelu, SEBlock, layer1_for_LR_Branch, layer2_for_LR_Branch, layer1_for_HR_Branch, layer2_for_HR_Branch

#------------------------------------------------------------------------------
#  MODNet Branches
#------------------------------------------------------------------------------

class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()

        enc_channels = backbone.enc_channels
        
        #self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.branch1 = layer1_for_LR_Branch(enc_channels[4], enc_channels[2])

        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.branch2 = layer2_for_LR_Branch(enc_channels[3], enc_channels[2])

        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)
        self.conv_lr8x3 = Conv2dIBNormRelu(enc_channels[2]*3, enc_channels[2], 1)

    def forward(self, enc32x, inference):
        #enc_features = self.backbone.forward(img)
        #enc32x = enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x1 = F.interpolate(enc32x, scale_factor=2.0, mode='bilinear', align_corners=False)
        #branch 1
        lr16x2 = self.conv_lr16x(lr16x1)
        lr8x1 = F.interpolate(lr16x2, scale_factor=2.0, mode='bilinear', align_corners=False)
        #branch 2
        lr8x2 = self.conv_lr8x(lr8x1)

        pred_semantic = torch.tensor([])
        pred_semantics = []
        if not inference:
            lr = self.conv_lr(lr8x2)
            pred_semantic = torch.sigmoid(lr)
        
        pred_semantic1, lr81 = self.branch1(lr16x1, inference)
        pred_semantic2, lr82 = self.branch2(lr8x1, inference)

        pred_semantics += [pred_semantic1]
        pred_semantics += [pred_semantic2]
        pred_semantics += [pred_semantic]
        lr_8x = self.conv_lr8x3(torch.cat((lr8x2, lr81, lr82), dim=1))

        return pred_semantics, lr_8x


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)
        
        #first branch
        self.branch1 = layer1_for_HR_Branch(3 * hr_channels + 3, hr_channels)
        
        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        #sencond branch
        self.branch2 = layer2_for_HR_Branch(2 * hr_channels, hr_channels)
        #combine the features extracted from 3 branches
        self.conv_hr2x3 = Conv2dIBNormRelu(3*hr_channels, hr_channels, 1)

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x = F.interpolate(img, scale_factor=1.0/2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(img, scale_factor=1.0/4, mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        lr4x = F.interpolate(lr8x, scale_factor=2.0, mode='bilinear', align_corners=False)

        #branch 1
        pred_detail1, hr2x1 = self.branch1(torch.cat((hr4x, lr4x, img4x), dim=1), img, inference)

        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2.0, mode='bilinear', align_corners=False)

        #branch 2
        pred_detail2, hr2x2 = self.branch2(torch.cat((hr2x, enc2x), dim=1), img, inference)

        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        pred_detail = torch.tensor([])
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2.0, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)
        
        pred_details = []
        pred_details += [pred_detail1]
        pred_details += [pred_detail2]
        pred_details += [pred_detail]
        
        hr2x = self.conv_hr2x3(torch.cat((hr2x, hr2x1, hr2x2), dim=1))
        return pred_details, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2.0, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2.0, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2.0, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        #pred_matte = torch.sigmoid(f)

        return f


#------------------------------------------------------------------------------
#  EnsembleModnet
#------------------------------------------------------------------------------

class EnsembleMODNet(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True, ensemble_size=3):
        super(EnsembleMODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)

        self.f_branch_ensemble = []
        for i in range(ensemble_size):
            f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)
            self.f_branch_ensemble += [[f_branch]]

        self.f_branch_ensemble = nn.ModuleList([nn.ModuleList(branch) for branch in self.f_branch_ensemble])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                

    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]
        pred_semantics, lr8x = self.lr_branch(enc32x, inference)
        pred_details, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_mattes = []
        for index, f_branch in enumerate(self.f_branch_ensemble):
            for branch in f_branch:
                pred_matte = branch(img, lr8x, hr2x)
            pred_mattes += [pred_matte]

        return pred_semantics, pred_details, pred_mattes
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
