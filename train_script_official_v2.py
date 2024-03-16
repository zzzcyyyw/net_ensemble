#使用深大dataloader和script

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

import glob

from srcs.models.modnet import MODNet

from utils import load_modnet 

from torch.utils.data import Dataset, DataLoader
from srcs.models.dataloader_from_composite import SalObjDataset, Rescale, RandomCropAndDownSample, ToTensor

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

if __name__ == '__main__':

    writer = SummaryWriter('./logs')

    # ------- 1. define loss function --------
    l1_loss = nn.L1Loss(reduce=False, size_average=False)
    l2_loss = nn.MSELoss(reduce=False, size_average=False)
    # alpha prediction loss: the abosolute difference between the ground truth alpha values and the
    # predicted alpha values at each pixel. However, due to the non-differentiable property of
    # absolute values, we use the following loss function to approximate it.
    def alpha_prediction_loss(y_pred, y_true):
        epsilon = 1e-6
        epsilon_sqr = epsilon ** 2
        mask = y_true[:, 0, :]
        diff = y_pred[:, 0, :] - y_true[:, 0, :]
        # diff = diff * mask
        num_pixels = torch.sum(mask)
        return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / (num_pixels + epsilon)


    # ------- 2. set the directory of training dataset --------
    epoch_num = 200
    bs = 16
    train_num = 0
    val_num = 0
    momentum = 0.9
    steps = [40, 80, 120, 160]
    weight_decay = 0.0005
    lr = 0.01
    gamma = 0.1

    resume = False
    resume_ckpt = './pretrained/modnet_portrait.ckpt'
    pre_train_dir = "./pretrained/modnet_pretrained.ckpt"

    log_save_path = "logs/modnet_v2/"
    mode_save_dir = "saved_model/modnet_v2_pretrain/"
    pre_train_dir = "pretrained/modnet_pretrained.ckpt"
    img_dir = './data/Composite/'
    mask_dir = './data/matte/'

    if not os.path.exists(mode_save_dir):
        os.makedirs(mode_save_dir)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    train_dataset = SalObjDataset(
                        img_dir=img_dir, 
                        mask_dir=mask_dir,
                        transform=transforms.Compose([
                            Rescale(512+32),
                            RandomCropAndDownSample(512),
                            ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)

    # ------- 3. define model --------
    # define the net
    #net = MODNet(backbone_pretrained=True)
    net = load_modnet(pre_train_dir)
    if torch.cuda.is_available():
        #net =  torch.nn.DataParallel(net).cuda()
        net = net.cuda()


    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=steps, gamma=gamma)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_sem_loss = 0.0
    running_det_loss = 0.0
    running_fus_loss = 0.0
    ite_num4val = 0
    print_frq = 500
    save_frq = 1 # save the model every 2000 iterations


    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(train_loader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, labels, md_masks, down_labels = data['image'], data['label'], data['md'], data['down_label']

            inputs = inputs.float()
            labels = labels.float()
            md_masks = md_masks.float()
            down_labels = down_labels.float()

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v, md_masks_v, down_labels_v = inputs.cuda(), labels.cuda(), md_masks.cuda(), down_labels.cuda()
            
            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred_semantic, pred_detail, pred_matte = net(inputs_v, False)
            detail_loss = (md_masks.cuda() * l1_loss(pred_detail, labels_v)).sum() / md_masks.cuda().sum() #no compositional loss

            semantic_loss = 0.5 * l2_loss(pred_semantic, down_labels_v).mean()
            Lc = alpha_prediction_loss(pred_matte, labels_v)
            fusion_loss = l1_loss(pred_matte, labels_v).mean() + Lc
            loss = semantic_loss + 10 * detail_loss + fusion_loss

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_det_loss += detail_loss.item()
            running_fus_loss += fusion_loss.item()
            running_sem_loss += semantic_loss.item()

            if ite_num % print_frq == 0:
                print("[epoch: %3d /%3d, batch: %5d /%5d, ite: %d] det_loss: %3f fus_loss: %3f sem_loss: %3f train_loss: %3f" % (
                    epoch + 1, epoch_num, (i + 1) * bs, train_num, ite_num, \
                        running_det_loss / ite_num4val, \
                        running_fus_loss / ite_num4val, \
                        running_sem_loss / ite_num4val, \
                        running_loss / ite_num4val))

                writer.add_scalar('lr', scheduler.get_last_lr()[0], ite_num / print_frq)
                writer.add_scalar('total loss', running_loss / ite_num4val, ite_num / print_frq)
                writer.add_scalar('semantic loss', running_sem_loss / ite_num4val, ite_num / print_frq)
                writer.add_scalar('detail loss', running_det_loss / ite_num4val, ite_num / print_frq)

                writer.add_image('semantic image', make_grid([pred_semantic[0], down_labels_v[0]]), ite_num / print_frq)
                writer.add_image('detail image', make_grid([pred_detail[0], md_masks_v[0], md_masks_v[0] * labels_v[0]]), ite_num / print_frq)
                writer.add_image('fusion image', make_grid([pred_matte[0], labels_v[0]]), ite_num / print_frq)
                writer.add_image('original image', make_grid(inputs_v[0]), ite_num / print_frq)

        
        #torch.save(net.state_dict(), model_dir + '/' + model_name + "_epoch_%d_train_%3f_tar_%3f.pth" % (
        #    epoch, running_loss / ite_num4val, running_loss / ite_num4val))
        checkpoint_name = "train_%d.pth" %(epoch+1)
        torch.save(net.state_dict(), mode_save_dir + '/' + checkpoint_name)

        running_loss = 0.0
        running_det_loss = 0.0
        running_sem_loss = 0.0
        running_fus_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0
        
        scheduler.step()