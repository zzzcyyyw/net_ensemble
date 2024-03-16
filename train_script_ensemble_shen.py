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
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

from srcs.models.modnet_ensemble_v2 import EnsembleMODNet

from torch.utils.data import Dataset, DataLoader
from srcs.models.dataloader_from_composite import SalObjDataset, Rescale, RandomCropAndDownSample, ToTensor
# from srcs.models.dataloader_v2 import SalObjDataset, Rescale, RandomCropAndDownSample, ToTensor
from utils import pre_train_to_ensemble


from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

if __name__ == '__main__':
    writer = SummaryWriter('./logs/Ensemble')

    alpha = 1.
    gamma = 2.0
    threshold = 1e-4
    w_sementic, w_detail, w_fusion = 1.0, 10.0, 1.0


    # ------- 1. define loss function --------
    l1_loss = nn.L1Loss(reduce=False, size_average=False)
    l2_loss = nn.MSELoss(reduce=False, size_average=False)

    def semantic_prediction_loss(pred_semantics, down_labels):
        loss = 0.0
        for i in range(down_labels.shape[0]):  # calculate loss on ith image
            label = down_labels[i].view(-1,1)
            weights = None
            loss_i = 0
            for idx, pred_semantic in enumerate(pred_semantics):
                pred_semantic_i = pred_semantic[i].view(-1,1)
                if weights is None:
                    weights = pred_semantic_i.detach()
                else:
                    weights = torch.cat([weights, pred_semantic_i.detach()], 1)
                #print("weights:", weights.shape)
                p = torch.mean(weights, 1).view(label.shape[0], 1)
                p = torch.abs(p-label)
                p = torch.clamp(p, min=0., max=1.0)
                max_p = torch.max(p)
                if max_p <= threshold:
                    p = torch.ones(label.shape)
                else:
                    p = p/(max_p+threshold)

                p = alpha*(p**gamma)
                loss_idx = p* F.mse_loss(pred_semantic_i, label) #  (pred_semantic_i-label)**2.0
                if idx==0:
                    loss_i = loss_idx                      #loss: the idx-th classifier's loss on i-th image
                else:
                    loss_i += loss_idx

            loss += loss_i.mean()
        loss /= down_labels.shape[0]
        #loss = torch.cat(loss, 0)
        #loss = torch.mean(loss)
        #print(loss)
        return loss

    def details_prediction_loss(pred_details, labels, md_masks):
        loss = 0

        for i in range(labels.shape[0]):  # calculate loss on ith image
            label = labels[i].view(-1,1)
            md_mask_i = md_masks[i].view(-1, 1)
            weights = None
            loss_i = 0
            for idx, pred_detail in enumerate(pred_details):
                pred_detail_i = pred_detail[i].view(-1,1)
                if weights is None:
                    weights = pred_detail_i.detach()
                else:
                    weights = torch.cat([weights, pred_detail_i.detach()], 1)
                p = torch.mean(weights, 1).view(label.shape[0], 1)
                p = torch.abs(p-label)
                p = torch.clamp(p, min=0., max=1.0)
                max_p = torch.max(p)

                if max_p <= threshold:
                    p = torch.ones(label.shape).cuda()
                else:
                    p = p/(max_p+threshold)
                #print("p:",p)
                #print("mask:", md_mask_i)
                p = alpha*(p**gamma)*md_mask_i                  #modulating factor
                loss_idx = p*torch.abs(pred_detail_i-label)
                loss_idx = loss_idx.sum()                      #loss: the idx-th classifier's loss on i-th image
                loss_i += loss_idx
            loss += loss_i
        loss /= (labels.shape[0]*md_masks.sum())
        #print("detail_loss:", loss)
        #loss = torch.cat(loss, 0)
        #loss = loss.sum()/md_masks.sum()
        return loss

    def alpha_prediction_loss(y_preds, y_true):
        epsilon = 1e-6
        epsilon_sqr = epsilon ** 2
        mask = y_true[:, 0, :]
        diff = 0
        for idx, y_pred in enumerate(y_preds):
            diff += y_pred[:, 0, :] - y_true[:, 0, :]
        # diff = diff * mask
        num_pixels = torch.sum(mask)
        return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / ((num_pixels + epsilon)*len(y_preds))

    def fusion_loss_fun(y_preds, y_true):
        diff = 0
        for idx, y_pred in enumerate(y_preds):
            diff += l1_loss(y_pred, y_true).mean()
        return diff
        #return diff/len(y_preds)



    # ------- 2. set the directory of training dataset --------

    model_name = 'ensemble' #'u2netp'


    epoch_num = 200
    bs = 16
    train_num = 0
    val_num = 0
    momentum = 0.9
    steps = [20, 30] #[10, 20, 30]
    weight_decay = 0.0005
    lr = 0.01
    gamma = 0.1

    resume = False
    resume_ckpt = './pretrained/modnet_portrait.ckpt'
    pre_train_dir = "./pretrained/modnet_pretrained.ckpt"

    log_save_path = "logs/ensemble_v2/"
    mode_save_dir = "saved_model/ensemble_v2/"
    img_dir = './data/Distinc_composite/'
    mask_dir = './data/Distinc_alpha/'


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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=16)

    #net = EnsembleMODNet(backbone_pretrained=True, ensemble_size=3)
    net = pre_train_to_ensemble(ensemble_size=3)
    if torch.cuda.is_available():
        net = nn.DataParallel(net)
        net = net.cuda()

    if resume:
        print('load pretrained model')
        net.load_state_dict(torch.load(resume_ckpt))

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
    print_frq = 200
    save_frq = 5 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(train_loader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, labels, md_masks, down_labels = data['image'], data['label'], data['md'], data['down_label']

            inputs, labels, md_masks, down_labels  = inputs.float(), labels.float(), md_masks.float(), down_labels.float()
            if torch.cuda.is_available():
                inputs_v, labels_v, md_masks_v, down_labels_v = inputs.cuda(), labels.cuda(), md_masks.cuda(), down_labels.cuda()

            #zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred_semantics, pred_details, pred_matte = net(inputs_v, False)
            #print(pred_semantics)
            #print(pred_semantics[0].shape)
            semantic_loss = semantic_prediction_loss(pred_semantics, down_labels_v)
            detail_loss = details_prediction_loss(pred_details, labels_v, md_masks_v)
            #detail_loss = (md_masks.cuda() * l1_loss(pred_detail, labels_v)).sum() / md_masks.cuda().sum() #no compositional loss
            #semantic_loss = 0.5 * l2_loss(pred_semantic, down_labels_v).mean()
            Lc = alpha_prediction_loss(pred_matte, labels_v)
            fusion_loss = fusion_loss_fun(pred_matte, labels_v) + Lc
            loss = semantic_loss*w_sementic + w_detail* detail_loss + w_fusion * fusion_loss

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

                writer.add_image('semantic image', make_grid([pred_semantics[0][0], down_labels_v[0]]), ite_num / print_frq)
                writer.add_image('detail image', make_grid([pred_details[0][0], md_masks_v[0], md_masks_v[0] * labels_v[0]]), ite_num / print_frq)
                writer.add_image('fusion image', make_grid([pred_matte[0][0], labels_v[0]]), ite_num / print_frq)
                writer.add_image('original image', make_grid(inputs_v[0]), ite_num / print_frq)

        checkpoint_name = "Ensemble_%d.pth" %(epoch+1)
        torch.save(net.state_dict(), mode_save_dir + '/' + checkpoint_name)

        running_loss = 0.0
        running_det_loss = 0.0
        running_sem_loss = 0.0
        running_fus_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0

        scheduler.step()
