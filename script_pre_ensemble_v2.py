import os
import torch
import torch.nn as nn
from srcs.models.modnet_ensemble_v2 import EnsembleMODNet
from trainer_ensemble import supervised_training_iter

from config import device, im_size, grad_clip, print_freq
from DIMDataset import DIMDataset
from utils import pre_train_to_ensemble_v2 

import numpy as np
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

ensemble_size = 4
lr = 0.01       # learn rate
epochs = 40     # total epochs
batch_size = 16
print_frq = 200
log_save_path = "logs/ensemble_pre_train_v2/"
mode_save_dir = "saved_model/ensemble_pre_train_v2"
pre_train_dir = "pretrained/modnet_pretrained.ckpt"
ensemble_dir = "pretrained/ensemble_pretrained.pth"

writer = SummaryWriter(log_save_path)

if not os.path.exists(mode_save_dir):
    os.makedirs(mode_save_dir)
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)


net = pre_train_to_ensemble_v2(ensemble_size, pre_train_dir, ensemble_dir)
if torch.cuda.is_available():
    net =  torch.nn.DataParallel(net).cuda()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

# Custom dataloaders
train_dataset = DIMDataset('train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

best_loss = 1000
running_loss = 0.0
running_sem_loss = 0.0
running_det_loss = 0.0
running_fus_loss = 0.0

ite_num = 0
ite_num4val = 0
for epoch in range(0, epochs):
    for idx, (image, gt_matte, trimap) in enumerate(train_loader):
        ite_num += 1
        ite_num4val += 1
        
        image, trimap, gt_matte = image.float(), trimap.float(), gt_matte.float()

        if torch.cuda.is_available():
            image, trimap, gt_matte = image.cuda(), trimap.cuda(), gt_matte.cuda()
        semantic_loss, detail_loss, matte_loss = \
            supervised_training_iter(net, optimizer, image, trimap, gt_matte)
        loss = semantic_loss + detail_loss + matte_loss

        running_loss += loss.item()
        running_det_loss += detail_loss.item()
        running_fus_loss += matte_loss.item()
        running_sem_loss += semantic_loss.item()

        if ite_num % print_frq == 0:
            print("[epoch: %3d /%3d, batch: %3d / %3d ] lr:%.4f  semantic_loss: %3f detail_loss: %3f fus_loss: %3f  train_loss: %3f" % (
                epoch + 1, epochs, idx, len(train_loader), lr_scheduler.get_lr()[0], semantic_loss.item(), \
				detail_loss.item(), matte_loss.item(), loss.item()))
            
            writer.add_scalar('lr', lr_scheduler.get_lr()[0], ite_num / print_frq)
            writer.add_scalar('total loss', running_loss / ite_num4val, ite_num / print_frq)
            writer.add_scalar('semantic loss', running_sem_loss / ite_num4val, ite_num / print_frq)
            writer.add_scalar('detail loss', running_det_loss / ite_num4val, ite_num / print_frq)
            writer.add_scalar('matte loss', running_fus_loss / ite_num4val, ite_num / print_frq)

    lr_scheduler.step()
    if running_fus_loss/ite_num4val<best_loss:
        best_loss = running_fus_loss/ite_num4val
        checkpoint_name = "best.pth"
        torch.save(net.state_dict(), mode_save_dir + '/' + checkpoint_name)

        running_loss = 0.0
        running_det_loss = 0.0
        running_sem_loss = 0.0
        running_fus_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0

    checkpoint_name = "train_%d.pth" %(epoch+1)
    torch.save(net.state_dict(), mode_save_dir + '/' + checkpoint_name)
writer.close()
