import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from srcs.models.modnet_torchscript import MODNet

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='saved_model/dis/', help='path of input images')
    parser.add_argument('--output_path', type=str, default='saved_model/dis_matte/', help='path of output images')
    parser.add_argument('--out_dir', type=str, default='saved_model/torch_script/', help='path of output images')
    args = parser.parse_args()
    mode_save_dir = "saved_model/train_122.pth"
    #mode_save_dir = "saved_model/torch_script/modnet_cpu.pt"
    
    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    modnet = MODNet(backbone_pretrained=True)
    ckpt = torch.load(mode_save_dir, map_location='cpu')

    # if use more than one GPU
    #if 'module.' in ckpt.keys():
    ckpt_single = OrderedDict()
    for k, v in ckpt.items():
        k = k.replace('module.', '')
        ckpt_single[k] = v
    modnet.load_state_dict(ckpt_single)
    modnet.eval()

    scripted_model = torch.jit.script(modnet)
    torch.jit.save(scripted_model, os.path.join(args.out_dir,'train_122.pt'))

