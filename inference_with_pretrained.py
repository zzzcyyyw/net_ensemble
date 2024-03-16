import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import sys
import os 
# print(os.getcwd())
# sys.path.append('../../../')
# print(sys.path)
from srcs.models.modnet_torchscript import MODNet
from collections import OrderedDict


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='saved_model/Image_test/', help='path of input images')
    #parser.add_argument('--input-path', type=str, default='data/Distinc_composite/', help='path of input images')
    #parser.add_argument('--input-path', type=str, default='saved_model/dis/', help='path of input images')
    
    parser.add_argument('--output-path', type=str, default='saved_model/dis_matte/', help='path of output images')
    #parser.add_argument('--output-path', type=str, default='saved_model/tao_matte/', help='path of output images')
    #parser.add_argument('--output-path', type=str, default='saved_model/cebela_matte/', help='path of output images')
    
    #parser.add_argument('--ckpt-path', type=str, default='./saved_models/mobilenetv2/best.pth', help='path of pre-trained MODNet')
    args = parser.parse_args()

    unloader = transforms.ToPILImage()

    mode_save_dir = "saved_model/train_4.pth"
    #mode_save_dir = "pretrained/modnet_pretrained.ckpt"

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            #transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    #modnet = nn.DataParallel(modnet)
    #modnet = modnet.cuda()
    ckpt = torch.load(mode_save_dir, map_location='cpu')
    modnet.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    
    #modnet.load_state_dict(ckpt)
    modnet.eval()
    
    

    # inference images
    images_tensors = []
    mattes_tensors = []
    im_names = os.listdir(args.input_path)
    im_names.sort()
    for im_name in im_names:
        file_name = im_name.split('.')
        file_name = file_name[-1]
        if file_name != 'jpg' and file_name != 'png':
            continue

        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(args.input_path, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]
        #im = im[..., ::-1].copy()  # RGB

        # convert image to PyTorch tensor

        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        #print(im.shape)

        # inference
        _, _, matte = modnet(im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        mattes_tensors += [matte.detach()]
        matte = matte[0][0].data.cpu().numpy()
        #matte_name = im_name.split('.')[0] + '.png'
        matte_name = im_name
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
        #plt.imshow(unloader(im.squeeze(0)))
        #plt.pause(0.001)
        images_tensors += [im]
        
    images_tensors = torch.cat(images_tensors, 0)
    mattes_tensors = torch.cat(mattes_tensors, 0)
    save_image(images_tensors, args.output_path + "/original_grid.jpg")
    save_image(mattes_tensors, args.output_path + "/ensemble_matte_grid.jpg")
