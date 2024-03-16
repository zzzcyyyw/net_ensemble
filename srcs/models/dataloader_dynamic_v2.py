#dataloader for dynamiclly generation

import os
import torch
import pandas as pd
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pdb, random
from torch.utils.data import Dataset, DataLoader
import random, os, cv2
from torchvision import transforms


# data loader
import glob
import torch
#from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import imghdr

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


# ==========================dataset load==========================
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):
        imidx, image, label, md_label, down_label = sample['imidx'], sample['image'], sample['label'], sample['md'], sample['down_label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, self.output_size)
        lbl = cv2.resize(label, self.output_size)
        lbl = lbl[:, :, np.newaxis]
        #cv2.imwrite('resacale_img.jpg', img)
        #cv2.imwrite('resacale_lbl.jpg', lbl)
        #down_lbl = cv2.resize(label, (self.output_size[0]/16, self.output_size[1]/16))


        return {'imidx':imidx, 'image':img,'label':lbl, 'md':md_label, 'down_label':down_label}


class RandomCropAndDownSample(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.kernel = np.ones((50,50), np.uint8)

    def __call__(self, sample):
        imidx, image, label, md_label, down_label = sample['imidx'], sample['image'], sample['label'], sample['md'], sample['down_label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = image[top: top + new_h, left: left + new_w]
        lbl = label[top: top + new_h, left: left + new_w]
        #lbl = lbl[:, :, np.newaxis]

        # downscale - blur
        # dow = cv2.resize(label, self.resize/16, self.resize/16))
        down_lbl = cv2.resize(lbl, (int(self.output_size[0] / 16), int(self.output_size[1] / 16)))
        #down_lbl = cv2.blur(down_lbl, (3, 3))
        down_lbl = cv2.GaussianBlur(down_lbl,(3,3), 0)
        down_lbl = down_lbl[:, :, np.newaxis]
        # dilate-erode
        dil = cv2.dilate(lbl, self.kernel, iterations=1)
        ero = cv2.erode(lbl, self.kernel, iterations=1)
        dil = dil[:, :, np.newaxis]
        ero = ero[:, :, np.newaxis]
        md_lbl = dil - ero  # md

        #cv2.imwrite('crop_img.jpg', img)
        #cv2.imwrite('crop_lbl.jpg', lbl)
        #cv2.imwrite('down_lbl.jpg', down_lbl)
        #cv2.imwrite('md_lbl.jpg', md_lbl)

        return {'imidx': imidx, 'image': img, 'label': lbl, 'md': md_lbl, 'down_label': down_lbl}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imidx, image, label, md_label, down_label = sample['imidx'], sample['image'], sample['label'], sample['md'], sample['down_label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)
        tmpDwlbl = np.zeros(down_label.shape)
        tmpMdlbl = np.zeros(md_label.shape)

        image = image / np.max(image)
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)
        if (np.max(down_label) < 1e-6):
            down_label = down_label
        else:
            down_label = down_label / np.max(down_label)
        if (np.max(md_label) < 1e-6):
            md_label = md_label
        else:
            md_label = md_label / np.max(md_label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.5) / 0.5
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.5) / 0.5
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.5) / 0.5
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.5) / 0.5
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.5) / 0.5
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.5) / 0.5

        tmpLbl[:, :, 0] = label[:, :, 0]
        tmpDwlbl[:, :, 0] = down_label[:, :, 0]
        tmpMdlbl[:, :, 0] = md_label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))
        tmpDwlbl = tmpDwlbl.transpose((2, 0, 1))
        tmpMdlbl = tmpMdlbl.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl), 'md': torch.from_numpy(tmpMdlbl), 'down_label': torch.from_numpy(tmpDwlbl)}


class SalObjDataset(Dataset):
    def __init__(self, fg_dir, matte_dir, bg_dir, fotor_fg, transform=None, bg_num=30):
        self.filenames = []
        self.fg_dir = fg_dir
        self.matte_dir = matte_dir
        self.bg_dir = bg_dir
        self.fotor_fg = fotor_fg

        name_bg = []
        for file in os.listdir(bg_dir):
            if file[-3:]!='png' and file[-3:]!='jpg':
                continue
            name_bg.append(file)

        for file in os.listdir(fg_dir):
            if file[-3:]!='png' and file[-3:]!='jpg':
                continue
            index = np.random.permutation(np.arange(len(name_bg)))
            for i in range(bg_num):
                filename = file + '--FOTOR--' + name_bg[index[i]]
                self.filenames.append(filename)
        #fotor
        for file in os.listdir(fotor_fg):
            if file[-3:]!='png' and file[-3:]!='jpg':
                continue
            self.filenames.append(file)
        self.transform = transform
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        fg_name = filename.split("--FOTOR--")[0]
        bg_name = filename.split("--FOTOR--")[1]

        if fg_name==bg_name:   #image from fotor, don't need to composite
            image = np.array(Image.open(os.path.join(self.fotor_fg, filename)))   # read in RGB mode
            image = np.asanyarray(image)
            label_3 = cv2.imread(os.path.join(self.matte_dir, fg_name), 0)
        else:
            image, label_3 = self.process(fg_name, bg_name)

        imname = fg_name
        imidx = np.array([idx])

        #print(label_3)
        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        #image = image[..., ::-1]#brg2rgb
        label = label[..., ::-1]

        sample = {'imidx': imidx, 'image': image, 'label': label, 'md': None, 'down_label':None}
        #print(image.shape)
        if self.transform:
            sample = self.transform(sample)
        #print(sample)
        return sample

    def process(self, fg_name, bg_name):
        #if imghdr.what(self.fg_dir + fg_name) == "png":
        #    Image.open(self.fg_dir + fg_name).convert("RGB").save(self.fg_dir + fg_name)
        
        im = np.array(Image.open(self.fg_dir + fg_name))
        im = np.asanyarray(im)
        im = im[:, :, [2, 1, 0]]    #RGB to BGR
        a = cv2.imread(self.matte_dir + fg_name, 0)
        
        h, w = im.shape[:2]
        bg = cv2.imread(self.bg_dir + bg_name)  #BGR
        bh, bw = bg.shape[:2]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = cv2.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)

        out = composite4(im, bg, a, w, h)    #BGR
        out = out[..., ::-1]                 #BGR to RGB
        return out, a

    def __len__(self):
        return len(self.filenames)

