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
    def __init__(self, img_dir, mask_dir, transform=None):
        self.imgs = []
        self.masks = []
        for root, dirs, files in sorted(os.walk(img_dir)):
            for file in files:
                if file[-3:] == "png" or file[-3:]== "jpg":
                    this_file_path = os.path.join(img_dir, file)
                    self.imgs.append(this_file_path)
                    mask_name = file.split("--COCO")[1]
                    this_mask_path = os.path.join(mask_dir, mask_name)
                    self.masks.append(this_mask_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        #im = Image.open(self.fg_dir + fg_name)
        
        #if imghdr.what(self.imgs[idx]) == "png":
        #    Image.open(self.imgs[idx]).convert("RGB").save(self.imgs[idx])
        image = np.array(Image.open(self.imgs[idx]))   # read in RGB mode
        image = np.asanyarray(image)
        #image = image[:, :, [2, 1, 0]] 


        #img = cv.imread(self.imgs[i])
        #alpha = cv.imread(self.masks[i], 0)
        #image = cv2.imread(self.imgs[idx])
        #alpha = cv.imread(self.masks[idx], 0)

        #image = cv2.imread(self.image_name_list[idx])
        imname = self.imgs[idx]
        imidx = np.array([idx])

        if (0 == len(self.masks)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = cv2.imread(self.masks[idx], 0)

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

    def __len__(self):
        return len(self.imgs)

