import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import shutil

from .config import im_size, unknown_code, fg_path, bg_path, a_path, num_valid

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

def safe_crop(mat, x, y, crop_size=(im_size, im_size)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.uint8)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.uint8)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (im_size, im_size):
        ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret

def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    iterations = np.random.randint(1, 20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap



# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y


class DIMDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = []
        self.masks = []
        for root, dirs, files in sorted(os.walk(img_dir)):
            for file in files:
                if file[-3:] == "png" or file[-3:]== "jpg":
                    this_file_path = os.path.join(img_dir, file)
                    self.imgs.append(this_file_path)
                    mask_name = file.split("--COCO")[0]
                    this_mask_path = os.path.join(mask_dir, mask_name)
                    self.masks.append(this_mask_path)
        self.transformer = data_transforms["train"]

    def __getitem__(self, i):
        img = cv.imread(self.imgs[i])
        alpha = cv.imread(self.masks[i], 0)

        # crop size 320:640:480 = 1:1:1
        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)
        x, y = random_choice(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)

        trimap = gen_trimap(alpha)
        
        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)

        x = torch.zeros((3, im_size, im_size), dtype=torch.float)
        img = img[..., ::-1].copy()  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x = img
        
        y = alpha / 255.0

        trimap = torch.from_numpy(trimap.copy() / 255.)
        
        return x, y, trimap

    def __len__(self):
        return len(self.imgs)
