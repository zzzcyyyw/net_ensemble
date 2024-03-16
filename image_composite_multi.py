import math
import time
from multiprocessing import Pool

import cv2 as cv
import numpy as np
import tqdm
from tqdm import tqdm
import os

fg_dir = './data/fg/'
alpha_dir = './data/matte/'
bg_dir = './data//bg/'
out_path = './data/Composite/'

num_bg = 40

name_bg = []
for file in os.listdir(bg_dir):
    if file[-3:]!='png' and file[-3:]!='jpg':
        continue
    name_bg.append(file)

name_fg = []
for file in os.listdir(fg_dir):
    if file[-3:]!='png' and file[-3:]!='jpg':
        continue
    name_fg.append(file)


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def process(im_name, bg_name):
    im = cv.imread(fg_dir + im_name)
    a = cv.imread(alpha_dir + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_dir + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    out = composite4(im, bg, a, w, h)
    filename = out_path + im_name + '--Fotor--' + bg_name
    cv.imwrite(filename, out)

def process_one_fg(count):
    index = np.random.permutation(np.arange(len(name_bg)))
    for i in range(num_bg):
        #process(name_fg[count], name_bg[i])
        process(name_fg[count], name_bg[index[i]])

def do_composite():
    start = time.time()
    with Pool(processes=10) as p:
        max_ = len(name_fg)
        print('num_fg_files: ' + str(max_))
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(process_one_fg, range(0, max_)))):
                pbar.update()

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))

if __name__ == '__main__':
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    do_composite()
