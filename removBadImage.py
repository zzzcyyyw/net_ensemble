import os

from numpy.lib.type_check import imag
import cv2
import numpy as np

path = 'saved_model/Image_test/'

imgList = os.listdir(path)

for file in imgList:
    print(file)
    if file.endswith('.jpg') or file.endswith('.jpg'):
        imgPath = os.path.join(path, file)
        if os.path.getsize(imgPath)==0:
            os.remove(imgPath)
            continue
        image = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)
        try:
            print(image.shape)
        except:
            os.remove(imgPath)