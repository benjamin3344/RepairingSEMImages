# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:11:58 2020

@author: benjamin

Reads all the images in a folder, performs cropping, and outputs a .npy file representing all the images.
"""
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

imgall1 = []

for path in pathlib.Path("./data/").iterdir():
    if path.is_file():
       print(path)
       img = cv2.imread(os.path.join("./", path))
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       crop = gray[0:600,0:1000].copy()
       imgall1.append(crop)

imgall = np.asarray(imgall1,dtype=np.int32)
imgall = imgall.reshape([-1,600,1000,1])
imgall = imgall.astype(np.uint8)
np.save('data.npy',imgall)
