import numpy as np
import os
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict


#X is images, y is labels

X = np.array([])
y = []
cwd = os.getcwd()
label = defaultdict()

with open("labels.txt") as f:
    lines = list(filter(None, f.read().split("\n")))

    for line in lines:
        pair = line.split(" ")
        label[pair[0]] = pair[1]

count = 0

for path, dirs, files in os.walk(cwd):
    for image in files:
        im_path = os.path.join(path,image)
        cur_dir = list(filter(None, path.split("\\")))[-1]
        #print(cur_dir)
        #print(im_path + "            ", end = "\r")
        ext = im_path.split(".")[-1]
        ext = ext.lower()

        if ext != "jpg" and ext != "png" and ext != "jpeg":
            continue

        if count % 20 <= 13:
            new_path = os.getcwd() + "/data/train/" + cur_dir + "/" + image #70
        elif count % 20 <= 16:
            new_path = os.getcwd() + "/data/validation/" + cur_dir + "/" + image #15
        else:
            new_path = os.getcwd() + "/data/test/" + cur_dir + "/" + image #15
        print(new_path)
        os.rename(im_path, new_path)
        count += 1
