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
        print(im_path + "            ", end = "\r")
        ext = im_path.split(".")[-1]
        ext = ext.lower()
        if ext != "jpg" and ext != "png" and ext != "jpeg":
            continue
        og_im = Image.open(im_path)
        im_file = os.path.basename(im_path)
        im_id = im_file.split("_")[0]
        category = label[im_id]
        np_im = np.array(og_im)
        y.append(category)
        np.concatenate(X,np_im)
        count += 1

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.28)
print(y_train)
temp = input("Hit Enter to Continue:")
print(X_train)
