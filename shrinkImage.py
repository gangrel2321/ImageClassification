from PIL import Image
import os
import sys

cwd = os.getcwd()
for path, dirs, files in os.walk(cwd):
    for file in files:
        beg = file.split(".")[0]
        ext = file.split(".")[-1]
        ext = ext.lower()
        if ext == "png" or ext == "jpg" or ext == "jpeg":
            im_path = os.path.join(path,file)
            og_im = Image.open(im_path)
            print(im_path)
            if og_im.size != (224,224):
                og_im = og_im.resize((224,224))
            if ext != "jpg":
                n_file = beg + ".jpg"
                im_path = os.path.join(path,n_file)
            og_im.save(im_path)
            og_im.close()
            if ext != "jpg":
                os.remove(os.path.join(path,file))
