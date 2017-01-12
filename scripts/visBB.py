import pdb
from pvtools import *
import numpy as np
import matplotlib.pyplot as plt

imageInput = "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp"
objFile = "/home/slundquist/mountData/kitti_iou_bin_obj/kitti_iou_bin_objFile.pvp"
labelFile = "/home/slundquist/mountData/kitti_iou_bin_obj/kitti_iou_bin_labelFile.pvp"

imageFile = readpvpfile(imageInput, lastFrame=20)
objPvp = readpvpfile(objFile, lastFrame=20)
labelFile = readpvpfile(labelFile, lastFrame=20)

#Grab the first 20 images and expand

imgs = imageFile["values"]
objs = np.reshape(objPvp["values"].toarray(), [20, 16, 64, 9, 2])
labels = np.reshape(labelFile["values"].toarray(), [20, 16, 64, 9, 5])

f = plt.figure()

for i in range(20):
    img = imgs[i, :, :, :]
    obj = objs[i, :, :, :, :]
    label = labels[i, :, :, :, :]
    posObj = np.nonzero(obj[:, :, :, 0])
    posLabels = label[posObj]
    for single_label in posLabels:
        [labelId, ymin, ymax, xmin, xmax] = single_label
        ymin = int(np.round(ymin))
        ymax = int(np.round(ymax))
        xmin = int(np.round(xmin))
        xmax = int(np.round(xmax))

        newImg = img.copy()
        #Scale image to be between 0 and 1
        newImg = (newImg-np.min(newImg))/(np.max(newImg)-np.min(newImg))
        #Draw box
        newImg[ymin:ymax, xmin, 0] = 1
        newImg[ymin:ymax, xmax, 0] = 1
        newImg[ymin, xmin:xmax, 0] = 1
        newImg[ymax, xmin:xmax, 0] = 1

        plt.imshow(newImg)
        plt.show()

pdb.set_trace()



pdb.set_trace()
