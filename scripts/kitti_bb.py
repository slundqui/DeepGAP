"""
Script to make output pvp based on iou of ground truth
Output will be len(windowSize) sparse pvp files, with the following dimensions:
[numImages, gtShapeY, gtShapeX, numClasses]
Each value will contain the max IOU of each anchor bb
"""

#import matplotlib
#matplotlib.use('Agg')

from dataObj.kitti_det import kittiDetBBObj
#from tf.VGGDetGap import VGGDetGap
import pdb
from bb_obj import bb_obj
from bb_mask import bb_mask

#Paths to list of filenames
trainImageList = "/shared/KITTI/objdet/training/genData/kitti_objdet_left_t0.txt"
#testImageList = "/shared/imageNet/DET/ILSVRC2015/ImageSets/DET/val.txt"

trainImagePrefix = "/shared/KITTI/objdet/training/image_2/"
#testImagePrefix =  "/shared/imageNet/DET/ILSVRC2015/Data/DET/val/"

trainGTPrefix = "/shared/KITTI/objdet/training/label_2/"
#testGTPrefix =  "/shared/imageNet/DET/ILSVRC2015/Annotations/DET/val/"

#clsMeta = "/shared/imageNet/devkit/data/meta_det.mat"

#Image is 256x256
#orientations = 1:1, 2:1, 1:2
windowSize=[
            (8, 8)  , (8, 16) , (16, 8) ,
            (16, 16), (32, 16), (16, 32),
            (32, 32), (64, 32), (32, 64),
            (64, 64), (64, 128),
           ]
iouThresh = .7

imageBatch = 256
gtShape = (16, 64)
outPrefix = "/home/slundquist/mountData/kitti_iou_obj/kitti_iou"

#Get object from which tensorflow will pull data from
trainDataObj = kittiDetBBObj(trainImageList, trainImagePrefix, trainGTPrefix, resizeMethod="crop", normStd=False, shuffle=False, seed=1234567)
#testDataObj = imageNetDetBBObj(testImageList, testImagePrefix, testGTPrefix, clsMeta, resizeMethod="crop", normStd=False, shuffle=False)

bb_mask(windowSize, gtShape, trainDataObj.inputShape, outPrefix)
bb_obj(trainDataObj, windowSize, imageBatch, gtShape, outPrefix, iouThresh)

