"""
Script to make output pvp based on iou of ground truth
Output will be len(windowSize) sparse pvp files, with the following dimensions:
[numImages, gtShapeY, gtShapeX, numClasses]
Each value will contain the max IOU of each anchor bb
"""

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.imagenet_det import imageNetDetBBObj
#from tf.VGGDetGap import VGGDetGap
import pdb
from bb_to_pvp import bb_to_pvp
from bb_mask import bb_mask

#Paths to list of filenames
trainImageList = "/shared/imageNet/DET/ILSVRC2015/ImageSets/DET/train.txt"
testImageList = "/shared/imageNet/DET/ILSVRC2015/ImageSets/DET/val.txt"

trainImagePrefix = "/shared/imageNet/DET/ILSVRC2015/Data/DET/train/"
testImagePrefix =  "/shared/imageNet/DET/ILSVRC2015/Data/DET/val/"

trainGTPrefix = "/shared/imageNet/DET/ILSVRC2015/Annotations/DET/train/"
testGTPrefix =  "/shared/imageNet/DET/ILSVRC2015/Annotations/DET/val/"

clsMeta = "/shared/imageNet/devkit/data/meta_det.mat"

DEBUG=False

#Image is 256x256
#orientations = 1:1, 2:1, 1:2
windowSize=[
            (32, 32), (64, 32), (32, 64),
            (64, 64), (128, 64), (64, 128),
            (128, 128), (256, 128), (128, 256),
            (256, 256)
           ]

imageBatch = 256
gtShape = (16, 16)
outPrefix = "/home/slundquist/mountData/imagenet_iou/imagenet_iou"

#Get object from which tensorflow will pull data from
trainDataObj = imageNetDetBBObj(trainImageList, trainImagePrefix, trainGTPrefix, clsMeta, resizeMethod="crop", normStd=False, shuffle=False, seed=1234567)
#testDataObj = imageNetDetBBObj(testImageList, testImagePrefix, testGTPrefix, clsMeta, resizeMethod="crop", normStd=False, shuffle=False)

bb_mask(windowSize, gtShape, trainDataObj.inputShape, outPrefix)
bb_to_pvp(trainDataObj, windowSize, imageBatch, gtShape, outPrefix)

