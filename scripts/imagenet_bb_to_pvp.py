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
import numpy as np
import scipy.sparse as sp
from pvtools import *
import pdb

import tensorflow as tf

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
testDataObj = imageNetDetBBObj(testImageList, testImagePrefix, testGTPrefix, clsMeta, resizeMethod="crop", normStd=False, shuffle=False)

dataShape = trainDataObj.inputShape
numClasses = trainDataObj.numClasses

#Make sure gtShape divides into output shape
assert(dataShape[0] % gtShape[0] == 0)
assert(dataShape[1] % gtShape[1] == 0)

strideY = dataShape[0]/gtShape[0]
strideX = dataShape[1]/gtShape[1]

#Placeholder for image
#inImage = tf.placeholder("float", shape=[None, trainDataObj.inputShape[0], trainDataObj.inputShape[1], trainDataObj.inputShape[2]], name="inImage")

#Placeholder for ground truth array
tf_inGt = tf.placeholder("float", shape=[None, trainDataObj.inputShape[0], trainDataObj.inputShape[1], 1], name="inGt")

with tf.device('/gpu:0'):
    tf_area_bb = tf.squeeze(tf.reduce_sum(tf_inGt, reduction_indices=[1, 2, 3], keep_dims=True), squeeze_dims=[3])

    #We avg pool per window size on each gt
    tf_iou = []
    for wSize in windowSize:
        (ySize, xSize) = wSize
        tf_area_wSize = ySize * xSize
        tf_avg_pool = tf.squeeze(tf.nn.avg_pool(tf_inGt, [1, ySize, xSize, 1], [1, strideY, strideX, 1], "SAME"), squeeze_dims=[3])
        #Undo average pool to find intersection
        tf_intersection = tf_avg_pool * tf_area_wSize
        #Save final node
        #We sum 2 bbs, minus intersection to find union
        tf_union = tf_area_bb + tf_area_wSize - tf_intersection
        tf_iou.append(tf_intersection/tf_union)

    #Pack into outer dimmension
    tf_iou_pack = tf.pack(tf_iou, axis=0)

init = tf.initialize_all_variables()

numIterations = int(np.ceil(float(trainDataObj.numImages) / imageBatch))
outSize = (trainDataObj.inputShape[0], trainDataObj.inputShape[1], 1)

numWindowSize = len(windowSize)

with tf.Session() as sess1:
    sess1.run(init)

    pvpFiles = []
    for wIdx in range(numWindowSize):
        outStr = outPrefix + str(windowSize[wIdx][0])+"x"+str(windowSize[wIdx][1])+".pvp"
        pvpFiles.append(pvpOpen(outStr, "w"))

    for it in range(numIterations):
        allBB = []
        bbPerImage = []
        bbObjId = []
        print "Getting BB,", it, "out of", numIterations
        for i in range(imageBatch):
            imgIdx = it*imageBatch + i
            if(i >= trainDataObj.numImages):
                break
            (img, bbs) = trainDataObj.nextImage(onlyGt = True)
            numBB = len(bbs)
            bbPerImage.append(numBB)
            allBB.extend(bbs)
            #if(DEBUG and i == 0):
            #    (y, x, f) = img.shape
            #    r_img = (img-img.min())/(img.max()-img.min())
            #    bbImg = np.zeros((y, x, f))
            #    for idx, b in enumerate(bbs):
            #        if(idx == 0):
            #            (i, top, bot, left, right) = b
            #            bbImg[top:bot, left:right] = 1
            #    plt.subplot(211)
            #    plt.imshow(r_img)
            #    plt.subplot(212)
            #    plt.imshow(bbImg)
            #    plt.show()

        numTotBB = len(allBB)
        gtArray = np.zeros((numTotBB,)+outSize)
        print "Buiding BB"
        for j, bb in enumerate(allBB):
            imgIdx = it*imageBatch + j
            (objid, up, down, left, right) = bb
            gtArray[j, up:down, left:right, :] = 1
            bbObjId.append(objid)
        print "Calculating IOU"
        feedDict = {tf_inGt: gtArray}
        #This is a matrix of [numWindows, numBB, gtShape[0], gtShape[1]]
        np_iou_pack = sess1.run(tf_iou_pack, feed_dict=feedDict)

        #We calculate per window shape
        print "Making and writing sparse matrix"
        for wIdx in range(numWindowSize):
            iou = np_iou_pack[wIdx, :, :, :]
            (numTotBB, gtY, gtX) = iou.shape
            #We want to translate this to a sparse matrix per image,
            #where we have 3 lists, data, rowidx, colidx
            outData = []
            outRowIdx = []
            outColIdx = []
            outTime = []
            windowIdx = 0
            for i, numBB in enumerate(bbPerImage):
                imgIdx = it*imageBatch + i
                outTime.append(imgIdx)
                for bbIdx in range(numBB):
                    bbIOU = iou[windowIdx, :, :]
                    bbId = bbObjId[windowIdx]
                    nzIdx = np.nonzero(bbIOU)
                    nnz = len(nzIdx[0])
                    (yIdx, xIdx) = nzIdx
                    #Convert y, x, and f (bbId) to linear index
                    linIdx = yIdx*gtShape[1]*numClasses + xIdx*numClasses + bbId
                    outData.extend(list(bbIOU[nzIdx]))
                    outColIdx.extend(list(linIdx))
                    outRowIdx.extend([i for j in range(nnz)])

                    windowIdx += 1
            #Sanity check
            assert(windowIdx == numTotBB)
            outSparse = sp.csr_matrix((outData, (outRowIdx, outColIdx)),
                shape=(imageBatch, gtShape[0]*gtShape[1]*numClasses))
            pvpData = {"values": outSparse, "time":outTime}
            pvpFiles[wIdx].write(pvpData, shape=(gtShape[0], gtShape[1], numClasses))








#params = {
#    #Base output directory
#    'outDir':          "/home/slundquist/mountData/DeepGAP/",
#    #Inner run directory
#    'runDir':          "/imagenet_det_vgg_ds/",
#    'tfDir':           "/tfout",
#    #Save parameters
#    'ckptDir':         "/checkpoints/",
#    'saveFile':        "/save-model",
#    'savePeriod':      10, #In terms of displayPeriod
#    #output plots directory
#    'plotDir':         "plots/",
#    'plotPeriod':      100, #With respect to displayPeriod
#    #Progress step
#    'progress':        1,
#    #Controls how often to write out to tensorboard
#    'writeStep':       100, #300,
#    #Flag for loading weights from checkpoint
#    'load':            False,
#    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/imagenet_det.ckpt",
#    #Input vgg file for preloaded weights
#    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
#    #Device to run on
#    'device':          '/gpu:0',
#    #####ISTA PARAMS######
#    #Num iterations
#    'outerSteps':      100, #1000000,
#    'innerSteps':      100, #300,
#    #Batch size
#    'batchSize':       8,
#    #Learning rate for optimizer
#    'learningRate':    1e-4,
#    'beta1' :          .9,
#    'beta2' :          .999,
#    'epsilon':         1e-8,
#    'numClasses': trainDataObj.numClasses+1,
#    'idxToName': trainDataObj.idxToName,
#    'preTrain': True,
#}
#
##Allocate tensorflow object
##This will build the graph
#tfObj = VGGDetGap(params, trainDataObj.inputShape)
#
#print "Done init"
#tfObj.runModel(trainDataObj, testDataObj = testDataObj)
#print "Done run"
#
#tfObj.closeSess()

