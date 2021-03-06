import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.image import vocDetObj
from tf.VGGDetGap import VGGDetGap
import numpy as np
import pdb

#Paths to list of filenames
trainImageList = "/shared/VOCdevkit/VOC2007/ImageSets/Main/train_trainval.txt"
testImageList = "/home/slundquist/mountData/voc/val.txt"

trainImagePrefix = "/shared/VOCdevkit/VOC2007/JPEGImages/"
testImagePrefix = "/shared/VOCdevkit/VOC2007/JPEGImages/"

trainGTPrefix = "/shared/VOCdevkit/VOC2007/Annotations/"
testGTPrefix =  "/shared/VOCdevkit/VOC2007/Annotations/"

#Get object from which tensorflow will pull data from
trainDataObj = vocDetObj(trainImageList, trainImagePrefix, trainGTPrefix, resizeMethod="aug", normStd=False, augument=True)
testDataObj = vocDetObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, augument=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/voc_det_vgg_ds/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      10, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      20, #With respect to displayPeriod
    #Progress step
    'progress':        1,
    #Controls how often to write out to tensorboard
    'writeStep':       100, #300,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/voc_det_vgg_ds.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:1',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      100, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       8,
    #Learning rate for optimizer
    'learningRate':    1e-5,
    'beta1' :          .9,
    'beta2' :          .999,
    'regStrength':     .01,
    'epsilon':         1e-8,
    'numClasses': trainDataObj.numClasses+1,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGDetGap(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

