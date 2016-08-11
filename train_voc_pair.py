import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.image import vocObj
from tf.VGGPair import VGGPair
import numpy as np
import pdb

#Paths to list of filenames
trainImageList = "/home/sheng/mountData/dataset/VOCdevkit/VOC2007/ImageSets/Main/train_trainval.txt"
testImageList = "/home/sheng/mountData/voc/val.txt"

trainImagePrefix = "/home/sheng/mountData/dataset/VOCdevkit/VOC2007/JPEGImages/"
testImagePrefix =  "/home/sheng/mountData/dataset/VOCdevkit/VOC2007/JPEGImages/"

trainGTPrefix = "/home/sheng/mountData/dataset/VOCdevkit/VOC2007/Annotations/"
testGTPrefix =  "/home/sheng/mountData/dataset/VOCdevkit/VOC2007/Annotations/"

#Get object from which tensorflow will pull data from
trainDataObj = vocObj(trainImageList, trainImagePrefix, trainGTPrefix, resizeMethod="aug", normStd=False, augument=True, singleObj=False)
testDataObj = vocObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, augument=False, singleObj=False)

params = {
    #Base output directory
    'outDir':          "/home/sheng/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/voc_vgg_pair/",
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
    'loadFile':        "/home/sheng/mountData/DeepGAP/saved/voc_vgg_pair.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/sheng/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      10000, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       8,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'beta1' :          .9,
    'beta2' :          .999,
    'regStrength':     .001,
    'epsilon':         1e-8,
    'numClasses': trainDataObj.numClasses,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGPair(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

