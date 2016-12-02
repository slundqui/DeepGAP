import matplotlib
matplotlib.use('Agg')
from dataObj.image import cifarObj
from tf.VGGGap import VGGGap
import numpy as np
import pdb

#Paths to list of filenames
trainImageList = "/home/slundquist/mountData/datasets/cifar/images/train.txt"
testImageList = "/home/slundquist/mountData/datasets/cifar/images/test.txt"

#Get object from which tensorflow will pull data from
trainDataObj = cifarObj(trainImageList, resizeMethod="pad")
testDataObj = cifarObj(testImageList, resizeMethod="pad")

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/cifar/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        10,
    #Controls how often to write out to tensorboard
    'writeStep':       100, #300,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/cifar.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-f.mat",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':   1000, #1000000,
    'innerSteps':   50, #300,
    #Batch size
    'batchSize':     128,
    #Learning rate for optimizer
    'learningRate':   1e-5,
    'numClasses': 10,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGGap(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

