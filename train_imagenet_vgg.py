import matplotlib
matplotlib.use('Agg')
from dataObj.image import imageNetObj
from tf.VGGGap import VGGGap
import numpy as np
import pdb

#Paths to list of filenames
trainImageList = "/home/slundquist/mountData/datasets/imagenet/train_cls.txt"
testImageList = "/home/slundquist/mountData/datasets/imagenet/val_cls.txt"

trainImagePrefix = "/shared/imageNet/CLS_LOC/ILSVRC2015/Data/CLS-LOC/train/"
testImagePrefix =  "/shared/imageNet/CLS_LOC/ILSVRC2015/Data/CLS-LOC/val/"

clsMeta = "/shared/imageNet/devkit/data/meta_clsloc.mat"

#Get object from which tensorflow will pull data from
trainDataObj = imageNetObj(trainImageList, trainImagePrefix, clsMeta, useClassDir = True, resizeMethod="crop", normStd=False)
testDataObj = imageNetObj(testImageList, testImagePrefix, clsMeta, useClassDir = False, resizeMethod="crop", normStd=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/imagenet_vgg/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      10, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      10, #With respect to displayPeriod
    #Progress step
    'progress':        1,
    #Controls how often to write out to tensorboard
    'writeStep':       50, #300,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/imagenet_vgg.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       8,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'beta1' :          .9,
    'beta2' :          .999,
    'epsilon':         1e-8,
    'numClasses': trainDataObj.numClasses,
    'idxToName': trainDataObj.idxToName,
    'preTrain': True,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGGap(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

