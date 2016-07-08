import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.image import imageNetDetObj
from tf.VGGDetGap import VGGDetGap
import numpy as np
import pdb

#Paths to list of filenames
trainImageList = "/shared/imageNet/DET/ILSVRC2015/ImageSets/DET/train.txt"
testImageList = "/shared/imageNet/DET/ILSVRC2015/ImageSets/DET/val.txt"

trainImagePrefix = "/shared/imageNet/DET/ILSVRC2015/Data/DET/train/"
testImagePrefix =  "/shared/imageNet/DET/ILSVRC2015/Data/DET/val/"

trainGTPrefix = "/shared/imageNet/DET/ILSVRC2015/Annotations/DET/train/"
testGTPrefix =  "/shared/imageNet/DET/ILSVRC2015/Annotations/DET/val/"

clsMeta = "/shared/imageNet/devkit/data/meta_det.mat"

#Get object from which tensorflow will pull data from
trainDataObj = imageNetDetObj(trainImageList, trainImagePrefix, trainGTPrefix, clsMeta, resizeMethod="crop", normStd=False)
testDataObj = imageNetDetObj(testImageList, testImagePrefix, testGTPrefix, clsMeta, resizeMethod="crop", normStd=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/imagenet_det_vgg/",
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
    'writeStep':       5, #300,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/imagenet_vgg_bias.ckpt",
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
    'epsilon':         1e-6,
    'numClasses': trainDataObj.numClasses+1,
    'idxToName': trainDataObj.idxToName,
    'preTrain': True,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGDetGap(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

