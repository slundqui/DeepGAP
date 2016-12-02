import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.image import vocObj
from tf.VGG import VGG
from plot.tagEval import calcMetric
import numpy as np
import pdb

#Paths to list of filenames
trainImageList = "/shared/VOCdevkit/VOC2007/ImageSets/Main/train_trainval.txt"
testImageList = "/home/slundquist/mountData/voc/test.txt"

trainImagePrefix = "/shared/VOCdevkit/VOC2007/JPEGImages/"
testImagePrefix = "/shared/VOCdevkit/VOC2007/JPEGImages/"

trainGTPrefix = "/shared/VOCdevkit/VOC2007/Annotations/"
testGTPrefix =  "/shared/VOCdevkit/VOC2007/Annotations/"

#Get object from which tensorflow will pull data from
trainDataObj = vocObj(trainImageList, trainImagePrefix, trainGTPrefix, resizeMethod="aug", normStd=False, augument=True)
testDataObj = vocObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, augument=False, shuffle=False, singleObj=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/eval_voc_vgg/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/voc_vgg.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      100, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       4,
    #Learning rate for optimizer
    'learningRate':    1e-5,
    'beta1' :          .9,
    'beta2' :          .999,
    'regStrength':     .01,
    'epsilon':         1e-8,
    'numClasses': trainDataObj.numClasses,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGG(params, trainDataObj.inputShape)

print "Done init"
estWhole = np.zeros((testDataObj.numImages, 20))

gt = np.zeros((testDataObj.numImages, 20))

assert(testDataObj.numImages % params["batchSize"] == 0)
for i in range(testDataObj.numImages/params["batchSize"]):
    print i*params["batchSize"], "out of", testDataObj.numImages
    (inImage, inGt) = testDataObj.getData(params["batchSize"])
    estOut = tfObj.evalModel(inImage, inGt = inGt, plot=False)
    (batch, drop) = estOut.shape
    tfObj.timestep += 1
    for b in range(batch):
        outIdx = i * batch + b
        estWhole[outIdx, :] = estOut[b, :]
        gt[outIdx, np.nonzero(inGt[b, :]==1)[0]] = 1

runDir = params["outDir"]+params["runDir"]
plotDir = runDir + params["plotDir"]
np.save(runDir + "estWhole.pkl", estWhole)
np.save(runDir + "gt.pkl", gt)

calcMetric(estWhole, gt, plotDir + "pvrWhole.png")
print "Done run"

tfObj.closeSess()

