import matplotlib
matplotlib.use('Agg')
from dataObj.pv_image import imageNetVidPvObj
from tf.SLPVid import SLPVid
import numpy as np
import pdb

#Paths to list of filenames
trainInputs = [
            "/home/slundquist/mountData/imagenet_pv/train3/S1_0.pvp",
            "/home/slundquist/mountData/imagenet_pv/train3/S1_1.pvp",
            "/home/slundquist/mountData/imagenet_pv/train3/S1_2.pvp",
            "/home/slundquist/mountData/imagenet_pv/train3/S1_3.pvp",
            ]
trainGts = [
            "/home/slundquist/mountData/imagenet_pv/train3/GroundTruth3.pvp",
        ]
trainFilenames = [
            "/home/slundquist/mountData/imagenet_pv/train3/timestamps/Frame3.txt",
        ]

fnPrefix = "/shared/imageNet/vid2015_128x64/"

#testImageLists = "/home/slundquist/mountData/datasets/imagenet/val_cls.txt"

#Get object from which tensorflow will pull data from
trainDataObj = imageNetVidPvObj(trainInputs, trainGts, trainFilenames, fnPrefix, shuffle=True)

#testDataObj = imageNetVidPvObj(testImageList, testImagePrefix, clsMeta, useClassDir = False, resizeMethod="crop", normStd=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/pv_imagenet_vid/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/imagenet_vgg_bias.ckpt",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       16,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'beta1' :          .9,
    'beta2' :          .999,
    'epsilon':         1e-8,
    'numClasses': trainDataObj.numClasses,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
}



#Allocate tensorflow object
#This will build the graph
tfObj = SLPVid(params, trainDataObj.inputShape)

print "Done init"
#tfObj.runModel(trainDataObj, testDataObj = testDataObj)
tfObj.runModel(trainDataObj)
print "Done run"

tfObj.closeSess()

