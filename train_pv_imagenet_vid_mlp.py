import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import imageNetVidPvObj
from tf.MLPVid import MLPVid
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
            "/home/slundquist/mountData/imagenet_pv/gtWrite_2x4/GroundTruth3.pvp",
        ]
trainFilenames = [
            "/home/slundquist/mountData/imagenet_pv/train3/timestamps/Frame3.txt",
        ]

testInputs = [
            "/home/slundquist/mountData/imagenet_pv/val_7objects3/S1_0.pvp",
            "/home/slundquist/mountData/imagenet_pv/val_7objects3/S1_1.pvp",
            "/home/slundquist/mountData/imagenet_pv/val_7objects3/S1_2.pvp",
            "/home/slundquist/mountData/imagenet_pv/val_7objects3/S1_3.pvp",
            ]

testGts = [
            "/home/slundquist/mountData/imagenet_pv/gtWrite_val_2x4/GroundTruth3.pvp",
        ]
testFilenames = [
            "/home/slundquist/mountData/imagenet_pv/val_7objects3/timestamps/Frame3.txt",
        ]

trainFnPrefix = "/shared/imageNet/vid2015_128x64/"
testFnPrefix = "/shared/imageNet/vid2015_128x64/val_7objects/"

#Get object from which tensorflow will pull data from
trainDataObj = imageNetVidPvObj(trainInputs, trainGts, trainFilenames, trainFnPrefix, shuffle=True)
testDataObj = imageNetVidPvObj(testInputs, testGts, testFilenames, testFnPrefix, shuffle=True)


params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/pv_imagenet_vid_2x4_mlp_dropout/",
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
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/pv_imagenet_vid_2x4_mlp_dropout.ckpt",
    #Device to run on
    'device':          '/gpu:1',
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
    'learningRateBias': 1e-6,
    #'numClasses': trainDataObj.numClasses,
    'numClasses': 8,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
    'lossWeight': trainDataObj.lossWeight,
    'gtShape': trainDataObj.gtShape,
    'gtSparse':True,
}



#Allocate tensorflow object
#This will build the graph
tfObj = MLPVid(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

