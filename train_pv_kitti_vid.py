import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import kittiVidPvObj
from tf.SLPVid import SLPVid
import numpy as np
import pdb

#Paths to list of filenames
trainInputs = [
            "/home/sheng/mountData/kitti_pv/objdet_train2/S1_0.pvp",
            "/home/sheng/mountData/kitti_pv/objdet_train2/S1_1.pvp",
            ]

trainGts = [
            "/home/sheng/mountData/kitti_pv/objdet_train2/GroundTruth2.pvp",
        ]
trainFilenames = [
            "/home/sheng/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp",
        ]
dncFilenames= [
            "/home/sheng/mountData/kitti_pv/objdet_train2/DNC2.pvp",
        ]

#trainFnPrefix = "/shared/KITTI/objdet/training/"

#Get object from which tensorflow will pull data from
trainDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=True, startIdx = 0, stopIdx = 6000)
testDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=True, startIdx=6000, stopIdx=-1)

params = {
    #Base output directory
    'outDir':          "/home/sheng/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/pv_kitti_vid_2x2/",
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
    'loadFile':        "/home/sheng/mountData/DeepGAP/saved/pv_imagenet_vid_2x4.ckpt",
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
    'learningRateBias': 1e-6,
    #'numClasses': trainDataObj.numClasses,
    'numClasses': trainDataObj.numClasses,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
    'lossWeight': None,
    'gtShape': trainDataObj.gtShape,
    'gtSparse': False
}

#Allocate tensorflow object
#This will build the graph
tfObj = SLPVid(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

