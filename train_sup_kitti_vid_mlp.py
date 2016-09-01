import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import kittiVidPvObj
from tf.SupVidMLP_kitti import SupVidMLP_kitti
import numpy as np
import pdb

#import os
#import signal
#def sig_handler(signum, frame):
#    print "segfault"
#    pdb.set_trace()
#signal.signal(signal.SIGSEGV, sig_handler)



#Paths to list of filenames
#Since we reshape from 6 to 3x2 (3 time, 2 stereo), left/right spin fastest
trainInputs = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft0.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameRight0.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft1.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameRight1.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameRight2.pvp",
            ]

trainGts = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/GroundTruth2Background.pvp",
        ]
trainFilenames = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp",
        ]

dncFilenames= [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/DNCPixels2.pvp",
        ]

trainRangeFn = "/shared/KITTI/objdet/training/genData/kitti_objdet_train_list.txt"
testRangeFn = "/shared/KITTI/objdet/training/genData/kitti_objdet_test_list.txt"

trainf = open(trainRangeFn, 'r')
trainLines = trainf.readlines()
trainf.close()
trainRange = [int(l) for l in trainLines]

testf = open(testRangeFn, 'r')
testLines = testf.readlines()
testf.close()
testRange = [int(l) for l in testLines]

#Get object from which tensorflow will pull data from
trainDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=True, rangeIdx=trainRange)
testDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=True, rangeIdx=testRange)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/sup_kitti_vid_4x8_mlp/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/sup_kitti_vid_4x8.ckpt",
    #Device to run on
    'device':          '/gpu:1',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      1000000, #1000000,
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
    'lossWeight': trainDataObj.lossWeight,
    'gtShape': trainDataObj.gtShape,
    'gtSparse': False,
    'regWeight': 0,
}

#Allocate tensorflow object
#This will build the graph
tfObj = SupVidMLP_kitti(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

