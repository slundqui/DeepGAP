import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import kittiVidPvObj
from tf.SupVid_kitti import SupVid_kitti
from tf.SupVidMLP_kitti import SupVidMLP_kitti
from tf.SupVidMLP2_kitti import SupVidMLP2_kitti
import numpy as np
import pdb

#Paths to list of filenames
#Since we reshape from 6 to 3x2 (3 time, 2 stereo), left/right spin fastest
trainInputs = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft0.pvp",
            #"/home/slundquist/mountData/kitti_pv/objdet_train2/FrameRight0.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft1.pvp",
            #"/home/slundquist/mountData/kitti_pv/objdet_train2/FrameRight1.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp",
            #"/home/slundquist/mountData/kitti_pv/objdet_train2/FrameRight2.pvp",
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

trainRangeFn = "/home/slundquist/mountData/kitti_pv/kitti_objdet_train_list.txt"
testRangeFn = "/home/slundquist/mountData/kitti_pv/kitti_objdet_test_list.txt"

trainf = open(trainRangeFn, 'r')
trainLines = trainf.readlines()
trainf.close()
trainRange = [int(l) for l in trainLines]

testf = open(testRangeFn, 'r')
testLines = testf.readlines()
testf.close()
testRange = [int(l) for l in testLines]

#Get object from which tensorflow will pull data from
trainDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=True, rangeIdx=trainRange, binClass=[1, 2, 3])
testDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=False, rangeIdx=testRange, binClass=[1, 2, 3])

stage1_params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/eval_sup_kitti_vid_time_4x8_boot_1_bin/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        1,
    #Controls how often to write out to tensorboard
    'writeStep':       50, #300,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/sup_kitti_vid_time_4x8_boot_1_bin/checkpoints/save-model-30100",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      302, #1000000,
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
    'stereo': False,
    'time': True,
}

#Allocate tensorflow object
#This will build the graph
tfObj = SupVid_kitti(stage1_params, trainDataObj.inputShape)

print "Done init"
tfObj.evalModelBatch(testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

#Reset index
testDataObj.imgIdx = 0
stage2_params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/eval_sup_kitti_vid_time_4x8_boot_2_bin/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        1,
    #Controls how often to write out to tensorboard
    'writeStep':       50, #300,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/sup_kitti_vid_time_4x8_boot_2_bin/checkpoints/save-model-30100",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      302, #1000000,
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
    'resLoad': False,
    'stereo': False,
    'time': True,
}

#Allocate tensorflow object
#This will build the graph
tfObj = SupVidMLP_kitti(stage2_params, trainDataObj.inputShape)

print "Done init"
tfObj.evalModelBatch(testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

#Reset index
testDataObj.imgIdx = 0
stage3_params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/eval_sup_kitti_vid_time_4x8_boot_3_bin/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      100, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      100, #With respect to displayPeriod
    #Progress step
    'progress':        1,
    #Controls how often to write out to tensorboard
    'writeStep':       50, #300,
    #Flag for loading weights from checkpoint
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/sup_kitti_vid_time_4x8_boot_3_bin/checkpoints/save-model-30100",
    #Device to run on
    'device':          '/gpu:0',
    #Num iterations
    'outerSteps':      1002, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       16,
    #Learning rate for optimizer
    'learningRate':    1e-3,
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
    'resLoad': False,
    'stereo': False,
    'time': True,
}

#Allocate tensorflow object
#This will build the graph
tfObj = SupVidMLP2_kitti(stage3_params, trainDataObj.inputShape)

print "Done init"
tfObj.evalModelBatch(testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

