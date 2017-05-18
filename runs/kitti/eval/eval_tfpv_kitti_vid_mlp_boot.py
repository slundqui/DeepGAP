import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import kittiVidPvObj
from tf.SLPVid import SLPVid
from tf.MLPVid import MLPVid
from tf.MLPVid2 import MLPVid2
import numpy as np
import pdb

#Paths to list of filenames
trainInputs = [
            "/home/slundquist/mountData/tfSparseCode/lca_adam_kitti_eval/time0.pvp",
            "/home/slundquist/mountData/tfSparseCode/lca_adam_kitti_eval/time1.pvp",
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

#trainFnPrefix = "/shared/KITTI/objdet/training/"

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
trainDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=False, rangeIdx=trainRange, binClass=[1, 2, 3])
testDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=False, rangeIdx=testRange, binClass=[1, 2, 3])

loadStrSuffix = [
        "10100",
        "10100",
        "10100",
        ]

device = "/gpu:1"

for i in range(6, 7):
    runSuffix = "pretrain_500_run"+str(i)
    stage1_params = {
        #Base output directory
        'outDir':          "/home/slundquist/mountData/DeepGAP/",
        #Inner run directory
        'runDir':          "/eval_tfpv_kitti_vid_4x8_boot_1_bin_"+runSuffix+"/",
        'tfDir':           "/tfout",
        #Save parameters
        'ckptDir':         "/checkpoints/",
        'saveFile':        "/save-model",
        'savePeriod':      100, #In terms of displayPeriod
        #output plots directory
        'plotDir':         "plots/",
        'plotPeriod':      500, #With respect to displayPeriod
        #Progress step
        'progress':        1,
        #Controls how often to write out to tensorboard
        'writeStep':       50, #300,
        #Flag for loading weights from checkpoint
        'load':            True,
        'loadFile':        "/media/data//slundquist/mountData/DeepGAP/tfpv_kitti_vid_4x8_boot_1_bin_" + runSuffix + "/checkpoints/save-model-" + loadStrSuffix[0],
        #Device to run on
        'device':          device,
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
        'numClasses': trainDataObj.numClasses,
        'idxToName': trainDataObj.idxToName,
        'preTrain': False,
        'lossWeight': trainDataObj.lossWeight,
        'gtShape': trainDataObj.gtShape,
        'gtSparse': False,
        'inputScale': 50,
        'regWeight': 0,
    }

    #Allocate tensorflow object
    #This will build the graph
    tfObj = SLPVid(stage1_params, trainDataObj.inputShape)

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
        'runDir':          "/eval_tfpv_kitti_vid_4x8_boot_2_bin_" + runSuffix + "/",
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
        'loadFile':        "/media/data/slundquist/mountData/DeepGAP/tfpv_kitti_vid_4x8_boot_2_bin_" + runSuffix + "/checkpoints/save-model-" + loadStrSuffix[1],
        #Device to run on
        'device':          device,
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
        'inputScale': 50,
        'regWeight': 0,
        'resLoad': False,
    }

    #Allocate tensorflow object
    #This will build the graph
    tfObj = MLPVid(stage2_params, trainDataObj.inputShape)

    print "Done init"
    tfObj.evalModelBatch(testDataObj = testDataObj)
    print "Done run"

    tfObj.closeSess()

    testDataObj.imgIdx = 0
    stage3_params = {
        #Base output directory
        'outDir':          "/home/slundquist/mountData/DeepGAP/",
        #Inner run directory
        'runDir':          "/eval_tfpv_kitti_vid_4x8_boot_3_bin_"+runSuffix + "/",
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
        'loadFile':        "/media/data/slundquist/mountData/DeepGAP/tfpv_kitti_vid_4x8_boot_3_bin_" + runSuffix + "/checkpoints/save-model-" + loadStrSuffix[2],
        #Device to run on
        'device':          device,
        #Num iterations
        'outerSteps':      300, #1000000,
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
        'inputScale': 50,
        'regWeight': 0,
        'resLoad': True,
        'regWeight': 0,
        'resLoad': False,
    }

    #Allocate tensorflow object
    #This will build the graph
    tfObj = MLPVid2(stage3_params, trainDataObj.inputShape)

    print "Done init"
    tfObj.evalModelBatch(testDataObj = testDataObj)
    print "Done run"

    tfObj.closeSess()




