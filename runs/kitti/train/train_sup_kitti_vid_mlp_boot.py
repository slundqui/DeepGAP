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
testDataObj = kittiVidPvObj(trainInputs, trainGts, trainFilenames, dncFilenames, None, shuffle=True, rangeIdx=testRange, binClass=[1, 2, 3])

numTotalSteps = [
        302,
        102,
        102,
        ]

loadStrSuffix = [
        None,
        "30100",
        "10100",
        ]

device = '/gpu:0'

for i in range(1, 7):
    #runSuffix = "run"+str(i)
    runSuffix = "pretrain_run"+str(i)
    #stage1_params = {
    #    #Base output directory
    #    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #    #Inner run directory
    #    'runDir':          "/sup_kitti_vid_4x8_boot_1_bin_" + runSuffix+ "/",
    #    'tfDir':           "/tfout",
    #    #Save parameters
    #    'ckptDir':         "/checkpoints/",
    #    'saveFile':        "/save-model",
    #    'savePeriod':      100, #In terms of displayPeriod
    #    #output plots directory
    #    'plotDir':         "plots/",
    #    'plotPeriod':      100, #With respect to displayPeriod
    #    #Progress step
    #    'progress':        1,
    #    #Controls how often to write out to tensorboard
    #    'writeStep':       50, #300,
    #    #Flag for loading weights from checkpoint
    #    'load':            False,
    #    'loadFile':        "/home/slundquist/mountData/DeepGAP/sup_kitti_vid_4x8_boot_1_bin_100_run3/checkpoints/save-model-5100",
    #    #Device to run on
    #    'device':          device,
    #    #####ISTA PARAMS######
    #    #Num iterations
    #    'outerSteps':      numTotalSteps[0], #1000000,
    #    'innerSteps':      100, #300,
    #    #Batch size
    #    'batchSize':       16,
    #    #Learning rate for optimizer
    #    'learningRate':    1e-4,
    #    'beta1' :          .9,
    #    'beta2' :          .999,
    #    'epsilon':         1e-8,
    #    'learningRateBias': 1e-6,
    #    #'numClasses': trainDataObj.numClasses,
    #    'numClasses': trainDataObj.numClasses,
    #    'idxToName': trainDataObj.idxToName,
    #    'preTrain': False, #True is do not train first layer weights
    #    'lossWeight': trainDataObj.lossWeight,
    #    'gtShape': trainDataObj.gtShape,
    #    'gtSparse': False,
    #    'regWeight': 0,
    #    'stereo': True,
    #    'time': True,
    #    'numFeatures': 3072,
    #    'plotInd': False,
    #    'plotFM': False,
    #    'augment': False,
    #    'augMean': .1,
    #    'augStd': .1,
    #    'loadHiddenWeights': False,
    #    'hiddenWeightsFile' : "/home/slundquist/mountData/tfSparseCode/lca_adam_kitti_weights/checkpoints/weights_300.npy",
    #}

    ##Allocate tensorflow object
    ##This will build the graph
    #tfObj = SupVid_kitti(stage1_params, trainDataObj.inputShape)

    #print "Done init"
    #tfObj.runModel(trainDataObj, testDataObj = testDataObj)
    #print "Done run"

    #tfObj.closeSess()

    stage2_params = {
        #Base output directory
        'outDir':          "/home/slundquist/mountData/DeepGAP/",
        #Inner run directory
        'runDir':          "/sup_kitti_vid_4x8_boot_2_bin_" + runSuffix + "/",
        'tfDir':           "/tfout",
        #Save parameters
        'ckptDir':         "/checkpoints/",
        'saveFile':        "/save-model",
        'savePeriod':      50, #In terms of displayPeriod
        #output plots directory
        'plotDir':         "plots/",
        'plotPeriod':      100, #With respect to displayPeriod
        #Progress step
        'progress':        1,
        #Controls how often to write out to tensorboard
        'writeStep':       50, #300,
        #Flag for loading weights from checkpoint
        'load':            True,
        'loadFile':        "/media/data/slundquist/mountData/DeepGAP/sup_kitti_vid_4x8_boot_1_bin_" + runSuffix + "/checkpoints/save-model-" + loadStrSuffix[1],
        #Device to run on
        'device':          device,
        #####ISTA PARAMS######
        #Num iterations
        'outerSteps':      numTotalSteps[1], #1000000,
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
        'resLoad': True,
        'stereo': True,
        'time': True,
        'numFeatures': 3072,
        'plotInd': False,
        'plotFM': False,
    }

    #Allocate tensorflow object
    #This will build the graph
    tfObj = SupVidMLP_kitti(stage2_params, trainDataObj.inputShape)

    print "Done init"
    tfObj.runModel(trainDataObj, testDataObj = testDataObj)
    print "Done run"

    tfObj.closeSess()


    stage3_params = {
        #Base output directory
        'outDir':          "/home/slundquist/mountData/DeepGAP/",
        #Inner run directory
        'runDir':          "/sup_kitti_vid_4x8_boot_3_bin_" + runSuffix + "/",
        'tfDir':           "/tfout",
        #Save parameters
        'ckptDir':         "/checkpoints/",
        'saveFile':        "/save-model",
        'savePeriod':      50, #In terms of displayPeriod
        #output plots directory
        'plotDir':         "plots/",
        'plotPeriod':      100, #With respect to displayPeriod
        #Progress step
        'progress':        1,
        #Controls how often to write out to tensorboard
        'writeStep':       50, #300,
        #Flag for loading weights from checkpoint
        'load':            True,
        'loadFile':        "/media/data/slundquist/mountData/DeepGAP/sup_kitti_vid_4x8_boot_2_bin_" + runSuffix + "/checkpoints/save-model-" + loadStrSuffix[2],
        #Device to run on
        'device':          device,
        #Num iterations
        'outerSteps':      numTotalSteps[2], #1000000,
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
        'regWeight': 0,
        'resLoad': True,
        'stereo': True,
        'time': True,
        'numFeatures': 3072,
        'plotInd': False,
        'plotFM': False,
    }

    #Allocate tensorflow object
    #This will build the graph
    tfObj = SupVidMLP2_kitti(stage3_params, trainDataObj.inputShape)

    print "Done init"
    tfObj.runModel(trainDataObj, testDataObj = testDataObj)
    print "Done run"

    tfObj.closeSess()

