import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import kittiVidPvObj
from tf.SupVGGFRCNN_kitti import VGG_FRCNN
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

gts= [
            "/home/slundquist/mountData/kitti_iou_bin_obj/kitti_iou_bin_objFile.pvp",
            "/home/slundquist/mountData/kitti_iou_bin_obj/kitti_iou_bin_labelFile.pvp",
        ]

imageFns = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp",
        ]

dncFilenames= [
            "/home/slundquist/mountData/kitti_iou_bin_obj/kitti_iou_bin_anchor.pvp",
            "/home/slundquist/mountData/kitti_iou_bin_obj/kitti_iou_bin_mask.pvp",
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

trainseed=None
testseed=None

#Get object from which tensorflow will pull data from
trainDataObj = kittiVidPvObj(trainInputs, gts, imageFns, dncFilenames, None, shuffle=True, rangeIdx=trainRange, seed=trainseed)
testDataObj = kittiVidPvObj(trainInputs, gts, imageFns, dncFilenames, None, shuffle=True, rangeIdx=testRange, seed=testseed)


params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/sup_kitti_vgg_frcnn/",
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
    'writeStep':       10, #300,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/save-model-19100",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      100000, #1000000,
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
    'regLossWeight': 10,
    'dncArea': trainDataObj.getDnc()[1],
    'anchors': trainDataObj.getDnc()[0],
    'imageShape': [trainDataObj.inputShape[1], trainDataObj.inputShape[2], trainDataObj.inputShape[3]],
    'detConfidenceThreshold': None, #0.7,
    'iouDetThreshold': 0.5, #0.7
    'nmsIouThreshold': 0.5,
    'maxBB': 100,
    'maxNegSamples': 256,
    'vggFile': "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGG_FRCNN(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

