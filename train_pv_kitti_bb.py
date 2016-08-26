import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from dataObj.pv_image import kittiVidPvObj
#from tf.SLPBBVid import SLPBBVid
from tf.MLPBBVid import MLPBBVid
import numpy as np
import pdb

#Paths to list of filenames
trainInputs = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/S1_0.pvp",
            "/home/slundquist/mountData/kitti_pv/objdet_train2/S1_1.pvp",
            ]

trainGts = [
            "/home/slundquist/mountData/kitti_iou/kitti_iou8x8.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou16x8.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou8x16.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou16x16.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou32x16.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou16x32.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou32x32.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou64x32.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou32x64.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou64x64.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou64x128.pvp",
        ]

trainFilenames = [
            "/home/slundquist/mountData/kitti_pv/objdet_train2/FrameLeft2.pvp",
        ]
dncFilenames= [
            "/home/slundquist/mountData/kitti_iou/kitti_iou8x8_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou16x8_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou8x16_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou16x16_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou32x16_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou16x32_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou32x32_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou64x32_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou32x64_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou64x64_mask.pvp",
            "/home/slundquist/mountData/kitti_iou/kitti_iou64x128_mask.pvp",
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

bbWindowSize=[
            (8, 8)  , (16, 8) , (8, 16) ,
            (16, 16), (32, 16), (16, 32),
            (32, 32), (64, 32), (32, 64),
            (64, 64), (64, 128),
           ]

bbStrides=[ #In encoding space
            (1, 1)  , (1, 1)  , (1, 1) ,
            (1, 1)  , (2, 1)  , (1, 2),
            (2, 2)  , (4, 2)  , (2, 4),
            (4, 4)  , (4, 8)  ,
           ]

dnc = trainDataObj.getDnc()
(windows, ny, nx, nf) = dnc.shape

pxToEncoding = .25
#We make a stride based on bbWindowSize

strideMask = np.zeros(dnc.shape)
for i, (sy, sx) in enumerate(bbStrides):
    iy = range(0, ny, sy)
    ix = range(0, nx, sx)
    for y in iy:
        for x in ix:
            strideMask[i, y, x, :] = 1

strideDnc = dnc*strideMask

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/pv_kitti_vid_bb/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/pv_kitti_vid_bb.ckpt",
    #Device to run on
    'device':          '/gpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       2,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'beta1' :          .9,
    'beta2' :          .999,
    'epsilon':         1e-8,
    'learningRateBias': 1e-6,
    'numClasses': trainDataObj.numClasses+1,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
    'lossWeight': trainDataObj.lossWeight,
    'gtShape': trainDataObj.gtShape,
    'imageShape': trainDataObj.imageShape,
    'gtSparse': True,
    'dncVals': strideDnc,
    'bbWindowSize': bbWindowSize,
    'iouThresh':.5,
    'minIouThresh':.1,
    'inputScale': 100,
    'gtStrideY':2,
    'gtStrideX':2,
}

#Allocate tensorflow object
#This will build the graph
tfObj = MLPBBVid(params, trainDataObj.inputShape)

print "Done init"
tfObj.runModel(trainDataObj, testDataObj = testDataObj)
print "Done run"

tfObj.closeSess()

