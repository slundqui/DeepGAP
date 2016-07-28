import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.image import vocDetObj
from tf.VGGDetGap import VGGDetGap
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
trainDataObj = vocDetObj(trainImageList, trainImagePrefix, trainGTPrefix, resizeMethod="aug", normStd=False, augument=True)
testDataObj = vocDetObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, augument=False, shuffle=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/eval_voc_det_ds/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/voc_det_vgg_ds.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:1',
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
    'numClasses': trainDataObj.numClasses+1,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGDetGap(params, trainDataObj.inputShape)

print "Done init"
est = np.zeros((testDataObj.numImages, 20))
gt = np.zeros((testDataObj.numImages, 20))

assert(testDataObj.numImages % params["batchSize"] == 0)
for i in range(testDataObj.numImages/params["batchSize"]):
    print i*params["batchSize"], "out of", testDataObj.numImages
    (inImage, inGt) = testDataObj.getData(params["batchSize"])
    outVals = tfObj.evalModel(inImage, inGt = inGt, plot=False)
    tfObj.timestep += 1
    (batch, c, y, x) = outVals.shape
    for b in range(batch):
        outIdx = i * batch + b
        v = outVals[b, :-1, :, :] #Throw out distractor
        v = v.reshape(c-1, y*x)
        #cv = np.max(v, axis=1)
        cv = np.mean(v, axis=1)

        ##Average confidence across winners
        #maxClass = np.argmax(v, axis=0)
        #cv = np.zeros((c-1))
        #for k in range(c-1):
        #    classIdxs = np.nonzero(maxClass == k)
        #    if(len(classIdxs[0]) == 0):
        #        cv[k] = 0
        #    else:
        #        cv[k] = np.mean(v[k, classIdxs])

        #maxIdxs = v.argsort(axis=1)[:, -4:][:, ::-1]
        #cv = np.zeros((c-1))
        #for k in range(c-1):
        #    maxVals = v[k, maxIdxs[k]]
        #    cv[k] = np.mean(maxVals)

        est[outIdx,:] = cv
        gt[outIdx, np.unique(np.nonzero(inGt[b, :, :, :-1]==1)[2])] = 1

runDir = params["outDir"]+params["runDir"]
plotDir = runDir + params["plotDir"]
np.save(runDir + "est.pkl", est)
np.save(runDir + "gt.pkl", gt)

calcMetric(est, gt, plotDir + "pvr.png")
print "Done run"

tfObj.closeSess()

