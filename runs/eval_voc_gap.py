import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataObj.image import vocObj
from tf.VGGGap import VGGGap
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
trainDataObj = vocObj(trainImageList, trainImagePrefix, trainGTPrefix, resizeMethod="aug", normStd=False, augument=True)
testDataObj = vocObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, augument=False, shuffle=False, singleObj=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/eval_voc_gap_vgg/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/voc_gap_vgg.ckpt",
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
    'numClasses': trainDataObj.numClasses,
    'idxToName': trainDataObj.idxToName,
    'preTrain': False,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGGap(params, trainDataObj.inputShape)

print "Done init"
estCam = np.zeros((testDataObj.numImages, 20))
estWhole = np.zeros((testDataObj.numImages, 20))

gt = np.zeros((testDataObj.numImages, 20))

histVals = np.zeros((20, testDataObj.numImages*14*14))
assert(testDataObj.numImages % params["batchSize"] == 0)
for i in range(testDataObj.numImages/params["batchSize"]):
    print i*params["batchSize"], "out of", testDataObj.numImages
    (inImage, inGt) = testDataObj.getData(params["batchSize"])
    (estOut, camOut) = tfObj.evalModel(inImage, inGt = inGt, plot=False)

    tfObj.timestep += 1
    (batch, c, y, x) = camOut.shape
    for b in range(batch):
        outIdx = i * batch + b
        v = camOut[b, :, :, :]
        #Subtract average
        #avgCam = np.mean(v, axis=0)
        #v = v - avgCam
        #Average 4 max confidence values
        v = v.reshape(c, y*x)
        sIdx = outIdx*y*x
        eIdx = (outIdx+1)*y*x
        histVals[:, sIdx:eIdx] = v
        cv = np.mean(v, axis=1)
        #cv = np.max(v, axis=1)

        ##Average confidence across winners
        #maxClass = np.argmax(v, axis=0)
        #cv = np.zeros((c))
        #for k in range(c):
        #    classIdxs = np.nonzero(maxClass == k)
        #    if(len(classIdxs[0]) == 0):
        #        cv[k] = 0
        #    else:
        #        cv[k] = np.mean(v[k, classIdxs])

        #maxIdxs = v.argsort(axis=1)[:, -4:][:, ::-1]
        #cv = np.zeros((c))
        #for k in range(c):
        #    maxVals = v[k, maxIdxs[k]]
        #    cv[k] = np.mean(maxVals)

        estCam[outIdx,:] = cv
        estWhole[outIdx, :] = estOut[b, :]
        gt[outIdx, np.nonzero(inGt[b, :]==1)[0]] = 1


runDir = params["outDir"]+params["runDir"]
plotDir = runDir + params["plotDir"]
np.save(runDir + "estCam.pkl", estCam)
np.save(runDir + "estWhole.pkl", estWhole)
np.save(runDir + "gt.pkl", gt)

for k in range(20):
    histV = histVals[k, :]
    plt.hist(histV, 20, log=True)
    plt.savefig(plotDir + "/" + params["idxToName"][k] + "_hist.png")
    plt.close()

plt.hist(histVals.flatten(), 20, log=True)
plt.savefig(plotDir + "/all_hist.png")


calcMetric(estCam, gt, plotDir + "pvrCam.png")
calcMetric(estWhole, gt, plotDir + "pvrWhole.png")
print "Done run"

tfObj.closeSess()

