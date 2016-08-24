import matplotlib
matplotlib.use('Agg')
from dataObj.image import evalObj
from tf.VGGDetGap import VGGDetGap
import numpy as np
import pdb
import sys

#Paths to list of filenames
if(len(sys.argv) != 2):
    print "Usage: python vgg_eval.py <filename>"
    assert(0)

inImageList = sys.argv[1]

#trainImageList = "/home/slundquist/mountData/datasets/imagenet/train_cls.txt"
#testImageList = "/home/slundquist/mountData/datasets/imagenet/val_cls.txt"

clsMeta = "/shared/imageNet/devkit/data/meta_det.mat"
#clsMeta = "/home/slundquist/mountData/DeepGAP/saved/meta_det.mat"

#Get object from which tensorflow will pull data from
evalDataObj = evalObj(inImageList, clsMeta, resizeMethod="crop", normStd=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/demo/",
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
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/imagenet_det.ckpt",
    #Input vgg file for preloaded weights
    #'vggFile':         "/home/slundquist/mountData/DeepGAP/saved/imagenet-vgg-verydeep-16.mat",
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/cpu:0',
    #####ISTA PARAMS######
    #Num iterations
    'outerSteps':      10000000, #1000000,
    'innerSteps':      100, #300,
    #Batch size
    'batchSize':       1,
    #Learning rate for optimizer
    'learningRate':    1e-4,
    'beta1' :          .9,
    'beta2' :          .999,
    'epsilon':         1e-8,
    'numClasses': 200+1,
    'idxToName': evalDataObj.idxToName,
    'preTrain': False,
    'regStrength': .001,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGDetGap(params, evalDataObj.inputShape)

print "Done init"
(outVals, outIdx) = tfObj.evalModelBatch(evalDataObj.getData(evalDataObj.numImages))
filenames = evalDataObj.imgFiles

outFile = params["outDir"] + params["runDir"] + "output.txt"
f = open(outFile, 'w')


for filename, val, idx in zip(filenames, outVals.tolist(), outIdx.tolist()):
    outStr = filename + ","
    for v, i in zip(val, idx):
        outStr += "[\"" + evalDataObj.idxToName[int(i)] + "\"]," + str(v) +","
    outStr += "\n"
    f.write(outStr)

print "Done run"

tfObj.closeSess()

