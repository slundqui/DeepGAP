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

#Path to imagenet det meta file
#clsMeta = "/shared/imageNet/devkit/data/meta_det.mat"
clsMeta = "/home/slundquist/mountData/DeepGAP/saved/meta_det.mat"

#Determines if the script makes plots
outputPlots = True

#Get object from which tensorflow will pull data from
evalDataObj = evalObj(inImageList, clsMeta, resizeMethod="crop", normStd=False)

params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/demo/",
    'tfDir':           "/tfout",
    #Save parameters - not used
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      10, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotPeriod':      1, #With respect to displayPeriod - not used
    #Progress step
    'progress':        1,
    #Controls how often to write out to tensorboard
    'writeStep':       1, #300,
    #Flag for loading weights from checkpoint
    #Point to pretrained model
    'load':            True,
    'loadFile':        "/home/slundquist/mountData/DeepGAP/saved/imagenet_det.ckpt",
    #Input vgg file for preloaded weights
    #'vggFile':         "/home/slundquist/mountData/DeepGAP/saved/imagenet-vgg-verydeep-16.mat",
    'vggFile':         "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:0',

    #Num iterations - not used
    'outerSteps':      10000000, #1000000,
    'innerSteps':      100, #300,
    #Batch size - keep at 1
    'batchSize':       1,
    #Learning rate for optimizer - not used
    'learningRate':    1e-4,
    'beta1' :          .9,
    'beta2' :          .999,
    'epsilon':         1e-8,
    'regStrength': .001,
    'preTrain': False,
    #Misc
    'numClasses': 200+1,
    'idxToName': evalDataObj.idxToName,
}

#Allocate tensorflow object
#This will build the graph
tfObj = VGGDetGap(params, evalDataObj.inputShape)

print "Done init"
(outVals, outIdx) = tfObj.evalModelBatch(evalDataObj.getData(evalDataObj.numImages), plot=outputPlots)
filenames = evalDataObj.imgFiles

#Open file, eval, and write to file
outFile = params["outDir"] + params["runDir"] + "output.txt"
f = open(outFile, 'w')

for i, (filename, val, idx) in enumerate(zip(filenames, outVals.tolist(), outIdx.tolist())):
    outStr = str(i) + ": " + filename + ","
    for v, i in zip(val, idx):
        outStr += "[\"" + evalDataObj.idxToName[int(i)] + "\"]," + str(v) +","
    outStr += "\n"
    f.write(outStr)

print "Done run"

tfObj.closeSess()

