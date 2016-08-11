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

clsMeta = "/home/sheng/mountData/DeepGAP/saved/meta_det.mat"

#Get object from which tensorflow will pull data from
evalDataObj = evalObj(inImageList, clsMeta, resizeMethod="crop", normStd=False)

params = {
    #Base output directory
    'outDir':          "/home/sheng/mountData/DeepGAP/",
    #Inner run directory
    'runDir':          "/imagenet_eval/",
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
    'loadFile':        "/home/sheng/mountData/DeepGAP/saved/imagenet_det.ckpt",
    #Input vgg file for preloaded weights
    'vggFile':         "/home/sheng/mountData/DeepGAP/saved/imagenet-vgg-verydeep-16.mat",
    #Device to run on
    'device':          '/gpu:0',
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
tfObj.evalModel(evalDataObj.getData(1), plot=True)

print "Done run"

tfObj.closeSess()

