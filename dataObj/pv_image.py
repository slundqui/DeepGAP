from image import imageObj, readList
from scipy.ndimage import imread
from pvtools import *
import random
import pdb
import numpy as np

class imageNetVidPvObj(imageObj):
    def __init__(self, trainInputs, trainGts, trainFilenames, fnPrefix, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True):

        self.resizeMethodParam=resizeMethod
        self.normStd = False

        self.numInputs = len(trainInputs)
        self.numGt = len(trainGts)

        self.fnPrefix = fnPrefix

        self.inputFiles = []
        self.gtFiles = []
        self.imgFilenames = []

        for inputFn in trainInputs:
            self.inputFiles.append(pvpOpen(inputFn, "r"))

        for gtFn in trainGts:
            self.gtFiles.append(pvpOpen(gtFn, "r"))

        for imageFn in trainFilenames:
            fnList = readList(imageFn)
            fnList = [self.fnPrefix + l.split('/')[-2] + "/" + l.split('/')[-1] for l in fnList]
            self.imgFilenames.append(fnList)

        assert(len(self.imgFilenames) == len(self.gtFiles))

        numInputFrames = np.min([f.header["nbands"] for f in self.inputFiles])
        numGtFrames = np.min([f.header["nbands"] for f in self.gtFiles])
        self.numImages = np.min([numInputFrames, numGtFrames])

        inHeader = self.inputFiles[0].header
        gtHeader =  self.gtFiles[0].header

        #Shape is 4d, [depth, height, width, channels]
        self.innerInputShape = (inHeader['ny'], inHeader['nx'], inHeader['nf'])
        self.innerGtShape = (gtHeader['ny'], gtHeader['nx'], gtHeader['nf'])
        self.innerImageShape = (64, 128, 3)
        self.inputShape = (self.numInputs, inHeader['ny'], inHeader['nx'], inHeader['nf'])
        self.gtShape = (self.numGt, gtHeader['ny'], gtHeader['nx'], gtHeader['nf'])
        self.imageShape  = (self.numGt, 64, 128, 3)

        self.numClasses = self.gtShape[3]

        self.idxToName = [
                "distractor",
                "motorcycle",
                "car",
                "bicycle",
                "bus",
                "airplane",
                "train",
                "watercraft",
                "antelope",
                "bear",
                "bird",
                "cattle",
                "dog",
                "domestic_cat",
                "elephant",
                "fox",
                "giant_panda",
                "hamster",
                "horse",
                "lion",
                "lizard",
                "monkey",
                "rabbit",
                "red_panda",
                "sheep",
                "snake",
                "squirrel",
                "tiger",
                "turtle",
                "whale",
                "zebra",]

        self.shuffleIdx = range(self.numImages)
        self.doShuffle = shuffle
        self.skip = skip
        self.getGT = getGT

        if(self.doShuffle):
            #Initialize random seed
            if(seed):
                #Seed random
                random.seed(seed)
            random.shuffle(self.shuffleIdx)
        #This function will also set self.maxDim
        #self.getMean()
        if(self.resizeMethodParam=="crop"):
            pass
        elif(self.resizeMethodParam=="pad"):
            pass
        elif(self.resizeMethodParam=="aug"):
            pass
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)

    def nextImage(self):
        startIdx = self.shuffleIdx[self.imgIdx]
        dataOut = np.zeros(self.inputShape)
        imgOut = np.zeros(self.imageShape)
        for i, f in enumerate(self.inputFiles):
            dataOut[i] = f.read(startIdx, startIdx+1)["values"].toarray().reshape(self.innerInputShape)

        for i, fns in enumerate(self.imgFilenames):
            fn = fns[startIdx]
            imgOut[i] = imread(fn)

        if(self.getGT):
            gtOut = np.zeros(self.gtShape)
            for i, f in enumerate(self.gtFiles):
                gtOut[i] = f.read(startIdx, startIdx+1)["values"].toarray().reshape(self.innerGtShape)

        #Update imgIdx
        self.imgIdx = self.imgIdx + self.skip

        if(self.imgIdx >= self.numImages):
            print "Rewinding"
            self.imgIdx = 0
            if(self.doShuffle):
                random.shuffle(self.shuffleIdx)
        if(self.getGT):
            return (dataOut, gtOut, imgOut)
        else:
            return (dataOut, imgOut)

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        outData = np.zeros((numExample,) + self.inputShape)
        outImg = np.zeros((numExample,) + self.imageShape)
        if(self.getGT):
            outGt = np.zeros((numExample,)+self.gtShape)

        for i in range(numExample):
            data = self.nextImage()
            if(self.getGT):
                outData[i] = data[0]
                outGt[i] = data[1]
                outImg[i] = data[2]
            else:
                outData[i] = data[0]
                outImg[i] = data[1]
        if(self.getGT):
            return (outData, outGt, outImg)
        else:
            return (outData, outImg)









