from image import imageObj, readList
from scipy.ndimage import imread
from pvtools import *
from scipy.sparse import csr_matrix, vstack
import random
import pdb
import numpy as np

class pvObj(imageObj):
    def __init__(self, trainInputs, trainGts, trainFilenames, fnPrefix, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True, startIdx=0, stopIdx=-1):

        self.resizeMethodParam=resizeMethod
        self.normStd = False

        self.numInputs = len(trainInputs)
        self.numGt = len(trainGts)

        self.fnPrefix = fnPrefix

        self.inputFiles = []
        self.gtFiles = []
        self.imgFilenames = []
        self.imgPvp = []

        for inputFn in trainInputs:
            self.inputFiles.append(pvpOpen(inputFn, "r"))

        for gtFn in trainGts:
            self.gtFiles.append(pvpOpen(gtFn, "r"))

        #If file is sparse
        if(self.gtFiles[0].header['filetype'] != 6 and self.gtFiles[0].header['filetype'] != 2):
            self.gtSparse = False
        else:
            self.gtSparse = True

        if(self.inputFiles[0].header['filetype'] != 6 and self.inputFiles[0].header['filetype'] != 2):
            self.dataSparse = False
        else:
            self.dataSparse = True


        for imageFn in trainFilenames:
            if(imageFn[-4:] == ".txt"):
                self.imgPvp.append(False)
                fnList = readList(imageFn)
                fnList = [self.fnPrefix + l.split('/')[-2] + "/" + l.split('/')[-1] for l in fnList]
                self.imgFilenames.append(fnList)
            elif(imageFn[-4:] == ".pvp"):
                self.imgPvp.append(True)
                self.imgFilenames.append(pvpOpen(imageFn, "r"))
            else:
                print "Fn prefix", imageFn[-4:], "not recognized"
                pdb.set_trace()

        assert(len(self.imgFilenames) == len(self.gtFiles))

        numInputFrames = np.min([f.header["nbands"] for f in self.inputFiles])
        numGtFrames = np.min([f.header["nbands"] for f in self.gtFiles])
        self.numImages = np.min([numInputFrames, numGtFrames])

        if(stopIdx == -1):
            stopIdx = self.numImages
        if(stopIdx >= self.numImages):
            stopIdx = self.numImages

        self.numData = stopIdx - startIdx

        self.shuffleIdx = range(startIdx, stopIdx)
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

        imgOut = np.zeros(self.imageShape)
        for i, fns in enumerate(self.imgFilenames):
            if(self.imgPvp[i]):
                imgOut[i] = fns.read(startIdx, startIdx+1)["values"][0, :, :, :]
            else:
                fn = fns[startIdx]
                imgOut[i] = imread(fn)

        if(self.dataSparse):
            dataOut = []
            for i, f in enumerate(self.inputFiles):
                out = f.read(startIdx, startIdx+1)["values"]
                dataOut.append(out)
            dataOut = vstack(dataOut, format="csr")
        else:
            dataOut = np.zeros(self.inputShape)
            for i, f in enumerate(self.inputFiles):
                dataOut[i] = f.read(startIdx, startIdx+1)["values"][0, :, :, :]

        if(self.getGT):
            if(self.gtSparse):
                gtOut = []
                for i, f in enumerate(self.gtFiles):
                    out = f.read(startIdx, startIdx+1)["values"]
                    gtOut.append(out)
                gtOut = vstack(gtOut, format="csr")
            else:
                gtOut = np.zeros(self.gtShape)
                for i, f in enumerate(self.gtFiles):
                    gtOut[i] = f.read(startIdx, startIdx+1)["values"][0, :, :, :]

        #Update imgIdx
        self.imgIdx = self.imgIdx + self.skip

        if(self.imgIdx >= self.numData):
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
        if(self.dataSparse):
            outData = []
        else:
            outData = np.zeros((numExample,) + self.inputShape)

        if(self.getGT):
            if(self.gtSparse):
                outGt = []
            else:
                outGt = np.zeros((numExample,)+self.gtShape)

        outImg = np.zeros((numExample,) + self.imageShape)

        for i in range(numExample):
            data = self.nextImage()
            if(self.dataSparse):
                outData.append(data[0])
            else:
                outData[i] = data[0]
            if(self.getGT):
                if(self.gtSparse):
                    outGt.append(data[1])
                else:
                    outGt[i] = data[1]
                outImg[i] = data[2]
            else:
                outImg[i] = data[1]

        if(self.dataSparse):
            outData = vstack(outData, format="csr")
        if(self.gtSparse):
            outGt = vstack(outGt, format="csr")

        if(self.getGT):
            return (outData, outGt, outImg)
        else:
            return (outData, outImg)


class imageNetVidPvObj(pvObj):
    def __init__(self, trainInputs, trainGts, trainFilenames, fnPrefix, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True, startIdx = 0, stopIdx=-1):
        super(imageNetVidPvObj, self).__init__(trainInputs, trainGts, trainFilenames, fnPrefix, resizeMethod, shuffle, skip, seed, getGT, startIdx, stopIdx)

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

        self.lossWeight= [
                0.57295454,
                0.94760136,
                0.94149609,
                0.96657907,
                0.95616652,
                0.89569667,
                0.80282838,
                0.92918675,
                0.99863669,
                0.99871325,
                0.99907137,
                1.        ,
                0.9962583 ,
                1.        ,
                0.9993381 ,
                1.        ,
                1.        ,
                1.        ,
                1.        ,
                0.99929612,
                1.        ,
                0.99913805,
                1.        ,
                1.        ,
                0.99978266,
                1.        ,
                1.        ,
                1.        ,
                1.        ,
                0.99725609,
                1.
                ]


class kittiVidPvObj(pvObj):
    def __init__(self, trainInputs, trainGts, trainFilenames, dncFilenames, fnPrefix, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True, startIdx = 0, stopIdx=-1):
        super(kittiVidPvObj, self).__init__(trainInputs, trainGts, trainFilenames, fnPrefix, resizeMethod, shuffle, skip, seed, getGT, startIdx, stopIdx)

        inHeader = self.inputFiles[0].header
        gtHeader =  self.gtFiles[0].header
        if(self.imgPvp[0] == True):
            imgHeader = self.imgFilenames[0].header
            self.innerImageShape = (imgHeader['ny'], imgHeader['nx'], imgHeader['nf'])
        else:
            [ny, nx, nf] = imread(self.imgFilenames[0][0]).shape
            self.innerImageShape = (ny, nx, nf)

        #TODO add dnc areas
        #self.dncFiles = []
        #for dncFn in dncFilenames:
        #    self.dncFiles.append(pvpOpen(dncFn, "r"))

        #if(dncFiles[0].header["filetype"] == 2 or dncFiles[0].header["filetype"]==6):
        #    self.dncSparse = True

        #self.numDnc = len(dncFiles)
        #assert(self.numDnc == self.numGt)

        #Shape is 4d, [depth, height, width, channels]
        self.innerInputShape = (inHeader['ny'], inHeader['nx'], inHeader['nf'])
        self.innerGtShape = (gtHeader['ny'], gtHeader['nx'], gtHeader['nf'])

        self.inputShape = (self.numInputs, inHeader['ny'], inHeader['nx'], inHeader['nf'])
        self.gtShape = (self.numGt, gtHeader['ny'], gtHeader['nx'], gtHeader['nf'])
        self.imageShape  = (self.numGt,) + self.innerImageShape
        self.numClasses = self.gtShape[3]


        self.idxToName = [
                "distractor",
                "Car",
                "Van",
                "Truck",
                "Pedestrian",
                "Person_sitting",
                "Cyclist",
                "Tram",
                "Misc",
                ]

        self.lossWeight= [
                0.34407706,
                0.89432339,
                0.95865095,
                0.90043874,
                0.99473026,
                0.95941077,
                0.97928871,
                0.96908012]

class imageNetVidSupObj(imageObj):
    #TODO fix this object to reduce copied code
    def __init__(self, trainInputs, trainGts, trainFilenames, fnPrefix, resizeMethod="crop", shuffle=True, skip=1, seed=None, getGT=True, getSparse=True):

        self.resizeMethodParam=resizeMethod
        self.normStd = False

        self.numInputs = len(trainInputs)
        self.numGt = len(trainGts)

        self.fnPrefix = fnPrefix

        self.getSparse = getSparse

        self.inputFilenames= []
        self.gtFiles = []
        self.imgPvp = []
        self.imgFilenames = []

        for inputFn in trainInputs:
            fnList = readList(inputFn)
            fnList = [self.fnPrefix + l.split('/')[-2] + "/" + l.split('/')[-1] for l in fnList]
            self.inputFilenames.append(fnList)

        for gtFn in trainGts:
            self.gtFiles.append(pvpOpen(gtFn, "r"))

        for imageFn in trainFilenames:
            if(imageFn[-4:] == ".txt"):
                self.imgPvp.append(False)
                fnList = readList(imageFn)
                fnList = [self.fnPrefix + l.split('/')[-2] + "/" + l.split('/')[-1] for l in fnList]
                self.imgFilenames.append(fnList)
            elif(imageFn[-4:] == ".pvp"):
                self.imgPvp.append(True)
                self.imgFilenames.append(pvpOpen(imageFn, "r"))
            else:
                print "Fn prefix", imageFn[-4:], "not recognized"
                pdb.set_trace()

        assert(len(self.imgFilenames) == len(self.gtFiles))

        numInputFrames = np.min([len(fnlist) for fnlist in self.inputFilenames])
        numGtFrames = np.min([f.header["nbands"] for f in self.gtFiles])
        self.numImages = np.min([numInputFrames, numGtFrames])

        gtHeader =  self.gtFiles[0].header

        #Shape is 4d, [depth, height, width, channels]
        self.innerInputShape = (64, 128, 3)
        self.innerGtShape = (gtHeader['ny'], gtHeader['nx'], gtHeader['nf'])
        self.innerImageShape = (64, 128, 3)
        self.inputShape = (self.numInputs,) + self.innerInputShape
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

        self.lossWeight= [
                0.57295454,
                0.94760136,
                0.94149609,
                0.96657907,
                0.95616652,
                0.89569667,
                0.80282838,
                0.92918675,
                0.99863669,
                0.99871325,
                0.99907137,
                1.        ,
                0.9962583 ,
                1.        ,
                0.9993381 ,
                1.        ,
                1.        ,
                1.        ,
                1.        ,
                0.99929612,
                1.        ,
                0.99913805,
                1.        ,
                1.        ,
                0.99978266,
                1.        ,
                1.        ,
                1.        ,
                1.        ,
                0.99725609,
                1.
                ]

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

        imgOut = np.zeros(self.imageShape)
        for i, fns in enumerate(self.imgFilenames):
            fn = fns[startIdx]
            imgOut[i] = imread(fn)

        dataOut = np.zeros(self.inputShape)
        for i, fns in enumerate(self.inputFilenames):
            fn = fns[startIdx]
            img = imread(fn)
            imgMean = np.mean(img)
            imgStd = np.std(img)
            dataOut[i] = (img-imgMean)/imgStd

        if(self.getSparse):
            if(self.getGT):
                gtOut = []
                for i, f in enumerate(self.gtFiles):
                    out = f.read(startIdx, startIdx+1)["values"]
                    gtOut.append(out)
                gtOut = vstack(gtOut, format="csr")
        else:
            if(self.getGT):
                gtOut = np.zeros(self.gtShape)
                for i, f in enumerate(self.gtFiles):
                    gtOut[i] = f.read(startIdx, startIdx+1)["values"].toarray().reshape(self.innerGtShape)

        #Update imgIdx
        self.imgIdx = self.imgIdx + self.skip

        if(self.imgIdx >= self.numData):
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
        if(self.getSparse):
            if(self.getGT):
                outGt = []
        else:
            if(self.getGT):
                outGt = np.zeros((numExample,)+self.gtShape)

        outImg = np.zeros((numExample,) + self.imageShape)

        for i in range(numExample):
            data = self.nextImage()
            outData[i] = data[0]
            if(self.getSparse):
                if(self.getGT):
                    outGt.append(data[1])
                    outImg[i] = data[2]
                else:
                    outImg[i] = data[1]
            else:
                if(self.getGT):
                    outGt[i] = data[1]
                    outImg[i] = data[2]
                else:
                    outImg[i] = data[1]
        if(self.getSparse):
            outGt = vstack(outGt, format="csr")

        if(self.getGT):
            return (outData, outGt, outImg)
        else:
            return (outData, outImg)


