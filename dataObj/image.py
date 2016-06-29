import scipy.io as spio
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random
from imagenet_clsloc_meta import loadMeta

def readList(filename):
    f = open(filename, 'r')
    allLines = f.readlines()
    f.close()
    #Remove newlines from all lines
    return [line[:-1] for line in allLines]


"""
An object that handles data input
"""
class imageObj(object):
    imgIdx = 0
    #inputShape = (32, 32, 3)
    maxDim = 0

    #Constructor takes a text file containing a list of absolute filenames
    #Will calculate the mean/std of image for normalization
    #resizeMethod takes 3 different types of input:
    #"crop" will resize the smallest dimension to inputShape,
    #and crop the other dimension in the center
    #"pad" will resize the largest dimension to inputShape, and pad the other dimension
    #"max" will find the max dimension of the list of images, and pad the surrounding area
    #Additionally, if inMaxDim is set with resizeMethod of "max", it will explicitly set
    #the max dimension to inMaxDim
    def __init__(self, imgList, resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None):
        self.resizeMethod=resizeMethod
        self.imgFiles = readList(imgList)
        self.numImages = len(self.imgFiles)
        self.shuffleIdx = range(self.numImages)
        self.doShuffle = shuffle
        self.skip = skip
        if(shuffle):
            #Initialize random seed
            if(seed):
                #Seed random
                random.seed(seed)
            random.shuffle(self.shuffleIdx)
        #This function will also set self.maxDim
        #self.getMean()
        if(self.resizeMethod=="crop"):
            pass
        elif(self.resizeMethod=="pad"):
            pass
        elif(self.resizeMethod=="max"):
            #self.inputShape=(self.maxDim, self.maxDim, 3)
            print "Resize method max Not implemented"
            assert(0)
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)

    ##Calculates the mean and standard deviation from the images
    ##Will also calculate the max dimension of image
    #def getMean(self):
    #    s = np.array(0).astype(np.float64)
    #    num = np.array(0).astype(np.float64)
    #    for (i, f) in enumerate(self.imgFiles):
    #        if(i % 100 == 0):
    #            print "Progress: " + str(float(i)/len(self.imgFiles)) + ": " + str(i) + " out of " + str(len(self.imgFiles))
    #        img = (imread(self.convertFilename(f)).astype(np.float32)/255)
    #        s += np.sum(img)
    #        num += np.array(img.size).astype(np.float64)
    #    self.mean = s / num
    #    print "img mean: ", self.mean

    #Function to resize image to inputShape
    def resizeImage(self, inImage):
        (ny, nx, nf) = inImage.shape
        if(self.resizeMethod == "crop"):
            if(ny > nx):
                #Get percentage of scale
                scale = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scale))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                cropTop = (targetNy-self.inputShape[0])/2
                outImage = scaleImage[cropTop:cropTop+self.inputShape[0], :, :]
            elif(ny <= nx):
                #Get percentage of scale
                scale = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scale))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                cropLeft = (targetNx-self.inputShape[1])/2
                outImage = scaleImage[:, cropLeft:cropLeft+self.inputShape[1], :]
        elif(self.resizeMethod == "pad"):
            if(ny > nx):
                #Get percentage of scale
                scale = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scale))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                padLeft = (self.inputShape[1]-targetNx)/2
                padRight = self.inputShape[1] - (padLeft + targetNx)
                outImage = np.pad(scaleImage, ((0, 0), (padLeft, padRight), (0, 0)), 'constant')
            elif(ny <= nx):
                #Get percentage of scale
                scale = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scale))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                padTop = (self.inputShape[0]-targetNy)/2
                padBot = self.inputShape[0] - (padTop + targetNy)
                outImage = np.pad(scaleImage, ((padTop, padBot), (0, 0), (0, 0)), 'constant')
        elif(self.resizeMethod=="max"):
            #We pad entire image with 0
            assert(ny <= self.inputShape[0])
            assert(nx <= self.inputShape[1])
            padTop   = (self.inputShape[0]-ny)/2
            padBot   = self.inputShape[0]-(padTop+ny)
            padLeft  = (self.inputShape[1]-nx)/2
            padRight = self.inputShape[1]-(padLeft+nx)
            outImage = np.pad(inImage, ((padTop, padBot), (padLeft, padRight), (0, 0)), 'constant')
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)
        return outImage

    #Reads image provided in the argument, resizes, and normalizes image
    #Returns the image
    def readImage(self, filename):
        image = imread(self.convertFilename(filename))
        #Check if b/w, and convert to color if necessary
        if(image.ndim == 2):
             image = np.transpose(np.tile(image, [3, 1, 1]), [1, 2, 0])
        image = (self.resizeImage(image).astype(np.float32)/255)
        #If image has alpha channel, remove that channel
        if(image.shape[2] == 4):
            image = image[:, :, 0:3]

        #Normalize
        if(self.normStd):
            image = (image-np.mean(image))/np.std(image)
        elif(self.mean):
            image = image-self.mean
        else:
            image = image-np.mean(image)
        gt = np.zeros((self.numClasses))
        gt[self.getClassIdx(filename)] = 1
        return (image, gt)

    #Grabs the next image in the list. Will shuffle images when rewinding
    def nextImage(self):
        startIdx = self.shuffleIdx[self.imgIdx]
        imgFile = self.imgFiles[startIdx]
        (outImg, outGt) = self.readImage(imgFile)
        #Update imgIdx
        self.imgIdx = self.imgIdx + self.skip

        if(self.imgIdx >= self.numImages):
            print "Rewinding"
            self.imgIdx = 0
            if(self.doShuffle):
                random.shuffle(self.shuffleIdx)
        return (outImg, outGt)

    ##Get all segments of current image. This is what evaluation calls for testing
    #def allImages(self):
    #    outData = np.zeros((self.numImages, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
    #    #outGt = np.zeros((self.numImages, 10))
    #    for i, imgFile in enumerate(self.imgFiles):
    #        data = self.readImage(imgFile)
    #        outData[i, :, :, :] = data
    #        #outGt[i, :] = data[1]
    #    return outData

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        outData = np.zeros((numExample, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
        outGt = np.zeros((numExample, self.numClasses))
        for i in range(numExample):
            (data, gt) = self.nextImage()
            outData[i, :, :, :] = data
            outGt[i, :] = gt
        return (outData, outGt)

class cifarObj(imageObj):
    inputShape = (32, 32, 3)
    numClasses = 10
    idxToName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mean = None
    def getClassIdx(self, filename):
        return(int(filename.split('/')[-2]))
    #Cifar filename is itself
    def convertFilename(self, filename):
        return filename

class imageNetObj(imageObj):
    #inputShape = (64, 128, 3)
    inputShape = (224, 224, 3)
    numClasses = 1000
    mean = 0.44864434289 #Mean of training set

    def __init__(self, imgList, imgPrefix, metaFilename, useClassDir, ext=".JPEG", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None):
        #Load metafile and store dict wnToIdx and list idxToName
        (self.wnToIdx, self.idxToName) = loadMeta(metaFilename)
        self.imgPrefix = imgPrefix
        self.useClassDir = useClassDir
        self.ext = ext
        self.normStd = normStd
        #Call superclass constructor
        super(imageNetObj, self).__init__(imgList, resizeMethod, normStd, shuffle, skip, seed)

    #Must append prefix to filenames and remove image idx
    def convertFilename(self, filename):
        #Split filename into file and ground truth
        (wnIdx, fn) = filename.split(" ")[0].split("/")
        #Depending on if useClassDir is set, we append the fn to imgPrefix with or without the wnIdx directory
        outFilename = self.imgPrefix + "/"
        if(self.useClassDir):
            outFilename += wnIdx + "/"
        outFilename += fn + self.ext
        return outFilename

    def getClassIdx(self, filename):
        #Split filename into file and ground truth
        (wnIdx, fn) = filename.split(" ")[0].split("/")
        #Convert wnIdx to gtIdx
        gtIdx = self.wnToIdx[wnIdx]
        return gtIdx

#if __name__ == "__main__":
#    trainImageList = "/home/slundquist/mountData/datasets/imagenet/train_cls.txt"
#    trainImagePrefix = "/nh/compneuro/Data/imageNet/CLS_LOC/ILSVRC2015/Data/CLS-LOC/train/"
#    clsMeta = "/nh/compneuro/Data/imageNet/devkit/data/meta_clsloc.mat"
#
#    imgnet = imageNetObj(trainImageList, trainImagePrefix, clsMeta, useClassDir=True)
#    imgnet.getMean()
#
