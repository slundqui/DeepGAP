import scipy.io as spio
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random
from imagenet_clsloc_meta import loadMeta
#from xml.dom import minidom
import xml.etree.ElementTree as ET
from os.path import isfile

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
    def __init__(self, imgList, resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, getGT=True, augument=False):
        self.resizeMethodParam=resizeMethod
        self.normStd = normStd
        if(imgList.split(".")[-1] == "txt"):
            self.imgFiles = readList(imgList)
        else:
            self.imgFiles = [imgList]
        self.numImages = len(self.imgFiles)

        self.shuffleIdx = range(self.numImages)
        self.doShuffle = shuffle
        self.skip = skip
        self.getGT = getGT
        self.gtShape = (self.numClasses,)
        self.augument = augument

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
    #We augument images here as necessary
    def resizeImage(self, inImage):
        try:
            (ny, nx, nf) = inImage.shape
        except:
            print inImage
            pdb.set_trace()

        if(self.augument):
            #Generate offset and flip
            self.flip = random.randint(0,1)
            if(self.flip):
                inImage = inImage[:, ::-1, :]
        else:
            self.flip = 0


        if(self.resizeMethodParam == "aug"):
            self.resizeMethod = random.choice(["crop", "pad"])
        else:
            self.resizeMethod = self.resizeMethodParam

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
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)
        return outImage

    #Reads image provided in the argument, resizes, and normalizes image
    #Returns the image
    def readImage(self, filename, onlyGt=False):
        if(onlyGt):
            gt = self.genGT(filename)
            return (None, gt)

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

        if(self.getGT):
            gt = self.genGT(filename)
            return (image, gt)
        else:
            return image

    def genGT(self, filename):
        gt = np.zeros(self.gtShape)
        gt[self.getClassIdx(filename)] = 1
        return gt

    #Grabs the next image in the list. Will shuffle images when rewinding
    def nextImage(self, onlyGt = False):
        startIdx = self.shuffleIdx[self.imgIdx]
        imgFile = self.imgFiles[startIdx]
        outData = self.readImage(imgFile, onlyGt)
        #Update imgIdx
        self.imgIdx = self.imgIdx + self.skip

        if(self.imgIdx >= self.numImages):
            print "Rewinding"
            self.imgIdx = 0
            if(self.doShuffle):
                random.shuffle(self.shuffleIdx)
        return outData

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
    def getData(self, numExample, onlyGt = False):
        if(onlyGt):
            outData = None
        else:
            outData = np.zeros((numExample,) + self.inputShape)
        if(self.getGT):
            outGt = np.zeros((numExample,)+self.gtShape)

        for i in range(numExample):
            data = self.nextImage(onlyGt)
            if(self.getGT):
                outData[i] = data[0]
                outGt[i] = data[1]
            elif(onlyGt):
                outGt[i] = data
            else:
                outData[i] = data
        if(self.getGT):
            return (outData, outGt)
        else:
            return outData

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

    def loadMetaFile(self, metaFilename):
        return loadMeta(metaFilename)

    def __init__(self, imgList, imgPrefix, metaFilename, useClassDir, ext=".JPEG", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, augument=False, getGT=True):
        #Load metafile and store dict wnToIdx and list idxToName
        (self.wnToIdx, self.idxToName) = self.loadMetaFile(metaFilename)
        #Add distractor to idxToName
        self.idxToName.append("distractor")
        self.imgPrefix = imgPrefix
        self.useClassDir = useClassDir
        self.ext = ext
        self.normStd = normStd
        #Call superclass constructor
        super(imageNetObj, self).__init__(imgList, resizeMethod, normStd, shuffle, skip, seed, augument=augument, getGT=getGT)

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

class imageNetDetObj(imageNetObj):
    numClasses = 200

    def __init__(self, imgList, imgPrefix, gtPrefix, metaFilename, ext=".JPEG", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, augument=False, getGT=True):
        #Load metafile and store dict wnToIdx and list idxToName
        self.imgPrefix = imgPrefix
        self.gtPrefix = gtPrefix
        #Call superclass constructor
        super(imageNetDetObj, self).__init__(imgList, imgPrefix, metaFilename, False, ext, resizeMethod, normStd, shuffle, skip, seed, augument=augument, getGT=getGT)
        #Class 200 is the distractor class
        self.gtShape = (14, 14, self.numClasses+1)

    #Must append prefix to filenames and remove image idx
    def convertFilename(self, filename):
        #Split filename into file and ground truth
        suffix = filename.split(" ")[0]
        #Depending on if useClassDir is set, we append the fn to imgPrefix with or without the wnIdx directory
        outFilename = self.imgPrefix + "/" + suffix + self.ext
        return outFilename

    def getClassIdx(self, filename):
        #This should never get called by this class
        assert(0)

    def genGT(self, filename):
        assert(len(self.gtShape) == 3)
        gt = np.zeros(self.gtShape)
        #Set default distractor class
        gt[:, :, -1] = 1

        #Split filename into file and ground truth
        suffix = filename.split(" ")[0]
        gtFilename = self.gtPrefix + "/" + suffix + ".xml"

        #If file does not exist, we mark it as a distractor
        if(not isfile(gtFilename)):
            return gt

        #Parse xml file
        tree = ET.parse(gtFilename)
        root = tree.getroot()

        #Get size of image
        nx = int(root.find('size').find('width').text)
        ny = int(root.find('size').find('height').text)
        #Get scale factor and crop/pad factor
        #offset is in terms before the resize
        if(self.resizeMethod=="crop"):
            if(nx > ny):
                scaleFactor = float(self.gtShape[0])/ny
                targetNx = int(round(nx * scaleFactor))
                xOffset = int(round(float(self.gtShape[1] - targetNx)/2))
                yOffset = 0
            else:
                scaleFactor = float(self.gtShape[1])/nx
                targetNy = int(round(ny * scaleFactor))
                xOffset = 0
                yOffset = int(round(float(self.gtShape[0] - targetNy)/2))
        elif(self.resizeMethod=="pad"):
            if(nx > ny):
                scaleFactor = float(self.gtShape[1])/nx
                targetNy = int(round(ny*scaleFactor))
                xOffset = 0
                yOffset = int(round(float(self.gtShape[0]-targetNy)/2))
            else:
                scaleFactor = float(self.gtShape[0])/ny
                targetNx = int(round(nx*scaleFactor))
                xOffset= int(round(float(self.gtShape[1]-targetNx)/2))
                yOffset = 0
        else:
            assert(0)
        #Get all objects
        objs = root.findall('object')
        for obj in objs:
            wnIdx = obj.find('name').text
            xmin = int(obj.find('bndbox').find('xmin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            #Add offset and scale
            scale_xmin = int(round(xmin*scaleFactor)+xOffset)
            scale_xmax = int(round(xmax*scaleFactor)+xOffset)
            scale_ymin = int(round(ymin*scaleFactor)+yOffset)
            scale_ymax = int(round(ymax*scaleFactor)+yOffset)
            #Flip x if needed
            if(self.flip):
                tmp = self.gtShape[1] - scale_xmin
                scale_xmin = self.gtShape[1] - scale_xmax
                scale_xmax = tmp

            #Check bounds
            #We only need to check min bounds, because numpy indexing auto truncates the upper dimension
            if(scale_xmin < 0):
                scale_xmin = 0
            if(scale_ymin < 0):
                scale_ymin = 0

            #Convert wnIdx to gtIdx
            gtIdx = self.wnToIdx[wnIdx]
            #Assign that index to be 1's, and take out distractor class
            gt[scale_ymin:scale_ymax, scale_xmin:scale_xmax, gtIdx] = 1
            gt[scale_ymin:scale_ymax, scale_xmin:scale_xmax, -1] = 0

        return gt

class evalObj(imageObj):
    inputShape = (224, 224, 3)
    numClasses = 200
    mean = None
    def __init__(self, imgFile, metaFilename, resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, augument=False):
        (self.wnToIdx, self.idxToName) = loadMeta(metaFilename)
        self.idxToName.append("distractor")
        super(evalObj, self).__init__(imgFile, resizeMethod, normStd, shuffle, skip, seed, getGT=False, augument=augument)
    def convertFilename(self, filename):
        return filename

class vocDetObj(imageNetDetObj):
    numClasses = 20
    def __init__(self, imgList, imgPrefix, gtPrefix, ext=".jpg", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, augument=False):
        super(vocDetObj, self).__init__(imgList, imgPrefix, gtPrefix, None, ext, resizeMethod, normStd, shuffle, skip, seed, augument=augument)
        self.gtShape = (7, 7, self.numClasses+1)

    def loadMetaFile(self, metaFilename):
        idxToName = [
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor', 'distractor']

        wnToIdx = {c:i for (i, c) in enumerate(idxToName)}
        return (wnToIdx, idxToName)

class vocObj(imageNetDetObj):
    numClasses = 20
    def __init__(self, imgList, imgPrefix, gtPrefix, ext=".jpg", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, singleObj=True, augument=False):
        super(vocObj, self).__init__(imgList, imgPrefix, gtPrefix, None, ext, resizeMethod, normStd, shuffle, skip, seed, augument=augument)
        self.gtShape = (self.numClasses,)
        self.singleObj = singleObj

    def loadMetaFile(self, metaFilename):
        idxToName = [
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

        wnToIdx = {c:i for (i, c) in enumerate(idxToName)}
        return (wnToIdx, idxToName)

    def genGT(self, filename):
        gt = np.zeros(self.gtShape)

        #Split filename into file and ground truth
        suffix = filename.split(" ")[0]
        gtFilename = self.gtPrefix + "/" + suffix + ".xml"

        #If file does not exist, we mark it as a distractor
        #This should not happen in VOC
        if(not isfile(gtFilename)):
            assert(0)

        #Parse xml file
        tree = ET.parse(gtFilename)
        root = tree.getroot()

        #Get all objects
        maxIdx = None
        maxArea = 0

        objs = root.findall('object')
        if(self.singleObj):
            for obj in objs:
                wnIdx = obj.find('name').text
                xmin = int(obj.find('bndbox').find('xmin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                ymax = int(obj.find('bndbox').find('ymax').text)

                bbArea = (xmax-xmin)*(ymax-ymin)
                if(bbArea >= maxArea):
                    maxAream = bbArea
                    maxIdx = self.wnToIdx[wnIdx]
            gt[maxIdx] = 1
        else:
            for obj in objs:
                wnIdx = obj.find('name').text
                gt[self.wnToIdx[wnIdx]] = 1

        return gt







#if __name__ == "__main__":
#    trainImageList = "/home/slundquist/mountData/datasets/imagenet/train_cls.txt"
#    trainImagePrefix = "/nh/compneuro/Data/imageNet/CLS_LOC/ILSVRC2015/Data/CLS-LOC/train/"
#    clsMeta = "/nh/compneuro/Data/imageNet/devkit/data/meta_clsloc.mat"
#
#    imgnet = imageNetObj(trainImageList, trainImagePrefix, clsMeta, useClassDir=True)
#    imgnet.getMean()
#
