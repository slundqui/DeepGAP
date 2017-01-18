from dataObj.image import imageObj
from os.path import isfile
import pdb
import xml.etree.ElementTree as ET
import string
import numpy as np
from scipy.misc import imresize
from PIL import Image

class kittiDetBBObj(imageObj):

    def __init__(self, imgList, imgPrefix, gtPrefix, resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, getGT=True, binClass=None, minSize=[3, 3]):

        self.binClass = binClass
        if(self.binClass is None):
            self.numClasses = 8
        else:
            self.numClasses = 2

        #Call superclass constructor
        super(kittiDetBBObj, self).__init__(imgList, resizeMethod, normStd, shuffle, skip, seed, augument=False , getGT=getGT)

        #Class 200 is the distractor class
        self.gtShape = None
        self.inputShape = (64, 256, 3)
        #self.imageShape = (276, 1242, 3)
        self.flip=False
        self.imgPrefix = imgPrefix
        self.gtPrefix = gtPrefix

        self.idxToName = [
                "Car",
                "Van",
                "Truck",
                "Pedestrian",
                "Person_sitting",
                "Cyclist",
                "Tram",
                "Misc",
                ]

        #We leave 0 as distractor class
        self.nameToIdx = {
            "DontCare":-1,
            "Car":0,
            "Van":1,
            "Truck":2,
            "Pedestrian":3,
            "Person_sitting":4,
            "Cyclist":5,
            "Tram":6,
            "Misc":7
        }

        self.minSize = minSize

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

        #Find y ratio and x ratio
        yRatio = float(self.inputShape[0])/ny
        xRatio = float(self.inputShape[1])/nx

        #We crop with anchor bottom center
        #i.e. y crop from bottom, x crop from center
        if(self.resizeMethod == "crop"):
            if(xRatio > yRatio):
                targetNy = int(round(ny * xRatio))
                scaleImage = imresize(inImage, (targetNy, self.inputShape[1]))
                cropTop = (targetNy-self.inputShape[0])
                outImage = scaleImage[cropTop:cropTop+self.inputShape[0], :, :]
            elif(xRatio <= yRatio):
                targetNx = int(round(nx * yRatio))
                scaleImage = imresize(inImage, (self.inputShape[0], targetNx))
                cropLeft = (targetNx-self.inputShape[1])/2
                outImage = scaleImage[:, cropLeft:cropLeft+self.inputShape[1], :]
        elif(self.resizeMethod == "pad"):
            assert(0)
        else:
            print "Method ", resizeMethod, "not supported"
            assert(0)
        return outImage

    def convertFilename(self, filename):
        suffix = filename.split("/")[-1]
        outfn = self.imgPrefix + suffix
        return outfn

    def genGT(self, filename):
        suffix = (filename.split("/")[-1]).split(".")[0] + ".txt"
        gtFilename = self.gtPrefix + suffix

        imageSuffix = filename.split("/")[-1]
        imageFn = self.imgPrefix + imageSuffix

        #Get size of image
        #This does not load the image into memory
        im = Image.open(imageFn)
        nx, ny= im.size

        #offset is in terms before the resize
        yRatio = float(self.inputShape[0])/ny
        xRatio = float(self.inputShape[1])/nx

        if(self.resizeMethod=="crop"):
            if(yRatio > xRatio):
                scaleFactor = yRatio
                targetNx = int(round(nx * scaleFactor))
                xOffset = int(round(float(self.inputShape[1] - targetNx)/2))
                yOffset = 0
            else:
                scaleFactor = xRatio
                targetNy = int(round(ny * scaleFactor))
                xOffset = 0
                #Anchor is at bottom
                yOffset = int(round(float(self.inputShape[0] - targetNy)))
        elif(self.resizeMethod=="pad"):
            assert(0)
        else:
            assert(0)

        #Output will be a list of lists of the following bb info
        #[id, top, bottom, left, right]
        outList = []

        gtFile = open(gtFilename, 'r')
        objs = gtFile.readlines()
        gtFile.close()

        #Get all objects
        for obj in objs:
            split = obj.split()
            gtIdx = self.nameToIdx[split[0]]
            #Dont care region
            if(gtIdx == -1):
                continue
            #We only return the obj
            if(self.binClass is not None):
                if(gtIdx not in self.binClass):
                    continue

            xmin = int(np.round(float(split[4])))
            ymin = int(np.round(float(split[5])))
            xmax = int(np.round(float(split[6])))
            ymax = int(np.round(float(split[7])))


            #Add offset and scale
            scale_xmin = int(round(xmin*scaleFactor)+xOffset)
            scale_xmax = int(round(xmax*scaleFactor)+xOffset)
            scale_ymin = int(round(ymin*scaleFactor)+yOffset)
            scale_ymax = int(round(ymax*scaleFactor)+yOffset)

            assert(not self.flip)

            #Check bounds
            if(scale_xmin < 0):
                scale_xmin = 0
            if(scale_ymin < 0):
                scale_ymin = 0
            if(scale_xmax >= self.inputShape[1]):
                scale_xmax = self.inputShape[1]-1
            if(scale_ymax >= self.inputShape[0]):
                scale_ymax = self.inputShape[0]-1

            #Check minimum size
            if((scale_ymax-scale_ymin) >= self.minSize[0] and
                    (scale_xmax-scale_xmin) >= self.minSize[1]):
                outList.append([gtIdx, scale_ymin, scale_ymax, scale_xmin, scale_xmax])

        return outList


