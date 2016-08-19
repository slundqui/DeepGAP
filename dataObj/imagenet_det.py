from dataObj.image import imageNetDetObj
from os.path import isfile
import pdb
import xml.etree.ElementTree as ET

class imageNetDetPVGT(imageNetDetObj):
    numClasses = 200
    def __init__(self, imgList, imgPrefix, gtPrefix, metaFilename, ext=".JPEG", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, getGT=True):

        #Call superclass constructor
        super(imageNetDetBBObj, self).__init__(imgList, imgPrefix, gtPrefix, metaFilename, ext, resizeMethod, normStd, shuffle, skip, seed, augument=False , getGT=getGT)

        #Class 200 is the distractor class
        self.gtShape = None
        self.inputShape = (256, 256, 3)
        self.flip=False

class imageNetDetBBObj(imageNetDetObj):
    numClasses = 200

    def __init__(self, imgList, imgPrefix, gtPrefix, metaFilename, ext=".JPEG", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, getGT=True):

        #Call superclass constructor
        super(imageNetDetBBObj, self).__init__(imgList, imgPrefix, gtPrefix, metaFilename, ext, resizeMethod, normStd, shuffle, skip, seed, augument=False , getGT=getGT)

        #Class 200 is the distractor class
        self.gtShape = None
        self.inputShape = (256, 256, 3)
        self.flip=False

    def genGT(self, filename):
        #Output will be a list of lists of the following bb info
        #[id, top, bottom, left, right]

        outList = []

        #Split filename into file and ground truth
        suffix = filename.split(" ")[0]
        gtFilename = self.gtPrefix + "/" + suffix + ".xml"

        #If file does not exist, we mark it as a distractor
        if(not isfile(gtFilename)):
            return outList

        #Parse xml file
        tree = ET.parse(gtFilename)
        root = tree.getroot()

        #Get size of image
        nx = int(root.find('size').find('width').text)
        ny = int(root.find('size').find('height').text)

        #offset is in terms before the resize
        yRatio = float(self.inputShape[0])/ny
        xRatio = float(self.inputShape[1])/nx

        #GT is in terms of orig croped/padded image

        #offset is in terms before the resize
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
                yOffset = int(round(float(self.inputShape[0] - targetNy)/2))
        elif(self.resizeMethod=="pad"):
            if(yRatio > xRatio):
                scaleFactor = xRatio
                targetNy = int(round(ny*scaleFactor))
                xOffset = 0
                yOffset = int(round(float(self.inputShape[0]-targetNy)/2))
            else:
                scaleFactor = yRatio
                targetNx = int(round(nx*scaleFactor))
                xOffset= int(round(float(self.inputShape[1]-targetNx)/2))
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

            assert(not self.flip)
            ##Flip x if needed
            #if(self.flip):
            #    tmp = self.inputShape[1] - scale_xmin
            #    scale_xmin = self.inputShape[1] - scale_xmax
            #    scale_xmax = tmp

            #Convert wnIdx to gtIdx
            gtIdx = self.wnToIdx[wnIdx]

            #Check bounds
            if(scale_xmin < 0):
                scale_xmin = 0
            if(scale_ymin < 0):
                scale_ymin = 0
            if(scale_xmax >= self.inputShape[1]):
                scale_xmax = self.inputShape[1]-1
            if(scale_ymax >= self.inputShape[0]):
                scale_ymax = self.inputShape[0]-1
            outList.append([gtIdx, scale_ymin, scale_ymax, scale_xmin, scale_xmax])

        return outList


