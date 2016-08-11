class imageNetDetBBObj(imageNetObj):
    numClasses = 200
    def __init__(self, imgList, imgPrefix, gtPrefix, metaFilename, ext=".JPEG", resizeMethod="crop", normStd=True, shuffle=True, skip=1, seed=None, augument=False, getGT=True):
        #Call superclass constructor
        super(imageNetDetBBObj, self).__init__(imgList, imgPrefix, metaFilename, False, ext, resizeMethod, normStd, shuffle, skip, seed, augument=augument, getGT=getGT)
        #Class 200 is the distractor class
        self.gtShape = None
        self.clsGtShape = [14, 14, 9, 2]
        self.regGtShape = [14, 14, 9, 4]

    def genGT(self, filename):
        clsGt = np.zeros(self.clsGtShape)
        regGt = np.zeros(self.regGtShape)

        #Split filename into file and ground truth
        suffix = filename.split(" ")[0]
        gtFilename = self.gtPrefix + "/" + suffix + ".xml"

        #If file does not exist, we mark it as a distractor
        if(not isfile(gtFilename)):
            clsGt[:, :, :, 1] = 1
            return (clsGt, regGt)

        #Parse xml file
        tree = ET.parse(gtFilename)
        root = tree.getroot()

        #Get size of image
        nx = int(root.find('size').find('width').text)
        ny = int(root.find('size').find('height').text)

        #GT is in terms of resized image

        #offset is in terms before the resize
        if(self.resizeMethod=="crop"):
            if(nx > ny):
                scaleFactor = float(self.inputShape[0])/ny
                targetNx = int(round(nx * scaleFactor))
                xOffset = int(round(float(self.inputShape[1] - targetNx)/2))
                yOffset = 0
            else:
                scaleFactor = float(self.inputShape[1])/nx
                targetNy = int(round(ny * scaleFactor))
                xOffset = 0
                yOffset = int(round(float(self.inputShape[0] - targetNy)/2))
        elif(self.resizeMethod=="pad"):
            if(nx > ny):
                scaleFactor = float(self.inputShape[1])/nx
                targetNy = int(round(ny*scaleFactor))
                xOffset = 0
                yOffset = int(round(float(self.inputShape[0]-targetNy)/2))
            else:
                scaleFactor = float(self.inputShape[0])/ny
                targetNx = int(round(nx*scaleFactor))
                xOffset= int(round(float(self.inputShape[1]-targetNx)/2))
                yOffset = 0
        else:
            assert(0)

        #Get all objects
        objs = root.findall('object')

        #For each object proposal region, we assign gt to it
        for y in range(self.clsGt

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
                tmp = self.inputShape[1] - scale_xmin
                scale_xmin = self.inputShape[1] - scale_xmax
                scale_xmax = tmp

            #We calculate iou

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


