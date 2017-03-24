import numpy as np
import pdb
import matplotlib.pyplot as plt

#imgInput is [batch, y, x, f]
#gtInput and est is [batch, y, x]
#boxSize is [y, x]
def plotDet(prefix, imgInput, gtInput, est, boxSize, thresh=.5, r=[2, 1, 0, 4, 7, 8, 10, 14]):
    #Copy input image to output buffer
    (nbatch, nyImage, nxImage, nfImage) = imgInput.shape

    if(r is None):
        r = range(nbatch)

    numPlots = len(r)

    #Create subplot
    f, axarr = plt.subplots(numPlots, 2)

    for (i, b) in enumerate(r):
        estImg = drawBoxes(imgInput[b], est[b], boxSize, thresh)
        gtImg = drawBoxes(imgInput[b], gtInput[b], boxSize, thresh)

        axarr[i,0].imshow(gtImg)
        #axarr[0].set_title("Ground Truth")
        axarr[i,0].xaxis.set_visible(False)
        axarr[i,0].yaxis.set_visible(False)

        axarr[i,1].imshow(estImg)
        #axarr[1].set_title("Estimate")
        axarr[i,1].xaxis.set_visible(False)
        axarr[i,1].yaxis.set_visible(False)

    f.tight_layout(h_pad=.3, w_pad=.01)

    plt.savefig(prefix+"_det.png")
    plt.close(f)


#Here, inImg is [y, x, f]. Est is [y, x]
def drawBoxes(inImg, est, boxSize, thresh=.5):
    (nyImage, nxImage, nfImage) = inImg.shape
    (nyEst, nxEst) = est.shape

    #Calculate scale
    estToImgScaleY = float(nyImage)/nyEst
    estToImgScaleX = float(nxImage)/nxEst

    #Copy image for output
    outputImg = inImg.copy()

    #Scale image to be between 0 and 1
    outputImg = (outputImg - outputImg.min())/(outputImg.max() - outputImg.min())

    #Find indices where est > thresh
    #We only draw boxes over thresh
    (yPosIdxs, xPosIdxs) = np.nonzero(est > thresh)

    #Scale to image size
    yImgPosIdx = yPosIdxs * estToImgScaleY
    xImgPosIdx = xPosIdxs * estToImgScaleX

    #Find bb based on calculated center and boxSize
    #We subtract the bottom by 1 to show
    ymin = yImgPosIdx
    ymax = ymin + boxSize[0]
    xmin = xImgPosIdx
    xmax = xmin + boxSize[1]

    ymin[np.nonzero(ymin < 0)] = 0
    ymin[np.nonzero(ymin >= nyImage)] = nyImage-1

    ymax[np.nonzero(ymax < 0)] = 0
    ymax[np.nonzero(ymax >= nyImage)] = nyImage-1

    xmin[np.nonzero(xmin < 0)] = 0
    xmin[np.nonzero(xmin >= nxImage)] = nxImage-1

    xmax[np.nonzero(xmax < 0)] = 0
    xmax[np.nonzero(xmax >= nxImage)] = nxImage-1

    #Cast as int
    ymin = ymin.astype(np.int32)
    ymax = ymax.astype(np.int32)
    xmin = xmin.astype(np.int32)
    xmax = xmax.astype(np.int32)

    #Mark boxes as green
    for (y1, y2, x1, x2) in zip(ymin, ymax, xmin, xmax):
        outputImg[y1:y2, x1, 1] = 1
        outputImg[y1:y2, x2, 1] = 1
        outputImg[y1, x1:x2, 1] = 1
        outputImg[y2, x1:x2, 1] = 1

    return outputImg
