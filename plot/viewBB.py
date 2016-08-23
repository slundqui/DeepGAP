import matplotlib
import matplotlib.pyplot as plt
import pdb
import numpy as np

bbColor = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]

#def drawBB(img, bbs, ws):
def drawBB(img, bbs):

    #Draw box per bb
    #for i, (bb, w) in enumerate(zip(bbs, ws)):
    for i, bb in enumerate(bbs):
        (top, bot, left, right) = bb
        bbColorIdx = i%len(bbColor)
        #Left
        img[top:bot, left, :] = bbColor[bbColorIdx]
        #Right
        img[top:bot, right, :] = bbColor[bbColorIdx]
        #Top
        img[top, left:right, :] = bbColor[bbColorIdx]
        #Bot
        img[bot, left:right, :] = bbColor[bbColorIdx]
    return img


#TODO make this function take an "offset" matrix for bb regression
def calcBB(img, inScores, bbWindows, thresh=0):

    (yImg, xImg, nf) = img.shape
    (wBB, yBB, xBB) = inScores.shape
    yScale = float(yImg)/yBB
    xScale = float(xImg)/xBB

    bbIdxs = np.nonzero(inScores> thresh)
    (wIdx, yIdx, xIdx) = bbIdxs
    outScores = inScores[bbIdxs]

    outBB = []
    outW = []

    for (w, y, x) in zip(wIdx, yIdx, xIdx):
        (wy, wx) = bbWindows[w]
        #Scale center
        yCenter = int(round(y * yScale))
        xCenter = int(round(x * xScale))
        #Calc udlr from center and window size
        yOff = wy/2
        xOff = wx/2

        top = yCenter - yOff
        bot = yCenter + yOff
        left = xCenter - xOff
        right = xCenter + xOff

        #Check bounds
        if(top < 0):
            top = 0
        elif(top >= yImg):
            top = yImg-1
        if(bot < 0):
            bot = 0
        elif(bot >= yImg):
            bot = yImg-1
        if(left < 0):
            left = 0
        elif(left >= xImg):
            left= xImg-1
        if(right < 0):
            right = 0
        elif(right >= xImg):
            right = xImg-1
        outBB.append([top, bot, left, right])
        outW.append(w)

    return (outBB, outScores, outW)

def plotBB(outPrefix, inImage, gt, est, idxs, vals, idxToName, bbWindows, sess, tfVars):
    (out, tf_bbs, tf_scores) = tfVars
    fontsize = 6
    matplotlib.rc('font', size=fontsize)
    (nbatch, ntime,  nyImage, nxImage, nfImage) = inImage.shape
    (nbatch, nw, nyEst, nxEst, nClass) = est.shape

    #We caluclate winners of est
    #estWin = np.argmax(est, axis=4)

    hasGt = False
    if(gt!=None):
        hasGt = True
        (nbatch, nw, nyGt, nxGt, nClass) = gt.shape
        yFactorGt = nyImage/nyGt
        xFactorGt = nxImage/nxGt

    (nbatch, numRank) = idxs.shape
    numTotal = numRank * 2

    for b in range(nbatch):
        image = inImage[b, 0, :, :, :]
        r_image = (image-image.min())/(image.max()-image.min())
        xTile = int(np.floor(np.sqrt(numTotal)))
        yTile = int(np.ceil(float(numTotal)/xTile))

        #Plot image first
        f, axarr = plt.subplots(yTile, xTile, figsize=(15, 10))

        if(hasGt):
            sortedGt = gt[b, :, :, :, idxs[b, :]+1]

        for y in range(yTile):
            for x in range(xTile):
                classIdx = y*xTile + x
                outImg = r_image.copy()
                if(classIdx < 6):
                    if(gt!=None):
                        (outBB, outScores, outW) = calcBB(outImg, sortedGt[classIdx, :, :, :], bbWindows)
                        outImg = drawBB(outImg, outBB)
                        axarr[y, x].imshow(outImg)
                        strLabel = idxToName[idxs[b, classIdx]+1].split(',')[0]
                        axarr[y, x].set_title(strLabel+ " gt", fontsize=fontsize)
                else:
                    classIdx = classIdx - 6
                    targetClass = idxs[b, classIdx]+1
                    batchEstScores = est[b, :, :, :, targetClass]
                    (outBB, outScores, outW) = calcBB(outImg, batchEstScores, bbWindows, thresh=.4)

                    if len(outBB) != 0:
                        feed_dict = {tf_bbs:outBB, tf_scores:outScores}
                        nms_bbs= sess.run(out, feed_dict=feed_dict)
                        outImg = drawBB(outImg, nms_bbs)
                    else:
                        outImg = drawBB(outImg, outBB)
                    axarr[y, x].imshow(outImg)
                    strLabel = idxToName[targetClass].split(',')[0]
                    axarr[y, x].set_title(strLabel + ": " + str('%.2g'%vals[b, classIdx]), fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(outPrefix + "_" + str(b) + ".png")
        plt.close(f)

