import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy.ndimage import zoom
import pdb
import colorsys
import operator

def plotCam(outPrefix, inImage, gtIdx, cam, idxs, vals, idxToName, weights):
    fontsize = 6
    matplotlib.rc('font', size=fontsize)
    (nbatch, nyImage, nxImage, nfImage) = inImage.shape
    (nbatch, totCam, nyCam, nxCam) = cam.shape
    numCam = len(idxs[0, :])

    assert(nyImage%nyCam == 0)
    assert(nxImage%nxCam == 0)

    yFactor = nyImage/nyCam
    xFactor = nxImage/nxCam

    colormap = cm.get_cmap('jet')

    maxWeight = np.max(weights)
    minWeight = np.min(weights)

    for b in range(nbatch):
        sortedCam = cam[b, idxs[b, :], :, :]

        numTotal = numCam + 1
        xTile = int(np.ceil(np.sqrt(numTotal)))
        yTile = int(np.ceil(float(numTotal)/xTile))
        image = inImage[b, :, :, :]
        norm_image = (image - np.min(image))/(np.max(image) - np.min(image))

        #Plot image first
        f, axarr = plt.subplots(yTile, xTile, figsize=(10, 7.5))
        avgCam = np.mean(cam[b, :, :, :], axis=0)

        #camImg = sortedCam[0, :, :]
        #camImg = camImg-avgCam
        #if(maxCam < minCam):
        #    tmp = maxCam
        #    maxCam = minCam
        #    minCam = tmp

        maxCam = np.max(sortedCam-avgCam)
        minCam = np.min(sortedCam-avgCam)

        for y in range(yTile):
            for x in range(xTile):
                camIdx = y*xTile + x - 1
                #First image is the orig image
                if(x == 0 and y == 0):
                    axx = axarr[0, 0].imshow(norm_image)
                    if(gtIdx!=None):
                        strLabel = idxToName[gtIdx[b]].split(',')[0]
                        axarr[0, 0].set_title(strLabel, fontsize=fontsize)
                    plt.colorbar(axx, ax = axarr[0, 0])
                #elif(camIdx < 5):
                else:
                    if(camIdx < numCam):
                        camImg = sortedCam[camIdx, :, :]
                        camImg = camImg-avgCam

                        #maxCam = np.max(camImg)
                        ##minCam = -maxCam
                        #minCam = np.min(camImg)

                        resizeCam = zoom(camImg, [yFactor, xFactor])
                        axarr[y, x].imshow(norm_image)
                        axx = axarr[y, x].imshow(resizeCam, cmap=colormap, vmax=maxCam, vmin=minCam, alpha=.6)
                        plt.colorbar(axx, ax = axarr[y, x])
                        #Take only label after first comma
                        strLabel = idxToName[idxs[b, camIdx]].split(',')[0]
                        axarr[y, x].set_title(strLabel + ": " + str('%.2g'%vals[b, camIdx]), fontsize=fontsize)
                #else:
                #    camIdx = camIdx - 5
                #    if(camIdx < numCam):
                #        [numWeights, numClass] = weights.shape
                #        plotWeights = weights[:, idxs[camIdx]]
                #        #axx = axarr[y, x].hist(plotWeights, 20, facecolor='green')
                #        axx = axarr[y, x].bar(range(numWeights), plotWeights)
                #        axarr[y, x].set_ylim([minWeight, maxWeight])
                #        strLabel = idxToName[idxs[camIdx]].split(',')[0]
                #        axarr[y, x].set_title(strLabel + ": " + str(vals[camIdx]))

        plt.tight_layout()

        plt.savefig(outPrefix + "_" + str(b) + ".png")
        plt.close(f)


def get_N_HexCol(N=5):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def plotDetCam(outPrefix, inImage, gt, cam, idxs, vals, idxToName, weights):
    fontsize = 6
    matplotlib.rc('font', size=fontsize)
    (nbatch, nyImage, nxImage, nfImage) = inImage.shape
    (nbatch, totCam, nyCam, nxCam) = cam.shape
    numCam = len(idxs[0, :])

    assert(nyImage%nyCam == 0)
    assert(nxImage%nxCam == 0)

    yFactor = nyImage/nyCam
    xFactor = nxImage/nxCam

    colormap = cm.get_cmap('jet')

    maxWeight = np.max(weights)
    minWeight = np.min(weights)

    reCam = cam.copy()
    reCam[:, -1, :, :] *= .5

    for b in range(nbatch):
        sortedCam = cam[b, idxs[b, :], :, :]

        numTotal = 2*numCam + 1
        xTile = int(np.ceil(np.sqrt(numTotal)))
        yTile = int(np.ceil(float(numTotal)/xTile))
        image = inImage[b, :, :, :]
        norm_image = (image - np.min(image))/(np.max(image) - np.min(image))

        #Plot image first
        f, axarr = plt.subplots(yTile, xTile, figsize=(10, 7.5))
        #avgCam = np.mean(cam[b, :, :, :], axis=0)

        #camImg = sortedCam[0, :, :]
        #camImg = camImg-avgCam
        #if(maxCam < minCam):
        #    tmp = maxCam
        #    maxCam = minCam
        #    minCam = tmp

        maxCam = np.max(sortedCam)
        minCam = np.min(sortedCam)

        for y in range(yTile):
            for x in range(xTile):
                camIdx = y*xTile + x - 1
                #First image is the orig image
                if(x == 0 and y == 0):
                    axx = axarr[0, 0].imshow(norm_image)
                    if(gt!=None):
                        strLabel = idxToName[np.argmax(np.mean(gt[b, :, :, :], axis=(0, 1)))].split(',')[0]
                        axarr[0, 0].set_title(strLabel, fontsize=fontsize)
                    plt.colorbar(axx, ax = axarr[0, 0])
                elif(camIdx < 5):
                    gtImg = gt[b, :, :, idxs[b, camIdx]]
                    resizeGT = zoom(gtImg, [yFactor, xFactor])
                    axarr[y, x].imshow(norm_image)
                    axx = axarr[y, x].imshow(resizeGT, cmap=colormap, vmax=1, vmin=0, alpha=.6)
                    plt.colorbar(axx, ax = axarr[y, x])
                    #Take only label after first comma
                    strLabel = idxToName[idxs[b, camIdx]].split(',')[0]
                    axarr[y, x].set_title(strLabel+ " gt", fontsize=fontsize)

                else:
                    camIdx = camIdx - 5
                    if(camIdx < numCam):
                        camImg = sortedCam[camIdx, :, :]

                        #maxCam = np.max(camImg)
                        ##minCam = -maxCam
                        #minCam = np.min(camImg)

                        resizeCam = zoom(camImg, [yFactor, xFactor])
                        axarr[y, x].imshow(norm_image)
                        axx = axarr[y, x].imshow(resizeCam, cmap=colormap, vmax=maxCam, vmin=0, alpha=.6)
                        plt.colorbar(axx, ax = axarr[y, x])
                        #Take only label after first comma
                        strLabel = idxToName[idxs[b, camIdx]].split(',')[0]
                        axarr[y, x].set_title(strLabel + ": " + str('%.2g'%vals[b, camIdx]), fontsize=fontsize)
                #else:
                #    camIdx = camIdx - 5
                #    if(camIdx < numCam):
                #        [numWeights, numClass] = weights.shape
                #        plotWeights = weights[:, idxs[camIdx]]
                #        #axx = axarr[y, x].hist(plotWeights, 20, facecolor='green')
                #        axx = axarr[y, x].bar(range(numWeights), plotWeights)
                #        axarr[y, x].set_ylim([minWeight, maxWeight])
                #        strLabel = idxToName[idxs[camIdx]].split(',')[0]
                #        axarr[y, x].set_title(strLabel + ": " + str(vals[camIdx]))

        plt.tight_layout()

        plt.savefig(outPrefix + "_" + str(b) + ".png")
        plt.close(f)

        #Find category per pixel and assign color
        #Decrease the weight of the distractor class
        camLabels = np.argmax(reCam[b, :, :, :], axis=0)
        gtLabels = np.argmax(gt[b, :, :, :], axis=2)

        uniqueCamLabels = np.unique(camLabels)
        uniqueGtLabels = np.unique(gtLabels)

        numUnique = len(np.unique(np.concatenate((uniqueCamLabels, uniqueGtLabels))))
        labelColors = get_N_HexCol(numUnique)

        #Assign color per label
        c = {}
        i=0
        for l in uniqueCamLabels:
            if(not l in c):
                c[l] = labelColors[i]
                i += 1

        for l in uniqueGtLabels:
            if(not l in c):
                c[l] = labelColors[i]
                i += 1

        outCam = np.zeros((nyCam, nxCam, 3))
        outGt = np.zeros((nyCam, nxCam, 3))

        numPerLabel = dict.fromkeys(c.keys(), 0)

        for (i, label) in enumerate(uniqueCamLabels):
            camIdxs = np.nonzero(camLabels == label)
            numPerLabel[label] += len(camIdxs[0])
            outCam[camIdxs[0], camIdxs[1], :] = matplotlib.colors.colorConverter.to_rgb(c[label])

        for (i, label) in enumerate(uniqueGtLabels):
            gtIdxs = np.nonzero(gtLabels == label)
            numPerLabel[label] += len(gtIdxs[0])
            outGt[gtIdxs[0], gtIdxs[1], :] = matplotlib.colors.colorConverter.to_rgb(c[label])

        rects = []
        labels = []
        #Rank by numPerLabel
        sortLabel = sorted(numPerLabel.items(), key=operator.itemgetter(1))
        #Reverse sort
        sortLabel = sortLabel[::-1]

        for (label, drop) in sortLabel:
            rects.append(matplotlib.patches.Rectangle((0, 0), 2, 2, fc=c[label]))
            labels.append(idxToName[label])

        resizeCam = zoom(outCam, [yFactor, xFactor, 1])
        resizeGt = zoom(outGt, [yFactor, xFactor, 1])

        f, axarr = plt.subplots(2, 1, figsize=(7.5, 10))
        axarr[0].imshow(norm_image)
        axarr[0].imshow(resizeCam, alpha=.9)
        axarr[0].set_title("Estimate")
        axarr[1].imshow(norm_image)
        axarr[1].imshow(resizeGt, alpha=.9)
        axarr[1].set_title("GT")

        plt.legend(rects, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.tight_layout()
        plt.savefig(outPrefix + "_" + str(b) + "_agg.png")
        plt.close(f)





