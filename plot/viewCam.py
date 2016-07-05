import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import zoom
import pdb

def plotCam(outFilename, inImage, gtIdx, cam, idxs, vals, idxToName, weights):
    (nyImage, nxImage, nfImage) = inImage.shape
    (numCam, nyCam, nxCam) = cam.shape
    assert(nyImage%nyCam == 0)
    assert(nxImage%nxCam == 0)

    yFactor = nyImage/nyCam
    xFactor = nxImage/nxCam

    numTotal = numCam*2 + 1
    xTile = int(np.ceil(np.sqrt(numTotal)))
    yTile = int(np.ceil(float(numTotal)/xTile))
    norm_image = (inImage - np.min(inImage))/(np.max(inImage) - np.min(inImage))

    #Find max/min and set imshow values to that
    #maxCam = np.max(cam)
    #minCam = np.min(cam)
    #if(maxCam < 0):
    #    maxCam = 0

    colormap = cm.get_cmap('jet')

    maxWeight = np.max(weights)
    minWeight = np.min(weights)
    #Plot image first
    f, axarr = plt.subplots(yTile, xTile, figsize=(20, 15))

    #avgCam = np.mean(cam, axis=0)

    #f, axarr = plt.subplots(yTile, xTile, sharex=True, sharey=True, figsize=(30, 22.5))
    for y in range(yTile):
        for x in range(xTile):
            camIdx = y*xTile + x - 1
            #First image is the orig image
            if(x == 0 and y == 0):
                axx = axarr[0, 0].imshow(norm_image)
                strLabel = idxToName[gtIdx].split(',')[0]
                axarr[0, 0].set_title(strLabel)
                plt.colorbar(axx, ax = axarr[0, 0])
            elif(camIdx < 5):
                if(camIdx < numCam):
                    camImg = cam[camIdx, :, :]
                    #camImg = camImg-avgCam
                    maxCam = np.max(camImg)
                    minCam = np.min(camImg)
                    rangeCam = maxCam-minCam
                    #minView = (rangeCam*.5)+minCam
                    minView = minCam

                    #resizeCam = np.repeat(np.repeat(camImg, yFactor, axis=0), xFactor, axis=1)
                    resizeCam = zoom(camImg, [yFactor, xFactor])
                    axarr[y, x].imshow(norm_image)
                    axx = axarr[y, x].imshow(resizeCam, cmap=colormap, vmax=maxCam, vmin=minView, alpha=.6)
                    plt.colorbar(axx, ax = axarr[y, x])
                    #Take only label after first comma
                    strLabel = idxToName[idxs[camIdx]].split(',')[0]
                    axarr[y, x].set_title(strLabel + ": " + str(vals[camIdx]))
            else:
                camIdx = camIdx - 5
                if(camIdx < numCam):
                    [numWeights, numClass] = weights.shape
                    plotWeights = weights[:, idxs[camIdx]]
                    #axx = axarr[y, x].hist(plotWeights, 20, facecolor='green')
                    axx = axarr[y, x].bar(range(numWeights), plotWeights)
                    axarr[y, x].set_ylim([minWeight, maxWeight])
                    strLabel = idxToName[idxs[camIdx]].split(',')[0]
                    axarr[y, x].set_title(strLabel + ": " + str(vals[camIdx]))

    plt.tight_layout()

    plt.savefig(outFilename)
    plt.close(f)

