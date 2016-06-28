import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.misc import imresize
import pdb

def plotCam(outFilename, inImage, gtIdx, cam, idxs, vals, idxToName):
    (nyImage, nxImage, nfImage) = inImage.shape
    (numCam, nyCam, nxCam) = cam.shape
    numTotal = numCam + 1
    xTile = int(np.floor(np.sqrt(numTotal)))
    yTile = int(np.ceil(float(numTotal)/xTile))
    resize_image = imresize(inImage, (nyCam, nxCam)).astype(np.float32)
    norm_image = (resize_image - np.min(resize_image))/(np.max(resize_image) - np.min(resize_image))

    #Find max/min and set imshow values to that
    maxCam = np.max(cam)
    minCam = np.min(cam)

    colormap = cm.get_cmap('jet')

    #Plot image first
    f, axarr = plt.subplots(yTile, xTile, sharex=True, sharey=True, figsize=(20, 15))
    for y in range(yTile):
        for x in range(xTile):
            camIdx = y*xTile + x - 1
            if(x == 0 and y == 0):
                axx = axarr[0, 0].imshow(norm_image)
                strLabel = idxToName[gtIdx].split(',')[0]
                axarr[0, 0].set_title(strLabel)
            else:
                if(camIdx < numCam):
                    axx = axarr[y, x].imshow(cam[camIdx, :, :], cmap=colormap, vmax=maxCam, vmin=minCam)
                    #Take only label after first comma
                    strLabel = idxToName[idxs[camIdx]].split(',')[0]
                    axarr[y, x].set_title(strLabel + ": " + str(vals[camIdx]))

    #Plot colorbar
    f.subplots_adjust(right=.8)
    cbar_ax = f.add_axes([.85, .15, .05, .7])
    f.colorbar(axx, cax=cbar_ax)
    plt.savefig(outFilename)
    plt.close(f)
