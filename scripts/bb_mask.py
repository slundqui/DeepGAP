import pdb
from pvtools import *
import numpy as np


def bb_mask(windowSize, gtShape, imageShape, outPrefix):
    numWindows = len(windowSize)

    #Make sure gtShape divides into output shape
    assert(imageShape[0] % gtShape[0] == 0)
    assert(imageShape[1] % gtShape[1] == 0)
    strideY = imageShape[0]/gtShape[0]
    strideX = imageShape[1]/gtShape[1]

    maskShape = (gtShape[0], gtShape[1], 1)

    for window in windowSize:
        #Convert windowSize from pixel space to gt space
        nyp = int(round(float(window[0])/strideY))
        nxp = int(round(float(window[1])/strideX))

        #Integer division, will floor
        yMargin = nyp/2
        xMargin = nxp/2

        mask = np.zeros((1,) + maskShape)
        #Valid locations are in the center
        mask[0, yMargin:-yMargin, xMargin:-xMargin, 0] = 1

        outFn = outPrefix + str(window[0])+"x"+str(window[1])+"_mask.pvp"

        data = {"values":mask, "time":[0]}
        writepvpfile(outFn, data)
