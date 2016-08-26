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

    maskShape = (gtShape[0], gtShape[1], len(windowSize))

    mask = np.zeros((1,) + maskShape)
    for i, window in enumerate(windowSize):
        #Convert windowSize from pixel space to gt space
        nyp = int(round(float(window[0])/strideY))
        nxp = int(round(float(window[1])/strideX))

        #Integer division, will floor
        yMargin = nyp/2
        xMargin = nxp/2

        #Valid locations are in the center
        mask[0, yMargin:-yMargin, xMargin:-xMargin, i] = 1

    outFn = outPrefix + "_mask.pvp"
    data = {"values":mask, "time":[0]}
    writepvpfile(outFn, data)
