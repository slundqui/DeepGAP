import numpy as np
import scipy.sparse as sp
from pvtools import *
import pdb

import tensorflow as tf

def bb_to_pvp(dataObj, windowSize, imageBatch, gtShape, outPrefix):
    dataShape = dataObj.inputShape
    numClasses = dataObj.numClasses

    #Make sure gtShape divides into output shape
    assert(dataShape[0] % gtShape[0] == 0)
    assert(dataShape[1] % gtShape[1] == 0)

    strideY = dataShape[0]/gtShape[0]
    strideX = dataShape[1]/gtShape[1]

    #Placeholder for ground truth array
    tf_inGt = tf.placeholder("float", shape=[None, dataObj.inputShape[0], dataObj.inputShape[1], 1], name="inGt")

    with tf.device('/cpu:0'):
        tf_area_bb = tf.squeeze(tf.reduce_sum(tf_inGt, reduction_indices=[1, 2, 3], keep_dims=True), squeeze_dims=[3])

        #We avg pool per window size on each gt
        tf_iou = []
        for wSize in windowSize:
            (ySize, xSize) = wSize
            tf_area_wSize = ySize * xSize
            #TF Average pool only averages over valid patches, so we pad with zeros
            tf_padGt = tf.pad(tf_inGt, [[0, 0], [ySize/2, (ySize/2)-1], [xSize/2, (xSize/2)-1], [0, 0]])
            tf_avg_pool = tf.squeeze(tf.nn.avg_pool(tf_padGt, [1, ySize, xSize, 1], [1, strideY, strideX, 1], "VALID"), squeeze_dims=[3])
            #Undo average pool to find intersection
            tf_intersection = tf_avg_pool * tf_area_wSize
            #Save final node
            #We sum 2 bbs, minus intersection to find union
            tf_union = tf_area_bb + tf_area_wSize - tf_intersection
            tf_iou.append(tf_intersection/tf_union)

        #Pack into outer dimmension
        tf_iou_pack = tf.pack(tf_iou, axis=0)

    init = tf.initialize_all_variables()

    numIterations = int(np.ceil(float(dataObj.numImages) / imageBatch))
    outSize = (dataObj.inputShape[0], dataObj.inputShape[1], 1)

    numWindowSize = len(windowSize)

    with tf.Session() as sess1:
        sess1.run(init)

        pvpFiles = []
        for wIdx in range(numWindowSize):
            outStr = outPrefix + str(windowSize[wIdx][0])+"x"+str(windowSize[wIdx][1])+".pvp"
            pvpFiles.append(pvpOpen(outStr, "w"))

        for it in range(numIterations):
            allBB = []
            bbPerImage = []
            bbObjId = []
            print "Getting BB,", it, "out of", numIterations
            for i in range(imageBatch):
                imgIdx = it*imageBatch + i
                if(i >= dataObj.numImages):
                    break
                (img, bbs) = dataObj.nextImage(onlyGt = True)
                numBB = len(bbs)
                bbPerImage.append(numBB)
                allBB.extend(bbs)

            numTotBB = len(allBB)
            gtArray = np.zeros((numTotBB,)+outSize)
            print "Buiding BB"
            for j, bb in enumerate(allBB):
                imgIdx = it*imageBatch + j
                (objid, up, down, left, right) = bb
                gtArray[j, up:down, left:right, :] = 1
                bbObjId.append(objid)
            print "Calculating IOU"
            feedDict = {tf_inGt: gtArray}
            #This is a matrix of [numWindows, numBB, gtShape[0], gtShape[1]]
            np_iou_pack = sess1.run(tf_iou_pack, feed_dict=feedDict)

            #We calculate per window shape
            print "Making and writing sparse matrix"
            for wIdx in range(numWindowSize):
                iou = np_iou_pack[wIdx, :, :, :]
                (numTotBB, gtY, gtX) = iou.shape
                #We want to translate this to a sparse matrix per image,
                #where we have 3 lists, data, rowidx, colidx
                outData = []
                outRowIdx = []
                outColIdx = []
                outTime = []
                windowIdx = 0
                for i, numBB in enumerate(bbPerImage):
                    imgIdx = it*imageBatch + i
                    outTime.append(imgIdx)
                    for bbIdx in range(numBB):
                        bbIOU = iou[windowIdx, :, :]
                        bbId = bbObjId[windowIdx]
                        nzIdx = np.nonzero(bbIOU)
                        nnz = len(nzIdx[0])
                        (yIdx, xIdx) = nzIdx
                        #Convert y, x, and f (bbId) to linear index
                        linIdx = yIdx*gtShape[1]*numClasses + xIdx*numClasses + bbId
                        outData.extend(list(bbIOU[nzIdx]))
                        outColIdx.extend(list(linIdx))
                        outRowIdx.extend([i for j in range(nnz)])

                        windowIdx += 1
                #Sanity check
                assert(windowIdx == numTotBB)
                outSparse = sp.csr_matrix((outData, (outRowIdx, outColIdx)),
                    shape=(imageBatch, gtShape[0]*gtShape[1]*numClasses))
                pvpData = {"values": outSparse, "time":outTime}
                pvpFiles[wIdx].write(pvpData, shape=(gtShape[0], gtShape[1], numClasses))
