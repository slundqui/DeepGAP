import numpy as np
import scipy.sparse as sp
from pvtools import *
import pdb

import tensorflow as tf

def bb_obj(dataObj, windowSize, imageBatch, gtShape, outPrefix, iouThresh, minIouThresh, device="/gpu:0"):
    dataShape = dataObj.inputShape
    numClasses = dataObj.numClasses

    #Make sure gtShape divides into output shape
    assert(dataShape[0] % gtShape[0] == 0)
    assert(dataShape[1] % gtShape[1] == 0)

    strideY = dataShape[0]/gtShape[0]
    strideX = dataShape[1]/gtShape[1]

    #Placeholder for image
    #inImage = tf.placeholder("float", shape=[None, dataObj.inputShape[0], dataObj.inputShape[1], dataObj.inputShape[2]], name="inImage")

    #Placeholder for ground truth array
    tf_inGt = tf.placeholder("float", shape=[None, dataObj.inputShape[0], dataObj.inputShape[1], 1], name="inGt")

    with tf.device(device):
        tf_area_bb = tf.squeeze(tf.reduce_sum(tf_inGt, reduction_indices=[1, 2, 3], keep_dims=True), squeeze_dims=[3])

        #We avg pool per window size on each gt
        tf_iou = []
        for wSize in windowSize:
            (ySize, xSize) = wSize
            tf_area_wSize = ySize * xSize
            tf_avg_pool = tf.squeeze(tf.nn.avg_pool(tf_inGt, [1, ySize, xSize, 1], [1, strideY, strideX, 1], "SAME"), squeeze_dims=[3])
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

    #Our output is 2 sparse arrays,
    #One is (numImage, windows, gtShape[0], gtShape[1]) binary file that cooresponds to obj or no obj
    #Second is (numImage, windows, gtShape[0], gtShape[1], 5) that cooresponds to (id, t, l, h, w) of bb

    with tf.Session() as sess1:
        sess1.run(init)

        pvpBBObjFile = pvpOpen(outPrefix+"_objFile.pvp", 'w')
        pvpBBLabelFile = pvpOpen(outPrefix+"_labelFile.pvp", 'w')

        for it in range(numIterations):
            allBB = []
            bbPerImage = []
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

            print "Calculating IOU"
            feedDict = {tf_inGt: gtArray}
            #This is a matrix of [numWindows, numBB, gtShape[0], gtShape[1]]
            np_iou_pack = sess1.run(tf_iou_pack, feed_dict=feedDict)
            (numWindows, numTotBB, gtY, gtX) = np_iou_pack.shape

            print "Making and writing sparse file"
            windowIdx = 0
            outBBObjList = []
            outBBLabelList = []
            time = []
            for i, numBB in enumerate(bbPerImage):
                imgIdx = it*imageBatch + i
                time.append(imgIdx)
                bbIOU = np_iou_pack[:, windowIdx:windowIdx+numBB, :, :]
                outBBObj = np.zeros((numWindows, gtY, gtX, 2))
                outBBLabel = np.zeros((numWindows, gtY, gtX, 5))
                for bbIdx in range(numBB):
                    bbWindowIOU = bbIOU[:, bbIdx, :, :]
                    #We assign a positive label to the anchor with highest IOU with ground truth
                    (w, y, x) = np.unravel_index(np.argmax(bbWindowIOU), bbWindowIOU.shape)

                    #This marks the IOU to find the max
                    outBBObj[w, y, x, 0] = bbWindowIOU[w, y, x]
                    outBBLabel[w, y, x, :] = allBB[windowIdx]

                    #We assign an anchor that has IOU >= iouThresh
                    #We take the max IOU as the new iou
                    iouHit = np.nonzero(bbWindowIOU >= iouThresh)
                    newIOU = bbWindowIOU[iouHit]
                    oldIOU = outBBObj[iouHit[0], iouHit[1], iouHit[2], 0]
                    replaceHits = np.array(newIOU > oldIOU)
                    if(len(replaceHits) > 0):
                        outBBObj[iouHit[0], iouHit[1], iouHit[2], 0] = np.where(replaceHits, newIOU, oldIOU)

                        newLabels = np.array(allBB[windowIdx])
                        oldLabels = outBBLabel[iouHit]
                        out = [newLabels if replace else oldLabels[j] for (j, replace) in enumerate(replaceHits.tolist())]
                        outBBLabel[iouHit] = out
                    windowIdx += 1

                #Calculate distractor class, which is anchors with <= minIouThresh for all bb
                if(numBB != 0):
                    maxIOU = np.max(bbIOU, axis=1)
                else:
                    #No bb, all zeros
                    maxIOU = np.zeros((numWindows, gtY, gtX))
                iouMiss = np.nonzero(maxIOU <= minIouThresh)
                outBBObj[iouMiss[0], iouMiss[1], iouMiss[2], 1] = 1

                #Set outBBObj to be binary
                nzIOUIdx = np.nonzero(outBBObj)
                outBBObj[nzIOUIdx] = 1

                #Permute windows dimension to last dimension
                outBBObj = np.transpose(outBBObj, [1, 2, 0, 3])
                outBBLabel = np.transpose(outBBLabel, [1, 2, 0, 3])

                #Make sparse matrix and store in outer array
                flat_outBBObj = np.reshape(outBBObj, (1, numWindows*gtY*gtX*2))
                flat_outBBLabel = np.reshape(outBBLabel, (1, numWindows*gtY*gtX*5))
                outBBObjList.append(sp.csr_matrix(flat_outBBObj))
                outBBLabelList.append(sp.csr_matrix(flat_outBBLabel))

            #Sanity check
            assert(windowIdx == numTotBB)

            sparseBBObjList = sp.vstack(outBBObjList)
            sparseBBLabelList = sp.vstack(outBBLabelList)

            objData = {"values": sparseBBObjList, "time":time}
            pvpBBObjFile.write(objData, shape=(gtShape[0], gtShape[1], numWindows*2))

            labelData = {"values": sparseBBLabelList, "time":time}
            pvpBBLabelFile.write(labelData, shape=(gtShape[0], gtShape[1], numWindows*5))
        #End iterations
    #End sessions
