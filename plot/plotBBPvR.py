import matplotlib.pyplot as plt
import numpy as np
import pdb


#outScore, outBB, and gtBB are all lists of length batchSize
#outScore is [numOutBB]
#outBB is [numOutBB, 4]
#gtBB is [numGtBB, 4]
def plotBBPvRBatch(outScores, outBBs, gtBBs, iouDetThreshold, filenamePrefix):
    numImages = len(outScores)
    assert(numImages == len(outBBs))
    assert(numImages == len(gtBBs))

    numThresholds = 100

    tp = np.zeros((numThresholds,))
    fp = np.zeros((numThresholds,))
    totalGT = 0
    for b in range(numImages):
        (numGT, drop) = gtBBs[b].shape
        (outTp, outFp, outNumCandidate) = calcMetric(outScores[b], outBBs[b], gtBBs[b], iouDetThreshold, numThresholds)
        (numCandidates, drop) = outBBs[b].shape
        tp += outTp
        fp += outFp
        totalGT += numGT

    #Calculate precision and recall
    precision = tp/(tp+fp)
    #If no selected items, set precision to 1
    precision[np.nonzero(tp+fp == 0)] = 1
    recall = tp/totalGT

    f1 = 2*((precision * recall)/(precision + recall))
    #Change nans to 0
    f1[np.nonzero(np.logical_not(np.isfinite(f1)))] = 0

    bestF1Idx = np.argmax(f1)
    thresholds = np.linspace(0, 1, num=numThresholds)
    bestThresh = thresholds[bestF1Idx]

    auc = 0
    for i in range(numThresholds-1):
        #height of trap is delta recall
        #a and b are precision
        dRecall = np.abs(recall[i] - recall[i+1])
        a = precision[i]
        b = precision[i+1]
        auc += ((a+b)/2) * dRecall

    f = plt.figure()
    plt.plot(recall, precision, linewidth=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title("Precision Vs Recall", fontsize = 30)
    threshStr = "%.2f" % bestThresh
    aucStr = "%.2f" % auc
    t = plt.text(.8, .8, 'bestT: ' + threshStr + "\nauc: " + aucStr, fontsize=12)
    filename = filenamePrefix + "_pvr.png"
    plt.savefig(filename, bbox_inches='tight', bbox_extra_artists=(t,))
    plt.close(f)
    return (precision, recall, f1, bestThresh)

def plotBBPvRBatch_Obsolete(outScores, outBBs, gtBBs, iouDetThreshold, filenamePrefix):
    print "Plotting PvR"
    batchSize = len(outScores)
    assert(batchSize == len(outBBs))
    assert(batchSize == len(gtBBs))

    #Make outScores and outBBs into one list, adding in a batch id into outBBs
    flatScore = np.array([s for sub in outScores for s in sub])
    idxOutBBs = np.array([[bi, b[0], b[1], b[2], b[3]] for (bi, sub) in enumerate(outBBs) for b in sub])

    #Calculate number of GTs

    numCandidate = len(flatScore)
    numGT = len([s for sub in gtBBs for s in sub])

    #Data structure for checking which GT boxes have been detected
    gtDet = [[False for s in sub] for sub in gtBBs]

    #Sort score in decreasing order
    sortIdx = np.argsort(flatScore)[::-1]
    sortScore = flatScore[sortIdx]
    sortOutBB = idxOutBBs[sortIdx]

    tp = np.zeros((numCandidate,))
    fp = np.zeros((numCandidate,))

    for iout, outBB in enumerate(sortOutBB):
        if(iout % 100 == 0):
            print iout, "out of", len(sortOutBB)
        (imageIdx, outYmin, outXmin, outYmax, outXmax) = outBB
        outArea = (outYmax - outYmin)*(outXmax-outXmin)
        imageIdx = int(imageIdx)
        iouMax = -np.inf
        iouMaxIdx = -1
        #Find max overlap with gt box
        for igt, gtBB in enumerate(gtBBs[imageIdx]):
            (gtYmin, gtXmin, gtYmax, gtXmax) = gtBB
            intYmin = np.max([outYmin, gtYmin])
            intXmin = np.max([outXmin, gtXmin])
            intYmax = np.min([outYmax, gtYmax])
            intXmax = np.min([outXmax, gtXmax])
            intH = intYmax - intYmin
            intW = intXmax - intXmin
            #If overlapping
            if intH > 0 and intW > 0:
                gtArea = (gtYmax - gtYmin) * (gtXmax - gtXmin)
                intArea = intH * intW
                unionArea = gtArea + outArea - intArea
                iou = intArea.astype(np.float32) / unionArea
                if iou > iouMax:
                    iouMax = iou
                    iouMaxIdx = igt

        #Assign detection
        if iouMax >= iouDetThreshold:
            assert(iouMaxIdx >= 0)
            #Check if gt box has already been detected
            if not gtDet[imageIdx][iouMaxIdx]:
                tp[iout] = 1
            else:
                fp[iout] = 1 #Already detected, count as false positive
        else:
            fp[iout] = 1

    cumTp = np.cumsum(tp)
    cumFp = np.cumsum(fp)

    recall = cumTp.astype(np.float32)/numGT
    precision = cumTp.astype(np.float32)/(cumTp+cumFp)

    f1 = 2*((precision * recall)/(precision + recall))
    #Change nans to 0
    f1[np.nonzero(np.logical_not(np.isfinite(f1)))] = 0

    bestF1Idx = np.argmax(f1)
    bestThresh = sortScore[bestF1Idx]

    #TODO calculate area under curve
    auc = 0
    for i in range(numCandidate-1):
        #height of trap is delta recall
        #a and b are precision
        dRecall = np.abs(recall[i] - recall[i+1])
        a = precision[i]
        b = precision[i+1]
        auc += ((a+b)/2) * dRecall

    f = plt.figure()
    plt.plot(recall, precision, linewidth=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title("Precision Vs Recall", fontsize = 30)
    threshStr = "%.2f" % bestThresh
    aucStr = "%.2f" % auc
    t = plt.text(.8, .8, 'bestT: ' + threshStr + "\nauc: " + aucStr, fontsize=12)
    filename = filenamePrefix + "_pvr.png"
    plt.savefig(filename, bbox_inches='tight', bbox_extra_artists=(t,))
    plt.close(f)
    return (precision, recall, f1, bestThresh)


def calcMetric(outScore, outBB, gtBB, iouDetThreshold, numThresholds=50):
    #Calculate IOU matrix of shape [numOutBB, numGTBB]
    #Expand out out and gt bb into seperate dimensions
    #such that each combination of out and gt bbs can be calculated
    e_outBB = np.expand_dims(outBB, axis=1)
    e_gtBB = np.expand_dims(gtBB, axis=0)

    outArea = (e_outBB[:, :, 2] - e_outBB[:, :, 0]) * (e_outBB[:, :, 3] - e_outBB[:, :, 1])
    gtArea = (e_gtBB[:, :, 2] - e_gtBB[:, :, 0]) * (e_gtBB[:, :, 3] - e_gtBB[:, :, 1])

    #np.maximum/minimum broadcasts the dimension into the singleton dimension
    intYMin = np.maximum(e_outBB[:, :, 0], e_gtBB[:, :, 0])
    intYMax = np.minimum(e_outBB[:, :, 2], e_gtBB[:, :, 2])
    intXMin = np.maximum(e_outBB[:, :, 1], e_gtBB[:, :, 1])
    intXMax = np.minimum(e_outBB[:, :, 3], e_gtBB[:, :, 3])

    intArea = np.maximum(intYMax-intYMin, 0) * np.maximum(intXMax-intXMin, 0)

    #Calculate union area
    unionArea = outArea + gtArea - intArea

    iou = intArea/unionArea

    (numGt, drop) = gtBB.shape

    #Make sure outScore is in the range of 0 and 1
    assert(np.max(outScore) <= 1)
    assert(np.min(outScore) >= 0)

    outTp = np.zeros((numThresholds,))
    outFp = np.zeros((numThresholds,))
    outNumCandidate = np.zeros((numThresholds,))
    #We now select specific outBBs based on scores and calculate pvr
    for i, t in enumerate(np.linspace(0, 1, num=numThresholds)):
        detIdx = np.nonzero(outScore > t)
        detIou = iou[detIdx[0], :]
        outNumCandidate[i] = len(detIdx[0])

        #Vector of length numCanididate that contains the gt idx of the max iou
        (drop, numDet) = detIou.shape
        if(numDet != 0):
            maxIouVal = np.max(detIou, axis=1)
            maxIouIdx = np.argmax(detIou, axis=1)
        else:
            maxIouVal = np.zeros((numGt,))
            maxIouIdx = np.zeros((numGt,))

        gtDet = [False for j in range(numGt)]
        for iouVal, iouIdx in zip(maxIouVal, maxIouIdx):
            if (iouVal >= iouDetThreshold):
                if(gtDet[iouIdx] is False):
                    outTp[i] += 1
                    gtDet[iouIdx] = True
                #Duplicate canidate bb
                else:
                    outFp[i] += 1
            #Insufficient iou
            else:
                outFp[i] += 1


    return (outTp, outFp, outNumCandidate)



