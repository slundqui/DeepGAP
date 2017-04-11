import numpy as np
import pdb

#Calculates precision vs recall for one threshold
def calcStats(est, gt, threshold):
    numExample = est.shape
    t_est = np.zeros((numExample))
    t_est[np.nonzero(est >= threshold)] = 1
    #We need tp, fn, and fp
    tp = float(len(np.nonzero(np.logical_and(gt == 1, t_est == 1))[0]))
    fn = float(len(np.nonzero(np.logical_and(gt == 1, t_est == 0))[0]))
    fp = float(len(np.nonzero(np.logical_and(gt == 0, t_est == 1))[0]))
    tn = float(len(np.nonzero(np.logical_and(gt == 0, t_est == 0))[0]))

    if(tp+fp == 0):
        #Precision is defined as 1 here
        precision = 1
    else:
        precision = tp/(tp+fp)

    if(tp+fn == 0):
        pdb.set_trace()
    if(fp+tn == 0):
        pdb.set_trace()

    #Also tpr
    recall = tp/(tp+fn)

    fpr = fp/(fp+tn)
    return (precision, recall, fpr)

#Calculates auc for a batch of files
def calcBatchAuc(estFList, gtFList, doPvr, numThresh = 50):
    numBatch = len(estFList)
    assert(numBatch == len(gtFList))
    precision = np.zeros((numBatch, numThresh))
    recall = np.zeros((numBatch, numThresh))
    fpr = np.zeros((numBatch, numThresh))

    for batchIdx, (fn, gtfn) in enumerate(zip(estFList, gtFList)):
        est = np.load(fn)
        gt = np.load(gtfn)
        estMin = est.min()
        estMax = est.max()

        thresh = np.linspace(estMin-1e-6, estMax+1e-6, num=numThresh)
        #We test a range of thresholds
        for j,t in enumerate(thresh):
            (p, r, f) = calcStats(est, gt, t)
            precision[batchIdx, j] = p
            recall[batchIdx, j] = r
            fpr[batchIdx, j] = f

    if(doPvr):
        auc = calcAuc(precision, recall)
    else:
        auc = calcAuc(recall, fpr)

    return (precision, recall, fpr, auc)


def calcAuc(precision, recall):
    assert(precision.shape == recall.shape)
    (numBatch, numThresh) = precision.shape
    auc = np.zeros((numBatch,))
    for j in range(1, numThresh):
        #Calculate area of trap
        height = abs(recall[:, j]-recall[:, j-1])
        auc += (height*(precision[:, j-1]+precision[:, j]))/2
    return auc

