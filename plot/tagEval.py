import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from dataObj.image import vocObj
import pdb


def calcMetric(est, gt, outfn, num=50, title=""):
    (numExample, numClass) = est.shape
    estMin = est.min()
    estMax = est.max()

    thresh = np.linspace(estMin, estMax, num=num)
    precision = np.zeros((num))
    recall = np.zeros((num))

    #We threshold the estimate to make a binary matrix
    for i,t in enumerate(thresh):
        t_est = np.zeros((numExample, numClass))
        t_est[np.nonzero(est >= t)] = 1

        #We need tp, fn, and fp
        tp = float(len(np.nonzero(np.logical_and(gt == 1, t_est == 1))[0]))
        fn = float(len(np.nonzero(np.logical_and(gt == 1, t_est == 0))[0]))
        fp = float(len(np.nonzero(np.logical_and(gt == 0, t_est == 1))[0]))

        precision[i] = tp/(tp+fp)
        recall[i] = tp/(tp+fn)

    f = plt.figure()
    plt.plot(recall, precision, linewidth=4)
    plt.plot([0, 1], [1, 0], 'r--', linewidth=4)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title(title + " AUC="+str('%.3f'%np.abs(np.trapz(precision, x=recall))), fontsize=30)
    plt.savefig(outfn)
    plt.close(f)



if __name__ == "__main__":
    idxToName = [
       'aeroplane', 'bicycle', 'bird', 'boat',
       'bottle', 'bus', 'car', 'cat', 'chair',
       'cow', 'diningtable', 'dog', 'horse',
       'motorbike', 'person', 'pottedplant',
       'sheep', 'sofa', 'train', 'tvmonitor']

    testImageList = "/shared/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    testImagePrefix = "/shared/VOCdevkit/VOC2007/JPEGImages/"
    testGTPrefix =  "/shared/VOCdevkit/VOC2007/Annotations/"
    #testDataObj = vocDetObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, shuffle=False, singleObj=False)
    testDataObj = vocObj(testImageList, testImagePrefix, testGTPrefix, resizeMethod="crop", normStd=False, shuffle=False, singleObj=False)

    numTest = 2500
    gt = np.zeros((numTest, 20))
    est = np.zeros((numTest, 20))

    testEstFile = "/home/slundquist/mountData/fcnnOut/voc2007Tag.txt"

    f = open(testEstFile, 'r')
    lines = f.readlines()

    for (i, l) in enumerate(lines):
        print i, "out of", numTest
        if(i >= numTest):
            break
        vals = l.split()[1:]
        vals = np.array([float(v) for v in vals])
        est[i, :] = vals
        (drop, s_gt) = testDataObj.getData(1)
        gt[i, :] = s_gt[0, :]

    runDir = "/home/slundquist/mountData/fcnnOut/"
    np.save(runDir + "est.pkl", est)
    np.save(runDir + "gt.pkl", gt)

    calcMetric(est, gt, "/home/slundquist/mountData/fcnnOut/fcnn_pvr.png", title="F-RCNN")








    #for i in range(numTest):
    #    gt[i, :] = s_gt








