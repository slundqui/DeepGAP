import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    baseDir = "/home/slundquist/mountData/DeepGAP/"
    estFiles = [
            "eval_voc_vgg/estWhole.pkl.npy",
            "eval_voc_gap_vgg/estWhole.pkl.npy",
            "eval_voc_det_vgg/est.pkl.npy",
            "fcnnOut/est.pkl.npy",
            ]
    labels = [
            "VGG",
            "VGG-GAP",
            "VGG-LOC",
            "VGG-FCRNN",
            ]

    num = 50

    #gtFile = baseDir + "/fcnnOut/gt.pkl.npy"
    gtFiles = [
            "/eval_voc_vgg/gt.pkl.npy",
            "/eval_voc_gap_vgg/gt.pkl.npy",
            "/eval_voc_det_vgg/gt.pkl.npy",
            "/fcnnOut/gt.pkl.npy",
            ]

    plt.figure(1)

    for estF, gtF, label in zip(estFiles, gtFiles, labels):
        fn = baseDir + estF
        gtfn = baseDir + gtF
        est = np.load(fn)
        gt = np.load(gtfn)
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

        plt.plot(recall, precision, linewidth=4, label=label)

    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title("Precision Vs Recall", fontsize = 30)
    #lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    lgd = plt.legend(loc=0)
    plt.savefig('pvr_all.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
