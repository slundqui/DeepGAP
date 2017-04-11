import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb
from util import calcStats, calcBatchAuc

if __name__ == "__main__":
    baseDir = "/media/data/slundquist/mountData/DeepGAP/"
    doPvr = True #pvr vs roc


    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/models_"
    titleSuffix = ""
    estFiles = [
            ["eval_rand_kitti_vid_4x8_boot_1_bin_run1/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run2/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_2_bin_run1/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run2/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_3_bin_run1/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run2/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run3/evalEstIdxs.npy",
            ],
            ]

    gtFiles = [
            ["eval_rand_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run2/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_2_bin_run1/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run2/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_3_bin_run1/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run2/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run3/evalGtIdxs.npy",
            ],
            ]

    barGroup = [(0,3,6), (1,4,7), (2,5,8)]
    barLabels = ["2 layer", "3 layer", "4 layer"]
    innerBarLabels = ["Rand", "Sparse", "Sup"]
    width = 0.25       # the width of the bars
    labels = [
            "Rand 2 layer",
            "Rand 3 layer",
            "Rand 4 layer",
            "Sparse 2 layer",
            "Sparse 3 layer",
            "Sparse 4 layer",
            "Sup 2 layer",
            "Sup 3 layer",
            "Sup 4 layer",
            ]
    colors = [
            (.8, .8, .8),
            (.4, .4, .4),
            (0, 0, 0),
            (1, .8, .8),
            (1, .4, .4),
            (1, 0, 0),
            (.8, .8, 1),
            (.4, .4, 1),
            (0, 0, 1),
    ]
    barColors = [colors[1], colors[5], colors[8]]

    randEstFiles = [
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run0.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run1.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run2.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run3.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run4.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run5.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run6.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run7.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run8.npy",
            "eval_randchance_kitti_vid_4x8/evalEstIdxs_run9.npy",
            ]

    randGtFile = "eval_randchance_kitti_vid_4x8/evalGtIdxs.npy"

    #gtFile = baseDir + "/fcnnOut/gt.pkl.npy"

    plt.figure(1)

    #Store auc
    auc = [None for outer in estFiles]

    #per plot loop
    for outerIdx, (estFList, gtFList, label, c) in enumerate(zip(estFiles, gtFiles, labels, colors)):
        if(type(estFList) is not list):
            estFList = [estFList]
        if(type(gtFList) is not list):
            gtFList = [gtFList]

        fn = [baseDir + estF for estF in estFList]
        gtfn = [baseDir + gtF for gtF in gtFList]

        (precision, recall, fpr, tmpAuc) = calcBatchAuc(fn, gtfn, doPvr)
        auc[outerIdx] = tmpAuc

        #Plot
        if(doPvr):
            plt.plot(recall[0], precision[0], linewidth=4, label=label, color=c)
        else:
            plt.plot(fpr[0], recall[0], linewidth=4, label=label, color=c)

    #Calculate for each random
    randFn = [baseDir + chanceEst for chanceEst in randEstFiles]
    randGtfn = [baseDir + randGtFile for drop in randEstFiles]
    (precision, recall, fpr, tmpAuc) = calcBatchAuc(randFn, randGtfn, doPvr)
    randChanceAuc = tmpAuc

    if(doPvr):
        plt.plot(recall[0], precision[0], 'k--', linewidth=4, label="chance")
    else:
        plt.plot(fpr[0], recall[0], 'k--', linewidth=4, label="chance")
    meanRandChanceAuc = np.mean(randChanceAuc)

    plt.ylim(0, 1)
    if(doPvr):
        plt.xlabel("Recall", fontsize=20)
        plt.ylabel("Precision", fontsize=20)
        plt.title("Precision Vs Recall" + titleSuffix, fontsize = 30)
    else:
        plt.xlabel("FPR", fontsize=20)
        plt.ylabel("TPR", fontsize=20)
        plt.title("ROC" + titleSuffix, fontsize = 30)

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #lgd = plt.legend(loc=0)
    if(doPvr):
        outName = outPrefix+'pvr.png'
    else:
        outName = outPrefix+'roc.png'

    plt.savefig(outName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

    #Calculate mean and stds for auc
    aucMean = []
    aucStd = []
    for runAuc in auc:
        aucMean.append(np.mean(runAuc))
        aucStd.append(np.std(runAuc))

    f = plt.figure()
    #Make bar graph of area under curve
    N = len(barGroup)
    ind = np.arange(N)  # the x locations for the groups
    fig, ax = plt.subplots()
    #Loop thorugh inner bars
    numInner = len(barGroup[0])
    rects = [None for i in range(numInner)]
    maxBar = 0

    for i in range(numInner):
        #Loop thorugh outer bars
        barVals = [aucMean[index[i]] for index in barGroup]
        colorVals = [colors[index[i]] for index in barGroup]

        errVals = [aucStd[index[i]] for index in barGroup]
        rects[i] = ax.bar(ind+(width*i), barVals, width, color=barColors[i], yerr=errVals,
                error_kw=dict(lw=5, capsize=8, capthick=2, ecolor='k'))

        xLoc = ind+(width*i)
        rects[i] = ax.bar(xLoc, barVals, width, color=barColors[i])
        #Add number label to top
        tmpMaxBar = np.max(barVals)
        if(tmpMaxBar > maxBar):
            maxBar = tmpMaxBar

    for i in range(numInner):
        xLoc = ind+(width*i)
        barVals = [aucMean[index[i]] for index in barGroup]
        for x, y in zip(xLoc, barVals):
            plt.text(x+(width/2), maxBar+.02, "{0:.2f}".format(y),
                    {'ha': 'center', 'va': 'bottom'}, rotation=90)

    #Plot random chance
    chanceAx = plt.plot([0, N], [meanRandChanceAuc, meanRandChanceAuc], 'k--', linewidth=4)
    plt.text(N, meanRandChanceAuc, "{0: .2f}\n Chance".format(meanRandChanceAuc),
            {'ha': 'left', 'va': 'center'})

    ax.set_ylabel('AUC')
    if(doPvr):
        ax.set_title('Area under PvR curve' + titleSuffix)
    else:
        ax.set_title('Area under ROC curve' + titleSuffix)

    ax.set_xticks(ind + (float(numInner)*width)/2)
    # add some text for labels, title and axes ticks
    ax.set_xticklabels(barLabels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.ylim(0, .8)
    #plt.ylim(0, 1)

    lgd = ax.legend([r[0] for r in rects], innerBarLabels, loc='upper left', bbox_to_anchor=(1, 1))

    if(doPvr):
        outName = outPrefix+'pvr_auc.png'
    else:
        outName = outPrefix+'roc_auc.png'


    plt.savefig(outName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.savefig('auc_all.png', bbox_extra_artists=(lgd,f))
    plt.clf()
