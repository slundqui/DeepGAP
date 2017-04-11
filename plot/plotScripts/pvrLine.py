import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb
from util import calcStats, calcBatchAuc

if __name__ == "__main__":
    baseDir = "/media/data/slundquist/mountData/DeepGAP/"
    doPvr = True #pvr vs roc

    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/trainVsAUC_"
    titleSuffix = ""

    #Innermost list contains batch
    estFiles = [
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_2000_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_2000_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_2000_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_2000_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_2000_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_2000_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_2000_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_2000_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_2000_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_4000_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_4000_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_4000_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_4000_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_4000_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_4000_run3/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_4000_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_4000_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_4000_run3/evalEstIdxs.npy",
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
            ["eval_sup_kitti_vid_4x8_boot_1_bin_2000_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_2000_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_2000_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_2000_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_2000_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_2000_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_2000_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_2000_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_2000_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_4000_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_4000_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_4000_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_4000_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_4000_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_4000_run3/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_4000_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_4000_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_4000_run3/evalEstIdxs.npy",
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
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_2000_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_2000_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_2000_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_2000_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_2000_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_2000_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_2000_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_2000_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_2000_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_4000_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_4000_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_4000_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_4000_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_4000_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_4000_run3/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_4000_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_4000_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_4000_run3/evalGtIdxs.npy",
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
            ["eval_sup_kitti_vid_4x8_boot_1_bin_2000_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_2000_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_2000_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_2000_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_2000_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_2000_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_2000_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_2000_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_2000_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_4000_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_4000_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_4000_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_4000_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_4000_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_4000_run3/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_4000_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_4000_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_4000_run3/evalGtIdxs.npy",
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

    #Inner group defines number of points on plot
    #Outer list defines number of lines
    lineGroup = [(0,3,6), (1,4,7), (2,5,8), (9,12,15), (10,13,16), (11, 14, 17)]
    #Labels of lines
    lineLabels = ["2 layer Sparse", "3 layer Sparse", "4 layer Sparse",
                  "2 layer Sup", "3 layer Sup", "4 layer Sup"]
    #x values of points
    xPoints = [2000, 4000, 6167]

    colors = [
            (1, .8, .8),
            (1, .4, .4),
            (1, 0, 0),
            (.8, .8, 1),
            (.4, .4, 1),
            (0, 0, 1),
    ]

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

    xmin = 1000
    xmax = 6500

    #Store auc
    auc = [None for outer in estFiles]

    #per plot loop
    for outerIdx, (estFList, gtFList) in enumerate(zip(estFiles, gtFiles)):
        if(type(estFList) is not list):
            estFList = [estFList]
        if(type(gtFList) is not list):
            gtFList = [gtFList]

        fn = [baseDir + estF for estF in estFList]
        gtfn = [baseDir + gtF for gtF in gtFList]

        (precision, recall, fpr, tmpAuc) = calcBatchAuc(fn, gtfn, doPvr)
        auc[outerIdx] = tmpAuc

        #TODO do plots here?
        ##Plot
        #if(doPvr):
        #    plt.plot(recall[0], precision[0], linewidth=4, label=label, color=c)
        #else:
        #    plt.plot(fpr[0], recall[0], linewidth=4, label=label, color=c)

    #Calculate for each random
    randFn = [baseDir + chanceEst for chanceEst in randEstFiles]
    randGtfn = [baseDir + randGtFile for drop in randEstFiles]
    (precision, recall, fpr, tmpAuc) = calcBatchAuc(randFn, randGtfn, doPvr)
    randChanceAuc = tmpAuc

    #if(doPvr):
    #    plt.plot(recall[0], precision[0], 'k--', linewidth=4, label="chance")
    #else:
    #    plt.plot(fpr[0], recall[0], 'k--', linewidth=4, label="chance")
    meanRandChanceAuc = np.mean(randChanceAuc)

    assert(len(xPoints) == len(auc[0]))
    auc = np.array(auc)
    aucMean = np.mean(auc, axis=1)
    aucStd = np.std(auc, axis=1)


    f = plt.figure()
    #Draw multple lines
    for i, lineIdx in enumerate(lineGroup):
        yPoints = aucMean[np.array(lineIdx)]
        yErr = aucStd[np.array(lineIdx)]
        label = lineLabels[i]
        color = colors[i]
        plt.errorbar(xPoints, yPoints, linewidth=2, label=label, color=color, yerr = yErr, fmt="-o")

    plt.xlabel("Training Examples", fontsize=20)
    if(doPvr):
        ylabel = "Area under PvR"
    else:
        ylabel = "Area under ROC"
    title = "Training vs " + ylabel
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title + titleSuffix, fontsize = 30)
    plt.xlim(xmin, xmax)
    plt.errorbar([xmin, xmax], [meanRandChanceAuc, meanRandChanceAuc], yerr=[0, 0],
            fmt="k--", label="Chance", linewidth=2)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if(doPvr):
        outName = outPrefix+'pvr.png'
    else:
        outName = outPrefix+'roc.png'

    plt.savefig(outName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()



    #Do pvr plots?


