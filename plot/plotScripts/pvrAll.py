import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb
from util import calcStats, calcBatchAuc

if __name__ == "__main__":
    baseDir = "/media/data/slundquist/mountData/DeepGAP/"
    doPvr = True #pvr vs roc
    doLinePlot = False

    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/models_pvr"
    titleSuffix = ""

    #barGroup = [(0,1,2,18,19,20), (3,4,5,21,22,23), (6,7,8,24,25,26), (9,10,11,27,28,29), (12,13,14,30,31,32), (15,16,17,33,34,35)]

    ##Number of groups labels
    #groupLabels = ["100", "500", "1000", "2000", "4000", "6167"]

    ##Inner group labels
    #innerBarLabels = ["Sparse 2 layer",
    #                  "Sparse 3 layer",
    #                  "Sparse 4 layer",
    #                  "Sup 2 layer",
    #                  "Sup 3 layer",
    #                  "Sup 4 layer",]
    #width = 0.15      # the width of the bars

    #barColors = [
    #        (1, .7, .7),
    #        (1, .3, .3),
    #        (.8, 0, 0),
    #        (.7, .7, 1),
    #        (.3, .3, 1),
    #        (0, 0, .8),
    #]

    #Outer list is for inner group
    #Inner list is for number of groups
    #barGroup = [(0,3,6,9,12), (1,4,7,10,13), (2,5,8,11,14)]
    barGroup = [(0,12,9,6,3), (1,13,10,7,4), (2,14,11,8,5)]
    #Inner group labels
    #innerBarLabels = ["CR", "SU", "CS", "CUFT", "CU"]
    innerBarLabels = ["DirectRand", "DirectUnsup", "DirectFinetune", "DirectSup", "SparseUnsup"]

    #Number of groups labels
    groupLabels = ["2 layer", "3 layer", "4 layer"]


    width = 0.18      # the width of the bars

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
            "SupFT 2 layer",
            "SupFT 3 layer",
            "SupFT 4 layer",
            "SupNoFT 2 layer",
            "SupNoFT 3 layer",
            "SupNoFT 4 layer",
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
            (.8, 1, .8),
            (.4, 1, .4),
            (0, 1, 0),
            (1, .8, 1),
            (1, .4, 1),
            (1, 0, 1),
    ]

    #barColors = [colors[1], colors[4], colors[7], colors[10], colors[13]]
    barColors = [colors[1], colors[13], colors[10], colors[7], colors[4]]


    estFiles = [
            ["eval_rand_kitti_vid_4x8_boot_1_bin_run1/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run2/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run3/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run4/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run5/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_2_bin_run1/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run2/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run3/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run4/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run5/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_3_bin_run1/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run2/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run3/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run4/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run5/evalEstIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run3/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run4/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run5/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run3/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run4/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run5/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_run1/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run2/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run3/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run4/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run5/evalEstIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run6/evalEstIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run1/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run2/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run3/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run4/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run5/evalEstIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run6/evalEstIdxs.npy",
            ],
            ]

    gtFiles = [
            ["eval_rand_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run2/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run3/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run4/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run5/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_1_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_2_bin_run1/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run2/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run3/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run4/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run5/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_2_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_rand_kitti_vid_4x8_boot_3_bin_run1/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run2/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run3/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run4/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run5/evalGtIdxs.npy",
             "eval_rand_kitti_vid_4x8_boot_3_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run3/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run4/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run5/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_1_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_2_bin_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run3/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run4/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run5/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_2_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_tfpv_kitti_vid_4x8_boot_3_bin_run1/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run2/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run3/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run4/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run5/evalGtIdxs.npy",
             "eval_tfpv_kitti_vid_4x8_boot_3_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_1_bin_pretrain_noft_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_2_bin_pretrain_noft_run6/evalGtIdxs.npy",
            ],
            ["eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run1/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run2/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run3/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run4/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run5/evalGtIdxs.npy",
             "eval_sup_kitti_vid_4x8_boot_3_bin_pretrain_noft_run6/evalGtIdxs.npy",
            ],
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

    #gtFile = baseDir + "/fcnnOut/gt.pkl.npy"

    #Store auc wrt fileidx
    auc = [None for i in range(len(estFiles))]

    if(doLinePlot):
        plt.figure(1)

    #per file
    for fileIdx, (estFList, gtFList) in enumerate(zip(estFiles, gtFiles)):
        if(type(estFList) is not list):
            estFList = [estFList]
        if(type(gtFList) is not list):
            gtFList = [gtFList]

        fn = [baseDir + estF for estF in estFList]
        gtfn = [baseDir + gtF for gtF in gtFList]

        (precision, recall, fpr, tmpAuc) = calcBatchAuc(fn, gtfn, doPvr)
        auc[fileIdx] = tmpAuc

        if(doLinePlot):
            label = labels[outerIdx]
            c = colors[outerIdx]
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
    meanRandChanceAuc = np.mean(randChanceAuc)

    if(doLinePlot):
        if(doPvr):
            plt.plot(recall[0], precision[0], 'k--', linewidth=4, label="chance")
        else:
            plt.plot(fpr[0], recall[0], 'k--', linewidth=4, label="chance")

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

    #Calculate median, min, max
    aucMed = []
    aucMin = []
    aucMax = []
    for runAuc in auc:
        aucMed.append(np.median(runAuc))
        aucMin.append(np.min(runAuc))
        aucMax.append(np.max(runAuc))

    f = plt.figure()

    #Make bar graph of area under curve
    numInnerGroup = len(barGroup[0])
    numGroups = len(barGroup)

    #Calculate the starting x location for each group
    ind = np.arange(numGroups)

    fig, ax = plt.subplots()

    rects = [None for i in range(numInnerGroup)]
    maxBar = 0

    #We plot each inner group at the same time
    for innerGroupIdx in range(numInnerGroup):
        barVals = [aucMed[idx[innerGroupIdx]] for idx in barGroup]
        barMin = [aucMed[idx[innerGroupIdx]] - aucMin[idx[innerGroupIdx]] for idx in barGroup]
        barMax = [aucMax[idx[innerGroupIdx]] - aucMed[idx[innerGroupIdx]] for idx in barGroup]
        errVals = np.array([barMin, barMax])

        #errVals = [aucStd[idx[innerGroupIdx]] for idx in barGroup]
        xLoc = ind + (width*innerGroupIdx)

        #Plot each group at the same time
        rects[innerGroupIdx] = ax.bar(xLoc, barVals, width, color=barColors[innerGroupIdx], yerr=errVals, error_kw=dict(lw=3, capsize=4, capthick=3, ecolor='k'))

        #Add number label to top
        tmpMaxBar = np.max(barVals)
        if(tmpMaxBar > maxBar):
            maxBar = tmpMaxBar

    #for innerGroupIdx in range(numInnerGroups):
    #    xLoc = ind[groupIdx] + (width*innerOffset)
    #    barVals = [aucMean[idx] for idx in barGroup[groupIdx]]
    #    for x, y in zip(xLoc, barVals):
    #        plt.text(x+(width/2), maxBar+.02, "{0:.2f}".format(y),
    #                {'ha': 'center', 'va': 'bottom'}, rotation=90)

    #Plot random chance
    chanceAx = plt.plot([0, numGroups], [meanRandChanceAuc, meanRandChanceAuc], 'k--', linewidth=4)
    plt.text(numGroups, meanRandChanceAuc, " Chance".format(meanRandChanceAuc),
            {'ha': 'left', 'va': 'center'})
    #plt.text(N, meanRandChanceAuc, "{0: .2f}\n Chance".format(meanRandChanceAuc),
    #        {'ha': 'left', 'va': 'center'})

    ax.set_ylabel('AUC')
    if(doPvr):
        ax.set_title('Area under PvR curve' + titleSuffix)
    else:
        ax.set_title('Area under ROC curve' + titleSuffix)

    #xticks = [ind[groupIdx] + (width*innerOffset/2) for groupIdx in range(numGroups)]
    ax.set_xticks(ind + (float(numInnerGroup)*width)/2)

    # add some text for labels, title and axes ticks
    ax.set_xticklabels(groupLabels)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.ylim(0, .8)
    #plt.ylim(0, 1)

    lgd = ax.legend([r[0] for r in rects], innerBarLabels, loc='upper left', bbox_to_anchor=(1, 1))

    if(doPvr):
        outName = outPrefix+'pvr_auc.png'
    else:
        outName = outPrefix+'roc_auc.png'


    #plt.savefig(outName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(outName, bbox_inches='tight')
    #plt.savefig('auc_all.png', bbox_extra_artists=(lgd,f))
    plt.clf()
