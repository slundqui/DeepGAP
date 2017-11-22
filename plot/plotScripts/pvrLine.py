import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb
from util import calcStats, calcBatchAuc

if __name__ == "__main__":
    baseDir = "/home/slundquist/mountData/DeepGAP/"
    doPvr = True #pvr vs roc
    showChance = False
    showErr = True

    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/trainVsAUC_all_"
    titleSuffix = ""
    #Inner group defines number of points on plot
    #Outer list defines number of lines
    lineGroup = [(0,3,6,9,12), (1,4,7,10,13), (2,5,8,11,14),
                 (15,18,21,24,27), (16,19,22,25,28), (17,20,23,26,29),
                 (30,33,36,39,42), (31,34,37,40,43), (32,35,38,41,44),
                 ]
    #Labels of lines
    lineLabels = ["2 layer SparseUnsup", "3 layer SparseUnsup", "4 layer SparseUnsup",
                  "2 layer DirectSup", "3 layer DirectSup", "4 layer DirectSup",
                  "2 layer DirectFinetune", "3 layer DirectFinetune", "4 layer DirectFinetune",
                  ]
    colors = [
            (1, .7, .7),
            (1, .3, .3),
            (.8, 0, 0),
            (.7, .7, 1),
            (.3, .3, 1),
            (0, 0, .8),
            (1, .7, 1),
            (1, .3, 1),
            (.8, 0, .8),
    ]
    fmt = [
            ':o',
            '--o',
            '-o',
            ':o',
            '--o',
            '-o',
            ':o',
            '--o',
            '-o',
    ]

    #Inner group defines number of points on plot
    #Outer list defines number of lines

    #outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/trainVsAUC_2layer_"
    #titleSuffix = " for 2 layer"
    #lineGroup = [lineGroup[0], lineGroup[3], lineGroup[6]]

    #outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/trainVsAUC_3layer_"
    #titleSuffix = " for 3 layer"
    #lineGroup = [lineGroup[1], lineGroup[4], lineGroup[7]]

    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/trainVsAUC_4layer_"
    titleSuffix = " for 4 layer"
    lineGroup = [lineGroup[2], lineGroup[5], lineGroup[8]]

    #Labels of lines
    lineLabels = ["SparseUnsup", "DirectSup", "DirectFinetune"]
    colors = [
            (1, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
    ]
    fmt = [
            '-o',
            '-o',
            '-o',
    ]

    #Innermost list contains batch

    #Sparse unsup
    aucFiles = []

    aucFiles = [
            ["/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_100_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_100_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_100_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_100_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_100_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_100_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_100_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_100_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_100_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_100_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_100_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_100_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_100_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_100_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_100_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_100_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_100_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_100_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_1000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_1000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_1000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_1000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_1000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_1000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_1000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_1000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_1000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_1000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_1000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_1000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_1000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_1000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_1000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_1000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_1000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_1000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_2000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_2000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_2000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_2000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_2000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_2000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_2000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_2000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_2000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_2000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_2000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_2000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_2000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_2000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_2000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_2000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_2000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_2000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_4000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_4000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_4000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_4000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_4000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_4000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_4000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_4000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_4000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_4000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_4000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_4000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_4000_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_4000_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_4000_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_4000_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_4000_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_4000_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_all_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_all_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_all_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_all_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_all_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_1_sparse_unsup_all_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_all_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_all_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_all_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_all_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_all_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_2_sparse_unsup_all_run6/auc.txt",
            ],
            ["/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_all_run1/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_all_run2/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_all_run3/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_all_run4/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_all_run5/auc.txt",
             "/sparse_unsup/tfpv_kitti_vid_boot_3_sparse_unsup_all_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_1_direct_sup_100_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_100_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_100_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_100_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_100_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_100_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_2_direct_sup_100_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_100_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_100_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_100_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_100_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_100_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_3_direct_sup_100_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_100_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_100_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_100_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_100_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_100_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_1_direct_sup_1000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_1000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_1000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_1000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_1000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_1000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_2_direct_sup_1000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_1000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_1000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_1000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_1000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_1000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_3_direct_sup_1000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_1000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_1000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_1000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_1000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_1000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_1_direct_sup_2000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_2000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_2000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_2000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_2000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_2000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_2_direct_sup_2000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_2000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_2000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_2000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_2000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_2000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_3_direct_sup_2000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_2000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_2000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_2000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_2000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_2000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_1_direct_sup_4000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_4000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_4000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_4000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_4000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_4000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_2_direct_sup_4000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_4000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_4000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_4000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_4000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_4000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_3_direct_sup_4000_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_4000_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_4000_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_4000_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_4000_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_4000_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_1_direct_sup_all_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_all_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_all_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_all_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_all_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_1_direct_sup_all_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_2_direct_sup_all_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_all_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_all_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_all_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_all_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_2_direct_sup_all_run6/auc.txt",
            ],
            ["/direct_sup/sup_kitti_vid_boot_3_direct_sup_all_run1/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_all_run2/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_all_run3/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_all_run4/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_all_run5/auc.txt",
             "/direct_sup/sup_kitti_vid_boot_3_direct_sup_all_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_100_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_100_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_100_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_100_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_100_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_100_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_100_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_100_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_100_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_100_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_100_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_100_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_100_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_100_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_100_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_100_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_100_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_100_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_1000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_1000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_1000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_1000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_1000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_1000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_1000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_1000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_1000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_1000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_1000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_1000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_1000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_1000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_1000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_1000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_1000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_1000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_2000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_2000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_2000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_2000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_2000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_2000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_2000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_2000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_2000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_2000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_2000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_2000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_2000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_2000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_2000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_2000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_2000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_2000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_4000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_4000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_4000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_4000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_4000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_4000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_4000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_4000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_4000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_4000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_4000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_4000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_4000_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_4000_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_4000_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_4000_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_4000_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_4000_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_all_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_all_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_all_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_all_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_all_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_1_direct_finetune_all_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_all_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_all_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_all_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_all_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_all_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_2_direct_finetune_all_run6/auc.txt",
            ],
            ["/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_all_run1/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_all_run2/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_all_run3/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_all_run4/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_all_run5/auc.txt",
             "/direct_finetune/sup_kitti_vid_boot_3_direct_finetune_all_run6/auc.txt",
            ],
            ]

    #x values of points
    xPoints = [100, 1000, 2000, 4000, 6167]

    xlim = [0, 6500]
    ylim = [.3, .7]

    #Store auc
    auc = [[] for outer in aucFiles]

    #per plot loop
    for outerIdx, aucFile in enumerate(aucFiles):
        #Read file and grab auc
        for fn in aucFile:
            with open(baseDir + fn, 'r') as f:
                auc[outerIdx].append(float(f.readlines()[0]))

    #assert(len(xPoints) == len(auc[0]))
    #auc = np.array(auc)
    aucMed = np.array([np.median(a) for a in auc])
    aucMin = np.array([np.min(a) for a in auc])
    aucMax = np.array([np.max(a) for a in auc])
    #aucMean = np.mean(auc, axis=1)
    #aucStd = np.std(auc, axis=1)

    f = plt.figure()
    plt.hold(1)

    #Draw multple lines
    for i, lineIdx in enumerate(lineGroup):
        #yPoints = aucMean[np.array(lineIdx)]
        yPoints = aucMed[np.array(lineIdx)]
        yMax = aucMax[np.array(lineIdx)]
        yMin = aucMin[np.array(lineIdx)]

        label = lineLabels[i]
        color = colors[i]
        plt.plot(xPoints, yPoints, fmt[i], linewidth=2, label=label, color=color)
        if(showErr):
            plt.fill_between(xPoints, yMin, yMax, alpha=.2, facecolor=color)
        #plt.errorbar(xPoints, yPoints, linewidth=2, label=label, color=color, yerr = yErr, fmt=fmt[i], capsize=5)

    if(showChance):
        plt.errorbar(xlim, [meanRandChanceAuc, meanRandChanceAuc], yerr=[0, 0],
            fmt="k--", label="Chance", linewidth=2)

    plt.xlabel("Training Examples", fontsize=20)
    if(doPvr):
        ylabel = "Area under PvR"
    else:
        ylabel = "Area under ROC"
    title = "Training vs AUC"
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title + titleSuffix, fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlim(xlim)
    plt.ylim(ylim)


    lgd = plt.legend(bbox_to_anchor=(.38, .36), loc=2, borderaxespad=0., fontsize=20)
    if(doPvr):
        outName = outPrefix+'pvr.png'
    else:
        outName = outPrefix+'roc.png'

    plt.savefig(outName, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()



    #Do pvr plots?


