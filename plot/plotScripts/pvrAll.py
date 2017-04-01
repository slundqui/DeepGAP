import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    baseDir = "/home/slundquist/mountData/DeepGAP/"

    estFiles = [
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
    barGroup = [(0,3), (1,4), (2,5)]
    barLabels = ["2 layer", "3 layer", "4 layer"]
    innerBarLabels = ["Sparse", "Sup"]
    width = 0.25       # the width of the bars
    labels = [
            "Sparse 2 layer",
            "Sparse 3 layer",
            "Sparse 4 layer",
            "Sup 2 layer",
            "Sup 3 layer",
            "Sup 4 layer",
            ]
    colors = [
            #(.8, .8, .8),
            #(.4, .4, .4),
            #(0, 0, 0),
            (1, .8, .8),
            (1, .4, .4),
            (1, 0, 0),
            (.8, .8, 1),
            (.4, .4, 1),
            (0, 0, 1),
    ]
    #barColors = [colors[1], colors[5], colors[8]]
    barColors = [colors[2], colors[5]]


    num = 50

    #gtFile = baseDir + "/fcnnOut/gt.pkl.npy"

    plt.figure(1)

    #Store auc
    auc = [[0 for inner in outer] for outer in estFiles]
    #per plot loop
    for outerIdx, (estFList, gtFList, label, c) in enumerate(zip(estFiles, gtFiles, labels, colors)):
        if(type(estFList) is not list):
            estFList = [estFList]
        if(type(gtFList) is not list):
            gtFList = [gtFList]


        for innerIdx, (estF, gtF) in enumerate(zip(estFList, gtFList)):
            fn = baseDir + estF
            gtfn = baseDir + gtF
            est = np.load(fn)
            gt = np.load(gtfn)
            numExample = est.shape
            estMin = est.min()
            estMax = est.max()

            thresh = np.linspace(estMin-1e-6, estMax+1e-6, num=num)
            precision = np.zeros((num))
            recall = np.zeros((num))

            #We threshold the estimate to make a binary matrix
            for j,t in enumerate(thresh):
                t_est = np.zeros((numExample))
                t_est[np.nonzero(est >= t)] = 1

                #We need tp, fn, and fp
                tp = float(len(np.nonzero(np.logical_and(gt == 1, t_est == 1))[0]))
                fn = float(len(np.nonzero(np.logical_and(gt == 1, t_est == 0))[0]))
                fp = float(len(np.nonzero(np.logical_and(gt == 0, t_est == 1))[0]))

                if(tp+fp == 0):
                    #Precision is defined as 1 here
                    precision[j] = 1
                else:
                    precision[j] = tp/(tp+fp)

                if(tp+fn == 0):
                    pdb.set_trace()

                recall[j] = tp/(tp+fn)

            plt.plot(recall, precision, linewidth=4, label=label, color=c)

            #Calculate auc
            for j in range(1, len(recall)):
                #Calculate area of trap
                height = abs(recall[j]-recall[j-1])
                auc[outerIdx][innerIdx] += (height*(precision[j-1]+precision[j]))/2


    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title("Precision Vs Recall", fontsize = 30)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #lgd = plt.legend(loc=0)
    plt.savefig('pvr_all.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
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
    for i in range(numInner):
        #Loop thorugh outer bars
        try:
            barVals = [aucMean[index[i]] for index in barGroup]
        except:
            pdb.set_trace()
        colorVals = [colors[index[i]] for index in barGroup]
        errVals = [aucStd[index[i]] for index in barGroup]
        rects[i] = ax.bar(ind+(width*i), barVals, width, color=barColors[i], yerr=errVals,
                error_kw=dict(lw=5, capsize=5, capthick=3, color='k'))

    ax.set_ylabel('AUC')
    ax.set_title('Area under PvR curve')
    ax.set_xticks(ind + (float(numInner)*width)/2)
    # add some text for labels, title and axes ticks
    ax.set_xticklabels(barLabels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.ylim(0, .8)

    lgd = ax.legend([r[0] for r in rects], innerBarLabels, loc='center left', bbox_to_anchor=(1, .5))
    plt.savefig('auc_all.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.savefig('auc_all.png', bbox_extra_artists=(lgd,f))
    plt.clf()
