import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    baseDir = "/home/slundquist/mountData/DeepGAP/"

    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/tfpv_input_"
    titleSuffix = " for 3 layer"

    estFiles = [
            #"eval_tfpv_kitti_vid_4x8_boot_1_bin/evalEstIdxs.npy",
            "eval_tfpv_kitti_vid_4x8_boot_2_bin/evalEstIdxs.npy",
            "eval_sup_kitti_vid_4x8_boot_2_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_4x8_boot_3_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_stereo_4x8_boot_1_bin/evalEstIdxs.npy",
            "eval_tfpv_kitti_vid_stereo_4x8_boot_2_bin/evalEstIdxs.npy",
            "eval_sup_kitti_vid_stereo_4x8_boot_2_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_stereo_4x8_boot_3_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_time_4x8_boot_1_bin/evalEstIdxs.npy",
            "eval_tfpv_kitti_vid_time_4x8_boot_2_bin/evalEstIdxs.npy",
            "eval_sup_kitti_vid_time_4x8_boot_2_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_time_4x8_boot_3_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_single_4x8_boot_1_bin/evalEstIdxs.npy",
            "eval_tfpv_kitti_vid_single_4x8_boot_2_bin/evalEstIdxs.npy",
            "eval_sup_kitti_vid_single_4x8_boot_2_bin/evalEstIdxs.npy",
            #"eval_tfpv_kitti_vid_single_4x8_boot_3_bin/evalEstIdxs.npy",
            ]


    gtFiles = [
            #"eval_tfpv_kitti_vid_4x8_boot_1_bin/evalGtIdxs.npy",
            "eval_tfpv_kitti_vid_4x8_boot_2_bin/evalGtIdxs.npy",
            "eval_sup_kitti_vid_4x8_boot_2_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_4x8_boot_3_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_stereo_4x8_boot_1_bin/evalGtIdxs.npy",
            "eval_tfpv_kitti_vid_stereo_4x8_boot_2_bin/evalGtIdxs.npy",
            "eval_sup_kitti_vid_stereo_4x8_boot_2_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_stereo_4x8_boot_3_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_time_4x8_boot_1_bin/evalGtIdxs.npy",
            "eval_tfpv_kitti_vid_time_4x8_boot_2_bin/evalGtIdxs.npy",
            "eval_sup_kitti_vid_time_4x8_boot_2_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_time_4x8_boot_3_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_single_4x8_boot_1_bin/evalGtIdxs.npy",
            "eval_tfpv_kitti_vid_single_4x8_boot_2_bin/evalGtIdxs.npy",
            "eval_sup_kitti_vid_single_4x8_boot_2_bin/evalGtIdxs.npy",
            #"eval_tfpv_kitti_vid_single_4x8_boot_3_bin/evalGtIdxs.npy",
            ]

    #How to group bars in bar plot based on idx
    #barGroup = [(0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11)]
    barGroup = [(0, 2, 4, 6), (1, 3, 5, 7)]
    barLabels = ["Sparse", "Sup"]
    innerBarLabels = ["All", "Stereo", "Time", "Single"]
    width = 0.20       # the width of the bars

    labels = [
            #"All 2 layer",
            "Sparse All 3 layer",
            "Sup All 3 layer",
            #"All 4 layer",
            #"Stereo 2 layer",
            "Sparse Stereo 3 layer",
            "Sup Stereo 3 layer",
            #"Stereo 4 layer",
            #"Time 2 layer",
            "Sparse Time 3 layer",
            "Sup Time 3 layer",
            #"Time 4 layer",
            #"Single 2 layer",
            "Sparse Single 3 layer",
            "Sup Single 3 layer",
            #"Single 4 layer",
            ]

    colors = [
            (1, .8, .8),
            #(1, .4, .4),
            (1, 0, 0),
            (.8, 1, .8),
            #(.4, 1, .4),
            (0, 1, 0),
            (.8, .8, 1),
            #(.4, .4, 1),
            (0, 0, 1),
            (.9, .9, .9),
            #(.4, .4, .4),
            (0, 0, 0),
    ]
    barColors = [colors[1], colors[3], colors[5], colors[6]]
    #barColors = [colors[2], colors[5], colors[8], colors[9]]



#    outPrefix = "/home/slundquist/mountData/DeepGAP/evalplots/tfpv_models_"
#
#    estFiles = [
#            "eval_rand_kitti_vid_4x8_boot_1_bin/evalEstIdxs.npy",
#            "eval_rand_kitti_vid_4x8_boot_2_bin/evalEstIdxs.npy",
#            "eval_rand_kitti_vid_4x8_boot_3_bin/evalEstIdxs.npy",
#            "eval_tfpv_kitti_vid_4x8_boot_1_bin/evalEstIdxs.npy",
#            "eval_tfpv_kitti_vid_4x8_boot_2_bin/evalEstIdxs.npy",
#            "eval_tfpv_kitti_vid_4x8_boot_3_bin/evalEstIdxs.npy",
#            "eval_sup_kitti_vid_4x8_boot_1_bin/evalEstIdxs.npy",
#            "eval_sup_kitti_vid_4x8_boot_2_bin/evalEstIdxs.npy",
#            "eval_sup_kitti_vid_4x8_boot_3_bin/evalEstIdxs.npy",
#            ]
#
#    gtFiles = [
#            "eval_rand_kitti_vid_4x8_boot_1_bin/evalGtIdxs.npy",
#            "eval_rand_kitti_vid_4x8_boot_2_bin/evalGtIdxs.npy",
#            "eval_rand_kitti_vid_4x8_boot_3_bin/evalGtIdxs.npy",
#            "eval_tfpv_kitti_vid_4x8_boot_1_bin/evalGtIdxs.npy",
#            "eval_tfpv_kitti_vid_4x8_boot_2_bin/evalGtIdxs.npy",
#            "eval_tfpv_kitti_vid_4x8_boot_3_bin/evalGtIdxs.npy",
#            "eval_sup_kitti_vid_4x8_boot_1_bin/evalGtIdxs.npy",
#            "eval_sup_kitti_vid_4x8_boot_2_bin/evalGtIdxs.npy",
#            "eval_sup_kitti_vid_4x8_boot_3_bin/evalGtIdxs.npy",
#            ]
#
#    barGroup = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
#    barLabels = ["2 layer", "3 layer", "4 layer"]
#    innerBarLabels = ["Rand", "Sparse", "Sup"]
#    width = 0.25       # the width of the bars
#    labels = [
#            "Rand 2 layer",
#            "Rand 3 layer",
#            "Rand 4 layer",
#            "Sparse 2 layer",
#            "Sparse 3 layer",
#            "Sparse 4 layer",
#            "Sup 2 layer",
#            "Sup 3 layer",
#            "Sup 4 layer",
#            ]
#
#    colors = [
#            (.9, .9, .9),
#            (.4, .4, .4),
#            (0, 0, 0),
#            (1, .8, .8),
#            (1, .4, .4),
#            (1, 0, 0),
#            (.8, .8, 1),
#            (.4, .4, 1),
#            (0, 0, 1),
#    ]
#    barColors = [colors[0], colors[5], colors[8]]

    num = 50

    #gtFile = baseDir + "/fcnnOut/gt.pkl.npy"

    plt.figure(1)

    #Store auc
    auc = np.zeros((len(estFiles)))
    #per plot loop
    for i, (estF, gtF, label, c) in enumerate(zip(estFiles, gtFiles, labels, colors)):
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
            auc[i] += (height*(precision[j-1]+precision[j]))/2

    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.title("Precision Vs Recall" + titleSuffix, fontsize = 30)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #lgd = plt.legend(loc=0)
    plt.savefig(outPrefix+'pvr.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

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
        barVals = [auc[index[i]] for index in barGroup]
        colorVals = [colors[index[i]] for index in barGroup]
        xLoc = ind+(width*i)
        rects[i] = ax.bar(xLoc, barVals, width, color=barColors[i])
        #Add number label to top
        for x, y in zip(xLoc, barVals):
            plt.text(x+(width/2), y+.01, "{0:.2f}".format(y),
                    {'ha': 'center', 'va': 'bottom'}, rotation=90)

    ax.set_ylim([0, .8])
    ax.set_ylabel('AUC')
    ax.set_title('Area under PvR curve' + titleSuffix)
    ax.set_xticks(ind + (float(numInner)*width)/2)
    # add some text for labels, title and axes ticks
    ax.set_xticklabels(barLabels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    lgd = ax.legend([r[0] for r in rects], innerBarLabels, loc='center left', bbox_to_anchor=(1, .5))
    plt.savefig(outPrefix+"auc.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.savefig('auc_all.png', bbox_extra_artists=(lgd,f))
    plt.clf()
