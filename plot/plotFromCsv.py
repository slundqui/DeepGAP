import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import exponential
import pdb

baseDir = "/home/slundquist/mountData/DeepGAP/"
outDir  = "/home/slundquist/mountData/DeepGAP/boot_plots/"

#baseDir = "/home/slundquist/mountData/DeepGAP/savedOutDirsKitti/"
#outDir  = "/home/slundquist/mountData/DeepGAP/savedOutDirsKitti/plots/"

leftFiles = [
        ("pv_kitti_vid_4x8_boot_1/csv/run_train,tag_accuracy.csv",
         "pv_kitti_vid_4x8_boot_1/csv/run_test,tag_accuracy.csv"),
        ("pv_kitti_vid_4x8_boot_1/csv/run_train,tag_Car F1.csv",
         "pv_kitti_vid_4x8_boot_1/csv/run_test,tag_Car F1.csv"),
        ("pv_kitti_vid_4x8_boot_1/csv/run_train,tag_Pedestrian F1.csv",
         "pv_kitti_vid_4x8_boot_1/csv/run_test,tag_Pedestrian F1.csv"),
        ("pv_kitti_vid_4x8_boot_1/csv/run_train,tag_distractor F1.csv",
         "pv_kitti_vid_4x8_boot_1/csv/run_test,tag_distractor F1.csv"),
        ("pv_kitti_vid_4x8_boot_2/csv/run_train,tag_accuracy.csv",
         "pv_kitti_vid_4x8_boot_2/csv/run_test,tag_accuracy.csv"),
        ("pv_kitti_vid_4x8_boot_2/csv/run_train,tag_Car F1.csv",
         "pv_kitti_vid_4x8_boot_2/csv/run_test,tag_Car F1.csv"),
        ("pv_kitti_vid_4x8_boot_2/csv/run_train,tag_Pedestrian F1.csv",
         "pv_kitti_vid_4x8_boot_2/csv/run_test,tag_Pedestrian F1.csv"),
        ("pv_kitti_vid_4x8_boot_2/csv/run_train,tag_distractor F1.csv",
         "pv_kitti_vid_4x8_boot_2/csv/run_test,tag_distractor F1.csv"),
        ("pv_kitti_vid_4x8_boot_3/csv/run_train,tag_accuracy.csv",
         "pv_kitti_vid_4x8_boot_3/csv/run_test,tag_accuracy.csv"),
        ("pv_kitti_vid_4x8_boot_3/csv/run_train,tag_Car F1.csv",
         "pv_kitti_vid_4x8_boot_3/csv/run_test,tag_Car F1.csv"),
        ("pv_kitti_vid_4x8_boot_3/csv/run_train,tag_Pedestrian F1.csv",
         "pv_kitti_vid_4x8_boot_3/csv/run_test,tag_Pedestrian F1.csv"),
        ("pv_kitti_vid_4x8_boot_3/csv/run_train,tag_distractor F1.csv",
         "pv_kitti_vid_4x8_boot_3/csv/run_test,tag_distractor F1.csv"),
        ]

rightFiles = [
        ("sup_kitti_vid_4x8_boot_1/csv/run_train,tag_accuracy.csv",
         "sup_kitti_vid_4x8_boot_1/csv/run_test,tag_accuracy.csv"),
        ("sup_kitti_vid_4x8_boot_1/csv/run_train,tag_Car F1.csv",
         "sup_kitti_vid_4x8_boot_1/csv/run_test,tag_Car F1.csv"),
        ("sup_kitti_vid_4x8_boot_1/csv/run_train,tag_Pedestrian F1.csv",
         "sup_kitti_vid_4x8_boot_1/csv/run_test,tag_Pedestrian F1.csv"),
        ("sup_kitti_vid_4x8_boot_1/csv/run_train,tag_distractor F1.csv",
         "sup_kitti_vid_4x8_boot_1/csv/run_test,tag_distractor F1.csv"),
        ("sup_kitti_vid_4x8_boot_2/csv/run_train,tag_accuracy.csv",
         "sup_kitti_vid_4x8_boot_2/csv/run_test,tag_accuracy.csv"),
        ("sup_kitti_vid_4x8_boot_2/csv/run_train,tag_Car F1.csv",
         "sup_kitti_vid_4x8_boot_2/csv/run_test,tag_Car F1.csv"),
        ("sup_kitti_vid_4x8_boot_2/csv/run_train,tag_Pedestrian F1.csv",
         "sup_kitti_vid_4x8_boot_2/csv/run_test,tag_Pedestrian F1.csv"),
        ("sup_kitti_vid_4x8_boot_2/csv/run_train,tag_distractor F1.csv",
         "sup_kitti_vid_4x8_boot_2/csv/run_test,tag_distractor F1.csv"),
        ("sup_kitti_vid_4x8_boot_3/csv/run_train,tag_accuracy.csv",
         "sup_kitti_vid_4x8_boot_3/csv/run_test,tag_accuracy.csv"),
        ("sup_kitti_vid_4x8_boot_3/csv/run_train,tag_Car F1.csv",
         "sup_kitti_vid_4x8_boot_3/csv/run_test,tag_Car F1.csv"),
        ("sup_kitti_vid_4x8_boot_3/csv/run_train,tag_Pedestrian F1.csv",
         "sup_kitti_vid_4x8_boot_3/csv/run_test,tag_Pedestrian F1.csv"),
        ("sup_kitti_vid_4x8_boot_3/csv/run_train,tag_distractor F1.csv",
         "sup_kitti_vid_4x8_boot_3/csv/run_test,tag_distractor F1.csv"),
        ]

#leftFiles = [
#        ("pv_kitti_vid_4x8_slp_noreg/csv/run_train,tag_accuracy.csv",
#         "pv_kitti_vid_4x8_slp_noreg/csv/run_test,tag_accuracy.csv"),
#        ("pv_kitti_vid_4x8_slp_noreg/csv/run_train,tag_Car F1.csv",
#         "pv_kitti_vid_4x8_slp_noreg/csv/run_test,tag_Car F1.csv"),
#        ("pv_kitti_vid_4x8_slp_noreg/csv/run_train,tag_Pedestrian F1.csv",
#         "pv_kitti_vid_4x8_slp_noreg/csv/run_test,tag_Pedestrian F1.csv"),
#        ("pv_kitti_vid_4x8_slp_noreg/csv/run_train,tag_distractor F1.csv",
#         "pv_kitti_vid_4x8_slp_noreg/csv/run_test,tag_distractor F1.csv"),
#        ("pv_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_accuracy.csv",
#         "pv_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_accuracy.csv"),
#        ("pv_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_Car F1.csv",
#         "pv_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_Car F1.csv"),
#        ("pv_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_Pedestrian F1.csv",
#         "pv_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_Pedestrian F1.csv"),
#        ("pv_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_distractor F1.csv",
#         "pv_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_distractor F1.csv"),
#        ("pv_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_accuracy.csv",
#         "pv_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_accuracy.csv"),
#        ("pv_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_Car F1.csv",
#         "pv_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_Car F1.csv"),
#        ("pv_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_Pedestrian F1.csv",
#         "pv_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_Pedestrian F1.csv"),
#        ("pv_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_distractor F1.csv",
#         "pv_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_distractor F1.csv"),
#        ]
#
#rightFiles = [
#        ("sup_kitti_vid_4x8_slp_noreg/csv/run_train,tag_accuracy.csv",
#         "sup_kitti_vid_4x8_slp_noreg/csv/run_test,tag_accuracy.csv"),
#        ("sup_kitti_vid_4x8_slp_noreg/csv/run_train,tag_Car F1.csv",
#         "sup_kitti_vid_4x8_slp_noreg/csv/run_test,tag_Car F1.csv"),
#        ("sup_kitti_vid_4x8_slp_noreg/csv/run_train,tag_Pedestrian F1.csv",
#         "sup_kitti_vid_4x8_slp_noreg/csv/run_test,tag_Pedestrian F1.csv"),
#        ("sup_kitti_vid_4x8_slp_noreg/csv/run_train,tag_distractor F1.csv",
#         "sup_kitti_vid_4x8_slp_noreg/csv/run_test,tag_distractor F1.csv"),
#        ("sup_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_accuracy.csv",
#         "sup_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_accuracy.csv"),
#        ("sup_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_Car F1.csv",
#         "sup_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_Car F1.csv"),
#        ("sup_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_Pedestrian F1.csv",
#         "sup_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_Pedestrian F1.csv"),
#        ("sup_kitti_vid_4x8_mlp_noreg/csv/run_train,tag_distractor F1.csv",
#         "sup_kitti_vid_4x8_mlp_noreg/csv/run_test,tag_distractor F1.csv"),
#        ("sup_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_accuracy.csv",
#         "sup_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_accuracy.csv"),
#        ("sup_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_Car F1.csv",
#         "sup_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_Car F1.csv"),
#        ("sup_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_Pedestrian F1.csv",
#         "sup_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_Pedestrian F1.csv"),
#        ("sup_kitti_vid_4x8_2xmlp_noreg/csv/run_train,tag_distractor F1.csv",
#         "sup_kitti_vid_4x8_2xmlp_noreg/csv/run_test,tag_distractor F1.csv"),
#        ]

leftLabel = "Sparse Coding"
rightLabel = "Supervised"

plotTitles = [
        "2 Layer Accuracy",
        "2 Layer Car F1",
        "2 Layer Pedestrian F1",
        "2 Layer Distractor F1",
        "3 Layer Accuracy",
        "3 Layer Car F1",
        "3 Layer Pedestrian F1",
        "3 Layer Distractor F1",
        "4 Layer Accuracy",
        "4 Layer Car F1",
        "4 Layer Pedestrian F1",
        "4 Layer Distractor F1",
        ]

def readCsv(filename):
    #First line is Wall time,Step,Value, so set comments as W
    vals = np.loadtxt(filename, delimiter=',', comments='W')
    #Remove wall time
    return vals[:, 1:]

def pruneTime(*arg):
    maxTimes = []
    for a in arg:
        maxTimes.append(np.max(a[:, 0]))
    maxTime = np.min(maxTimes)
    out = []
    for a in arg:
        out.append(a[np.nonzero(a[:, 0] <= maxTime)[0], :])
    return out

tau = 1000
def expSmooth(*arg):
    out = []
    for a in arg:
        x = a[:, 0]
        y = a[:, 1]
        numData = x.shape[0]
        #Find dt
        dt = x[2]-x[1]
        alpha = float(dt)/tau
        expVal = np.zeros(a.shape)
        expVal[:, 0] = x
        expVal[0, 1] = y[0]
        for t in range(1, numData):
            expVal[t, 1] = alpha * y[t] + (1-alpha) * expVal[t-1, 1]
        out.append(expVal)
    return out

def plotSubfig(ax, actualTrain, actualTest, smoothTrain, smoothTest, label):
    ax.plot(actualTrain[:, 0], actualTrain[:, 1], color=[.8, .8, 1])
    ax.plot(actualTest[:, 0], actualTest[:, 1], color=[1, .8, .8])
    ax.plot(smoothTrain[:, 0], smoothTrain[:, 1], color=[0, 0, 1], linewidth=2)
    ax.plot(smoothTest[:, 0], smoothTest[:, 1], color=[1, 0, 0], linewidth=2)
    ax.set_title(label, fontsize=16)
    ax.set_ylim((0, 1))
    ax.set_ylabel("Percent", fontsize=14)
    ax.set_xlabel("Timestep", fontsize=14)
    ax.grid(True)


for i, (leftCsv, rightCsv) in enumerate(zip(leftFiles, rightFiles)):
    print plotTitles[i]
    leftTrainCsvFile  = baseDir + leftCsv[0]
    leftTestCsvFile   = baseDir + leftCsv[1]
    rightTrainCsvFile = baseDir + rightCsv[0]
    rightTestCsvFile  = baseDir + rightCsv[1]

    leftTrain = readCsv(leftTrainCsvFile)
    leftTest = readCsv(leftTestCsvFile)
    rightTrain = readCsv(rightTrainCsvFile)
    rightTest = readCsv(rightTestCsvFile)

    (leftTrainPrune, leftTestPrune, rightTrainPrune, rightTestPrune) = pruneTime(leftTrain, leftTest, rightTrain, rightTest)
    (leftTrainSmooth, leftTestSmooth, rightTrainSmooth, rightTestSmooth) = expSmooth(leftTrainPrune, leftTestPrune, rightTrainPrune, rightTestPrune)

    f, axarr = plt.subplots(1, 2, figsize=(16, 8))
    plotSubfig(axarr[0], leftTrainPrune, leftTestPrune, leftTrainSmooth, leftTestSmooth, leftLabel)
    plotSubfig(axarr[1], rightTrainPrune, rightTestPrune, rightTrainSmooth, rightTestSmooth, rightLabel)

    st = plt.suptitle(plotTitles[i], fontsize=20)

    rects = [matplotlib.patches.Rectangle((0, 0), 2, 2, fc='b'),
             matplotlib.patches.Rectangle((0, 0), 2, 2, fc='r'),
            ]
    labels= ["Train", "Test"]
    lgd = plt.legend(rects, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(outDir + plotTitles[i] + ".png", bbox_extra_artists=(lgd, st,), bbox_inches='tight')

    plt.close("all")







