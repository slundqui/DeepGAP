import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import exponential
import pdb

baseDir = "/home/slundquist/mountData/DeepGAP/"
outDir  = "/home/slundquist/mountData/DeepGAP/boot_plots_new/"

#baseDir = "/home/slundquist/mountData/DeepGAP/savedOutDirsKitti/"
#outDir  = "/home/slundquist/mountData/DeepGAP/savedOutDirsKitti/plots/"

#List of list of tuples
#First list number of plots per figure
#Second list individual figures per element
#Tuple is (train, test)

#labels = ["All", "Stereo", "Time", "Single"]
#files = [
#        [("tfpv_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
#        ],
#        [("tfpv_kitti_vid_stereo_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_stereo_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_stereo_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
#        ],
#        [("tfpv_kitti_vid_time_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_time_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_time_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_time_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_time_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_time_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_time_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_time_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_time_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_time_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_time_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
#        ],
#        [("tfpv_kitti_vid_single_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_single_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_single_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_single_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_single_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_single_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_single_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
#          "tfpv_kitti_vid_single_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
#          "tfpv_kitti_vid_single_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
#         ("tfpv_kitti_vid_single_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
#          "tfpv_kitti_vid_single_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
#        ],
#]

labels = ["Random", "Sparse Coding", "Supervised"]
files = [
        [("rand_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
          "rand_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
         ("rand_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
          "rand_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
         ("rand_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
          "rand_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
         ("rand_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
          "rand_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
         ("rand_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
          "rand_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
         ("rand_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
          "rand_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
         ("rand_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
          "rand_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
         ("rand_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
          "rand_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
         ("rand_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
          "rand_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
         ],

        [("tfpv_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
         "tfpv_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
        ("tfpv_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
         "tfpv_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
        ("tfpv_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
         "tfpv_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
        ("tfpv_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
         "tfpv_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
        ("tfpv_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
         "tfpv_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
        ("tfpv_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
         "tfpv_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
        ("tfpv_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
         "tfpv_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
        ("tfpv_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
         "tfpv_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
        ("tfpv_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
         "tfpv_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
        ],
        [("sup_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_accuracy.csv",
         "sup_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_accuracy.csv"),
        ("sup_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_Car F1.csv",
         "sup_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_Car F1.csv"),
        ("sup_kitti_vid_4x8_boot_1_bin/csv/run_train,tag_distractor F1.csv",
         "sup_kitti_vid_4x8_boot_1_bin/csv/run_test,tag_distractor F1.csv"),
        ("sup_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_accuracy.csv",
         "sup_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_accuracy.csv"),
        ("sup_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_Car F1.csv",
         "sup_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_Car F1.csv"),
        ("sup_kitti_vid_4x8_boot_2_bin/csv/run_train,tag_distractor F1.csv",
         "sup_kitti_vid_4x8_boot_2_bin/csv/run_test,tag_distractor F1.csv"),
        ("sup_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_accuracy.csv",
         "sup_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_accuracy.csv"),
        ("sup_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_Car F1.csv",
         "sup_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_Car F1.csv"),
        ("sup_kitti_vid_4x8_boot_3_bin/csv/run_train,tag_distractor F1.csv",
         "sup_kitti_vid_4x8_boot_3_bin/csv/run_test,tag_distractor F1.csv"),
        ]
    ]



legend_labels= ["Train", "Test"]

plotTitles = [
        "2 Layer Accuracy",
        "2 Layer Car F1",
        "2 Layer Distractor F1",
        "3 Layer Accuracy",
        "3 Layer Car F1",
        "3 Layer Distractor F1",
        "4 Layer Accuracy",
        "4 Layer Car F1",
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


numSubplots = len(files)
numFiles = len(files[0])

for i in range(numFiles):
    print plotTitles[i]
    allVals = []
    for s in range(numSubplots):
        trainCsvFile  = baseDir + files[s][i][0]
        testCsvFile   = baseDir + files[s][i][1]
        trainVals = readCsv(trainCsvFile)
        testVals  = readCsv(testCsvFile)
        allVals.extend([trainVals, testVals])

    pruneVals = pruneTime(*allVals)
    #(leftTrainPrune, leftTestPrune, rightTrainPrune, rightTestPrune) = pruneTime(leftTrain, leftTest, rightTrain, rightTest)
    smoothVals = expSmooth(*pruneVals)
    #(leftTrainSmooth, leftTestSmooth, rightTrainSmooth, rightTestSmooth) = expSmooth(leftTrainPrune, leftTestPrune, rightTrainPrune, rightTestPrune)

    f, axarr = plt.subplots(1, numSubplots, figsize=(16, 8))
    for s in range(numSubplots):
        plotSubfig(axarr[s], pruneVals[s*2], pruneVals[s*2+1], smoothVals[s*2], smoothVals[s*2+1], labels[s])

    st = plt.suptitle(plotTitles[i], fontsize=20)

    rects = [matplotlib.patches.Rectangle((0, 0), 2, 2, fc='b'),
             matplotlib.patches.Rectangle((0, 0), 2, 2, fc='r'),
            ]

    lgd = plt.legend(rects, legend_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(outDir + plotTitles[i] + ".png", bbox_extra_artists=(lgd, st,), bbox_inches='tight')

    plt.close("all")







