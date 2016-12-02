import numpy as np
import matplotlib.pyplot as plt

baseDir = "/home/slundquist/mountData/DeepGAP"
#Outer list contains models
#Inner list contains different runs
evalDirs = [
        [
            "eval_pv_kitti_vid_4x8_boot_1",
            "eval_pv_kitti_vid_4x8_boot_2",
            "eval_pv_kitti_vid_4x8_boot_3",
        ],
        [
            "eval_sup_kitti_vid_4x8_boot_1",
            "eval_sup_kitti_vid_4x8_boot_2",
            "eval_sup_kitti_vid_4x8_boot_3",
        ],
]

numModels = len(evalDirs)
numRuns = len(evalDirs[0])

