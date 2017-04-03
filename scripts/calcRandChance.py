import numpy as np

gtFile = "/home/slundquist/mountData/DeepGAP/eval_rand_kitti_vid_4x8_boot_1_bin_run1/evalGtIdxs.npy"

outDir = "/home/slundquist/mountData/DeepGAP/eval_randchance_kitti_vid_4x8/"

gt = np.load(gtFile)
np.save(outDir+"/evalGtIdxs.npy", gt)

gtShape = gt.shape

numRandTrials = 10

for trainIdx in range(numRandTrials):
    randEst = np.random.uniform(gtShape)
    #Save est
    np.save(outDir+"/evalEstIdxs_run"+str(trainIdx)+".npy", randEst)
