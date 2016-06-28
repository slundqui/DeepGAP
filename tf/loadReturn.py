import numpy as np
from scipy.io import loadmat
import pdb
#import matplotlib.pyplot as plt

def convertW(inW):
    #inW is in the shape of [W, H, inF, outF]
    #We need to translate to [H, W, inF, outF]
    return np.transpose(inW, (1, 0, 2, 3))

def convertB(inB):
    return inB[:, 0]

def loadWeights(inFile):
    m = loadmat(inFile)

    outdict = {}

    outdict["conv1_w"] = convertW(m["layers"][0, 0][0, 0][2][0, 0])
    outdict["conv1_b"] = convertB(m["layers"][0, 0][0, 0][2][0, 1])
    outdict["conv2_w"] = convertW(m["layers"][0, 4][0, 0][2][0, 0])
    outdict["conv2_b"] = convertB(m["layers"][0, 4][0, 0][2][0, 1])
    outdict["conv3_w"] = convertW(m["layers"][0, 8][0, 0][2][0, 0])
    outdict["conv3_b"] = convertB(m["layers"][0, 8][0, 0][2][0, 1])
    outdict["conv4_w"] = convertW(m["layers"][0, 10][0, 0][2][0, 0])
    outdict["conv4_b"] = convertB(m["layers"][0, 10][0, 0][2][0, 1])
    outdict["conv5_w"] = convertW(m["layers"][0, 12][0, 0][2][0, 0])
    outdict["conv5_b"] = convertB(m["layers"][0, 12][0, 0][2][0, 1])
    outdict["fc1_w"] = convertW(m["layers"][0, 15][0, 0][2][0, 0]).reshape(6*6*256, 4096)
    outdict["fc1_b"] = convertB(m["layers"][0, 15][0, 0][2][0, 1])
    outdict["fc2_w"] = convertW(m["layers"][0, 17][0, 0][2][0, 0])[0, 0, :, :]
    outdict["fc2_b"] = convertB(m["layers"][0, 17][0, 0][2][0, 1])
    outdict["fc3_w"] = convertW(m["layers"][0, 19][0, 0][2][0, 0])[0, 0, :, :]
    outdict["fc3_b"] = convertB(m["layers"][0, 19][0, 0][2][0, 1])

    return outdict


if __name__ == "__main__":
    inputFile = "/home/sheng/mountData/pretrain/imagenet-vgg-f.mat"
    outdict = loadWeights(inputFile)
