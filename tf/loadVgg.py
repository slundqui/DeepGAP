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

    #conv1
    outdict["conv1_1_w"] = convertW(m["layers"][0, 0][0, 0][2][0, 0])
    outdict["conv1_1_b"] = convertB(m["layers"][0, 0][0, 0][2][0, 1])
    outdict["conv1_2_w"] = convertW(m["layers"][0, 2][0, 0][2][0, 0])
    outdict["conv1_2_b"] = convertB(m["layers"][0, 2][0, 0][2][0, 1])

    #conv2
    outdict["conv2_1_w"] = convertW(m["layers"][0, 5][0, 0][2][0, 0])
    outdict["conv2_1_b"] = convertB(m["layers"][0, 5][0, 0][2][0, 1])
    outdict["conv2_2_w"] = convertW(m["layers"][0, 7][0, 0][2][0, 0])
    outdict["conv2_2_b"] = convertB(m["layers"][0, 7][0, 0][2][0, 1])

    #conv3
    outdict["conv3_1_w"] = convertW(m["layers"][0, 10][0, 0][2][0, 0])
    outdict["conv3_1_b"] = convertB(m["layers"][0, 10][0, 0][2][0, 1])
    outdict["conv3_2_w"] = convertW(m["layers"][0, 12][0, 0][2][0, 0])
    outdict["conv3_2_b"] = convertB(m["layers"][0, 12][0, 0][2][0, 1])
    outdict["conv3_3_w"] = convertW(m["layers"][0, 14][0, 0][2][0, 0])
    outdict["conv3_3_b"] = convertB(m["layers"][0, 14][0, 0][2][0, 1])

    #conv4
    outdict["conv4_1_w"] = convertW(m["layers"][0, 17][0, 0][2][0, 0])
    outdict["conv4_1_b"] = convertB(m["layers"][0, 17][0, 0][2][0, 1])
    outdict["conv4_2_w"] = convertW(m["layers"][0, 19][0, 0][2][0, 0])
    outdict["conv4_2_b"] = convertB(m["layers"][0, 19][0, 0][2][0, 1])
    outdict["conv4_3_w"] = convertW(m["layers"][0, 21][0, 0][2][0, 0])
    outdict["conv4_3_b"] = convertB(m["layers"][0, 21][0, 0][2][0, 1])

    #conv5
    outdict["conv5_1_w"] = convertW(m["layers"][0, 24][0, 0][2][0, 0])
    outdict["conv5_1_b"] = convertB(m["layers"][0, 24][0, 0][2][0, 1])
    outdict["conv5_2_w"] = convertW(m["layers"][0, 26][0, 0][2][0, 0])
    outdict["conv5_2_b"] = convertB(m["layers"][0, 26][0, 0][2][0, 1])
    outdict["conv5_3_w"] = convertW(m["layers"][0, 28][0, 0][2][0, 0])
    outdict["conv5_3_b"] = convertB(m["layers"][0, 28][0, 0][2][0, 1])

    #FC
    outdict["fc6_w"] = convertW(m["layers"][0, 31][0, 0][2][0, 0]).reshape(7*7*512, 4096)
    outdict["fc6_b"] = convertB(m["layers"][0, 31][0, 0][2][0, 1])
    outdict["fc7_w"] = convertW(m["layers"][0, 33][0, 0][2][0, 0])[0, 0, :, :]
    outdict["fc7_b"] = convertB(m["layers"][0, 33][0, 0][2][0, 1])
    outdict["fc8_w"] = convertW(m["layers"][0, 35][0, 0][2][0, 0])[0, 0, :, :]
    outdict["fc8_b"] = convertB(m["layers"][0, 35][0, 0][2][0, 1])

    return outdict


if __name__ == "__main__":
    inputFile = "/home/slundquist/mountData/pretrain/imagenet-vgg-verydeep-16.mat"
    outdict = loadWeights(inputFile)
