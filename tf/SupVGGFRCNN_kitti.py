import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.plotBBPvR import plotBBPvRBatch
from base import TFObj
import scipy.sparse as sp
from tensorflow.python.ops import control_flow_ops
from SupFRCNN_kitti import FRCNN
import sys
#import matplotlib.pyplot as plt

class VGG_FRCNN(FRCNN):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        #numConvLayers not used here
        params['numConvLayers'] = None
        super(VGG_FRCNN, self).loadParams(params)
        self.vggFile = params['vggFile']

    def defineEncodingVars(self):
        if(self.vggFile):
            npWeights = loadWeights(self.vggFile)
        else:
            print "Must load from weights"
            assert(0)

        self.W_conv1_1 = weight_variable_fromnp(npWeights["conv1_1_w"], "w_conv1_1")
        self.B_conv1_1 = weight_variable_fromnp(npWeights["conv1_1_b"], "b_conv1_1")
        self.W_conv1_2 = weight_variable_fromnp(npWeights["conv1_2_w"], "w_conv1_2")
        self.B_conv1_2 = weight_variable_fromnp(npWeights["conv1_2_b"], "b_conv1_2")

        self.W_conv2_1 = weight_variable_fromnp(npWeights["conv2_1_w"], "w_conv2_1")
        self.B_conv2_1 = weight_variable_fromnp(npWeights["conv2_1_b"], "b_conv2_1")
        self.W_conv2_2 = weight_variable_fromnp(npWeights["conv2_2_w"], "w_conv2_2")
        self.B_conv2_2 = weight_variable_fromnp(npWeights["conv2_2_b"], "b_conv2_2")

        self.W_conv3_1 = weight_variable_fromnp(npWeights["conv3_1_w"], "w_conv3_1")
        self.B_conv3_1 = weight_variable_fromnp(npWeights["conv3_1_b"], "b_conv3_1")
        self.W_conv3_2 = weight_variable_fromnp(npWeights["conv3_2_w"], "w_conv3_2")
        self.B_conv3_2 = weight_variable_fromnp(npWeights["conv3_2_b"], "b_conv3_2")
        self.W_conv3_3 = weight_variable_fromnp(npWeights["conv3_3_w"], "w_conv3_3")
        self.B_conv3_3 = weight_variable_fromnp(npWeights["conv3_3_b"], "b_conv3_3")

        self.W_conv4_1 = weight_variable_fromnp(npWeights["conv4_1_w"], "w_conv4_1")
        self.B_conv4_1 = weight_variable_fromnp(npWeights["conv4_1_b"], "b_conv4_1")
        self.W_conv4_2 = weight_variable_fromnp(npWeights["conv4_2_w"], "w_conv4_2")
        self.B_conv4_2 = weight_variable_fromnp(npWeights["conv4_2_b"], "b_conv4_2")
        self.W_conv4_3 = weight_variable_fromnp(npWeights["conv4_3_w"], "w_conv4_3")
        self.B_conv4_3 = weight_variable_fromnp(npWeights["conv4_3_b"], "b_conv4_3")

        self.W_conv5_1 = weight_variable_fromnp(npWeights["conv5_1_w"], "w_conv5_1")
        self.B_conv5_1 = weight_variable_fromnp(npWeights["conv5_1_b"], "b_conv5_1")
        self.W_conv5_2 = weight_variable_fromnp(npWeights["conv5_2_w"], "w_conv5_2")
        self.B_conv5_2 = weight_variable_fromnp(npWeights["conv5_2_b"], "b_conv5_2")
        self.W_conv5_3 = weight_variable_fromnp(npWeights["conv5_3_w"], "w_conv5_3")
        self.B_conv5_3 = weight_variable_fromnp(npWeights["conv5_3_b"], "b_conv5_3")

        #These are class vars, but defined here to load from npWeights
        self.class_fc_1_weight = weight_variable_fromnp(npWeights['fc6_w'], "class_fc_1_weight")
        self.class_fc_1_bias   = weight_variable_fromnp(npWeights['fc6_b'], "class_fc_1_bias")
        self.class_fc_2_weight = weight_variable_fromnp(npWeights['fc7_w'], "class_fc_2_weight")
        self.class_fc_2_bias   = weight_variable_fromnp(npWeights['fc7_b'], "class_fc_2_bias")

    def defineClassVars(self):
        self.class_obj_weight = weight_variable_xavier([4096, 2], "class_obj_weight")
        self.class_obj_bias = bias_variable([2], "class_obj_bias")
        self.class_reg_weight = weight_variable_xavier([4096, 4], "class_reg_weight")
        self.class_reg_bias = bias_variable([4], "class_reg_bias")

    def defineEncoding(self):
        with tf.name_scope("inputOps"):
            self.singleImage = self.inputImage[:, 4, :, :, :]

        with tf.name_scope("Conv1Ops"):
            self.h_conv1_1 = tf.nn.relu(conv2d(self.singleImage, self.W_conv1_1, "conv1_1", stride=[1, 1, 1, 1]) + self.B_conv1_1)
            self.h_conv1_2 = tf.nn.relu(conv2d(self.h_conv1_1, self.W_conv1_2, "conv1_1", stride=[1, 1, 1, 1]) + self.B_conv1_2)
            self.h_pool1 = maxpool_2x2(self.h_conv1_2, "pool1")

        with tf.name_scope("Conv2Ops"):
            self.h_conv2_1 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2_1, "conv2_1") + self.B_conv2_1)
            self.h_conv2_2 = tf.nn.relu(conv2d(self.h_conv2_1, self.W_conv2_2, "conv2_2") + self.B_conv2_2)
            self.h_pool2 = maxpool_2x2(self.h_conv2_2, "pool2")

        with tf.name_scope("Conv3Ops"):
            self.h_conv3_1 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3_1, "conv3_1") + self.B_conv3_1)
            self.h_conv3_2 = tf.nn.relu(conv2d(self.h_conv3_1, self.W_conv3_2, "conv3_2") + self.B_conv3_2)
            self.h_conv3_3 = tf.nn.relu(conv2d(self.h_conv3_2, self.W_conv3_3, "conv3_2") + self.B_conv3_3)
            #self.h_pool3 = maxpool_2x2(self.h_conv3_3, "pool3")

        with tf.name_scope("Conv4Ops"):
            #self.h_conv4_1 = tf.nn.relu(conv2d(self.h_pool3, self.W_conv4_1, "conv4_1") + self.B_conv4_1)
            self.h_conv4_1 = tf.nn.relu(conv2d(self.h_conv3_3, self.W_conv4_1, "conv4_1") + self.B_conv4_1)
            self.h_conv4_2 = tf.nn.relu(conv2d(self.h_conv4_1, self.W_conv4_2, "conv4_2") + self.B_conv4_2)
            self.h_conv4_3 = tf.nn.relu(conv2d(self.h_conv4_2, self.W_conv4_3, "conv4_2") + self.B_conv4_3)
            #self.h_pool4 = maxpool_2x2(self.h_conv4_3, "pool4")


        with tf.name_scope("Conv5Ops"):

            #self.h_conv5_1 = tf.nn.relu(conv2d(self.h_pool4, self.W_conv5_1, "conv5_1") + self.B_conv5_1)
            self.h_conv5_1 = tf.nn.relu(conv2d(self.h_conv4_2, self.W_conv5_1, "conv5_1") + self.B_conv5_1)
            self.h_conv5_2 = tf.nn.relu(conv2d(self.h_conv5_1, self.W_conv5_2, "conv5_2") + self.B_conv5_2)
            self.h_conv5_3 = tf.nn.relu(conv2d(self.h_conv5_2, self.W_conv5_3, "conv5_2") + self.B_conv5_3)

        return self.h_conv5_3

    def defineOptimizer(self):
        with tf.name_scope("Opt"):
            optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss,
                    var_list=[
                        self.rpn_conv_weight,
                        self.rpn_conv_bias,
                        self.rpn_obj_weight,
                        self.rpn_obj_bias,
                        self.rpn_reg_weight,
                        self.rpn_reg_bias,
                    ]
                    )
        return optimizerAll

    def defineSummaries(self):
        #Summaries
        tf.scalar_summary('loss', self.loss, name="loss")
        tf.scalar_summary('clsLoss', self.clsLoss, name="clsLoss")
        tf.scalar_summary('regLoss', self.regLoss, name="regLoss")
        tf.scalar_summary('posInstances', self.numPosInstances, name="regLoss")
        tf.scalar_summary('negInstances', self.numNegInstances, name="regLoss")
        tf.scalar_summary('subPosInstances', self.subPosInstances, name="regLoss")
        tf.scalar_summary('subNegInstances', self.subNegInstances, name="regLoss")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('maskObjGt', self.maskObjGt, name="maskObjGt")
        tf.histogram_summary('subMaskObjGt', self.subMaskObjGt, name="subMaskObjGt")
        tf.histogram_summary('relBBGt', self.relBBGt, name="relBBGt")
        tf.histogram_summary('subRelBBGt', self.subRelBBGt, name="relBBGt")

        tf.histogram_summary('gtTy', self.subGtTy, name="vis_gtTy")
        tf.histogram_summary('gtTx', self.subGtTx, name="vis_gtTx")
        tf.histogram_summary('gtTh', self.subGtTh, name="vis_gtTh")
        tf.histogram_summary('gtTw', self.subGtTw, name="vis_gtTw")

        tf.histogram_summary('rpnObj', self.rpnObj, name="vis_rpnObj")
        tf.histogram_summary('subRpnObj', self.subRpnObj, name="vis_subRpnObj")
        tf.histogram_summary('rpnReg', self.rpnReg, name="vis_rpnReg")
        tf.histogram_summary('subRpnReg', self.subRpnReg, name="vis_subRpnReg")

        tf.histogram_summary('outTy', self.outTy, name="vis_outTy")
        tf.histogram_summary('outTx', self.outTx, name="vis_outTx")
        tf.histogram_summary('outTh', self.outTh, name="vis_outTh")
        tf.histogram_summary('outTw', self.outTw, name="vis_outTw")

        #Weight and bias hists
        tf.histogram_summary('rpn_conv_weight', self.rpn_conv_weight, name="vis_rpn_conv_weight")
        tf.histogram_summary('rpn_conv_bias', self.rpn_conv_bias, name="vis_rpn_conv_bias")
        tf.histogram_summary('rpn_obj_weight', self.rpn_obj_weight, name="vis_rpn_obj_weight")
        tf.histogram_summary('rpn_obj_bias', self.rpn_obj_bias, name="vis_rpn_obj_bias")
        tf.histogram_summary('rpn_reg_weight', self.rpn_reg_weight, name="vis_rpn_reg_weight")
        tf.histogram_summary('rpn_reg_bias', self.rpn_reg_bias, name="vis_rpn_reg_bias")

        #Image output
        tf.summary.image('candidateBB', self.outEstImg, max_outputs=3)
        tf.summary.image('topCandidateBB', self.outTopEstImg, max_outputs=3)
        tf.summary.image('gtBB', self.outGtImg, max_outputs=3)
        tf.summary.image('subNegBB', self.outNegSubImg, max_outputs=3)

