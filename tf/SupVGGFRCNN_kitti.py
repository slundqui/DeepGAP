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
import sys
#import matplotlib.pyplot as plt

class VGG_FRCNN(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(VGG_FRCNN, self).loadParams(params)
        #GT shape here is defined for obj/noobj
        #stored as (numGt, y, x, windows*2)
        self.gtShape = params['gtShape']
        self.dncArea = params['dncArea']
        self.anchors = params['anchors']
        self.regLossWeight = params['regLossWeight']
        self.imageShape = params['imageShape']
        self.detConfidenceThreshold = params['detConfidenceThreshold']
        self.iouDetThreshold = params['iouDetThreshold']
        self.nmsIouThreshold = params['nmsIouThreshold']
        self.maxBB = params['maxBB']
        self.maxNegSamples = params['maxNegSamples']
        self.bestCurrThresh = .5
        self.vggFile = params['vggFile']

    def defineVars(self):
        ##Define all variables outside of scope
        ##Hidden 1st layer weights
        #self.h_weight = weight_variable_xavier([2, 15, 32, 6, 3072], "hidden_weight")
        #self.h_bias = bias_variable([3072], "hidden_bias")

        ##Convolution weights
        #self.conv_weights = []
        #self.conv_bias = []
        #inputFeatures = 3072

        #for i in range(self.numConvLayers):
        #    #First convolution weights
        #    self.conv_weights.append(weight_variable_xavier([3, 3, inputFeatures, 512], "conv"+str(i) + "_weight"))
        #    self.conv_bias.append(bias_variable([512], "conv"+str(i)+"_bias"))
        #    inputFeatures = 512

        #Two sibling fully connected layers
        #Obj/no obj
        #gtShape[3] encompases both windows and obj/no obj class

        self.W_convFrcnn= weight_variable_xavier([3, 3, 512, 512], "W_convFrcnn")
        self.B_convFrcnn= weight_variable_xavier([512], "B_convFrcnn")

        self.fc_obj_weight = weight_variable_xavier([1, 1, 512, self.gtShape[3]], "fc_obj_weight")
        self.fc_obj_bias= bias_variable([self.gtShape[3]], "fc_obj_bias" )
        #BB Regression
        self.fc_reg_weight = weight_variable_xavier([1, 1, 512, (self.gtShape[3]/2) * 4], "fc_reg_weight")
        self.fc_reg_bias = bias_variable([(self.gtShape[3]/2)*4], "fc_reg_bias")

    #Builds the model.
    def buildModel(self, inputShape):
        if(self.vggFile):
            npWeights = loadWeights(self.vggFile)
        else:
            print "Must load from weights"
            assert(0)

        #Running on GPU
        with tf.device(self.device):
            self.defineVars()
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]], "inputImage")
                self.singleImage = self.inputImage[:, 4, :, :, :]

                #self.stereoImage = tf.reshape(self.permuteImage,
                        #[self.batchSize, 3, inputShape[1], inputShape[2], inputShape[3]*2])

                #self.padInput = tf.pad(self.stereoImage, [[0, 0], [0, 0], [7, 7], [15, 15], [0, 0]])

            with tf.name_scope("groundTruth"):
                self.objIndices = tf.placeholder("int64", [2, None], "objIndices")
                self.objValues = tf.placeholder("float32", [None], "objValues")
                self.labelIndices = tf.placeholder("int64", [2, None], "labelIndices")
                self.labelValues = tf.placeholder("float32", [None], "dataValues")

                self.flatObj = tf.sparse_tensor_to_dense(tf.SparseTensor(
                        tf.transpose(self.objIndices, [1, 0]),
                        self.objValues,
                        [self.batchSize, self.gtShape[1] * self.gtShape[2] * self.gtShape[3]]
                    ),
                    validate_indices=False
                    )

                self.objGt = tf.reshape(self.flatObj, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 2])
                #Mask objGt with provided mask
                mask = np.expand_dims(self.dncArea, 0)
                mask = np.expand_dims(mask, 4)
                self.maskObjGt = self.objGt * mask

                #Count number of pos and neg samples
                self.numPosInstances = tf.reduce_sum(self.maskObjGt[:, :, :, :, 0])
                self.numNegInstances = tf.reduce_sum(self.maskObjGt[:, :, :, :, 1])

                self.flatLabels = tf.sparse_tensor_to_dense(tf.SparseTensor(
                        tf.transpose(self.labelIndices, [1, 0]),
                        self.labelValues,
                        [self.batchSize, self.gtShape[1] * self.gtShape[2] * self.gtShape[3]/2 * 5]
                    ),
                    validate_indices=False
                    )

                #gt is in shape of [id, ymin, ymax, xmin, xmax]
                self.bbAbsGt = tf.reshape(self.flatLabels, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 5])

                #Translate into relative locations based on anchors
                self.absoluteBB = self.bbAbsGt[:, :, :, :, 1:]

                self.absoluteAnchor = np.expand_dims(self.anchors, 0)
                self.absoluteAnchor = np.reshape(self.absoluteAnchor, [1, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 4])

                #Calculate w/h
                absoluteAnchorH = np.abs(self.absoluteAnchor[:, :, :, :, 1] - self.absoluteAnchor[:, :, :, :, 0])
                absoluteAnchorW = np.abs(self.absoluteAnchor[:, :, :, :, 3] - self.absoluteAnchor[:, :, :, :, 2])

                posIdx = tf.to_int32(tf.where(self.maskObjGt[:, :, :, :, 0] > .5))

                self.gtTy = (self.absoluteBB[:, :, :, :, 0]-self.absoluteAnchor[:, :, :, :, 0])/(absoluteAnchorH + 1e-8)
                self.check_gtTy = tf.check_numerics(self.gtTy, "gtTy value error")
                self.subGtTy = tf.gather_nd(self.gtTy, posIdx)

                self.gtTx = (self.absoluteBB[:, :, :, :, 2]-self.absoluteAnchor[:, :, :, :, 2])/(absoluteAnchorW + 1e-8)
                self.check_gtTx = tf.check_numerics(self.gtTx, "gtTx value error")
                self.subGtTx = tf.gather_nd(self.gtTx, posIdx)

                self.gtTh = tf.log(((self.absoluteBB[:, :, :, :, 1] - self.absoluteBB[:, :, :, :, 0])/(absoluteAnchorH + 1e-8))+1e-8)
                self.check_gtTh = tf.check_numerics(self.gtTh, "gtTh value error")
                self.subGtTh = tf.gather_nd(self.gtTh, posIdx)

                self.gtTw = tf.log(((self.absoluteBB[:, :, :, :, 3] - self.absoluteBB[:, :, :, :, 2])/(absoluteAnchorW + 1e-8))+1e-8)
                self.check_gtTw = tf.check_numerics(self.gtTw, "gtTh value error")
                self.subGtTw = tf.gather_nd(self.gtTw, posIdx)

                self.relBBGt = tf.pack([self.check_gtTy, self.check_gtTx, self.check_gtTh, self.check_gtTw], 4)
                #self.relBBGt = tf.pack([self.gtTy, self.gtTx, self.gtTh, self.gtTw], 4)

            with tf.name_scope("Conv1Ops"):
                self.W_conv1_1 = weight_variable_fromnp(npWeights["conv1_1_w"], "w_conv1_1")
                self.B_conv1_1 = weight_variable_fromnp(npWeights["conv1_1_b"], "b_conv1_1")
                self.W_conv1_2 = weight_variable_fromnp(npWeights["conv1_2_w"], "w_conv1_2")
                self.B_conv1_2 = weight_variable_fromnp(npWeights["conv1_2_b"], "b_conv1_2")

                self.h_conv1_1 = tf.nn.relu(conv2d(self.singleImage, self.W_conv1_1, "conv1_1", stride=[1, 1, 1, 1]) + self.B_conv1_1)
                self.h_conv1_2 = tf.nn.relu(conv2d(self.h_conv1_1, self.W_conv1_2, "conv1_1", stride=[1, 1, 1, 1]) + self.B_conv1_2)
                self.h_pool1 = maxpool_2x2(self.h_conv1_2, "pool1")

            with tf.name_scope("Conv2Ops"):
                self.W_conv2_1 = weight_variable_fromnp(npWeights["conv2_1_w"], "w_conv2_1")
                self.B_conv2_1 = weight_variable_fromnp(npWeights["conv2_1_b"], "b_conv2_1")
                self.W_conv2_2 = weight_variable_fromnp(npWeights["conv2_2_w"], "w_conv2_2")
                self.B_conv2_2 = weight_variable_fromnp(npWeights["conv2_2_b"], "b_conv2_2")

                self.h_conv2_1 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2_1, "conv2_1") + self.B_conv2_1)
                self.h_conv2_2 = tf.nn.relu(conv2d(self.h_conv2_1, self.W_conv2_2, "conv2_2") + self.B_conv2_2)
                self.h_pool2 = maxpool_2x2(self.h_conv2_2, "pool2")

            with tf.name_scope("Conv3Ops"):
                self.W_conv3_1 = weight_variable_fromnp(npWeights["conv3_1_w"], "w_conv3_1")
                self.B_conv3_1 = weight_variable_fromnp(npWeights["conv3_1_b"], "b_conv3_1")
                self.W_conv3_2 = weight_variable_fromnp(npWeights["conv3_2_w"], "w_conv3_2")
                self.B_conv3_2 = weight_variable_fromnp(npWeights["conv3_2_b"], "b_conv3_2")
                self.W_conv3_3 = weight_variable_fromnp(npWeights["conv3_3_w"], "w_conv3_3")
                self.B_conv3_3 = weight_variable_fromnp(npWeights["conv3_3_b"], "b_conv3_3")

                self.h_conv3_1 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3_1, "conv3_1") + self.B_conv3_1)
                self.h_conv3_2 = tf.nn.relu(conv2d(self.h_conv3_1, self.W_conv3_2, "conv3_2") + self.B_conv3_2)
                self.h_conv3_3 = tf.nn.relu(conv2d(self.h_conv3_2, self.W_conv3_3, "conv3_2") + self.B_conv3_3)
                #self.h_pool3 = maxpool_2x2(self.h_conv3_3, "pool3")

            with tf.name_scope("Conv4Ops"):
                self.W_conv4_1 = weight_variable_fromnp(npWeights["conv4_1_w"], "w_conv4_1")
                self.B_conv4_1 = weight_variable_fromnp(npWeights["conv4_1_b"], "b_conv4_1")
                self.W_conv4_2 = weight_variable_fromnp(npWeights["conv4_2_w"], "w_conv4_2")
                self.B_conv4_2 = weight_variable_fromnp(npWeights["conv4_2_b"], "b_conv4_2")
                self.W_conv4_3 = weight_variable_fromnp(npWeights["conv4_3_w"], "w_conv4_3")
                self.B_conv4_3 = weight_variable_fromnp(npWeights["conv4_3_b"], "b_conv4_3")

                #self.h_conv4_1 = tf.nn.relu(conv2d(self.h_pool3, self.W_conv4_1, "conv4_1") + self.B_conv4_1)
                self.h_conv4_1 = tf.nn.relu(conv2d(self.h_conv3_3, self.W_conv4_1, "conv4_1") + self.B_conv4_1)
                self.h_conv4_2 = tf.nn.relu(conv2d(self.h_conv4_1, self.W_conv4_2, "conv4_2") + self.B_conv4_2)
                self.h_conv4_3 = tf.nn.relu(conv2d(self.h_conv4_2, self.W_conv4_3, "conv4_2") + self.B_conv4_3)
                #self.h_pool4 = maxpool_2x2(self.h_conv4_3, "pool4")


            with tf.name_scope("Conv5Ops"):
                self.W_conv5_1 = weight_variable_fromnp(npWeights["conv5_1_w"], "w_conv5_1")
                self.B_conv5_1 = weight_variable_fromnp(npWeights["conv5_1_b"], "b_conv5_1")
                self.W_conv5_2 = weight_variable_fromnp(npWeights["conv5_2_w"], "w_conv5_2")
                self.B_conv5_2 = weight_variable_fromnp(npWeights["conv5_2_b"], "b_conv5_2")
                self.W_conv5_3 = weight_variable_fromnp(npWeights["conv5_3_w"], "w_conv5_3")
                self.B_conv5_3 = weight_variable_fromnp(npWeights["conv5_3_b"], "b_conv5_3")

                #self.h_conv5_1 = tf.nn.relu(conv2d(self.h_pool4, self.W_conv5_1, "conv5_1") + self.B_conv5_1)
                self.h_conv5_1 = tf.nn.relu(conv2d(self.h_conv4_2, self.W_conv5_1, "conv5_1") + self.B_conv5_1)
                self.h_conv5_2 = tf.nn.relu(conv2d(self.h_conv5_1, self.W_conv5_2, "conv5_2") + self.B_conv5_2)
                self.h_conv5_3 = tf.nn.relu(conv2d(self.h_conv5_2, self.W_conv5_3, "conv5_2") + self.B_conv5_3)

            with tf.name_scope("rcnn"):
                self.h_conv_rcnn= tf.nn.relu(conv2d(self.h_conv5_3, self.W_convFrcnn, "conv5_1") + self.B_convFrcnn)

            with tf.name_scope("fcObj"):
                tmpFcObj = tf.nn.conv2d(self.h_conv_rcnn, self.fc_obj_weight, [1, 1, 1, 1], padding="SAME") + self.fc_obj_bias
                #Expand out window dim
                self.fcObj = tf.nn.softmax(tf.reshape(tmpFcObj, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 2]))

            with tf.name_scope("fcReg"):
                tmpFcReg = tf.nn.conv2d(self.h_conv_rcnn, self.fc_reg_weight, [1, 1, 1, 1], padding="SAME") + self.fc_reg_bias
                self.fcReg = tf.reshape(tmpFcReg, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 4])

            with tf.name_scope("calcMetric"):
                #Remap to ymin, ymax, xmin, xmax
                self.outTy = self.fcReg[:, :, :, :, 0]
                self.outTx = self.fcReg[:, :, :, :, 1]
                self.outTh = self.fcReg[:, :, :, :, 2]
                self.outTw = self.fcReg[:, :, :, :, 3]

                ymin = self.outTy*(absoluteAnchorH+1e-8) + self.absoluteAnchor[:, :, :, :, 0]
                xmin = self.outTx*(absoluteAnchorW+1e-8) + self.absoluteAnchor[:, :, :, :, 2]
                ysize = tf.exp(self.outTh-1e-8) * (absoluteAnchorH+1e-8)
                xsize = tf.exp(self.outTw-1e-8) * (absoluteAnchorW+1e-8)
                ymax = ymin + ysize
                xmax = xmin + xsize
                self.absoluteOutBB = tf.pack([ymin, ymax, xmin, xmax], 4)

                #Translate absoulteOutBB into relative locations and transpose to what tf is looking for
                yminRel = tf.to_float(ymin)/self.imageShape[0]
                ymaxRel = tf.to_float(ymax)/self.imageShape[0]
                xminRel = tf.to_float(xmin)/self.imageShape[1]
                xmaxRel = tf.to_float(xmax)/self.imageShape[1]
                self.tfOutBB = tf.pack([yminRel, xminRel, ymaxRel, xmaxRel], 4)

                #Run nms on output
                self.nmsOutBB = []
                self.nmsOutScore = []
                for b in range(self.batchSize):
                    (tmpOutBB, tmpOutScore) = runNms(self.fcObj[b, :, :, :, :], self.tfOutBB[b, :, :, :, :], self.maxBB, self.nmsIouThreshold)
                    self.nmsOutBB.append(tmpOutBB)
                    self.nmsOutScore.append(tmpOutScore)

                self.inThreshold = tf.placeholder(tf.float32)

                self.outPosBB = calcBatchBB(self.nmsOutBB, self.nmsOutScore, self.inThreshold, self.maxBB)

                #Do the same with gt boxes
                gt_yminRel = tf.to_float(self.absoluteBB[:, :, :, :, 0])/self.imageShape[0]
                gt_ymaxRel = tf.to_float(self.absoluteBB[:, :, :, :, 1])/self.imageShape[0]
                gt_xminRel = tf.to_float(self.absoluteBB[:, :, :, :, 2])/self.imageShape[1]
                gt_xmaxRel = tf.to_float(self.absoluteBB[:, :, :, :, 3])/self.imageShape[1]
                self.gt_tfOutBB = tf.pack([gt_yminRel, gt_xminRel, gt_ymaxRel, gt_xmaxRel], 4)

                #Flatten gt to match
                self.gtBBs = []
                self.gtScores = []
                for b in range(self.batchSize):
                    self.gtScores.append(tf.reshape(self.maskObjGt[b, :, :, :, 0], [-1]))
                    self.gtBBs.append(tf.reshape(self.gt_tfOutBB[b, :, :, :, :], [-1, 4]))

                self.outGtBB = calcBatchBB(self.gtBBs, self.gtScores, 0.5, self.maxBB)

                #4 for left camera of 2nd frame
                outImg = self.inputImage[:, 4, :, :, :]
                self.outEstImg = tf.image.draw_bounding_boxes(outImg, self.outPosBB)
                self.outGtImg = tf.image.draw_bounding_boxes(outImg, self.outGtBB)

                #Calculate mean average precision
                #(precision, recall) = calcPvR(self.fcObj, self.tfOutBB, self.maskObjGt, self.gt_tfOutBB, self.batchSize, self.maxBB, self.iouDetThreshold, nms=True, nms_iou_threshold=self.nmsIouThreshold)

            with tf.name_scope("Loss"):
                #Define loss
                #We limit negative examples to maxNegVal
                posIdx = tf.to_int32(tf.where(self.maskObjGt[:, :, :, :, 0] > .5))
                negIdx = tf.to_int32(tf.where(self.maskObjGt[:, :, :, :, 1] > .5))
                #Shuffle negIdx and pick subset
                #Note that this will break if there does not exist maxNegVal neg examples in batch
                subNegIdx = tf.random_shuffle(negIdx)[:self.maxNegSamples, :]

                #Calculate relative anchors
                ymin = self.absoluteAnchor[:, :, :, :, 0]/self.imageShape[0]
                ymax = self.absoluteAnchor[:, :, :, :, 1]/self.imageShape[0]
                xmin = self.absoluteAnchor[:, :, :, :, 2]/self.imageShape[1]
                xmax = self.absoluteAnchor[:, :, :, :, 3]/self.imageShape[1]
                #Permute to tf
                self.relativeAnchor = tf.pack([ymin, xmin, ymax, xmax], 4)

                #Generate image output based on this subset
                negAnchors = []
                for b in range(self.batchSize):
                    imgSubNegIdxIdx = tf.where(tf.equal(subNegIdx[:, 0], b))
                    imgSubNegIdx = tf.gather_nd(subNegIdx, imgSubNegIdxIdx)
                    batchNegAnchors = tf.gather_nd(self.relativeAnchor[0, :, :, :, :], imgSubNegIdx[:, 1:])
                    #Expand out dim
                    #Pad 1 at bottom in case of zero posBBs
                    padNegAnchors = tf.pad(batchNegAnchors, [[0, 1], [0, 0]], mode='CONSTANT')
                    expandNegAnchors = tf.expand_dims(padNegAnchors, -1)
                    reshapeNegAnchors = tf.image.resize_image_with_crop_or_pad(expandNegAnchors, 10, 4)[:, :, 0]
                    negAnchors.append(reshapeNegAnchors)
                #Pack list
                negAnchors = tf.to_float(tf.pack(negAnchors, 0))
                #Draw subset (10 per image)
                self.outNegSubImg = tf.image.draw_bounding_boxes(outImg, negAnchors)

                self.subMaskObjGt = tf.concat(0,
                        [tf.gather_nd(self.maskObjGt, posIdx), tf.gather_nd(self.maskObjGt, subNegIdx)])
                self.subFcObj = tf.concat(0,
                        [tf.gather_nd(self.fcObj, posIdx), tf.gather_nd(self.fcObj, subNegIdx)])

                self.subFcReg = tf.concat(0,
                        [tf.gather_nd(self.fcReg, posIdx), tf.gather_nd(self.fcReg, subNegIdx)])
                self.subRelBBGt = tf.concat(0,
                        [tf.gather_nd(self.relBBGt, posIdx), tf.gather_nd(self.relBBGt, subNegIdx)])

                #Count numbers after subset
                self.subPosInstances = tf.reduce_sum(self.subMaskObjGt[:, 0])
                self.subNegInstances = tf.reduce_sum(self.subMaskObjGt[:, 1])

                ##Code for all neg examples
                #self.clsLoss = tf.reduce_mean(-tf.reduce_sum(self.maskObjGt * tf.log(self.fcObj+1e-10), reduction_indices=[4]))
                #self.clsLoss = tf.check_numerics(self.clsLoss, "clsLoss value error")
                #sL1 = smoothL1(self.fcReg-self.relBBGt)
                #masksL1 = tf.expand_dims(self.maskObjGt[:, :, :, :, 0], 4) * sL1

                #Code for sub neg examples
                self.clsLoss = tf.reduce_mean(-tf.reduce_sum(self.subMaskObjGt * tf.log(self.subFcObj+1e-10), reduction_indices=[1]))
                sL1 = smoothL1(self.subFcReg-self.subRelBBGt)
                masksL1 = tf.expand_dims(self.subMaskObjGt[:, 0], 1) * sL1


                masksL1 = tf.check_numerics(masksL1, "masksL1 value error")

                #regLoss only applies to positive objs
                self.regLoss = tf.reduce_mean(masksL1)

                self.loss = self.clsLoss + self.regLossWeight * self.regLoss

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss,
                        var_list=[
                            self.W_convFrcnn,
                            self.B_convFrcnn,
                            self.fc_obj_weight,
                            self.fc_obj_bias,
                            self.fc_reg_weight,
                            self.fc_reg_bias,
                        ]
                        )

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

        tf.histogram_summary('fcObj', self.fcObj, name="vis_fcObj")
        tf.histogram_summary('subFcObj', self.subFcObj, name="vis_subFcObj")
        tf.histogram_summary('fcReg', self.fcReg, name="vis_fcReg")
        tf.histogram_summary('subFcReg', self.subFcReg, name="vis_subFcReg")

        tf.histogram_summary('outTy', self.outTy, name="vis_outTy")
        tf.histogram_summary('outTx', self.outTx, name="vis_outTx")
        tf.histogram_summary('outTh', self.outTh, name="vis_outTh")
        tf.histogram_summary('outTw', self.outTw, name="vis_outTw")

        #Weight and bias hists
        tf.histogram_summary('W_convFrcnn', self.W_convFrcnn, name="vis_W_convFrcnn")
        tf.histogram_summary('B_convFrcnn', self.B_convFrcnn, name="vis_B_convFrcnn")
        tf.histogram_summary('fc_obj_weight', self.fc_obj_weight, name="vis_fc_obj_weight")
        tf.histogram_summary('fc_obj_bias', self.fc_obj_bias, name="vis_fc_obj_bias")
        tf.histogram_summary('fc_reg_weight', self.fc_reg_weight, name="vis_fc_reg_weight")
        tf.histogram_summary('fc_reg_bias', self.fc_reg_bias, name="vis_fc_reg_bias")

        #Image output
        tf.summary.image('candidateBB', self.outEstImg, max_outputs=3)
        tf.summary.image('gtBB', self.outGtImg, max_outputs=3)
        tf.summary.image('subNegBB', self.outNegSubImg, max_outputs=3)

    def getLoadVars(self):
        v = tf.global_variables()
        #return [var for var in v if (not "gap" in var.name) and (not "GAP" in var.name) ]
        return v


    def genFeedDict(self, data):
        if(self.detConfidenceThreshold is None):
            thresh = self.bestCurrThresh
        else:
            thresh = self.detConfidenceThreshold

        feedDict = {self.inputImage: data[0], self.inThreshold: self.detConfidenceThreshold}
        if(data[1] is not None):
            (spObjGt, spRegGt) = data[1]

            (objGtY, objGtX, objGtVals) = sp.find(spObjGt)
            feedDict[self.objIndices] = [objGtY, objGtX]
            feedDict[self.objValues] = objGtVals

            (regGtY, regGtX, regGtVals) = sp.find(spRegGt)
            feedDict[self.labelIndices] = [regGtY, regGtX]
            feedDict[self.labelValues] = regGtVals
        return feedDict

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        #Define session
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = self.genFeedDict(data)

            #Run optimizer
            try:
                self.sess.run(self.optimizerAll, feed_dict=feedDict)
            except:
                print "Opt error:", sys.exc_info()[0]
                np_gtTw = self.sess.run(self.gtTw, feed_dict=feedDict)
                np_absoluteBB = self.sess.run(self.absoluteBB, feed_dict=feedDict)
                np_absoluteAnchor = self.absoluteAnchor

                pdb.set_trace()

            if(i%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if(i%self.progress == 0):
                print "Timestep ", self.timestep
            self.timestep+=1

        #Plot first to get best threshold
        if(plot):
            filename = self.plotDir + "train_" + str(self.timestep)
            self.bestCurrThresh = self.evalAndPlotPvR(feedDict, filename)

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True):
        feedDict = self.genFeedDict([inData, inGt])

        #outVals = self.vis_cam.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)

        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            self.evalAndPlotPvR(feedDict, filename)

        #return outVals

    def evalAndPlotPvR(self, feedDict, filename):
        #These scores contain ALL output bbs
        #TODO nmsOutScore is a list of tensors, does this work?
        npOutScore = self.sess.run(self.nmsOutScore, feed_dict=feedDict)
        npOutBB = self.sess.run(self.nmsOutBB, feed_dict=feedDict)
        npGtScore = self.sess.run(self.gtScores, feed_dict=feedDict)
        npGtBB = self.sess.run(self.gtBBs, feed_dict=feedDict)
        #Parse npGtBB for positive GT only
        npPosGtBB = []
        for b in range(self.batchSize):
            npPosGtBB.append(npGtBB[b][np.nonzero(npGtScore[b] > .5)])
        (precision, recall, f1, bestThresh) = plotBBPvRBatch(npOutScore, npOutBB, npPosGtBB, self.iouDetThreshold, filename)
        return bestThresh

