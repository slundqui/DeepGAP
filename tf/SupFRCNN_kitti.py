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

class FRCNN(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(FRCNN, self).loadParams(params)
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
        self.numConvLayers = params['numConvLayers']
        self.bestCurrThresh = .5
        self.roiPoolSize = params['roiPoolSize']

    def defineEncodingVars(self):
        #Define all variables outside of scope
        #Hidden 1st layer weights
        self.h_weight = weight_variable_xavier([2, 15, 32, 6, 3072], "hidden_weight")
        self.h_bias = bias_variable([3072], "hidden_bias")

        #Convolution weights
        self.conv_weights = []
        self.conv_bias = []
        inputFeatures = 3072
        for i in range(self.numConvLayers):
            #First convolution weights
            self.conv_weights.append(weight_variable_xavier([3, 3, inputFeatures, 512], "conv"+str(i) + "_weight"))
            self.conv_bias.append(bias_variable([512], "conv"+str(i)+"_bias"))
            inputFeatures = 512

    def defineRpnVars(self):
        self.rpn_conv_weight = weight_variable_xavier([3, 3, 512, 512], 'rpn_conv_weight')
        self.rpn_conv_bias = bias_variable([512], 'rpn_conv_bias')

        #Two sibling fully connected layers
        #Obj/no obj
        #gtShape[3] encompases both windows and obj/no obj class
        self.rpn_obj_weight = weight_variable_xavier([1, 1, 512, self.gtShape[3]], "rpn_obj_weight")
        self.rpn_obj_bias= bias_variable([self.gtShape[3]], "rpn_obj_bias" )

        #BB Regression
        self.rpn_reg_weight = weight_variable_xavier([1, 1, 512, (self.gtShape[3]/2) * 4], "rpn_reg_weight")
        self.rpn_reg_bias = bias_variable([(self.gtShape[3]/2)*4], "fc_reg_bias")

    def defineClassVars(self):
        #Classification pipelin
        self.class_fc_1_weight = weight_variable_xavier([self.roiPoolSize[0]*self.roiPoolSize[1] * 512, 4096], "class_fc_1_weight")
        self.class_fc_1_bias = bias_variable([4096], "class_fc_1_bias")
        self.class_fc_2_weight = weight_variable_xavier([4096, 4096], "class_fc_2_weight")
        self.class_fc_2_bias = bias_variable([4096], "class_fc_2_bias")

        #class
        self.class_obj_weight = weight_variable_xavier([4096, self.gtShape[3]], "class_obj_weight")
        self.class_reg_weight = weight_variable_xavier([4096, (self.gtShape[3]/2)*4], "class_reg_weight")

    def defineEncoding(self):
        with tf.name_scope("Hidden"):
            self.h_hidden = tf.nn.relu(tf.nn.conv3d(self.padInput, self.h_weight, [1, 1, 4, 4, 1], padding="VALID") + self.h_bias)

        with tf.name_scope("pool"):
            yPool = int(np.ceil(float(16)/self.gtShape[1]))
            xPool = int(np.ceil(float(64)/self.gtShape[2]))

            self.timePooled = tf.reduce_max(self.h_hidden, reduction_indices=1)
            #self.inputPooled = tf.nn.max_pool(self.timePooled, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")

        with tf.name_scope("conv"):
            #inLayer = self.inputPooled
            inLayer = self.timePooled
            self.convLayers = []

            for i in range(self.numConvLayers):
                tmpConvLayer = tf.nn.relu(tf.nn.conv2d(inLayer, self.conv_weights[i], [1, 1, 1, 1], padding="SAME") + self.conv_bias[i])
                dropoutConvLayer = tf.nn.dropout(tmpConvLayer, self.keep_prob)
                self.convLayers.append(dropoutConvLayer)
                inLayer = dropoutConvLayer
            outLayer = self.convLayers[-1]
        return outLayer

    def defineOptimizer(self):
        with tf.name_scope("Opt"):
            optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
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
        tf.histogram_summary('wrtBBGt', self.wrtBBGt, name="wrtBBGt")
        tf.histogram_summary('subRelBBGt', self.subRelBBGt, name="wrtBBGt")

        tf.histogram_summary('gtTy', self.subGtTy, name="vis_gtTy")
        tf.histogram_summary('gtTx', self.subGtTx, name="vis_gtTx")
        tf.histogram_summary('gtTh', self.subGtTh, name="vis_gtTh")
        tf.histogram_summary('gtTw', self.subGtTw, name="vis_gtTw")

        tf.histogram_summary('h_hidden', self.h_hidden, name="vis_h_hidden")

        #Conv layer histograms
        for i in range(self.numConvLayers):
            convStr = "conv"+str(i)
            tf.histogram_summary('h_'+convStr       , self.convLayers[i], name="vis_h_"+convStr)
            tf.histogram_summary(convStr + "_weight", self.conv_weights[i], name="vis_"+convStr+"_weight")
            tf.histogram_summary(convStr + '_bias'  , self.conv_bias[i], name="vis_"+convStr+"bias")

        tf.histogram_summary('rpnObj', self.rpnObj, name="vis_rpnObj")
        tf.histogram_summary('subRpnObj', self.subRpnObj, name="vis_subRpnObj")
        tf.histogram_summary('rpnReg', self.rpnReg, name="vis_rpnReg")
        tf.histogram_summary('subRpnReg', self.subRpnReg, name="vis_subRpnReg")

        tf.histogram_summary('outTy', self.outTy, name="vis_outTy")
        tf.histogram_summary('outTx', self.outTx, name="vis_outTx")
        tf.histogram_summary('outTh', self.outTh, name="vis_outTh")
        tf.histogram_summary('outTw', self.outTw, name="vis_outTw")

        #Weight and bias hists
        tf.histogram_summary('h_weight', self.h_weight, name="vis_h_weight")
        tf.histogram_summary('h_bias', self.h_bias, name="vis_h_bias")

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



    #Builds the model.
    def buildModel(self, inputShape):
        #Running on GPU
        with tf.device(self.device):
            self.defineEncodingVars()
            self.defineRpnVars()
            self.defineClassVars()
            self.keep_prob = tf.placeholder(tf.float32)

            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]], "inputImage")
                #We split the time dimension to stereo and concatenate with feature dim
                self.reshapeImage = tf.reshape(self.inputImage,
                        [self.batchSize, 3, 2, inputShape[1], inputShape[2], inputShape[3]])
                self.permuteImage = tf.transpose(self.reshapeImage, [0, 1, 3, 4, 5, 2])
                self.stereoImage = tf.reshape(self.permuteImage,
                        [self.batchSize, 3, inputShape[1], inputShape[2], inputShape[3]*2])
                self.padInput = tf.pad(self.stereoImage, [[0, 0], [0, 0], [7, 7], [15, 15], [0, 0]])

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

                #gt is in shape of [id, ymin, xmin, ymax, xmax]
                self.bbAbsGt = tf.reshape(self.flatLabels, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 5])

                #Translate into relative locations based on anchors
                self.absoluteBB = self.bbAbsGt[:, :, :, :, 1:]

                self.absoluteAnchor = np.expand_dims(self.anchors, 0)
                self.absoluteAnchor = np.reshape(self.absoluteAnchor, [1, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 4])

                self.wrtBBGt = absBBtoWrtBB(absoluteBB, absoluteAnchor)

            with tf.name_scope("Encoding"):
                self.outEncodingLayer = self.defineEncoding()

            with tf.name_scope("RPN"):
                self.h_rpn_conv = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(self.outEncodingLayer, self.rpn_conv_weight, [1, 1, 1, 1], padding="SAME") + self.rpn_conv_bias), self.keep_prob)

                tmpRpnObj = tf.nn.conv2d(self.h_rpn_conv, self.rpn_obj_weight, [1, 1, 1, 1], padding="SAME") + self.rpn_obj_bias
                #Expand out window dim
                self.rpnObj = tf.nn.softmax(tf.reshape(tmpRpnObj, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 2]))

                tmpRpnReg = tf.nn.conv2d(self.h_rpn_conv, self.rpn_reg_weight, [1, 1, 1, 1], padding="SAME") + self.rpn_reg_bias
                self.rpnReg = tf.reshape(tmpRpnReg, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 4])

                self.absoluteOutBB = wrtBBtoAbsBB(self.rpnReg, self.absoluteAnchor)
                self.relOutBB = absBBtoRelBB(self.absoluteOutBB, self.imageShape)

                #Run nms on output
                self.rpnOutBB = []
                self.rpnOutScore = []
                #self.rpnGtBB = []
                #self.rpnGtObj= []

                for b in range(self.batchSize):
                    (tmpOutBB, tmpOutScore, tmpIdx) = runNms(self.rpnObj[b, :, :, :, :], self.relOutBB[b, :, :, :, :], 2*self.maxBB, self.nmsIouThreshold)

                    #Pick the top self.maxBB proposals
                    (topScores, topIdx) = tf.nn.top_k(tmpOutScore, k=self.maxBB)
                    topBBs = tf.gather(tmpOutBB, topIdx)

                    self.rpnOutBB.append(topBBs)
                    self.rpnOutScore.append(topScores)

            with tf.name_scope("ClassificationGt"):
                #Given proposals, we want to recalculate the grouth truth wrt the proposals



            with tf.name_scope("Classification"):
                #rpnOutBB and rpnOutScore are final object proposals

                #Concatenate boxes in the first dimension
                self.rpnConcatBB = tf.reshape(tf.pack(self.rpnOutBB, 0), [-1, 4])
                numBoxes = self.maxBB*self.batchSize
                box_ind = [int(np.floor(float(i)/self.maxBB)) for i in range(numBoxes)]
                #Crop and resize to 14x14
                self.resized_roi = tf.image.crop_and_resize(self.outEncodingLayer, self.rpnConcatBB, box_ind, [self.roiPoolSize[0]*2, self.roiPoolSize[1]*2])
                self.pooled_roi = tf.nn.max_pool(self.resized_roi, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

                self.reshaped_roi = tf.reshape(self.pooled_roi, [numBoxes, -1])
                self.h_class_fc_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.reshaped_roi, self.class_fc_1_weight, name="h_class_fc_1") + self.class_fc_1_bias, "h_class_fc_1"), self.keep_prob)
                self.h_class_fc_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.h_class_fc_1, self.class_fc_2_weight, name="h_class_fc_2") + self.class_fc_2_bias, "h_class_fc_2"), self.keep_prob)

                #Split into obj and reg
                tmp_class_obj = tf.nn.softmax(tf.matmul(self.h_class_fc_2, self.class_obj_weight, name="h_class_obj") + self.class_obj_bias)

                #TODO does reg use an activation function?
                tmp_class_reg = tf.matmul(self.h_class_fc_2, self.class_reg_weight, name="h_class_reg") + self.class_reg_bias

                #Reshape to extract batch dimension
                self.h_class_obj = tf.reshape(tmp_class_obj, [self.batchSize, -1, 2])
                self.h_class_reg = tf.reshape(tmp_class_reg, [self.batchSize, -1, 4])

                #Remap to ymin, ymax, xmin, xmax
                #self.
                #TODO feed this through nms

            with tf.name_scope("calcMetric"):
                posGtIdx = tf.to_int32(tf.where(self.maskObjGt[:, :, :, :, 0] > .5))

                self.subGtTy = tf.gather_nd(self.wrtBBGt[..., 0], posGtIdx)
                self.subGtTx = tf.gather_nd(self.wrtBBGt[..., 1], posGtIdx)
                self.subGtTh = tf.gather_nd(self.wrtBBGt[..., 2], posGtIdx)
                self.subGtTw = tf.gather_nd(self.wrtBBGt[..., 3], posGtIdx)




                self.inThreshold = tf.placeholder(tf.float32)

                pdb.set_trace()
                self.outPosBB = calcBatchBB(self.h_class_reg, self.rpnOutScore, self.inThreshold, self.maxBB, self.batchSize)
                self.bestBBs = calcBestBatchBB(self.rpnOutBB, self.rpnOutScore, 5, self.batchSize)

                #self.outPosBB = calcBatchBB(self.rpnOutBB, self.rpnOutScore, self.inThreshold, self.maxBB)
                #self.bestBBs = calcBestBatchBB(self.rpnOutBB, self.rpnOutScore, 5)

                #Do the same with gt boxes
                self.gt_tfOutBB = absBBtoRelBB(self.absoluteBB, self.imageShape)

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
                self.outTopEstImg = tf.image.draw_bounding_boxes(outImg, self.bestBBs)
                self.outGtImg = tf.image.draw_bounding_boxes(outImg, self.outGtBB)

            with tf.name_scope("RpnLoss"):
                #Define loss
                #We limit negative examples to maxNegVal
                posIdx = tf.to_int32(tf.where(self.maskObjGt[:, :, :, :, 0] > .5))
                negIdx = tf.to_int32(tf.where(self.maskObjGt[:, :, :, :, 1] > .5))
                #Shuffle negIdx and pick subset
                #Note that this will break if there does not exist maxNegVal neg examples in batch
                subNegIdx = tf.random_shuffle(negIdx)[:self.maxNegSamples, :]

                #Calculate relative anchors
                self.relativeAnchor = absBBtoRelBB(self.absoluteAnchor, self.imageShape)

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
                self.subRelBBGt = tf.concat(0,
                        [tf.gather_nd(self.wrtBBGt, posIdx), tf.gather_nd(self.wrtBBGt, subNegIdx)])

                self.subRpnObj = tf.concat(0,
                        [tf.gather_nd(self.rpnObj, posIdx), tf.gather_nd(self.rpnObj, subNegIdx)])
                self.subRpnReg = tf.concat(0,
                        [tf.gather_nd(self.rpnReg, posIdx), tf.gather_nd(self.rpnReg, subNegIdx)])

                #Count numbers after subset
                self.subPosInstances = tf.reduce_sum(self.subMaskObjGt[:, 0])
                self.subNegInstances = tf.reduce_sum(self.subMaskObjGt[:, 1])

                ##Code for all neg examples
                #self.clsLoss = tf.reduce_mean(-tf.reduce_sum(self.maskObjGt * tf.log(self.fcObj+1e-10), reduction_indices=[4]))
                #self.clsLoss = tf.check_numerics(self.clsLoss, "clsLoss value error")
                #sL1 = smoothL1(self.fcReg-self.wrtBBGt)
                #masksL1 = tf.expand_dims(self.maskObjGt[:, :, :, :, 0], 4) * sL1

                #Code for sub neg examples
                self.clsLoss = tf.reduce_mean(-tf.reduce_sum(self.subMaskObjGt * tf.log(self.subRpnObj+1e-10), reduction_indices=[1]))
                sL1 = smoothL1(self.subRpnReg-self.subRelBBGt)
                masksL1 = tf.expand_dims(self.subMaskObjGt[:, 0], 1) * sL1


                masksL1 = tf.check_numerics(masksL1, "masksL1 value error")

                #regLoss only applies to positive objs
                self.regLoss = tf.reduce_mean(masksL1)

                self.loss = self.clsLoss + self.regLossWeight * self.regLoss

            with tf.name_scope("Opt"):
                self.optimizer = self.defineOptimizer()

        self.defineSummaries()

    def getLoadVars(self):
        v = tf.global_variables()
        #return [var for var in v if (not "gap" in var.name) and (not "GAP" in var.name) ]
        return v

    def genFeedDict(self, data, dropoutProb):
        if(self.detConfidenceThreshold is None):
            thresh = self.bestCurrThresh
        else:
            thresh = self.detConfidenceThreshold

        feedDict = {self.inputImage: data[0], self.inThreshold: self.detConfidenceThreshold, self.keep_prob: dropoutProb}
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
            feedDict = self.genFeedDict(data, .5)

            #Run optimizer
            try:
                self.sess.run(self.optimizer, feed_dict=feedDict)
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

            npOutScore = self.sess.run(self.rpnOutScore, feed_dict=feedDict)
            npOutBB = self.sess.run(self.rpnOutBB, feed_dict=feedDict)
            npGtScore = self.sess.run(self.gtScores, feed_dict=feedDict)
            npGtBB = self.sess.run(self.gtBBs, feed_dict=feedDict)
            npPosGtBB = []
            for b in range(self.batchSize):
                npPosGtBB.append(npGtBB[b][np.nonzero(npGtScore[b] > .5)])

            self.bestCurrThresh = self.evalAndPlotPvR(npOutBB, npOutScore, npPosGtBB, filename)

        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True, writeOutput=True):
        feedDict = self.genFeedDict([inData, inGt], 1.0)

        #outVals = self.vis_cam.eval(feed_dict=feedDict, session=self.sess)
        #These scores contain ALL output bbs
        npOutScore = self.sess.run(self.rpnOutScore, feed_dict=feedDict)
        npOutBB = self.sess.run(self.rpnOutBB, feed_dict=feedDict)

        if(inGt != None):
            npGtScore = self.sess.run(self.gtScores, feed_dict=feedDict)
            npGtBB = self.sess.run(self.gtBBs, feed_dict=feedDict)
            #Parse npGtBB for positive GT only
            npPosGtBB = []
            for b in range(self.batchSize):
                npPosGtBB.append(npGtBB[b][np.nonzero(npGtScore[b] > .5)])
            if(writeOutput):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.test_writer.add_summary(summary, self.timestep)

            if(plot):
                filename = self.plotDir + "test_" + str(self.timestep)
                self.evalAndPlotPvR(npOutBB, npOutScore, npPosGtBB, filename)
            return (npOutBB, npOutScore, npGtBB)
        else:
            return (npOutBB, npOutScore)

    def evalModelBatch(self, testDataObj, plot=True):
        numData = len(testDataObj.shuffleIdx)
        #Ceil of numData/batchSize
        numIt = int(np.ceil(float(numData)/self.batchSize))
        outBB = []
        outScore = []
        gtBB = []
        for it in range(numIt):
            print "Eval all:", it, " out of ", numIt
            data = testDataObj.getData(self.batchSize)
            if(it == 0):
                writeOutput = True
            else:
                writeOutput = False
            (tmpOutBB, tmpOutScore, tmpGtBB) = self.evalModel(data[0], data[1], plot=False, writeOutput=writeOutput)
            outBB.extend(tmpOutBB)
            outScore.extend(tmpOutScore)
            gtBB.extend(tmpGtBB)
        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep) + "_all"
            self.evalAndPlotPvR(outBB, outScore, gtBB, filename)

    def evalAndPlotPvR(self, npOutBB, npOutScore, npGtBB, filename):
        (precision, recall, f1, bestThresh) = plotBBPvRBatch(npOutScore, npOutBB, npGtBB, self.iouDetThreshold, filename)
        return bestThresh

