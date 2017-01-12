import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
from base import TFObj
import scipy.sparse as sp
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

    def defineVars(self):
        #Define all variables outside of scope
        #Hidden 1st layer weights
        self.h_weight = weight_variable_xavier([2, 15, 32, 6, 3072], "hidden_weight")
        self.h_bias = bias_variable([3072], "hidden_bias")
        #First convolution weights
        self.conv1_weight = weight_variable_xavier([3, 3, 3072, 512], "conv1_weight")
        self.conv1_bias = bias_variable([512], "conv1_bias")
        #Two sibling fully connected layers
        #Obj/no obj
        #gtShape[3] encompases both windows and obj/no obj class
        self.fc_obj_weight = weight_variable_xavier([1, 1, 512, self.gtShape[3]], "fc_obj_weight")
        self.fc_obj_bias= bias_variable([self.gtShape[3]], "fc_obj_bias" )
        #BB Regression
        self.fc_reg_weight = weight_variable_xavier([1, 1, 512, (self.gtShape[3]/2) * 4], "fc_reg_weight")
        self.fc_reg_bias = bias_variable([(self.gtShape[3]/2)*4], "fc_reg_bias")

    #Builds the model.
    def buildModel(self, inputShape):
        #Running on GPU
        with tf.device(self.device):
            self.defineVars()
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
                absoluteBB = self.bbAbsGt[:, :, :, :, 1:]
                absoluteAnchor = np.expand_dims(self.anchors, 0)
                absoluteAnchor = np.reshape(absoluteAnchor, [1, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 4])

                #Calculate w/h
                absoluteAnchorH = np.abs(absoluteAnchor[:, :, :, :, 1] - absoluteAnchor[:, :, :, :, 0])
                absoluteAnchorW = np.abs(absoluteAnchor[:, :, :, :, 3] - absoluteAnchor[:, :, :, :, 2])


                self.gtTy = (absoluteBB[:, :, :, :, 0]-absoluteAnchor[:, :, :, :, 0])/(absoluteAnchorH + 1e-6)
                self.gtTx = (absoluteBB[:, :, :, :, 2]-absoluteAnchor[:, :, :, :, 2])/(absoluteAnchorW + 1e-6)

                self.gtTh = tf.log(((absoluteBB[:, :, :, :, 1] - absoluteBB[:, :, :, :, 0])/(absoluteAnchorH + 1e-6))+1e-6)
                self.gtTw = tf.log(((absoluteBB[:, :, :, :, 3] - absoluteBB[:, :, :, :, 2])/(absoluteAnchorW + 1e-6))+1e-6)

                self.relBbGt = tf.pack([self.gtTy, self.gtTx, self.gtTh, self.gtTw], 4)

            with tf.name_scope("Hidden"):
                self.h_hidden = tf.nn.relu(tf.nn.conv3d(self.padInput, self.h_weight, [1, 1, 4, 4, 1], padding="VALID") + self.h_bias)

            with tf.name_scope("conv1"):
                yPool = int(np.ceil(float(16)/self.gtShape[1]))
                xPool = int(np.ceil(float(64)/self.gtShape[2]))

                self.timePooled = tf.reduce_max(self.h_hidden, reduction_indices=1)
                self.inputPooled = tf.nn.max_pool(self.timePooled, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")
                self.h_conv1 = tf.nn.relu(tf.nn.conv2d(self.inputPooled, self.conv1_weight, [1, 1, 1, 1], padding="SAME") + self.conv1_bias)

            #ADD MORE LAYERS HERE

            with tf.name_scope("fcObj"):
                tmpFcObj = tf.nn.conv2d(self.h_conv1, self.fc_obj_weight, [1, 1, 1, 1], padding="SAME") + self.fc_obj_bias
                #Expand out window dim
                self.fcObj = tf.nn.softmax(tf.reshape(tmpFcObj, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 2]))

            with tf.name_scope("fcReg"):
                tmpFcReg = tf.nn.conv2d(self.h_conv1, self.fc_reg_weight, [1, 1, 1, 1], padding="SAME") + self.fc_reg_bias
                self.fcReg = tf.reshape(tmpFcReg, [self.batchSize, self.gtShape[1], self.gtShape[2], self.gtShape[3]/2, 4])

                #Remap to ymin, ymax, xmin, xmax
                self.outTy = self.fcReg[:, :, :, :, 0]
                self.outTx = self.fcReg[:, :, :, :, 1]
                self.outTh = self.fcReg[:, :, :, :, 2]
                self.outTw = self.fcReg[:, :, :, :, 3]

                ymin = self.outTy*(absoluteAnchorH) + absoluteAnchor[:, :, :, :, 0]
                xmin = self.outTx*(absoluteAnchorW) + absoluteAnchor[:, :, :, :, 2]
                ysize = tf.exp(self.outTh) * (absoluteAnchorH)
                xsize = tf.exp(self.outTw) * (absoluteAnchorW)
                ymax = ymin + ysize
                xmax = xmin + xsize
                self.absoluteOutBb = tf.pack([ymin, ymax, xmin, xmax], 4)

            with tf.name_scope("Loss"):
                #Define loss
                self.clsLoss = tf.reduce_mean(-tf.reduce_sum(self.maskObjGt * tf.log(self.fcObj+1e-6), reduction_indices=[4]))

                sL1 = smoothL1(self.fcReg-self.relBbGt)
                self.regLoss = tf.reduce_mean(tf.expand_dims(self.maskObjGt[:, :, :, :, 0], 4) * sL1)

                self.loss = self.clsLoss + self.regLossWeight * self.regLoss

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="loss")
        tf.scalar_summary('clsLoss', self.clsLoss, name="clsLoss")
        tf.scalar_summary('regLoss', self.regLoss, name="regLoss")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('maskObjGt', self.maskObjGt, name="maskObjGt")
        tf.histogram_summary('relBbGt', self.relBbGt, name="relBbGt")

        tf.histogram_summary('gtTy', self.gtTy, name="vis_gtTy")
        tf.histogram_summary('gtTx', self.gtTx, name="vis_gtTx")
        tf.histogram_summary('gtTh', self.gtTh, name="vis_gtTh")
        tf.histogram_summary('gtTw', self.gtTw, name="vis_gtTw")

        #Conv layer histograms
        tf.histogram_summary('h_hidden', self.h_hidden, name="vis_h_hidden")
        tf.histogram_summary('h_conv1', self.h_conv1, name="vis_h_conv1")
        tf.histogram_summary('fcObj', self.fcObj, name="vis_fcObj")
        tf.histogram_summary('fcReg', self.fcReg, name="vis_fcReg")

        tf.histogram_summary('outTy', self.outTy, name="vis_outTy")
        tf.histogram_summary('outTx', self.outTx, name="vis_outTx")
        tf.histogram_summary('outTh', self.outTh, name="vis_outTh")
        tf.histogram_summary('outTw', self.outTw, name="vis_outTw")

        #Weight and bias hists
        tf.histogram_summary('h_weight', self.h_weight, name="vis_h_weight")
        tf.histogram_summary('h_bias', self.h_bias, name="vis_h_bias")
        tf.histogram_summary('conv1_weight', self.conv1_weight, name="vis_conv1_weight")
        tf.histogram_summary('conv1_bias', self.conv1_bias, name="vis_conv1_bias")
        tf.histogram_summary('fc_obj_weight', self.fc_obj_weight, name="vis_fc_obj_weight")
        tf.histogram_summary('fc_obj_bias', self.fc_obj_bias, name="vis_fc_obj_bias")
        tf.histogram_summary('fc_reg_weight', self.fc_reg_weight, name="vis_fc_reg_weight")
        tf.histogram_summary('fc_reg_bias', self.fc_reg_bias, name="vis_fc_reg_bias")

    def getLoadVars(self):
        v = tf.global_variables()
        #return [var for var in v if (not "gap" in var.name) and (not "GAP" in var.name) ]
        return v


    def genFeedDict(self, data):
        feedDict = {self.inputImage: data[0]}
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
            self.sess.run(self.optimizerAll, feed_dict=feedDict)

            if(i%self.writeStep == 0):
                summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
                self.train_writer.add_summary(summary, self.timestep)
            if(i%self.progress == 0):
                print "Timestep ", self.timestep
            self.timestep+=1
        if(save):
            save_path = self.saver.save(self.sess, self.saveFile, global_step=self.timestep, write_meta_graph=False)
            print("Model saved in file: %s" % save_path)
        if(plot):
            filename = self.plotDir + "train_" + str(self.timestep)
            self.evalAndPlotBB(feedDict, filename)

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
            self.evalAndPlotBB(feedDict, filename)

        #return outVals

    def evalAndPlotBB(self, feedDict, filename):
        #TODO nms these
        npFcObj = self.sess.run(self.fcObj, feed_dict=feedDict)
        npFcReg = self.sess.run(self.absoluteOutBb, feed_dict=feedDict)
        #TODO plot



