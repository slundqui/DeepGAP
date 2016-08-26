import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
from base import TFObj
#import matplotlib.pyplot as plt

class FRCNN(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(VGGDetGap, self).loadParams(params)

        self.vggFile = params['vggFile']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.gtShape = params['gtShape']

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        if(self.vggFile):
            npWeights = loadWeights(self.vggFile)
        else:
            print "Must load from weights"
            assert(0)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2]], "inputImage")

            with tf.name_scope("groundTruth"):
                self.objIndices = tf.placeholder("int64", [2, None], "objIndices")
                #TODO faster to make this vector of 1's faster?
                self.objValues = tf.placeholder("float32", [None], "objValues")
                self.labelIndices = tf.placeholder("int64", [2, None], "labelIndices")
                self.labelValues = tf.placeholder("float32", [None], "dataValues")

                self.flatObj = tf.sparse_tensor_to_dense(tf.SparseTensor(
                        tf.transpose(self.objIndices, [1, 0]),
                        self.objValues,
                        [self.batchSize, gtShape[0] * gtShape[1]]
                    ),
                    validate_indices=False
                    )

                self.ObjGt = tf.reshape(self.flatObj, [self.batchSize, gtShape[0], gtShape[1], 1])

                #Expand out to be binary (obj, no obj) depending on iou thresh

                self.flatLabels = tf.sparse_tensor_to_dense(tf.SparseTensor(
                        tf.transpose(self.labelIndices, [1, 0]),
                        self.labelValues,
                        [self.batchSize, gtShape[0] * gtShape[1] * 5]
                    ),
                    validate_indices=False
                    )

                self.LabelGt = tf.reshape(self.flatObj, [self.batchSize, gtShape[0], gtShape[1], 5])


                #gt is in shape of [top, left, height, width]
                self.clsRpGt = node_variable([self.batchSize, 9, 14, 14, 2], "clsGT")
                self.regRpGt = node_variable([self.batchSize, 9, 14, 14, 4], "regGT")



            with tf.name_scope("Conv1Ops"):
                self.W_conv1_1 = weight_variable_fromnp(npWeights["conv1_1_w"], "w_conv1_1")
                self.B_conv1_1 = weight_variable_fromnp(npWeights["conv1_1_b"], "b_conv1_1")
                self.W_conv1_2 = weight_variable_fromnp(npWeights["conv1_2_w"], "w_conv1_2")
                self.B_conv1_2 = weight_variable_fromnp(npWeights["conv1_2_b"], "b_conv1_2")

                self.h_conv1_1 = tf.nn.relu(conv2d(self.inputImage, self.W_conv1_1, "conv1_1", stride=[1, 1, 1, 1]) + self.B_conv1_1)
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
                self.h_pool3 = maxpool_2x2(self.h_conv3_3, "pool3")

            with tf.name_scope("Conv4Ops"):
                self.W_conv4_1 = weight_variable_fromnp(npWeights["conv4_1_w"], "w_conv4_1")
                self.B_conv4_1 = weight_variable_fromnp(npWeights["conv4_1_b"], "b_conv4_1")
                self.W_conv4_2 = weight_variable_fromnp(npWeights["conv4_2_w"], "w_conv4_2")
                self.B_conv4_2 = weight_variable_fromnp(npWeights["conv4_2_b"], "b_conv4_2")
                self.W_conv4_3 = weight_variable_fromnp(npWeights["conv4_3_w"], "w_conv4_3")
                self.B_conv4_3 = weight_variable_fromnp(npWeights["conv4_3_b"], "b_conv4_3")

                self.h_conv4_1 = tf.nn.relu(conv2d(self.h_pool3, self.W_conv4_1, "conv4_1") + self.B_conv4_1)
                self.h_conv4_2 = tf.nn.relu(conv2d(self.h_conv4_1, self.W_conv4_2, "conv4_2") + self.B_conv4_2)
                self.h_conv4_3 = tf.nn.relu(conv2d(self.h_conv4_2, self.W_conv4_3, "conv4_2") + self.B_conv4_3)
                self.h_pool4 = maxpool_2x2(self.h_conv4_3, "pool4")

            with tf.name_scope("Conv5Ops"):
                self.W_conv5_1 = weight_variable_fromnp(npWeights["conv5_1_w"], "w_conv5_1")
                self.B_conv5_1 = weight_variable_fromnp(npWeights["conv5_1_b"], "b_conv5_1")
                self.W_conv5_2 = weight_variable_fromnp(npWeights["conv5_2_w"], "w_conv5_2")
                self.B_conv5_2 = weight_variable_fromnp(npWeights["conv5_2_b"], "b_conv5_2")
                self.W_conv5_3 = weight_variable_fromnp(npWeights["conv5_3_w"], "w_conv5_3")
                self.B_conv5_3 = weight_variable_fromnp(npWeights["conv5_3_b"], "b_conv5_3")

                self.h_conv5_1 = tf.nn.relu(conv2d(self.h_pool4, self.W_conv5_1, "conv5_1") + self.B_conv5_1)
                self.h_conv5_2 = tf.nn.relu(conv2d(self.h_conv5_1, self.W_conv5_2, "conv5_2") + self.B_conv5_2)
                self.h_conv5_3 = tf.nn.relu(conv2d(self.h_conv5_2, self.W_conv5_3, "conv5_2") + self.B_conv5_3)

            with tf.name_scope("regionProposal"):
                self.W_convRP = weight_variable_xavier([3, 3, 512, 512], "w_convRP", conv=True)
                self.B_convRP = bias_variable([512], "b_convRP")

                self.h_convRP = tf.nn.relu(conv2d(self.h_conv5_4, self.W_convRP, "convRP") + self.B_convRP)

                #Here, 36 represents 4 coordinates of 9 boxes
                #We represent this with coordinates spinning fastest
                #so we can reshape the output 36 features as [..., 9, 4]
                self.W_regRP = weight_variable_xavier([1, 1, 512, 36], "w_regRP", conv=True)
                self.B_regRP = bias_variable([36], "b_regRP", 0)
                self.h_reg = conv2d(self.h_convRP, self.W_regRP, 'regRP') + self.B_regRP
                self.h_reshape_reg = tf.reshape(h_reg, [self.batchSize, 14, 14, 9, 4])

                #Here, 18 represents 2 classes (obj or no obj) of 9 boxes
                #We represent this with class spinning fastest
                #so we can reshape the output 36 features as [..., 9, 2]
                self.W_clsRP = weight_variable_xavier([1, 1, 512, 18], "w_clsRP", conv=True)
                self.B_clsRP = bias_variable([18], "b_clsRP", 0)
                self.h_cls = conv2d(self.h_convRP, self.W_clsRP, 'clsRP') + self.B_clsRP
                self.h_reshape_cls = tf.reshape(h_cls, [self.batchSize, 14, 14, 9, 2])
                self.h_softmax_cls = pixelSofmtax5d(self.h_reshape_cls)

            with tf.name_scope("Loss"):
                #Define loss
                self.clsLoss = tf.reduce_mean(-tf.reduce_sum(self.clsGt * tf.log(self.h_softmax_cls+self.epsilon), reduction_indices=[4]))

                self.regLoss = tf.reduce_mean(self.clsGt[:, :, :, :, 0] * smoothL1(self.h_reshape_reg - self.regGt))

                self.loss = clsLoss + self.regLossWeight * self.regLoss

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss)
                ##self.optimizerAll = tf.train.MomentumOptimizer(self.learningRate, momentum=self.beta1).minimize(self.loss)
                #self.optimizerPre = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                #        var_list=[
                #            self.W_gap,
                #            self.B_gap,
                #            ]
                #        )

        #Summaries
        tf.scalar_summary('loss', self.loss, name="lossSum")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('clsGT', self.clsGT, name="clsGT_vis")
        tf.histogram_summary('regGT', self.regGT, name="regGT_vis")
        #Conv layer histograms
        tf.histogram_summary('conv1_1', self.h_conv1_1, name="conv1_1_vis")
        tf.histogram_summary('conv1_2', self.h_conv1_2, name="conv1_2_vis")
        tf.histogram_summary('conv2_1', self.h_conv2_1, name="conv2_1_vis")
        tf.histogram_summary('conv2_2', self.h_conv2_2, name="conv2_2_vis")
        tf.histogram_summary('conv3_1', self.h_conv3_1, name="conv3_1_vis")
        tf.histogram_summary('conv3_2', self.h_conv3_2, name="conv3_2_vis")
        tf.histogram_summary('conv3_3', self.h_conv3_3, name="conv3_3_vis")
        tf.histogram_summary('conv4_1', self.h_conv4_1, name="conv4_1_vis")
        tf.histogram_summary('conv4_2', self.h_conv4_2, name="conv4_2_vis")
        tf.histogram_summary('conv4_3', self.h_conv4_3, name="conv4_3_vis")
        tf.histogram_summary('conv5_1', self.h_conv5_1, name="conv5_1_vis")
        tf.histogram_summary('conv5_2', self.h_conv5_2, name="conv5_2_vis")
        tf.histogram_summary('conv5_3', self.h_conv5_3, name="conv5_3_vis")
        tf.histogram_summary('convRP', self.h_convRP, name="convRP_vis")
        tf.histogram_summary('reg', self.h_reg, name="reg_vis")
        tf.histogram_summary('cls', self.h_softmax_cls, name="cls_vis")
        #Weight and bias hists
        tf.histogram_summary('w_conv1_1', self.W_conv1_1, name="w_conv1_1_vis")
        tf.histogram_summary('b_conv1_1', self.B_conv1_1, name="b_conv1_1_vis")
        tf.histogram_summary('w_conv1_2', self.W_conv1_2, name="w_conv1_2_vis")
        tf.histogram_summary('b_conv1_2', self.B_conv1_2, name="b_conv1_2_vis")
        tf.histogram_summary('w_conv2_1', self.W_conv2_1, name="w_conv2_1_vis")
        tf.histogram_summary('b_conv2_1', self.B_conv2_1, name="b_conv2_1_vis")
        tf.histogram_summary('w_conv2_2', self.W_conv2_2, name="w_conv2_2_vis")
        tf.histogram_summary('b_conv2_2', self.B_conv2_2, name="b_conv2_2_vis")
        tf.histogram_summary('w_conv3_1', self.W_conv3_1, name="w_conv3_1_vis")
        tf.histogram_summary('b_conv3_1', self.B_conv3_1, name="b_conv3_1_vis")
        tf.histogram_summary('w_conv3_2', self.W_conv3_2, name="w_conv3_2_vis")
        tf.histogram_summary('b_conv3_2', self.B_conv3_2, name="b_conv3_2_vis")
        tf.histogram_summary('w_conv3_3', self.W_conv3_3, name="w_conv3_3_vis")
        tf.histogram_summary('b_conv3_3', self.B_conv3_3, name="b_conv3_3_vis")
        tf.histogram_summary('w_conv4_1', self.W_conv4_1, name="w_conv4_1_vis")
        tf.histogram_summary('b_conv4_1', self.B_conv4_1, name="b_conv4_1_vis")
        tf.histogram_summary('w_conv4_2', self.W_conv4_2, name="w_conv4_2_vis")
        tf.histogram_summary('b_conv4_2', self.B_conv4_2, name="b_conv4_2_vis")
        tf.histogram_summary('w_conv4_3', self.W_conv4_3, name="w_conv4_3_vis")
        tf.histogram_summary('b_conv4_3', self.B_conv4_3, name="b_conv4_3_vis")
        tf.histogram_summary('w_conv5_1', self.W_conv5_1, name="w_conv5_1_vis")
        tf.histogram_summary('b_conv5_1', self.B_conv5_1, name="b_conv5_1_vis")
        tf.histogram_summary('w_conv5_2', self.W_conv5_2, name="w_conv5_2_vis")
        tf.histogram_summary('b_conv5_2', self.B_conv5_2, name="b_conv5_2_vis")
        tf.histogram_summary('w_conv5_3', self.W_conv5_3, name="w_conv5_3_vis")
        tf.histogram_summary('b_conv5_3', self.B_conv5_3, name="b_conv5_3_vis")
        tf.histogram_summary('w_convRP', self.W_convRP, name="w_convRP_vis")
        tf.histogram_summary('b_convRP', self.B_convRP, name="b_convRP_vis")
        tf.histogram_summary('w_regRP', self.W_regRP, name="w_regRP_vis")
        tf.histogram_summary('b_regRP', self.B_regRP, name="b_regRP_vis")
        tf.histogram_summary('w_clsRP', self.W_clsRP, name="w_clsRP_vis")
        tf.histogram_summary('b_clsRP', self.B_clsRP, name="b_clsRP_vis")

    def getLoadVars(self):
        v = tf.all_variables()
        #return [var for var in v if (not "gap" in var.name) and (not "GAP" in var.name) ]
        return v

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        #Define session
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1]}
            #Run optimizer
            if(self.preTrain):
                self.sess.run(self.optimizerPre, feed_dict=feedDict)
            else:
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
        #if(plot):
        #    filename = self.plotDir + "train_" + str(self.timestep)
        #    self.evalAndPlotBB(feedDict, filename)

    #def evalAndPlotBB(self, feedDict, prefix):
    #    print "Plotting"
    #    #We need feed_dict here
    #    cam = self.sess.run(self.vis_cam, feed_dict=feedDict)
    #    img = feedDict[self.inputImage]
    #    if self.gt in feedDict:
    #        gt = feedDict[self.gt]
    #    else:
    #        gt = None
    #    camIdxs = self.sess.run(self.eval_idx, feed_dict=feedDict)
    #    camVals = self.sess.run(self.eval_vals, feed_dict=feedDict)
    #    np_w_gap = self.sess.run(self.W_gap, feed_dict=feedDict)
    #    plotDetCam(prefix, img, gt, cam, camIdxs, camVals, self.idxToName, np_w_gap)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            numGt = inGt.shape[0]
            assert(numData == numGt)
            feedDict = {self.inputImage: inData, self.gt: inGt}
        else:
            feedDict = {self.inputImage: inData}

        outVals = self.vis_cam.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)

        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            self.evalAndPlotCam(feedDict, filename)

        return outVals

