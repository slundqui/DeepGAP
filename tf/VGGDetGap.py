import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
from base import TFObj
#import matplotlib.pyplot as plt

class VGGDetGap(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(VGGDetGap, self).loadParams(params)

        self.vggFile = params['vggFile']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.regStrength = params['regStrength']

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
                self.gt = node_variable([self.batchSize, 7, 7, self.numClasses], "gt")
                self.norm_gt = self.gt/tf.reduce_sum(self.gt, reduction_indices=3, keep_dims=True)
                #Model variables for convolutions

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
                self.h_pool5 = tf.nn.avg_pool(self.h_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")

            #16 comes from 4 2x2 pooling
            #self.h_conv5_shape = [self.batchSize, inputShape[0]/32, inputShape[1]/32, 512]
            self.h_conv5_shape = [self.batchSize, 7, 7, 512]
            with tf.name_scope("GAP"):
                self.h_gap = tf.reduce_mean(self.h_pool5, reduction_indices=[1, 2])
                self.W_gap = weight_variable_xavier([512, self.numClasses], "w_gap", conv=False)
                self.B_gap = bias_variable([self.numClasses], "b_gap")
                self.est = tf.nn.softmax(tf.matmul(self.h_gap, self.W_gap)+self.B_gap)

            with tf.name_scope("CAM"):
                self.h_reshape_gap = tf.reshape(self.h_pool5, [self.batchSize*self.h_conv5_shape[1]*self.h_conv5_shape[2], -1])
                self.flat_cam = tf.matmul(self.h_reshape_gap, self.W_gap) + self.B_gap
                self.reshape_cam = tf.reshape(self.flat_cam, [self.batchSize, self.h_conv5_shape[1], self.h_conv5_shape[2], -1])
                self.softmax_cam = pixelSoftmax(self.reshape_cam)
                self.vis_cam = tf.transpose(self.softmax_cam, [0, 3, 1, 2])
                self.classRank = tf.reduce_mean(self.vis_cam, reduction_indices=[2, 3])

            with tf.name_scope("Loss"):
                #Define loss
                #self.loss = tf.reduce_mean(-tf.reduce_sum(self.gt * tf.log(self.est+self.epsilon), reduction_indices=[1]))
                self.loss = tf.reduce_mean(-tf.reduce_sum(self.norm_gt * tf.log(self.softmax_cam+self.epsilon), reduction_indices=[3]))
                self.regLoss = self.loss + self.regStrength * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                self.nan_check_loss = tf.verify_tensor_all_finite(self.loss, msg="check_nan")

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss)
                #self.optimizerAll = tf.train.MomentumOptimizer(self.learningRate, momentum=self.beta1).minimize(self.loss)
                self.optimizerPre = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.W_gap,
                            self.B_gap,
                            ]
                        )

            with tf.name_scope("Metric"):
                self.correct = tf.equal(tf.argmax(tf.reduce_mean(self.gt, reduction_indices=[1, 2]), 1), tf.argmax(self.est, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        #Cannot be on GPU
        #Remove distractor class
        rankNoDist = self.classRank[:, :self.numClasses-1]
        (self.eval_vals, self.eval_idx) = tf.nn.top_k(rankNoDist, k=5)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="lossSum")
        tf.scalar_summary('accuracy', self.accuracy, name="accSum")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('gt', self.gt, name="gt_vis")
        tf.histogram_summary('norm_gt', self.norm_gt, name="norm_gt_vis")
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
        tf.histogram_summary('cam', self.flat_cam, name="flat_cam")
        tf.histogram_summary('softmax_cam', self.softmax_cam, name="softmax_cam")
        tf.histogram_summary('gap', self.h_gap, name="gap_vis")
        tf.histogram_summary('est', self.est, name="est_vis")
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
        tf.histogram_summary('w_gap', self.W_gap, name="w_gap_vis")
        tf.histogram_summary('b_gap', self.B_gap, name="b_gap_vis")

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
        if(plot):
            filename = self.plotDir + "train_" + str(self.timestep)
            self.evalAndPlotCam(feedDict, filename)

    def evalAndPlotCam(self, feedDict, prefix):
        print "Plotting"
        #We need feed_dict here
        cam = self.sess.run(self.vis_cam, feed_dict=feedDict)
        img = feedDict[self.inputImage]
        if self.gt in feedDict:
            gt = feedDict[self.gt]
        else:
            gt = None
        camIdxs = self.sess.run(self.eval_idx, feed_dict=feedDict)
        camVals = self.sess.run(self.eval_vals, feed_dict=feedDict)
        np_w_gap = self.sess.run(self.W_gap, feed_dict=feedDict)
        plotDetCam(prefix, img, gt, cam, camIdxs, camVals, self.idxToName)

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

        outVals = self.eval_vals.eval(feed_dict=feedDict, session=self.sess)
        outIdx = self.eval_idx.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)

        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            self.evalAndPlotCam(feedDict, filename)
        return (outVals, outIdx)

    #Evaluates inData, but in batchSize batches for memory efficiency
    #If an inGt is provided, will calculate summary as test set
    def evalModelBatch(self, inData, inGt=None):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)

        #Split up numData into batchSize and evaluate est data
        tfInVals = np.zeros((self.batchSize, ny, nx, nf))

        outVals = np.zeros((numData, 5))
        outIdx = np.zeros((numData, 5))

        #Ceil of numData/batchSize
        numIt = int(numData/self.batchSize) + 1

        #Only write summary on first it

        startOffset = 0
        for it in range(numIt):
            print it, " out of ", numIt
            #Calculate indices
            startDataIdx = startOffset
            endDataIdx = startOffset + self.batchSize
            startTfValIdx = 0
            endTfValIdx = self.batchSize

            #If out of bounds
            if(endDataIdx >= numData):
                #Calculate offset
                offset = endDataIdx - numData
                #Set endDataIdx to max value
                endDataIdx = numData
                #Set endTfValIdx to less than max value
                endTfValIdx -= offset

            tfInVals[startTfValIdx:endTfValIdx, :, :, :] = inData[startDataIdx:endDataIdx, :, :, :]

            (oVal, oIdx) = self.evalModel(tfInVals, inGt, plot=False)
            outVals[startDataIdx:endDataIdx, :] = oVal
            outIdx[startDataIdx:endDataIdx, :] = oIdx

            if(inGt != None and it == 0):
                tfInGt = inGt[startDataIdx:endDataIdx, :]
                summary = self.sess.run(self.mergedSummary, feed_dict={self.inputImage: tfInVals, self.gt: tfInGt})
                self.test_writer.add_summary(summary, self.timestep)

            startOffset += self.batchSize
        #Return output data
        return (outVals, outIdx)

