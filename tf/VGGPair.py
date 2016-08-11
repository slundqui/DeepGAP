import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
from base import TFObj
#import matplotlib.pyplot as plt

class VGGPair(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(VGGPair, self).loadParams(params)

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
                self.gt = node_variable([self.batchSize, self.numClasses], "gt")
                self.notGt = 1-self.gt

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
                self.h_pool5 = maxpool_2x2(self.h_conv5_3, "pool5")

        #with tf.device('cpu:0'):

            self.keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope("FC6"):
                self.W_fc6 = weight_variable_fromnp(npWeights["fc6_w"], "w_fc6")
                self.B_fc6 = weight_variable_fromnp(npWeights["fc6_b"], "b_fc6")
                h_pool5_flat = tf.reshape(self.h_pool5, [self.batchSize, 7*7*512])
                self.h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, self.W_fc6, name="fc6") + self.B_fc6, "fc6_relu")
                self.drop_h_fc6 = tf.nn.dropout(self.h_fc6, self.keep_prob)

        #with tf.device(self.device):
            with tf.name_scope("FC7"):
                self.W_fc7 = weight_variable_fromnp(npWeights["fc7_w"], "w_fc7")
                self.B_fc7 = weight_variable_fromnp(npWeights["fc7_b"], "b_fc7")
                self.h_fc7 = tf.nn.relu(tf.matmul(self.drop_h_fc6, self.W_fc7, name="fc7") + self.B_fc7, "fc7_relu")
                self.drop_h_fc7 = tf.nn.dropout(self.h_fc7, self.keep_prob)

            with tf.name_scope("FC8"):
                self.W_fc8 = weight_variable_xavier([4096, 20], "w_fc8")
                self.B_fc8 = bias_variable([20], "b_fc8")
                #self.est = tf.nn.softmax(tf.matmul(self.drop_h_fc7, self.W_fc8, name="fc8") + self.B_fc8)
                self.est = tf.matmul(self.drop_h_fc7, self.W_fc8, name="fc8") + self.B_fc8

            with tf.name_scope("Loss"):
                #Grab positive and negative classes as needed
                self.posArray = tf.expand_dims(self.gt*self.est, dim=1)
                self.negArray = tf.expand_dims(self.notGt*self.est, dim=2)
                #tf broadcasting should take care of this to make this a [batch, class, class] matrix
                self.cSum = (1 - self.posArray) + self.negArray
                self.pairLoss = tf.reduce_mean(self.cSum, reduction_indices=[1, 2])
                self.loss = tf.reduce_mean(tf.nn.relu(self.pairLoss))

                #self.loss = tf.reduce_mean(-tf.reduce_sum(self.gt * tf.log(self.est+self.epsilon), reduction_indices=[1]))
                #self.regLoss = self.loss + self.regStrength * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                #self.nan_check_loss = tf.verify_tensor_all_finite(self.loss, msg="check_nan")

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss)
                #self.optimizerAll = tf.train.MomentumOptimizer(self.learningRate, momentum=self.beta1).minimize(self.loss)
                self.optimizerPre = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.W_fc6,
                            self.B_fc6,
                            self.W_fc7,
                            self.B_fc7,
                            self.W_fc8,
                            self.B_fc8,
                            ]
                        )

            with tf.name_scope("Metric"):
                self.correct = tf.equal(tf.argmax(self.gt, 1), tf.argmax(self.est, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        #Cannot be on GPU
        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.est, k=5)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="lossSum")
        tf.scalar_summary('accuracy', self.accuracy, name="accSum")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('gt', self.gt, name="gt_vis")
        tf.histogram_summary('notGt', self.notGt, name="notGt_vis")
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
        tf.histogram_summary('fc6', self.h_fc6, name="fc6_vis")
        tf.histogram_summary('fc7', self.h_fc7, name="fc7_vis")
        tf.histogram_summary('est', self.est, name="est_vis")
        tf.histogram_summary('posArray', self.posArray, name="est_vis")
        tf.histogram_summary('negArray', self.negArray, name="est_vis")
        tf.histogram_summary('cSum', self.cSum, name="est_vis")
        tf.histogram_summary('pairLoss', self.pairLoss, name="est_vis")
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
        tf.histogram_summary('w_fc6', self.W_fc6, name="w_fc6_vis")
        tf.histogram_summary('b_fc6', self.B_fc6, name="b_fc6_vis")
        tf.histogram_summary('w_fc7', self.W_fc7, name="w_fc7_vis")
        tf.histogram_summary('b_fc7', self.B_fc7, name="b_fc7_vis")
        tf.histogram_summary('w_fc8', self.W_fc7, name="w_fc8_vis")
        tf.histogram_summary('b_fc8', self.B_fc7, name="b_fc8_vis")


    def getLoadVars(self):
        v = tf.all_variables()
        #return [var for var in v if ("fc8" in var.name)]
        return v


    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        #Define session
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1], self.keep_prob: .5}
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

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            numGt = inGt.shape[0]
            assert(numData == numGt)
            feedDict = {self.inputImage: inData, self.gt: inGt, self.keep_prob: 1}
        else:
            feedDict = {self.inputImage: inData, self.keep_prob: 1}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)

        return outVals

