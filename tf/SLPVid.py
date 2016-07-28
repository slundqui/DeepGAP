import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
from base import TFObj
#import matplotlib.pyplot as plt

class SLPVid(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(SLPVid, self).loadParams(params)

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]], "inputImage")
                self.gt = node_variable([self.batchSize, 1, 2, 2, self.numClasses], "gt")
                self.norm_gt = self.gt/tf.reduce_sum(self.gt, reduction_indices=4, keep_dims=True)

            with tf.name_scope("Pool"):
                yPool = inputShape[1]/2
                xPool = inputShape[2]/2
                #We pad inputPooled to get to gt temporal shape of 7
                #self.padInput = tf.pad(self.inputImage, [[0, 0], [1, 2], [0, 0], [0, 0], [0, 0]])
                #Pool over spatial dimensions to be 2x2
                self.inputPooled = tf.nn.max_pool3d(self.inputImage, ksize=[1, 1, yPool, xPool, 1], strides=[1, 1, yPool, xPool, 1], padding="SAME")

                self.weight = weight_variable_xavier([4, 1, 1, inputShape[3], self.numClasses], "weight")
                self.bias = bias_variable([self.numClasses], "bias" )

                self.h_conv = tf.nn.conv3d(self.inputPooled, self.weight, [1, 1, 1, 1, 1], padding="VALID") + self.bias
                self.cam = tf.nn.conv3d(self.inputImage, self.weight, [1, 1, 1, 1, 1], padding="VALID") + self.bias

                #Reshape batch and time together
                #self.reshape_cam = tf.transpose(tf.reshape(self.cam, [self.batchSize*7, 16, 32, 31]), [0, 3, 1, 2])
                self.reshape_cam = tf.transpose(tf.reshape(self.cam, [self.batchSize, 16, 32, 31]), [0, 3, 1, 2])

                #Get ranking from h_conv
                self.classRank = tf.reduce_mean(self.reshape_cam, reduction_indices=[2, 3])

                self.est = pixelSoftmax5d(self.h_conv)

            with tf.name_scope("Loss"):
                #Define loss
                self.loss = tf.reduce_mean(-tf.reduce_sum(self.norm_gt * tf.log(self.est+self.epsilon), reduction_indices=4))

            with tf.name_scope("Opt"):
                #Define optimizer
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss)

        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.classRank, k=5)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="lossSum")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('norm_gt', self.gt, name="gt_vis")
        #Conv layer histograms
        tf.histogram_summary('h_conv', self.h_conv, name="conv1_vis")
        tf.histogram_summary('est', self.est, name="est_vis")
        #Weight and bias hists
        tf.histogram_summary('weight', self.weight, name="weight_vis")
        tf.histogram_summary('bias', self.bias, name="bias_vis")


    def getLoadVars(self):
        v = tf.all_variables()
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
            self.evalAndPlotCam(feedDict, data[2], filename)

    def evalAndPlotCam(self, feedDict, img, prefix):
        print "Plotting"
        #We need feed_dict here
        cam = self.sess.run(self.reshape_cam, feed_dict=feedDict)
        if self.gt in feedDict:
            gt = feedDict[self.gt]
        else:
            gt = None
        camIdxs = self.sess.run(self.eval_idx, feed_dict=feedDict)
        camVals = self.sess.run(self.eval_vals, feed_dict=feedDict)
        #We conlidate the time and batch dim into 1
        (batch, t, y, x, f) = img.shape
        reshape_img = np.reshape(img, [batch*t, y, x, f])
        (batch, t, y, x, f) = gt.shape
        reshape_gt = np.reshape(gt, [batch*t, y, x, f])

        plotDetCam(prefix, reshape_img, reshape_gt, cam, camIdxs, camVals, self.idxToName, distIdx=0)


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

