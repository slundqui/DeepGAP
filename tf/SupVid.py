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

class SupVid(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(SupVid, self).loadParams(params)

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.learningRateBias = params['learningRateBias']
        self.lossWeight = params['lossWeight']

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        #TODO add this as argument to buildModel
        gtShape = [1, 2, 4, 31]

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]], "inputImage")
                #self.gt = node_variable([self.batchSize, 1, 8, 16, self.numClasses], "gt")

                #Input shape should be [batch, 7, 64, 128, 3]
                self.inputImage = node_variable((self.batchSize,)+inputShape, "inputImage")
                self.padInput = tf.pad(self.inputImage, [[0, 0], [0, 0], [7, 7], [15, 15], [0, 0]])

                self.gtIndices = tf.placeholder("int64", [2, None], "gtIndices")
                self.gtValues = node_variable([None], "gtValues")

                self.pre_gt = tf.sparse_tensor_to_dense(tf.SparseTensor(
                        tf.transpose(self.gtIndices, [1, 0]),
                        self.gtValues,
                        [self.batchSize*gtShape[0], gtShape[1]*gtShape[2]*gtShape[3]]
                        ))
                self.all_gt = tf.reshape(self.pre_gt, [self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]])
                self.gt = self.all_gt[:, :, :, :, 0:8]

                #self.norm_gt = self.gt/tf.reduce_sum(self.gt, reduction_indices=4, keep_dims=True)

            with tf.name_scope("Hidden"):
                self.h_weight = weight_variable_xavier([4, 16, 32, 3, 3072], "hidden_weight")
                self.h_bias = bias_variable([3072], "hidden_bias")
                self.h_hidden= tf.nn.relu(tf.nn.conv3d(self.padInput, self.h_weight, [1, 1, 4, 4, 1], padding="VALID") + self.h_bias)

            with tf.name_scope("Pool"):
                #Pool over spatial dimensions to be 2x2
                self.inputPooled = tf.nn.max_pool3d(self.h_hidden, ksize=[1, 1, 8, 8, 1], strides=[1, 1, 8, 8, 1], padding="SAME")

                self.camPooled = tf.nn.max_pool3d(self.h_hidden, ksize=[1, 1, 8, 8, 1], strides=[1, 1, 1, 1, 1], padding="SAME")

                self.weight = weight_variable_xavier([4, 1, 1, 3072, self.numClasses], "weight")
                self.bias = bias_variable([self.numClasses], "bias" )

                self.h_conv = tf.nn.conv3d(self.inputPooled, self.weight, [1, 1, 1, 1, 1], padding="VALID") + self.bias
                self.cam = tf.nn.conv3d(self.camPooled, self.weight, [1, 1, 1, 1, 1], padding="VALID") + self.bias

                #Reshape batch and time together
                #self.reshape_cam = tf.transpose(tf.reshape(self.cam, [self.batchSize*7, 16, 32, 31]), [0, 3, 1, 2])
                self.reshape_cam = tf.transpose(tf.reshape(self.cam, [self.batchSize, 16, 32, self.numClasses]), [0, 3, 1, 2])

                #Get ranking from h_conv
                self.classRank = tf.reduce_mean(self.reshape_cam, reduction_indices=[2, 3])

                self.est = pixelSoftmax5d(self.h_conv)
                #self.est = self.h_conv

            with tf.name_scope("Loss"):
                self.flat_gt = tf.reshape(self.gt, [-1, self.numClasses])
                self.flat_est = tf.reshape(self.est, [-1, self.numClasses])

                gtClass = tf.argmax(self.flat_gt, 1)
                estClass = tf.argmax(self.flat_est, 1)
                correct = tf.equal(gtClass, estClass)
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

                self.classF1 = []
                for c in range(self.numClasses):
                    classGT = tf.equal(gtClass, c)
                    classEst = tf.equal(estClass, c)
                    classTP = tf.reduce_sum(tf.cast(tf.logical_and(classGT, classEst), tf.float32))
                    classFP = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(classGT), classEst), tf.float32))
                    classFN = tf.reduce_sum(tf.cast(tf.logical_and(classGT, tf.logical_not(classEst)), tf.float32))

                    precision = classTP/(classTP+classFP+self.epsilon)
                    recall = classTP/(classTP+classFN+self.epsilon)
                    self.classF1.append((2*precision*recall)/(precision+recall+self.epsilon))

                self.loss = tf.reduce_mean(-tf.reduce_sum(self.lossWeight[0:8] * self.gt* tf.log(self.est+self.epsilon), reduction_indices=4))
                #self.loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.gt - self.est), reduction_indices=[1, 2, 3, 4]))

            with tf.name_scope("Opt"):
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.h_weight,
                            self.weight,
                        ]
                        )
                self.optimizerBias = tf.train.GradientDescentOptimizer(self.learningRateBias).minimize(self.loss,
                        var_list=[
                            self.h_bias,
                            self.bias,
                        ]
                        )

                self.optimizerPre = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.weight,
                        ]
                        )
                self.optimizerPreBias = tf.train.GradientDescentOptimizer(self.learningRateBias).minimize(self.loss,
                        var_list=[
                            self.bias,
                        ]
                        )

        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.classRank, k=5)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="accuracy")
        tf.scalar_summary('accuracy', self.accuracy, name="accuracy")
        for c in range(self.numClasses):
            className = self.idxToName[c]
            tf.scalar_summary(className+' F1', self.classF1[c])

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('inputPooled', self.inputPooled, name="image_vis")
        tf.histogram_summary('gt', self.gt, name="gt_vis")
        #Conv layer histograms
        tf.histogram_summary('h_conv', self.h_conv, name="conv1_vis")
        tf.histogram_summary('h_hidden', self.h_hidden, name="hidden_vis")
        tf.histogram_summary('est', self.est, name="est_vis")
        #Weight and bias hists
        tf.histogram_summary('h_weight', self.h_weight, name="weight_vis")
        tf.histogram_summary('h_bias', self.h_bias, name="bias_vis")
        tf.histogram_summary('weight', self.weight, name="weight_vis")
        tf.histogram_summary('bias', self.bias, name="bias_vis")


    def getLoadVars(self):
        v = tf.all_variables()
        return v

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            (gtOutY, gtOutX, gtVals) = sp.find(data[1])
            feedDict = {self.inputImage:data[0],
                        self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals}

            #feedDict = {self.inputImage: data[0], self.gt: data[1]}
            #Run optimizer
            if(self.preTrain):
                self.sess.run(self.optimizerPre, feed_dict=feedDict)
                self.sess.run(self.optimizerPreBias, feed_dict=feedDict)
            else:
                self.sess.run(self.optimizerAll, feed_dict=feedDict)
                self.sess.run(self.optimizerBias, feed_dict=feedDict)

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
            gtShape = dataObj.gtShape
            gt = np.reshape(data[1].toarray(), (self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]))
            self.evalAndPlotCam(feedDict, data, gt, filename)

    def evalAndPlotCam(self, feedDict, data, gt, prefix):
        print "Plotting"

        #We need feed_dict here
        cam = self.sess.run(self.reshape_cam, feed_dict=feedDict)

        img = data[2]
        camIdxs = self.sess.run(self.eval_idx, feed_dict=feedDict)
        camVals = self.sess.run(self.eval_vals, feed_dict=feedDict)
        #We conlidate the time and batch dim into 1
        (batch, imgT, imgY, imgX, imgF) = img.shape
        (batch, gtT, gtY, gtX, gtF) = gt.shape
        reshape_img = np.reshape(img, [batch*imgT, imgY, imgX, imgF])
        reshape_gt = np.reshape(gt, [batch*gtT, gtY, gtX, gtF])

        plotDetCam(prefix, reshape_img, reshape_gt, cam, camIdxs, camVals, self.idxToName, distIdx=0)

    def runModel(self, trainDataObj, testDataObj=None):
        for i in range(self.outerSteps):
           #Plot flag
           if(i%self.plotPeriod == 0):
               plot = True
           else:
               plot = False
           if(testDataObj):
               #Evaluate test frame, providing gt so that it writes to summary
               (evalData, gtData, imgData) = testDataObj.getData(self.batchSize)
               self.evalModel(evalData, gtData, imgData, testDataObj.gtShape, plot=plot)
               print "Done test eval"
           #Train
           if(i%self.savePeriod == 0):
               self.trainModel(trainDataObj, save=True, plot=plot)
           else:
               self.trainModel(trainDataObj, save=False, plot=plot)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt, inImg, gtShape, plot=True):

        if(inGt != None):
            (gtOutY, gtOutX, gtVals) = sp.find(inGt)
            feedDict = {self.inputImage:inData,
                        self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals}
        else:
            feedDict ={self.inputImage:inData}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            gt = np.reshape(inGt.toarray(), (self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]))
            data = (inData, inGt, inImg)
            self.evalAndPlotCam(feedDict, data, gt, filename)

        return outVals

