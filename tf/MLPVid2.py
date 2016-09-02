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

class MLPVid2(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(MLPVid2, self).loadParams(params)

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.learningRateBias = params['learningRateBias']
        self.lossWeight = params['lossWeight']
        self.gtShape = params['gtShape']
        self.gtSparse = params['gtSparse']
        self.inputScale = params['inputScale']
        self.regWeight = params['regWeight']
        self.resLoad = params['resLoad']

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]], "inputImage")
                #self.gt = node_variable([self.batchSize, 1, 8, 16, self.numClasses], "gt")

                #We represent inputImage and gt as sparse matrices, with indices/values
                self.dataIndices = tf.placeholder("int64", [2, None], "dataIndices")
                self.dataValues = node_variable([None], "dataValues")

                self.pre_inputImage = tf.sparse_tensor_to_dense(tf.SparseTensor(
                        tf.transpose(self.dataIndices, [1, 0]),
                        self.dataValues,
                        [self.batchSize*inputShape[0], inputShape[1]*inputShape[2]*inputShape[3]]
                        ), validate_indices=False)

                self.inputImage = self.inputScale * tf.reshape(self.pre_inputImage, [self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]])

                if(self.gtSparse):
                    self.gtIndices = tf.placeholder("int64", [2, None], "gtIndices")
                    self.gtValues = node_variable([None], "gtValues")

                    self.pre_gt = tf.sparse_tensor_to_dense(tf.SparseTensor(
                            tf.transpose(self.gtIndices, [1, 0]),
                            self.gtValues,
                            [self.batchSize*self.gtShape[0], self.gtShape[1]*self.gtShape[2]*self.gtShape[3]]
                            ), validate_indices=False)
                    self.gt = tf.reshape(self.pre_gt, [self.batchSize, self.gtShape[0], self.gtShape[1], self.gtShape[2], self.gtShape[3]])
                else:
                    self.gt=tf.placeholder("float32", [self.batchSize, self.gtShape[0], self.gtShape[1], self.gtShape[2], self.gtShape[3]])

                self.select_gt = tf.squeeze(self.gt[:, :, :, :, 0:self.numClasses], squeeze_dims=[1])

                #self.norm_gt = self.gt/tf.reduce_sum(self.gt, reduction_indices=4, keep_dims=True)

            with tf.name_scope("conv1"):
                yPool = 2
                xPool = 2
                self.timePooled = tf.reduce_max(self.inputImage, reduction_indices=1)
                self.inputPooled = tf.nn.max_pool(self.timePooled, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")

                self.conv1_w = tf.Variable(tf.zeros([1, 1, inputShape[3], inputShape[3]]), "conv1_w")
                self.conv1_b = bias_variable([3072], "conv1_b")

                self.h_res1 = tf.nn.relu(tf.nn.conv2d(self.inputPooled, self.conv1_w, [1, 1, 1, 1], padding="SAME") + self.conv1_b)
                self.h_conv1 = tf.nn.relu(self.inputPooled + self.h_res1)

            with tf.name_scope("reg1"):
                self.keep_prob = tf.placeholder(tf.float32)
                self.h_dropout1 = tf.nn.dropout(self.h_conv1, self.keep_prob)

            with tf.name_scope("conv2"):
                yPool = 2
                xPool = 2

                self.h_conv1_pool= tf.nn.max_pool(self.h_dropout1, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")

                self.conv2_w = weight_variable_xavier([1, 1, inputShape[3], inputShape[3]], "conv2_w")
                self.conv2_b = bias_variable([inputShape[3]], "conv2_b" )

                self.h_res2 = tf.nn.relu(tf.nn.conv2d(self.h_conv1_pool, self.conv2_w, [1, 1, 1, 1], padding="SAME") + self.conv2_b)
                self.h_conv2 = tf.nn.relu(self.h_conv1_pool + self.h_res2)

            with tf.name_scope("reg2"):
                self.h_dropout2 = tf.nn.dropout(self.h_conv2, self.keep_prob)

            with tf.name_scope("conv3"):
                yPool = int(np.ceil(float(inputShape[1])/(self.gtShape[1] * 4)))
                xPool = int(np.ceil(float(inputShape[2])/(self.gtShape[2] * 4)))

                self.h_conv2_pool= tf.nn.max_pool(self.h_dropout2, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")
                self.camPooled = tf.nn.max_pool(self.h_dropout2, ksize=[1, yPool, xPool, 1], strides=[1, 1, 1, 1], padding="SAME")

                self.conv3_w = weight_variable_xavier([1, 1, inputShape[3], self.numClasses], "weight")
                self.conv3_b = bias_variable([self.numClasses], "bias" )

                self.h_conv3 = tf.nn.conv2d(self.h_conv2_pool, self.conv3_w, [1, 1, 1, 1], padding="SAME") + self.conv3_b

                #We evaluate pooling with smaller stride here
                self.cam = tf.nn.conv2d(self.camPooled, self.conv3_w, [1, 1, 1, 1], padding="SAME") + self.conv3_b

                #Reshape batch and time together
                #self.reshape_cam = tf.transpose(tf.reshape(self.cam, [self.batchSize*7, 16, 32, 31]), [0, 3, 1, 2])
                self.reshape_cam = tf.transpose(self.cam, [0, 3, 1, 2])

                #Get ranking from h_conv
                self.classRank = tf.reduce_mean(self.reshape_cam, reduction_indices=[2, 3])

                self.est = pixelSoftmax(self.h_conv3)
                #self.est = self.h_conv

            with tf.name_scope("Loss"):
                self.flat_gt = tf.reshape(self.select_gt, [-1, self.numClasses])
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

                self.weightRegLoss = tf.reduce_sum(tf.square(self.conv3_w)) + tf.reduce_sum(tf.square(self.conv2_w)) + tf.reduce_sum(tf.square(self.conv1_w))

                if(self.lossWeight == None):
                    self.loss = tf.reduce_mean(-tf.reduce_sum(self.select_gt * tf.log(self.est+self.epsilon), reduction_indices=3)) + self.regWeight * self.weightRegLoss
                else:
                    self.loss = tf.reduce_mean(-tf.reduce_sum(self.lossWeight[0:self.numClasses] * self.select_gt * tf.log(self.est+self.epsilon), reduction_indices=3)) + self.regWeight * self.weightRegLoss

            with tf.name_scope("Opt"):
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.conv1_w,
                            self.conv2_w,
                            self.conv3_w,
                            #self.beta,
                            #self.gamma
                        ]
                        )
                self.optimizerBias = tf.train.GradientDescentOptimizer(self.learningRateBias).minimize(self.loss,
                        var_list=[
                            self.conv1_b,
                            self.conv2_b,
                            self.conv3_b,
                        ]
                        )

        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.classRank, k=5)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="loss")
        tf.scalar_summary('accuracy', self.accuracy, name="accuracy")
        for c in range(self.numClasses):
            className = self.idxToName[c]
            tf.scalar_summary(className+' F1', self.classF1[c])

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('inputPooled', self.inputPooled, name="image_vis")
        tf.histogram_summary('gt', self.select_gt, name="gt_vis")
        #Conv layer histograms
        tf.histogram_summary('h_conv1', self.h_conv1, name="conv1_vis")
        #tf.histogram_summary('h_norm_conv1', self.h_norm_conv1, name="norm_conv1_vis")
        tf.histogram_summary('h_conv2', self.h_conv2, name="conv2_vis")
        tf.histogram_summary('h_conv3', self.h_conv3, name="conv3_vis")
        tf.histogram_summary('est', self.est, name="est_vis")
        #Weight and bias hists

        tf.histogram_summary('conv1_w', self.conv1_w, name="conv1_w_vis")
        tf.histogram_summary('conv1_b', self.conv1_b, name="conv1_b_vis")
        tf.histogram_summary('conv2_w', self.conv2_w, name="conv2_w_vis")
        tf.histogram_summary('conv2_b', self.conv2_b, name="conv2_b_vis")
        tf.histogram_summary('conv3_w', self.conv2_w, name="conv3_w_vis")
        tf.histogram_summary('conv3_b', self.conv2_b, name="conv3_b_vis")


    def getLoadVars(self):
        v = tf.all_variables()
        if(self.resLoad):
            v = [var for var in v if ("weight" in var.name) or ("bias" in var.name)]
        return v

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            (dataOutY, dataOutX, dataVals) = sp.find(data[0])

            if(self.gtSparse):
                (gtOutY, gtOutX, gtVals) = sp.find(data[1])
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                            self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals,
                            self.keep_prob:.5
                            }
            else:
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                        self.gt:data[1],
                        self.keep_prob:.5
                        }

            #feedDict = {self.inputImage: data[0], self.gt: data[1]}
            #Run optimizer
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
            if(self.gtSparse):
                gt = np.reshape(data[1].toarray(), (self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]))
            else:
                gt = data[1]
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
        (dataOutY, dataOutX, dataVals) = sp.find(inData)
        if(inGt != None):
            if(self.gtSparse):
                (gtOutY, gtOutX, gtVals) = sp.find(inGt)
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                        self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals,
                        self.keep_prob:1.0
                        }
            else:
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                        self.gt:inGt,
                       self.keep_prob:1.0
                       }
        else:
            feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                    self.keep_prob:1.0
                    }

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            if(self.gtSparse):
                gt = np.reshape(inGt.toarray(), (self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]))
            else:
                gt = inGt
            data = (inData, inGt, inImg)
            self.evalAndPlotCam(feedDict, data, gt, filename)

        return outVals

