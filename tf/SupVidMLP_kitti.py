import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
from plot.plotFeaturemaps import plotFeaturemaps
from plot.plotWeights import plot_weights
from base import TFObj
import scipy.sparse as sp
#import matplotlib.pyplot as plt

class SupVidMLP_kitti(TFObj):
    def makeDirs(self):
        super(SupVidMLP_kitti, self).makeDirs()
        makeDir(self.weightDir)
        makeDir(self.featureMapDir)

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(SupVidMLP_kitti, self).loadParams(params)

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.learningRateBias = params['learningRateBias']
        self.lossWeight = params['lossWeight']
        self.gtShape = params['gtShape']
        self.gtSparse = params['gtSparse']
        self.regWeight = params['regWeight']
        self.resLoad = params['resLoad']
        self.stereo = params['stereo']
        self.time = params['time']
        self.numFeatures = params['numFeatures']
        self.plotInd = params['plotInd']
        self.plotFM = params['plotFM']

        self.weightDir = self.plotDir + "/weight/"
        self.featureMapDir = self.plotDir + "/featuremap/"

    def defineVars(self):
        #Define all variables outside of scope
        if(self.stereo):
            numInputFeatures = 6
        else:
            numInputFeatures = 3
        if(self.time):
            numTimeInputs = 2
        else:
            numTimeInputs = 1

        self.h_weight = weight_variable_xavier([numTimeInputs, 15, 32, numInputFeatures, self.numFeatures], "V1_W")
        self.h_bias = bias_variable([self.numFeatures], "hidden_bias")
        self.conv1_w = weight_variable([1, 1, self.numFeatures, self.numFeatures], "conv1_w", 1e-6)
        self.conv1_b = weight_variable([self.numFeatures], "conv1_b", 1e-6)
        self.class_weight = weight_variable_xavier([1, 1, self.numFeatures, self.numClasses], "class_weight")
        self.class_bias = bias_variable([self.numClasses], "class_bias" )

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        #Running on GPU
        with tf.device(self.device):
            self.defineVars()
            with tf.name_scope("inputOps"):
                #self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]], "inputImage")
                #self.gt = node_variable([self.batchSize, 1, 8, 16, self.numClasses], "gt")

                self.inputImage = node_variable((self.batchSize,)+inputShape, "inputImage")
                if(self.stereo):
                    #We split the time dimension to stereo and concatenate with feature dim
                    numTime = inputShape[0]/2
                    self.reshapeImage = tf.reshape(self.inputImage,
                            [self.batchSize, numTime, 2, inputShape[1], inputShape[2], inputShape[3]])
                    self.permuteImage = tf.transpose(self.reshapeImage, [0, 1, 3, 4, 5, 2])
                    self.image = tf.reshape(self.permuteImage,
                            [self.batchSize, numTime, inputShape[1], inputShape[2], inputShape[3]*2])
                else:
                    self.image = tf.reshape(self.inputImage,
                            [self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]])

                self.padInput = tf.pad(self.image, [[0, 0], [0, 0], [7, 7], [15, 15], [0, 0]])
                #Reshape time dimension to feature dimension

                if(self.gtSparse):
                    self.gtIndices = tf.placeholder("int64", [2, None], "gtIndices")
                    self.gtValues = node_variable([None], "gtValues")

                    self.pre_gt = tf.sparse_tensor_to_dense(tf.SparseTensor(
                            tf.transpose(self.gtIndices, [1, 0]),
                            self.gtValues,
                            [self.batchSize*self.gtShape[0], self.gtShape[1]*self.gtShape[2]*self.numClasses]),
                            validate_indices=False
                            )
                    self.gt = tf.reshape(self.pre_gt, [self.batchSize, self.gtShape[0], self.gtShape[1], self.gtShape[2], self.numClasses])
                else:
                    self.gt=tf.placeholder("float32", [self.batchSize, self.gtShape[0], self.gtShape[1], self.gtShape[2], self.numClasses])

                self.select_gt = tf.squeeze(self.gt[:, :, :, :, 0:self.numClasses], squeeze_dims=[1])

                #self.norm_gt = self.gt/tf.reduce_sum(self.gt, reduction_indices=4, keep_dims=True)

            with tf.name_scope("Hidden"):
                if(self.time):
                    self.h_hidden= tf.nn.relu(tf.nn.conv3d(self.padInput, self.h_weight, [1, 1, 4, 4, 1], padding="VALID") + self.h_bias)
                    self.timePooled = tf.reduce_max(self.h_hidden, reduction_indices=1)
                else:
                    self.squeezeInput = tf.squeeze(self.padInput, axis=1)
                    self.squeezeWeight = tf.squeeze(self.h_weight, axis=0)
                    self.h_hidden = tf.nn.relu(tf.nn.conv2d(self.squeezeInput, self.squeezeWeight, [1, 4, 4, 1], padding="VALID") + self.h_bias)
                    self.timePooled = self.h_hidden

            with tf.name_scope("conv1"):
                yPool = 2
                xPool = 2
                self.hiddenPooled = tf.nn.max_pool(self.timePooled, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")

                self.h_res = tf.nn.relu(tf.nn.conv2d(self.hiddenPooled, self.conv1_w, [1, 1, 1, 1], padding="SAME") + self.conv1_b)
                self.h_conv1 = self.hiddenPooled + self.h_res

            with tf.name_scope("reg"):
                self.keep_prob = tf.placeholder(tf.float32)
                self.h_dropout = tf.nn.dropout(self.h_conv1, self.keep_prob)

            with tf.name_scope("conv2"):
                yPool = int(np.ceil(float(16)/(self.gtShape[1]*2)))
                xPool = int(np.ceil(float(64)/(self.gtShape[2]*2)))

                #Pool over spatial dimensions to be 2x2
                self.h_conv1_pool= tf.nn.max_pool(self.h_dropout, ksize=[1, yPool, xPool, 1], strides=[1, yPool, xPool, 1], padding="SAME")

                self.camPooled = tf.nn.max_pool(self.h_dropout, ksize=[1, yPool, xPool, 1], strides=[1, 1, 1, 1], padding="SAME")

                self.h_conv2 = tf.nn.conv2d(self.h_conv1_pool, self.class_weight, [1, 1, 1, 1], padding="SAME") + self.class_bias

                #We evaluate pooling with smaller stride here
                self.cam = tf.nn.conv2d(self.camPooled, self.class_weight, [1, 1, 1, 1], padding="SAME") + self.class_bias

                #Reshape batch and time together
                #self.reshape_cam = tf.transpose(tf.reshape(self.cam, [self.batchSize*7, 16, 32, 31]), [0, 3, 1, 2])
                self.reshape_cam = tf.transpose(self.cam, [0, 3, 1, 2])

                #Get ranking from h_conv
                self.classRank = tf.reduce_mean(self.reshape_cam, reduction_indices=[2, 3])

                self.est = pixelSoftmax(self.h_conv2)
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

                self.weightRegLoss = tf.reduce_sum(tf.square(self.class_weight)) + tf.reduce_sum(tf.square(self.conv1_w)) + tf.reduce_sum(tf.square(self.h_weight))

                self.loss = tf.reduce_mean(-tf.reduce_sum(self.lossWeight[0:self.numClasses] * self.select_gt* tf.log(self.est+self.epsilon), reduction_indices=3)) + self.regWeight * self.weightRegLoss

                #self.loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.gt - self.est), reduction_indices=[1, 2, 3, 4]))

            with tf.name_scope("Opt"):
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.h_weight,
                            self.conv1_w,
                            self.class_weight,
                        ]
                        )
                self.optimizerBias = tf.train.GradientDescentOptimizer(self.learningRateBias).minimize(self.loss,
                        var_list=[
                            self.h_bias,
                            self.conv1_b,
                            self.class_bias,
                        ]
                        )
                self.optimizerPre = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            #self.h_weight,
                            self.conv1_w,
                            self.class_weight,
                        ]
                        )
                self.optimizerPreBias = tf.train.GradientDescentOptimizer(self.learningRateBias).minimize(self.loss,
                        var_list=[
                            #self.h_bias,
                            self.conv1_b,
                            self.class_bias,
                        ]
                        )

        numK = min(5, self.numClasses)
        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.classRank, k=numK)

        #Summaries
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        for c in range(self.numClasses):
            className = self.idxToName[c]
            tf.summary.scalar(className+' F1', self.classF1[c])

        tf.summary.histogram('input', self.inputImage)
        tf.summary.histogram('hiddenPooled', self.hiddenPooled)
        tf.summary.histogram('gt', self.select_gt)
        #Conv layer histograms
        tf.summary.histogram('h_hidden', self.h_hidden)
        tf.summary.histogram('h_conv1', self.h_conv1)
        tf.summary.histogram('h_conv2', self.h_conv2)
        tf.summary.histogram('est', self.est)
        #Weight and bias hists
        tf.summary.histogram('h_weight', self.h_weight)
        tf.summary.histogram('h_bias', self.h_bias)
        tf.summary.histogram('conv1_w', self.conv1_w)
        tf.summary.histogram('conv1_b', self.conv1_b)
        tf.summary.histogram('class_weight', self.class_weight)
        tf.summary.histogram('class_bias', self.class_bias)

    def getLoadVars(self):
        v = tf.global_variables()
        #v = tf.global_variables()
        if(self.resLoad):
            #Load first and 3rd layers
            v = [var for var in v if (("V1_W" in var.name) or ("hidden_bias" in var.name) or ("class_weight" in var.name) or ("class_bias" in var.name)) and ("Adam" not in var.name)]
        else:
            v = [var for var in v if ("Adam" not in var.name)]
        return v

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            if(self.gtSparse):
                (gtOutY, gtOutX, gtVals) = sp.find(data[1])
                feedDict = {self.inputImage:data[0],
                            self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals,
                            self.keep_prob:.5
                            }
            else:
                feedDict = {self.inputImage:data[0],
                            self.gt:data[1],
                            self.keep_prob:.5
                            }

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
            if(self.gtSparse):
                gt = np.reshape(data[1].toarray(), (self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]))
            else:
                gt = data[1]
            print "Plotting"
            self.evalAndPlotWeights(feedDict, filename)
            #self.evalAndPlotCam(feedDict, data, gt, filename)

    def evalAndPlotWeights(self, feedDict, prefix):
        np_weights = self.sess.run(self.h_weight, feed_dict=feedDict)
        np_act = self.sess.run(self.h_hidden, feed_dict=feedDict)
        (ntime, ny, nx, nfns, nf) = np_weights.shape
        if(self.stereo):
            np_weights_reshape = np.reshape(np_weights, (ntime, ny, nx, 3, 2, nf))
            for s in range(2):
                if(s == 0):
                    suffix = "_left"
                elif(s == 1):
                    suffix = "_right"
                for t in range(ntime):
                    plotWeights = np_weights_reshape[t, :, :, :, s, :]
                    plot_weights(plotWeights, prefix, suffix+"_time"+str(t), [3, 0, 1, 2], np_act, plotInd=self.plotInd)
        else:
            for t in range(ntime):
                suffix = "_time" + str(t)
                plotWeights = np_weights[t, :, :, :, :]
                plot_weights(plotWeights, prefix, suffix, [3, 0, 1, 2], np_act, plotInd=self.plotInd)

    def evalAndPlotCam(self, feedDict, data, gt, prefix):

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

        if(inGt is not None):
            if(self.gtSparse):
                (gtOutY, gtOutX, gtVals) = sp.find(inGt)
                feedDict = {self.inputImage:inData,
                            self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals,
                            self.keep_prob:1.0
                            }
            else:
                feedDict = {self.inputImage:inData,
                        self.gt:inGt,
                        self.keep_prob:1.0
                        }
        else:
            feedDict ={self.inputImage:inData,
                        self.keep_prob:1.0
                      }

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt is not None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            if(self.gtSparse):
                gt = np.reshape(inGt.toarray(), (self.batchSize, gtShape[0], gtShape[1], gtShape[2], gtShape[3]))
            else:
                gt = inGt
            data = (inData, inGt, inImg)
            #self.evalAndPlotCam(feedDict, data, gt, filename)

        return outVals

