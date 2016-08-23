import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewBB import plotBB
from base import TFObj
import scipy.sparse as sp
#import matplotlib.pyplot as plt

class SLPBBVid(TFObj):

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        super(SLPBBVid, self).loadParams(params)

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.learningRateBias = params['learningRateBias']
        self.lossWeight = params['lossWeight']
        self.gtShape = params['gtShape']
        self.imageShape = params['imageShape']
        self.gtSparse = params['gtSparse']
        self.dncVals = params['dncVals']
        self.bbWindowSize = params['bbWindowSize']
        self.iouThresh = params['iouThresh']
        self.minIouThresh = params['minIouThresh']
        self.inputScale = params['inputScale']
        self.gtStrideY = params['gtStrideY']
        self.gtStrideX = params['gtStrideX']

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
                        [self.batchSize*inputShape[0], inputShape[1]*inputShape[2]*inputShape[3]]),
                        validate_indices=False
                        )

                self.inputImage = self.inputScale * tf.reshape(self.pre_inputImage, [self.batchSize, inputShape[0], inputShape[1], inputShape[2], inputShape[3]])

                if(self.gtSparse):
                    self.gtIndices = tf.placeholder("int64", [2, None], "gtIndices")
                    self.gtValues = node_variable([None], "gtValues")

                    self.pre_gt = tf.sparse_tensor_to_dense(tf.SparseTensor(
                            tf.transpose(self.gtIndices, [1, 0]),
                            self.gtValues,
                            [self.batchSize*self.gtShape[0], self.gtShape[1]*self.gtShape[2]*self.gtShape[3]]),
                            validate_indices=False
                            )
                    self.gt = tf.reshape(self.pre_gt, [self.batchSize, self.gtShape[0], self.gtShape[1], self.gtShape[2], self.gtShape[3]])
                else:
                    self.gt=tf.placeholder("float32", [self.batchSize, self.gtShape[0], self.gtShape[1], self.gtShape[2], self.gtShape[3]])

                #Binarize gt values
                self.bool_gt = tf.greater_equal(self.gt , self.iouThresh)
                self.bin_gt = tf.cast(self.bool_gt, "float32")

                #Distractor should be [minIouThresh, iouThresh)
                self.bool_dist = tf.logical_and(tf.greater_equal(self.gt, self.minIouThresh), tf.logical_not(self.bool_gt))
                self.distractor = tf.reduce_max(tf.cast(self.bool_dist, "float32"), reduction_indices=[4], keep_dims=True)

                self.dist_gt = tf.concat(4, [self.distractor, self.bin_gt])
                self.dnc = tf.expand_dims(tf.constant(self.dncVals, dtype="float32", name="dnc"),0)
                self.masked_gt = self.dist_gt * self.dnc

                #We downsample gt for a bigger stride
                targetNy = self.gtShape[1]/self.gtStrideY
                targetNx = self.gtShape[2]/self.gtStrideX

                stride_gt1 = tf.reshape(self.masked_gt, [self.batchSize, self.gtShape[0], targetNy, self.gtStrideY, self.gtShape[2], self.numClasses])[:, :, :, 0, :, :]
                self.stride_gt= tf.reshape(stride_gt1, [self.batchSize, self.gtShape[0], targetNy, targetNx, self.gtStrideX, self.numClasses])[:, :, :, :, 0, :]

                #Make a boolean mask for vectors with all 0s
                self.careMask = tf.cast(tf.reduce_max(self.stride_gt, reduction_indices=[4]), "bool")

            with tf.name_scope("Pool"):
                yStride = int(np.ceil(float(inputShape[1])/targetNy))
                xStride = int(np.ceil(float(inputShape[2])/targetNx))

                wyScale = float(inputShape[1])/self.imageShape[1]
                wxScale = float(inputShape[2])/self.imageShape[2]

                #Pool over time dimension
                self.inputPooled = tf.reduce_max(self.inputImage, reduction_indices=1)
                pooled = []
                for w in self.bbWindowSize:
                    (wy, wx) = w
                    swy = int(np.ceil(wy*wyScale))
                    swx = int(np.ceil(wx*wxScale))

                    #TODO do pooling over quadrates here
                    #self.pad_input = tf.pad(self.inputPooled, [[0, 0], [swy/2, swy/2], [swx/2, swx/2], [0, 0]])
                    #pdb.set_trace()
                    #swy_sub = swy/2
                    #swx_sub = swx/2

                    pooled.append(tf.nn.max_pool(self.inputPooled, [1, swy, swx, 1], [1, yStride, xStride, 1], padding="SAME"))

                #Concatenate into batch dimension
                self.catPooled = tf.concat(0, pooled)

            with tf.name_scope("Conv"):
                self.weight = weight_variable_xavier([1, 1, inputShape[3], self.numClasses], "weight")
                self.bias = bias_variable([self.numClasses], "bias")

                self.h_conv = tf.nn.conv2d(self.catPooled, self.weight, [1, 1, 1, 1], padding="SAME") + self.bias

                #Get ranking from h_conv
                self.softmax_est = pixelSoftmax(self.h_conv)
                self.est = tf.reshape(self.softmax_est, [self.batchSize, self.gtShape[0], targetNy, targetNx, self.numClasses])

                #Remove distractor from est
                self.classRank = tf.reduce_mean(self.est[:, :, :, :, 1:], reduction_indices=[1, 2, 3])

            with tf.name_scope("Metric"):
                self.maskGt = tf.boolean_mask(self.stride_gt, self.careMask)
                self.maskEst = tf.boolean_mask(self.est, self.careMask)

                gtClass = tf.argmax(self.maskGt, 1)
                estClass = tf.argmax(self.maskEst, 1)
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

            with tf.name_scope("Loss"):
                if(self.lossWeight == None):
                    self.loss = tf.reduce_mean(-tf.reduce_sum(self.maskGt* tf.log(self.maskEst+self.epsilon), reduction_indices=1))
                else:
                    self.loss = tf.reduce_mean(-tf.reduce_sum(self.lossWeight[0:self.numClasses] * self.maskGt* tf.log(self.maskEst+self.epsilon), reduction_indices=1))
                #self.loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.gt - self.est), reduction_indices=[1, 2, 3, 4]))

            with tf.name_scope("Opt"):
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss,
                        var_list=[
                            self.weight,
                        ]
                        )
                self.optimizerBias = tf.train.GradientDescentOptimizer(self.learningRateBias).minimize(self.loss,
                        var_list=[
                            self.bias,
                        ]
                        )

            with tf.name_scope("NMS"):
                self.nms_boxes = tf.placeholder("float32", [None, 4], "nms_boxes")
                self.nms_scores = tf.placeholder("float32", [None], "nms_scores")
                nbb = tf.shape(self.nms_scores)
                #Cut by a quarter
                self.nms_bb_idx = tf.image.non_max_suppression(self.nms_boxes, self.nms_scores, 30)
                self.nms_bb = tf.gather(self.nms_boxes, self.nms_bb_idx)



        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.classRank, k=6)
        self.numCare = tf.reduce_sum(tf.cast(self.careMask, "float32"))

        #Summaries
        tf.scalar_summary('loss', self.loss, name="loss_vis")
        tf.scalar_summary('accuracy', self.accuracy, name="accuracy_vis")
        for c in range(self.numClasses):
            className = self.idxToName[c]
            tf.scalar_summary(className+' F1', self.classF1[c])
        tf.scalar_summary('numCare', self.numCare, name="numCare_vis")

        tf.histogram_summary('input', self.inputImage, name="image_vis")
        tf.histogram_summary('inputPooled', self.inputPooled, name="imagePooled_vis")
        tf.histogram_summary('catPooled', self.catPooled, name="catPooled_vis")
        tf.histogram_summary('gt', self.gt, name="gt_vis")
        tf.histogram_summary('stride_gt', self.stride_gt, name="stride_gt_vis")
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
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            (dataOutY, dataOutX, dataVals) = sp.find(data[0])

            if(self.gtSparse):
                (gtOutY, gtOutX, gtVals) = sp.find(data[1])
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                            self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals}
            else:
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                        self.gt:data[1]}

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

            self.evalAndPlotBB(feedDict, data[2], filename)

    def evalAndPlotBB(self, feedDict, img, prefix):
        print "Plotting"

        np_est = self.sess.run(self.est, feed_dict=feedDict)
        np_gt = self.sess.run(self.stride_gt, feed_dict=feedDict)

        classRankIdxs = self.sess.run(self.eval_idx, feed_dict=feedDict)
        classRankVals = self.sess.run(self.eval_vals, feed_dict=feedDict)

        plotBB(prefix, img, np_gt, np_est, classRankIdxs, classRankVals, self.idxToName, self.bbWindowSize, self.sess, (self.nms_bb, self.nms_boxes, self.nms_scores))

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
                        self.gtIndices:[gtOutY, gtOutX], self.gtValues:gtVals}
            else:
                feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals,
                        self.gt:inGt}
        else:
            feedDict = {self.dataIndices:[dataOutY, dataOutX], self.dataValues:dataVals}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        if(plot):
            filename = self.plotDir + "test_" + str(self.timestep)
            self.evalAndPlotBB(feedDict, inImg, filename)

        return outVals

