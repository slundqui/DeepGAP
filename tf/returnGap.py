import pdb
import numpy as np
import tensorflow as tf
from returnVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotCam
#import matplotlib.pyplot as plt

class returnGap:

    #Global timestep
    timestep = 0
    plotTimestep = 0

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, inputShape):
        self.loadParams(params)
        self.makeDirs()
        #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess = tf.Session()
        self.buildModel(inputShape)

    #Sets dictionary of params to member variables
    def loadParams(self, params):
        #Initialize tf parameters here
        self.outDir = params['outDir']
        self.runDir = self.outDir + params['runDir']
        self.ckptDir = self.runDir + params['ckptDir']
        self.plotDir = self.runDir + params['plotDir']
        self.tfDir = self.runDir + params['tfDir']
        self.saveFile = self.ckptDir + params['saveFile']

        self.load = params['load']
        self.loadFile = params['loadFile']
        self.vggFile = params['vggFile']
        self.outerSteps = params['outerSteps']
        self.innerSteps = params['innerSteps']
        self.savePeriod = params['savePeriod']
        self.plotPeriod = params['plotPeriod']
        self.writeStep = params['writeStep']

        self.device = params['device']
        self.batchSize = params['batchSize']
        self.learningRate = params['learningRate']
        self.numClasses = params['numClasses']

        self.progress = params['progress']
        self.idxToName = params['idxToName']

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        if not os.path.exists(self.runDir):
           os.makedirs(self.runDir)
        if not os.path.exists(self.plotDir):
           os.makedirs(self.plotDir)
        if not os.path.exists(self.ckptDir):
           os.makedirs(self.ckptDir)

    def runModel(self, trainDataObj, testDataObj=None):
        #Load summary
        self.writeSummary()
        for i in range(self.outerSteps):
           #Plot flag
           if(i%self.plotPeriod == 0):
               plot = True
           else:
               plot=False
           if(testDataObj):
               #Evaluate test frame, providing gt so that it writes to summary
               (evalData, gtData) = testDataObj.getData(self.batchSize)
               self.evalModel(evalData, gtData, plot=plot)
               print "Done test eval"
           #Train
           if(i%self.savePeriod == 0):
               self.trainModel(trainDataObj, save=True, plot=plot, pre=False)
           else:
               self.trainModel(trainDataObj, save=False, plot=plot, pre=False)



    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self, inputShape):
        if(self.vggFile):
            npWeights = loadWeights(self.vggFile)

        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("inputOps"):
                #Get convolution variables as placeholders
                self.inputImage = node_variable([self.batchSize, inputShape[0], inputShape[1], inputShape[2]], "inputImage")
                self.gt = node_variable([self.batchSize, self.numClasses], "gt")
                #Model variables for convolutions

            with tf.name_scope("Conv1Ops"):
                if(self.vggFile):
                    self.W_conv1 = weight_variable_fromnp(npWeights["conv1_w"], "w_conv1")
                    self.B_conv1 = weight_variable_fromnp(npWeights["conv1_b"], "b_conv1")
                else:
                    self.W_conv1 = weight_variable_fromnp(np.zeros((11, 11, 3, 64), dtype=np.float32), "w_conv1")
                    self.B_conv1 = weight_variable_fromnp(np.zeros((64), dtype=np.float32), "b_conv1")
                    #self.W_conv1 = weight_variable_xavier([11, 11, 3, 64], "w_conv1", conv=True)
                    #self.B_conv1 = bias_variable([64], "b_conv1")
                #self.h_conv1 = tf.nn.relu(conv2d(self.inputImage, self.W_conv1, "conv1", stride=[1, 4, 4, 1]) + self.B_conv1)
                self.h_conv1 = tf.nn.relu(conv2d(self.inputImage, self.W_conv1, "conv1", stride=[1, 1, 1, 1]) + self.B_conv1)
                #self.h_norm1 = tf.nn.local_response_normalization(self.h_conv1, name="LRN1")
                self.h_pool1 = maxpool_2x2(self.h_conv1, "pool1")

            with tf.name_scope("Conv2Ops"):
                if(self.vggFile):
                    self.W_conv2 = weight_variable_fromnp(npWeights["conv2_w"], "w_conv2")
                    self.B_conv2 = weight_variable_fromnp(npWeights["conv2_b"], "b_conv2")
                else:
                    self.W_conv2 = weight_variable_fromnp(np.zeros((5, 5, 64, 256), dtype=np.float32), "w_conv2")
                    self.B_conv2 = weight_variable_fromnp(np.zeros((256), dtype=np.float32), "b_conv2")
                    #self.W_conv2 = weight_variable_xavier([5, 5, 64, 256], "w_conv2", conv=True)
                    #self.B_conv2 = bias_variable([256], "b_conv2")
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, "conv2") + self.B_conv2)
                #self.h_norm2 = tf.nn.local_response_normalization(self.h_conv2, name="LRN2")
                self.h_pool2 = maxpool_2x2(self.h_conv2, "pool2")

            with tf.name_scope("Conv3Ops"):
                if(self.vggFile):
                    self.W_conv3 = weight_variable_fromnp(npWeights["conv3_w"], "w_conv3")
                    self.B_conv3 = weight_variable_fromnp(npWeights["conv3_b"], "b_conv3")
                else:
                    self.W_conv3 = weight_variable_fromnp(np.zeros((3, 3, 256, 256), dtype=np.float32), "w_conv3")
                    self.B_conv3 = weight_variable_fromnp(np.zeros((256), dtype=np.float32), "b_conv3")
                    #self.W_conv3 = weight_variable_xavier([3, 3, 256, 256], "w_conv3", conv=True)
                    #self.B_conv3 = bias_variable([256], "b_conv3")
                self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3, "conv3") + self.B_conv3, name="relu3")

            with tf.name_scope("Conv4Ops"):
                if(self.vggFile):
                    self.W_conv4 = weight_variable_fromnp(npWeights["conv4_w"], "w_conv4")
                    self.B_conv4 = weight_variable_fromnp(npWeights["conv4_b"], "b_conv4")
                else:
                    self.W_conv4 = weight_variable_fromnp(np.zeros((3, 3, 256, 256), dtype=np.float32), "w_conv4")
                    self.B_conv4 = weight_variable_fromnp(np.zeros((256), dtype=np.float32), "b_conv4")
                    #self.W_conv4 = weight_variable_xavier([3, 3, 256, 256], "w_conv4", conv=True)
                    #self.B_conv4 = bias_variable([256], "b_conv4")
                self.h_conv4 = tf.nn.relu(conv2d(self.h_conv3, self.W_conv4, "conv4") + self.B_conv4, name="relu4")

            with tf.name_scope("Conv5Ops"):
                if(self.vggFile):
                    self.W_conv5 = weight_variable_fromnp(npWeights["conv5_w"], "w_conv5")
                    self.B_conv5 = weight_variable_fromnp(npWeights["conv5_b"], "b_conv5")
                else:
                    self.W_conv5 = weight_variable_fromnp(np.zeros((3, 3, 256, 256), dtype=np.float32), "w_conv5")
                    self.B_conv5 = weight_variable_fromnp(np.zeros((256), dtype = np.float32), "b_conv5")
                    #self.W_conv5 = weight_variable_xavier([3, 3, 256, 256], "w_conv5", conv=True)
                    #self.B_conv5 = bias_variable([256], "b_conv5")
                self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5, "conv5") + self.B_conv5)
                #self.h_pool5 = maxpool_2x2(self.h_conv5, "pool5")

            #placeholder for specifying dropout
            #self.keep_prob = tf.placeholder(tf.float32)

            #8 comes from 4 stride in conv1, 2 stride in pool1, 2 stride in pool2, 2 stride in pool5
            self.h_conv5_shape = [self.batchSize, inputShape[0]/4, inputShape[1]/4, 256]
            with tf.name_scope("GAP"):
                self.h_gap = tf.reduce_mean(self.h_conv5, reduction_indices=[1, 2])
                self.W_gap = weight_variable_xavier([256, self.numClasses], "w_gap", conv=False)
                self.est = tf.nn.softmax(tf.matmul(self.h_gap, self.W_gap))

            with tf.name_scope("CAM"):
                self.h_reshape_gap = tf.reshape(self.h_conv5, [self.batchSize*self.h_conv5_shape[1]*self.h_conv5_shape[2], -1])
                self.flat_cam = tf.matmul(self.h_reshape_gap, self.W_gap)
                self.reshape_cam = tf.reshape(self.flat_cam, [self.batchSize, self.h_conv5_shape[1], self.h_conv5_shape[2], -1])
                self.cam = tf.transpose(self.reshape_cam, [0, 3, 1, 2])

            with tf.name_scope("Loss"):
                #Define loss
                self.loss = tf.reduce_mean(-tf.reduce_sum(self.gt * tf.log(self.est), reduction_indices=[1]))

            with tf.name_scope("Opt"):
                #Define optimizer
                #self.optimizerAll = tf.train.AdagradOptimizer(self.learningRate).minimize(self.loss)
                #self.optimizerFC = tf.train.AdagradOptimizer(self.learningRate).minimize(self.loss,
                self.optimizerAll = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
                #self.optimizerFC = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss,
                #        var_list=[
                #            self.W_fc1,
                #            self.B_fc1,
                #            self.W_fc2,
                #            self.B_fc2,
                #            self.W_fc3,
                #            self.B_fc3]
                #        )

            with tf.name_scope("Metric"):
                self.correct = tf.equal(tf.argmax(self.gt, 1), tf.argmax(self.est, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        #Cannot be on GPU
        (self.eval_vals, self.eval_idx) = tf.nn.top_k(self.est, k=10)

        #Summaries
        tf.scalar_summary('loss', self.loss, name="lossSum")
        tf.scalar_summary('accuracy', self.accuracy, name="accSum")

        tf.histogram_summary('input', self.inputImage, name="image")
        tf.histogram_summary('gt', self.gt, name="gt")
        tf.histogram_summary('conv1', self.h_pool1, name="conv1")
        tf.histogram_summary('conv2', self.h_pool2, name="conv2")
        tf.histogram_summary('conv3', self.h_conv3, name="conv3")
        tf.histogram_summary('conv4', self.h_conv4, name="conv4")
        tf.histogram_summary('conv5', self.h_conv5, name="conv5")
        tf.histogram_summary('gap', self.h_gap, name="gap")
        tf.histogram_summary('est', self.est, name="est")
        tf.histogram_summary('w_conv1', self.W_conv1, name="w_conv1")
        tf.histogram_summary('b_conv1', self.B_conv1, name="b_conv1")
        tf.histogram_summary('w_conv2', self.W_conv2, name="w_conv2")
        tf.histogram_summary('b_conv2', self.B_conv2, name="b_conv2")
        tf.histogram_summary('w_conv3', self.W_conv3, name="w_conv3")
        tf.histogram_summary('b_conv3', self.B_conv3, name="b_conv3")
        tf.histogram_summary('w_conv4', self.W_conv4, name="w_conv4")
        tf.histogram_summary('b_conv4', self.B_conv4, name="b_conv4")
        tf.histogram_summary('w_conv5', self.W_conv5, name="w_conv5")
        tf.histogram_summary('b_conv5', self.B_conv5, name="b_conv5")
        tf.histogram_summary('w_gap', self.W_gap, name="w_gap")

        #tf.image_summary("cam", self.sortedCamImg, max_images=10)
        #tf.image_summary("input", self.inputImage, max_images=1)

        #Define saver
        self.saver = tf.train.Saver()

        #Initialize
        #Load checkpoint if flag set
        if(self.load):
           self.loadModel()
           ##We only load weights, so we need to initialize A
           #un_vars = list(tf.get_variable(name) for name in self.sess.run(tf.report_uninitialized_variables(tf.all_variables())))
           #tf.initialize_variables(un_vars)
        else:
           self.initSess()

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self):
        self.mergedSummary = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(self.tfDir + "/train", self.sess.graph)
        self.test_writer = tf.train.SummaryWriter(self.tfDir + "/test")

    def closeSess(self):
        self.sess.close()

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot, pre=False):
        #Define session
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1]}
            #Run optimizer
            if(pre):
                self.sess.run(self.optimizerFC, feed_dict=feedDict)
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
            filename = self.plotDir + "train_" + str(self.timestep) + ".png"
            self.evalAndPlotCam(feedDict, filename)


    def evalAndPlotCam(self, feedDict, filename):
        #We need feed_dict here
        cam = self.sess.run(self.cam, feed_dict=feedDict)
        img = feedDict[self.inputImage][0, :, :, :]
        gtIdx = np.argmax(feedDict[self.gt][0, :])
        camIdxs = self.sess.run(self.eval_idx, feed_dict=feedDict)[0, :]
        camVals = self.sess.run(self.eval_vals, feed_dict=feedDict)[0, :]
        sortedCam = cam[0, camIdxs, :, :]
        plotCam(filename, img, gtIdx, sortedCam, camIdxs, camVals, self.idxToName)

    #Evaluates all of inData at once
    #If an inGt is provided, will calculate summary as test set
    def evalModel(self, inData, inGt = None, plot=True):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            (numGt, drop) = inGt.shape
            assert(numData == numGt)
            feedDict = {self.inputImage: inData, self.gt: inGt}
        else:
            feedDict = {self.inputImage: inData}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)
        if(plot and inGt != None):
            filename = self.plotDir + "test_" + str(self.timestep) + ".png"
            self.evalAndPlotCam(feedDict, filename)

        return outVals

    ##Evaluates inData, but in miniBatchSize batches for memory efficiency
    ##If an inGt is provided, will calculate summary as test set
    #def evalModelBatch(self, inData, inGt=None):
    #    (numData, ny, nx, nf) = inData.shape
    #    if(inGt != None):
    #        (numGt, drop) = inGt.shape
    #        assert(numData == numGt)

    #    #Split up numData into miniBatchSize and evaluate est data
    #    tfInVals = np.zeros((self.batchSize, ny, nx, nf))
    #    outData = np.zeros((numData, 1))

    #    #Ceil of numData/batchSize
    #    numIt = int(numData/self.batchSize) + 1

    #    #Only write summary on first it

    #    startOffset = 0
    #    for it in range(numIt):
    #        print it, " out of ", numIt
    #        #Calculate indices
    #        startDataIdx = startOffset
    #        endDataIdx = startOffset + miniBatchSize
    #        startTfValIdx = 0
    #        endTfValIdx = miniBatchSize

    #        #If out of bounds
    #        if(endDataIdx >= numData):
    #            #Calculate offset
    #            offset = endDataIdx - numData
    #            #Set endDataIdx to max value
    #            endDataIdx = numData
    #            #Set endTfValIdx to less than max value
    #            endTfValIdx -= offset

    #        tfInVals[startTfValIdx:endTfValIdx, :, :, :] = inData[startDataIdx:endDataIdx, :, :, :]
    #        feedDict = {self.inputImage: tfInVals}
    #        tfOutVals = self.est.eval(feed_dict=feedDict, session=self.sess)
    #        outData[startDataIdx:endDataIdx, :] = tfOutVals[startTfValIdx:endTfValIdx, :]

    #        if(inGt != None and it == 0):
    #            tfInGt = inGt[startDataIdx:endDataIdx, :]
    #            summary = self.sess.run(self.mergedSummary, feed_dict={self.inputImage: tfInVals, self.gt: tfInGt})
    #            self.test_writer.add_summary(summary, self.timestep)

    #        startOffset += miniBatchSize

    #    #Return output data
    #    return outData

    #Loads a tf checkpoint
    def loadModel(self):
        self.saver.restore(self.sess, self.loadFile)
        print("Model %s loaded" % self.loadFile)

