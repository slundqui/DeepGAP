import pdb
import numpy as np
import tensorflow as tf
from loadVgg import loadWeights
from utils import *
import os
from plot.viewCam import plotDetCam
import scipy.sparse as sp
#import matplotlib.pyplot as plt

class TFObj(object):
    #Global timestep
    timestep = 0
    plotTimestep = 0

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, inputShape):
        self.loadParams(params)
        self.makeDirs()

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuPercent)
        #config = tf.ConfigProto(gpu_options=gpu_options)
        config = tf.ConfigProto()

        #config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        self.sess = tf.Session(config=config)
        self.buildModel(inputShape)
        self.initialize()
        self.writeSummary()

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
        self.outerSteps = params['outerSteps']
        self.innerSteps = params['innerSteps']
        self.savePeriod = params['savePeriod']
        self.plotPeriod = params['plotPeriod']
        self.writeStep = params['writeStep']

        self.device = params['device']
        #self.gpuPercent = params['gpuPercent']
        self.batchSize = params['batchSize']
        self.learningRate = params['learningRate']
        self.numClasses = params['numClasses']

        self.progress = params['progress']
        self.idxToName = params['idxToName']
        self.preTrain = params['preTrain']

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        if not os.path.exists(self.runDir):
           os.makedirs(self.runDir)
        if not os.path.exists(self.plotDir):
           os.makedirs(self.plotDir)
        if not os.path.exists(self.ckptDir):
           os.makedirs(self.ckptDir)

    def runModel(self, trainDataObj, testDataObj=None):
        for i in range(self.outerSteps):
           #Plot flag
           if(i%self.plotPeriod == 0):
               plot = True
           else:
               plot = False
           if(testDataObj):
               #Evaluate test frame, providing gt so that it writes to summary
               (evalData, gtData) = testDataObj.getData(self.batchSize)
               self.evalModel(evalData, gtData, plot=plot)
               print "Done test eval"
           #Train
           if(i%self.savePeriod == 0):
               self.trainModel(trainDataObj, save=True, plot=plot)
           else:
               self.trainModel(trainDataObj, save=False, plot=plot)

    def buildModel(self, inputShape):
        print "Cannot call base class buildModel"
        assert(0)

    def getLoadVars(self):
        return tf.all_variables()

    def initialize(self):
        ##Define saver
        load_v = self.getLoadVars()
        ##Load specific variables, save all variables
        self.loader = tf.train.Saver(var_list=load_v)
        self.saver = tf.train.Saver()

        #Initialize
        self.initSess()
        #Load checkpoint if flag set
        if(self.load):
           self.loadModel()

    def guarantee_initialized_variables(self, session, list_of_variables = None):
        if list_of_variables is None:
            list_of_variables = tf.all_variables()
        uninitialized_variables = list(tf.get_variable(name) for name in
                                       session.run(tf.report_uninitialized_variables(list_of_variables)))
        session.run(tf.initialize_variables(uninitialized_variables))
        return uninitialized_variables

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
        tf.reset_default_graph()

    #Trains model for numSteps
    #If pre is False, will train entire network
    #If pre is True, will train only fully connected network
    def trainModel(self, dataObj, save, plot):
        #Define session
        for i in range(self.innerSteps):
            #Get data from dataObj
            data = dataObj.getData(self.batchSize)
            feedDict = {self.inputImage: data[0], self.gt: data[1]}
            self.sess.run(self.optimizer, feed_dict=feedDict)
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
    def evalModel(self, inData, inGt = None, inImage = None, gtShape=None, plot=True):
        (numData, ny, nx, nf) = inData.shape
        if(inGt != None):
            numGt = inGt.shape[0]
            assert(numData == numGt)
            feedDict = {self.inputImage: inData, self.gt: inGt}
        else:
            feedDict = {self.inputImage: inData}

        outVals = self.est.eval(feed_dict=feedDict, session=self.sess)
        if(inGt != None):
            summary = self.sess.run(self.mergedSummary, feed_dict=feedDict)
            self.test_writer.add_summary(summary, self.timestep)

        return outVals

    #Evaluates inData, but in batchSize batches for memory efficiency
    #If an inGt is provided, will calculate summary as test set
    def evalModelBatch(self, testDataObj):
        numData = len(testDataObj.shuffleIdx)
        #Ceil of numData/batchSize
        numIt = int(np.ceil(float(numData)/self.batchSize))
        gtShape = testDataObj.gtShape
        numTiles = 1
        #last index of gtshape is gt
        for dim in gtShape[:-1]:
            numTiles *= dim

        outGt = np.zeros((numData * numTiles))
        outEst = np.zeros((numData * numTiles))

        startDataOffset = 0
        for it in range(numIt):
            print it, " out of ", numIt

            #Calculate indices
            endDataOffset = startDataOffset + self.batchSize

            startDataTfIdx = 0
            endDataTfIdx = self.batchSize

            startGtIdx = startDataOffset * numTiles
            endGtIdx = startGtIdx + self.batchSize * numTiles


            #If out of bounds
            if(endDataOffset >= numData):
                #Calculate offset
                offsetData = endDataOffset - numData
                #Set endDataIdx to max value
                endGtIdx = numData*numTiles
                #Set endTfValIdx to less than max value
                endDataTfIdx -= offsetData

            (inData, inGt, drop) = testDataObj.getData(self.batchSize)
            #TODO
            #assert(sp.issparse(inData))
            assert(not sp.issparse(inGt))

            #TODO fix this to be more general
            #gtVals = np.argmax(inGt, axis=inGt.ndim-1)[startDataTfIdx:endDataTfIdx].flatten()
            gtVals = inGt[startDataTfIdx:endDataTfIdx, :, :, 1].flatten()
            outGt[startGtIdx:endGtIdx] = gtVals

            #Eval data
            est = self.evalModel(inData, None, None, None, False)
            #TODO fix this to be more general
            #estVals = np.argmax(estGt, axis=estGt.ndim-1)[startDataTfIdx:endDataTfIdx].flatten()
            estVals = est[startDataTfIdx:endDataTfIdx, :, :, 1].flatten()
            outEst[startGtIdx:endGtIdx] = estVals

            startDataOffset += self.batchSize

        fn = self.runDir + "evalGtIdxs.npy"
        np.save(fn, outGt)
        fn = self.runDir + "evalEstIdxs.npy"
        np.save(fn, outEst)

        #Return output data
        return (outGt, outEst)


    #Loads a tf checkpoint
    def loadModel(self):
        self.loader.restore(self.sess, self.loadFile)
        #self.guarantee_initialized_variables(self.sess)

        print("Model %s loaded" % self.loadFile)





