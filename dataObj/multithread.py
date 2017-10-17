import threading

class multithread(object):
     def loadData(self, batchSize):
         self.loadBuf = self.dataObj.getData(batchSize)

     def __init__(self, dataObj, batchSize):
         self.dataObj = dataObj
         self.batchSize = batchSize
         #This is needed by tf code
         self.inputShape = dataObj.inputShape
         self.gtShape = dataObj.gtShape
         #self.numExamples = dataObj.numExamples
         self.numClasses = dataObj.numClasses
         self.idxToName = dataObj.idxToName
         self.lossWeight = dataObj.lossWeight

         #Start first thread
         self.loadThread = threading.Thread(target=self.loadData, args=(self.batchSize,))
         self.loadThread.start()

     #This function doesn't actually need numExample and getMeta, but this api matches that of
     #image. So all we do here is assert numExample and getMeta are the same
     def getData(self, numExample):
         assert(numExample == self.batchSize)
         #Block loadThread here
         self.loadThread.join()
         #Store loaded data into local variable
         #This should copy, not access reference
         returnBuf = self.loadBuf[:]
         #Launch new thread to load new buffer
         self.loadThread = threading.Thread(target=self.loadData, args=(self.batchSize,))
         self.loadThread.start()
         #Return stored buffer
         return returnBuf
