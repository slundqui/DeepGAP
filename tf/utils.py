import numpy as np
import tensorflow as tf
import pdb
from scipy import sparse

#bbs and scores should be a list of length batchSize
#bbs items should be [numBB, 4]
#scores items should be [numBB]
def calcBatchBB(bbs, scores, detConfThreshold, numBB):
    outBBList = []
    for (batchBB, batchScores) in zip(bbs, scores):
        posBB = calcBB(batchBB, batchScores, detConfThreshold)
        #Pad or crop to have exactly numBB of bbs
        #Expand into image shape, and spoof as image to pad/crop into numBB
        #Pad 1 at bottom in case of zero posBBs
        padBB = tf.pad(posBB, [[0, 1], [0, 0]], mode='CONSTANT')
        expandBB = tf.expand_dims(padBB, -1)
        reshapedBB = tf.image.resize_image_with_crop_or_pad(expandBB, numBB, 4)[:, :, 0]
        outBBList.append(reshapedBB)
    #Outputs [batch, numBB, 4]
    return tf.pack(outBBList, 0)

#Note that this assumes objVals and bbVals are from 1 batch
#i.e. 4 dimensional in [y, x, window, 2] and [y, x, windows, 4] respectively
#Returns 2 lists, one for output bbs and one for the score of each bb
def runNms(objVals, bbVals, maxNumBB, nms_iou_threshold=None):
    scores = tf.reshape(objVals[:, :, :, 0], [-1])
    bbs = tf.reshape(bbVals, [-1, 4])
    nmsIdx = tf.image.non_max_suppression(bbs, scores, maxNumBB, iou_threshold=nms_iou_threshold)
    outBbs = tf.gather(bbs, nmsIdx)
    outScores = tf.gather(scores, nmsIdx)
    return (outBbs, outScores)

#bb should be [numBB, 4]
#scores should be [numBB]
def calcBB(bb, score, detConfThreshold):
    posBBIdx = tf.where(tf.greater(score, detConfThreshold))
    posBB = tf.gather_nd(bb, posBBIdx)
    return posBB

def calcPvR(estObj, estBB, gtObj, gtBB, batchSize, maxNumBB, det_iou_threshold, nms=True, nms_iou_threshold=0.5):

    precision = []
    recall = []
    for b in range(batchSize):
        bEstObj = estObj[b, :, :, :, :]
        bEstBB = estBB[b, :, :, :, :]
        bGtObj = gtObj[b, :, :, :, :]
        bGtBB = gtBB[b, :, :, :, :]

        gtProposals = calcBB(bGtObj, bGtBB, 0.5, maxNumBB, nms=False)
        expandGtProposals = tf.expand_dims(gtProposals, 0)
        gtArea = (expandGtProposals[:, :, 2] - expandGtProposals[:, :, 0]) *                  (expandGtProposals[:, :, 3] - expandGtProposals[:, :, 1])

        batchPrecision = []
        batchRecall = []
        #How to calculate best threshold?
        for t in np.linspace(0, 1):
            estProposals = calcBB(bEstObj, bEstBB, t, maxNumBB, nms=nms, nms_iou_threshold=nms_iou_threshold)
            expandEstProposals = tf.expand_dims(gtProposals, 1)
            estArea = (expandEstProposals[:, :, 2] - expandEstProposals[:, :, 0]) * (expandEstProposals[:, :, 3] - expandEstProposals[:, :, 1])
            #Calculate iou
            intYMin = tf.maximum(expandGtProposals[:, :, 0], expandEstProposals[:, :, 0])
            intYMax = tf.minimum(expandGtProposals[:, :, 2], expandEstProposals[:, :, 2])
            intXMin = tf.maximum(expandGtProposals[:, :, 1], expandEstProposals[:, :, 1])
            intXMax = tf.minimum(expandGtProposals[:, :, 3], expandEstProposals[:, :, 3])

            intArea = tf.nn.relu((intYMax-intYMin) * (intXMax-intXMin))
            unionArea = gtArea + estArea - intArea
            iou = tf.to_float(intArea)/unionArea
            #IOU is in the shape of [numEstBB, numGtBB]
            bool_iou = iou > det_iou_threshold
            #For every gt box, if any est bbs overlaps sufficiently
            tp = tf.reduce_sum(tf.cast(tf.reduce_any(bool_iou, axis=0), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_not(bool_iou), axis=1), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_not(bool_iou), axis=0), tf.float32))

            #When tp+fp == 0, precision is 1
            batchPrecision.append(tf.select(tp+fp == 0, 1.0, (tp)/(tp+fp)))
            batchRecall.append((tp)/(tp+fn))

        precision.append(tf.pack(batchPrecision, 0))
        recall.append(tf.pack(batchRecall, 0))

        #Calculate auc, TODO
    return [tf.pack(precision, 0), tf.pack(recall, 0)]

def standard_batch_norm(l, x, n_out, phase_train, scope='BN'):
    """
    Batch normalization on feedforward maps.
    Args:
        x:           Vector
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope+l):
        #beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float64 ), name='beta', trainable=True, dtype=tf.float64 )
        #gamma = tf.Variable(tf.constant(1.0, shape=[n_out],dtype=tf.float64 ), name='gamma', trainable=True, dtype=tf.float64 )
        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float32)
        beta = tf.get_variable(name='beta'+l, dtype=tf.float64, initializer=init_beta, regularizer=None, trainable=True)
        gamma = tf.get_variable(name='gamma'+l, dtype=tf.float64, initializer=init_gamma, regularizer=None, trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return (normed, beta, gamma)


def smoothL1(inNode):
    absV = tf.abs(inNode)
    LTvalue = 0.5 * tf.square(inNode)
    Evalue = absV - .5
    return tf.select(absV < 1, LTvalue, Evalue)

def pixelSoftmax(inNode):
    exp = tf.exp(inNode - tf.reduce_max(inNode, reduction_indices=3, keep_dims=True))
    #Calculate sum across feature dimension
    norm = tf.reduce_sum(exp, reduction_indices=3, keep_dims=True) + 1e-8
    return tf.truediv(exp, norm)

def pixelSoftmax5d(inNode):
    exp = tf.exp(inNode - tf.reduce_max(inNode, reduction_indices=4, keep_dims=True))
    #Calculate sum across feature dimension
    norm = tf.reduce_sum(exp, reduction_indices=4, keep_dims=True) + 1e-8
    return tf.truediv(exp, norm)

def convertToSparse5d(m):
    [nb, nt, ny, nx, nf] = m.shape
    mreshape = np.reshape(m, (nb, nt*ny*nx*nf))
    return sparse.csr_matrix(mreshape)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

#Helper functions for initializing weights
def weight_variable_fromnp(inNp, inName):
    shape = inNp.shape
    return tf.Variable(inNp, name=inName)

def weight_variable(shape, inName, inStd):
    initial = tf.truncated_normal_initializer(stddev=inStd)
    return tf.get_variable(inName, shape, initializer=initial)

def bias_variable(shape, inName, biasInitConst=.01):
   initial = tf.constant_initializer(biasInitConst)
   return tf.get_variable(inName, shape, initializer=initial)

def weight_variable_xavier(shape, inName, conv=False):
   #initial = tf.truncated_normal(shape, stddev=weightInitStd, name=inName)
   if conv:
       initial = tf.contrib.layers.xavier_initializer_conv2d()
   else:
       initial = tf.contrib.layers.xavier_initializer()
   return tf.get_variable(inName, shape, initializer=initial)

#Helper functions for creating input nodes
def node_variable(shape, inName):
   return tf.placeholder("float", shape=shape, name=inName)

#Helper functions for creating convolutions and pooling
def conv2d(x, W, inName, stride = None):
    if stride:
        return tf.nn.conv2d(x, W, strides=stride, padding='SAME', name=inName)
    else:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=inName)

def conv2d_oneToMany(x, W, outShape, inName, tStride):
    return tf.nn.conv2d_transpose(x, W, outShape, [1, tStride, tStride, 1], padding='SAME', name=inName)

def maxpool_2x2(x, inName):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=inName)

def conv3d(x, w, inName):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=inName)

#Transposes data to permute strides to the output feature dimension
def transpose5dData(x, xShape, strideT, strideY, strideX):
    [nb, nt, ny, nx, nf] = xShape
    print "Building output indices for conv3d"
    #Build gather indices for output
    #Must be in shape of target output data
    dataIdxs = np.zeros((nb, nt/strideT, ny/strideY, nx/strideX, nf*strideT*strideY*strideX, 5)).astype(np.int32)
    for iib in range(nb):
        for iit in range(nt):
            for iiy in range(ny):
                for iix in range(nx):
                    for iif in range(nf):
                        #Calculate input indices given output indices
                        oob = iib
                        oot = iit/strideT
                        ooy = iiy/strideY
                        oox = iix/strideX
                        kernelIdx = (iit%strideT)*strideY*strideX + (iiy%strideY)*strideX + (iix%strideX)
                        oof = iif + nf*kernelIdx
                        dataIdxs[oob, oot, ooy, oox, oof, :] = [iib, iit, iiy, iix, iif]
    return tf.gather_nd(x, dataIdxs)

#Undo transepost5dData
def undoTranspose5dData(x, xShape, strideT, strideY, strideX):
    #These shapes are in terms of the orig image
    [nb, nt, ny, nx, nf] = xShape
    print "Building output indices for conv3d"
    #Build gather indices for output
    #Must be in shape of target output data
    dataIdxs = np.zeros((nb, nt, ny, nx, nf, 5)).astype(np.int32)
    for oob in range(nb):
        for oot in range(nt):
            for ooy in range(ny):
                for oox in range(nx):
                    for oof in range(nf):
                        #Calculate input indices given output indices
                        iib = oob
                        iit = oot/strideT
                        iiy = ooy/strideY
                        iix = oox/strideX
                        kernelIdx = (oot%strideT)*strideY*strideX + (ooy%strideY)*strideX + (oox%strideX)
                        iif = oof + nf*kernelIdx
                        dataIdxs[oob, oot, ooy, oox, oof, :] = [iib, iit, iiy, iix, iif]
    return tf.gather_nd(x, dataIdxs)

#Transposes weight data for viewing
def transpose5dWeight(w, wShape, strideT, strideY, strideX):
    print "Building weight indices for conv3d"
    #These shapes are in terms of the already strided values
    [ntp, nyp, nxp, nifp, nofp] = wShape
    #Translate to target output shape
    ntp *= strideT
    nyp *= strideY
    nxp *= strideX
    nofp = nofp/(strideT*strideX*strideY)

    #Build gather indices for weights
    #Must be in shape of target output weights
    weightIdxs = np.zeros((ntp, nyp, nxp, nifp, nofp, 5)).astype(np.int32)
    #Adding kernel number to end of features
    for otp in range(ntp):
        for oyp in range(nyp):
            for oxp in range(nxp):
                for oifp in range(nifp):
                    for oofp in range(nofp):
                        #Calculate output indices given input indices
                        #Must reverse, as we're using conv2d as transpose conv2d
                        #otp = int((ntp-itp-1)/strideT)
                        #oyp = int((nyp-iyp-1)/strideY)
                        #oxp = int((nxp-ixp-1)/strideX)
                        #oifp = iifp #Input features stay the same
                        itp = int((ntp - otp-1)/strideT)
                        iyp = int((nyp - oyp-1)/strideY)
                        ixp = int((nxp - oxp-1)/strideX)
                        iifp=oifp
                        #oofp uses iofp as offset, plus an nf stride based on which kernel it belongs to
                        kernelIdx = (otp%strideT)*strideY*strideX + (oyp%strideY)*strideX + (oxp%strideX)
                        iofp = oofp + nofp * kernelIdx
                        weightIdxs[otp, oyp, oxp, oifp, oofp, :] = [itp, iyp, ixp, iifp, iofp]
    return tf.gather_nd(w, weightIdxs)

def conv3d_oneToMany(x, xShape, w, wShape, strideT, strideY, strideX, inName):
    [ntp, nyp, nxp, nifp, nofp] = wShape
    [nb, nt, ny, nx, nf] = xShape

    #stride must be divisible by both weights and input
    assert(ntp%strideT == 0)
    assert(nyp%strideY == 0)
    assert(nxp%strideX == 0)
    assert(nt%strideT == 0)
    assert(ny%strideY == 0)
    assert(nx%strideX == 0)

    assert(nifp == nf)

    print "Building weight indices for conv3d"
    #Build gather indices for weights
    #Must be in shape of target output weights
    weightIdxs = np.zeros((int(ntp/strideT), int(nyp/strideY), int(nxp/strideX), nifp, nofp*strideT*strideX*strideY, 5)).astype(np.int32)
    #Adding kernel number to end of features
    for itp in range(ntp):
        for iyp in range(nyp):
            for ixp in range(nxp):
                for iifp in range(nifp):
                    for iofp in range(nofp):
                        #Calculate output indices given input indices
                        #Must reverse, as we're using conv2d as transpose conv2d
                        otp = int((ntp-itp-1)/strideT)
                        oyp = int((nyp-iyp-1)/strideY)
                        oxp = int((nxp-ixp-1)/strideX)
                        oifp = iifp #Input features stay the same
                        #oofp uses iofp as offset, plus an nf stride based on which kernel it belongs to
                        kernelIdx = (itp%strideT)*strideY*strideX + (iyp%strideY)*strideX + (ixp%strideX)
                        oofp = iofp + nofp * kernelIdx
                        weightIdxs[otp, oyp, oxp, oifp, oofp, :] = [itp, iyp, ixp, iifp, iofp]


    print "Building output indices for conv3d"
    #Build gather indices for output
    #Must be in shape of target output data
    dataIdxs = np.zeros((nb, nt*strideT, ny*strideY, nx*strideX, nofp, 5)).astype(np.int32)
    for oob in range(nb):
        for oot in range(nt*strideT):
            for ooy in range(ny*strideY):
                for oox in range(nx*strideX):
                    for oof in range(nofp):
                        #Calculate input indices given output indices
                        iib = oob
                        iit = oot/strideT
                        iiy = ooy/strideY
                        iix = oox/strideX
                        kernelIdx = (oot%strideT)*strideY*strideX + (ooy%strideY)*strideX + (oox%strideX)
                        iif = oof + nofp*kernelIdx
                        dataIdxs[oob, oot, ooy, oox, oof, :] = [iib, iit, iiy, iix, iif]

    #Build convolution structure
    w_reshape = tf.gather_nd(w, weightIdxs)
    o_reshape = tf.nn.conv3d(x, w_reshape, strides=[1, 1, 1, 1, 1], padding='SAME', name=inName)
    o = tf.gather_nd(o_reshape, dataIdxs)
    return o

if __name__ == "__main__":
    #For conv2d
    weightShapeOrig = (6, 6, 6, 1, 1)
    stride = 2
    inputShape = (1, 8, 8, 8, 1)

    npWeightArray = np.zeros(weightShapeOrig).astype(np.float32)
    for itp in range(6):
        for iyp in range(6):
            for ixp in range(6):
                idx = itp*36 + iyp*6 + ixp
                npWeightArray[itp, iyp, ixp, 0, 0] = idx

    npInputArray = np.zeros(inputShape).astype(np.float32)
    npInputArray[0, 3, 3, 3, 0] = 1

    #Tensorflow test
    sess=tf.InteractiveSession()
    W = tf.Variable(npWeightArray)
    I = tf.Variable(npInputArray)
    O = conv3d_oneToMany(I, inputShape, W, weightShapeOrig, 2, 2, 2, "test")
    sess.run(tf.initialize_all_variables())

    npI = I.eval()[0, :, :, :, 0]
    npW = W.eval()[:, :, :, 0, 0]
    npO = O.eval()[0, :, :, :, 0]

    pdb.set_trace()
