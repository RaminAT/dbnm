# -*- coding: utf-8 -*-
# An implementation of DBN module using python2
import os
import sys
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import base64
import matplotlib.pyplot as plt
import requests
from sklearn import preprocessing
from sklearn import model_selection


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    if target_type in (np.int64, np.int32):
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)



def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)


def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


def encode_numeric_range(df, name, normalized_low=0, normalized_high=1, 
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
        * (normalized_high - normalized_low) + normalized_low

import numpy as num
import gnumpy as gnp

class Binary(object):
    def activate(self, netInput):
        return netInput.sigmoid()
    def sampleStates(self, acts):
        return gnp.rand(*acts.shape) <= acts

class Gaussian(object):
    def activate(self, netInput):
        return netInput
    def sampleStates(self, acts): #
        return acts + gnp.randn(*acts.shape)

class ReLU(object):
    def __init__(self, krizNoise = False):
        self.krizNoise = krizNoise
    def activate(self, netInput):
        return netInput*(netInput > 0)
    def sampleStates(self, acts):
        if self.krizNoise:
            return self.activate(acts + gnp.randn(*acts.shape))
        tiny = 1e-30
        stddev = gnp.sqrt(acts.sigmoid() + tiny)
        return self.activate( acts + stddev*gnp.randn(*acts.shape) )

def CD1(vis, visToHid, visBias, hidBias, visUnit = Binary(), hidUnit = Binary()):

    posHid = hidUnit.activate(gnp.dot(vis, visToHid) + hidBias)
    posHidStates = hidUnit.sampleStates(posHid)
    
    negVis = visUnit.activate(gnp.dot(posHidStates, visToHid.T) + visBias)
    negHid = hidUnit.activate(gnp.dot(negVis, visToHid) + hidBias)
    
    visHidStats = gnp.dot(vis.T, posHid) - gnp.dot(negVis.T, negHid)
    visBiasStats = vis.sum(axis=0).reshape(*visBias.shape) - negVis.sum(axis=0).reshape(*visBias.shape)
    hidBiasStats = posHid.sum(axis=0).reshape(*hidBias.shape) - negHid.sum(axis=0).reshape(*hidBias.shape)

    return visHidStats, hidBiasStats, visBiasStats, negVis
  
from sys import stderr
class Counter(object):
    def __init__(self, step=10):
        self.cur = 0
        self.step = step

    def tick(self):
        self.cur += 1
        if self.cur % self.step == 0:
            stderr.write( str(self.cur ) )
            stderr.write( "\r" )
            stderr.flush()
        
    def done(self):
        stderr.write( str(self.cur ) )
        stderr.write( "\n" )
        stderr.flush()
class Progress(object):
    def __init__(self, numSteps):
        self.total = numSteps
        self.cur = 0
        self.curPercent = 0
    def tick(self):
        self.cur += 1
        newPercent = (100*self.cur)/self.total
        if newPercent > self.curPercent:
            self.curPercent = newPercent
            stderr.write( str(self.curPercent)+"%" )
            stderr.write( "\r" )
            stderr.flush()
    def done(self):
        stderr.write( '100%' )
        stderr.write( "\n" )
        stderr.flush()
def ProgressLine(line):
    stderr.write(line)
    stderr.write( "\r" )
    stderr.flush()

def main():
    from time import sleep
    for i in range(500):
        s = str(2.379*i)
        ProgressLine(s)
        sleep(0.02)
    c = Counter(5)
    for i in range(500):
        c.tick()
        sleep(.005)
    c.done()
    p = Progress(5000)
    for i in range(5000):
        p.tick()
        sleep(.0005)
    p.done()


if __name__ == "__main__":
    main()
    
    

import numpy as num
import gnumpy as gnp
class Sigmoid(object):
    def activation(self, netInput):
        return netInput.sigmoid()
    def dEdNetInput(self, acts):
        return acts*(1-acts)
    def error(self, targets, netInput, acts = None):
        return (netInput.log_1_plus_exp()-targets*netInput).sum()
    def HProd(self, vect, acts):
        return vect*acts*(1-acts)
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
class Tanh(object):
    def activation(self, netInput):
        return gnp.tanh(netInput)
    def dEdNetInput(self, acts):
        return 1-acts*acts

class ReLU(object):
    def activation(self, netInput):
        return netInput*(netInput > 0)
    def dEdNetInput(self, acts):
        return acts > 0

class Linear(object):
    def activation(self, netInput):
        return netInput
    def dEdNetInput(self, acts):
        return 1
    def error(self, targets, netInput, acts = None):
        diff = targets-netInput
        return 0.5*(diff*diff).sum()
    def HProd(self, vect, acts):
        return vect
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets

class Softmax(object):
    def activation(self, netInput):
        Zshape = (netInput.shape[0],1)
        acts = netInput - netInput.max(axis=1).reshape(*Zshape)
        acts = acts.exp()
        return acts/acts.sum(axis=1).reshape(*Zshape)
    def HProd(self, vect, acts):
        return acts*(vect-(acts*vect).sum(1).reshape(-1,1))
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    def error(self, targets, netInput, acts = None):
        ntInpt = netInput - netInput.max(axis=1).reshape(netInput.shape[0],1)
        logZs = ntInpt.exp().sum(axis=1).log().reshape(-1,1)
        err = targets*(ntInpt - logZs)
        return -err.sum()
import numpy as num
import gnumpy as gnp
import itertools
class DummyProgBar(object):
    def __init__(self, *args): pass
    def tick(self): pass
    def done(self): pass

def initWeightMatrix(shape, scale, maxNonZeroPerColumn = None, uniform = False):
    fanIn = shape[0] if maxNonZeroPerColumn==None else min(maxNonZeroPerColumn, shape[0])
    if uniform:
        W = scale*(2*num.random.rand(*shape)-1)
    else:
        W = scale*num.random.randn(*shape)
    for j in range(shape[1]):
        perm = num.random.permutation(shape[0])
        W[perm[fanIn:],j] *= 0
    return W

def validShapes(weights, biases):
    if len(weights) + 1 == len(biases):
        t1 = all(b.shape[0] == 1 for b in biases)
        t2 = all(wA.shape[1] == wB.shape[0] for wA, wB in zip(weights[:-1], weights[1:]))
        t3 = all(w.shape[1] == hb.shape[1] for w, hb in zip(weights, biases[1:]))
        t4 = all(w.shape[0] == vb.shape[1] for w, vb in zip(weights, biases[:-1]))
        return t1 and t2 and t3 and t4
    return False

def garrayify(arrays):
    return [ar if isinstance(ar, gnp.garray) else gnp.garray(ar) for ar in arrays]

def numpyify(arrays):
    return [ar if isinstance(ar, num.ndarray) else ar.as_numpy_array(dtype=num.float32) for ar in arrays]

def loadDBN(path, outputActFunct, realValuedVis = False, useReLU = False):
    fd = open(path, 'rb')
    d = num.load(fd)
    weights = garrayify(d['weights'].flatten())
    biases = garrayify(d['biases'].flatten())
    genBiases = []
    if 'genBiases' in d:
        genBiases = garrayify(d['genBiases'].flatten())
    fd.close()
    return DBN(weights, biases, genBiases, outputActFunct, realValuedVis, useReLU)

def buildDBN(layerSizes, scales, fanOuts, outputActFunct, realValuedVis, useReLU = False, uniforms = None):
    shapes = [(layerSizes[i-1],layerSizes[i]) for i in range(1, len(layerSizes))]
    assert(len(scales) == len(shapes) == len(fanOuts))
    if uniforms == None:
        uniforms = [False for s in shapes]
    assert(len(scales) == len(uniforms))
    
    initialBiases = [gnp.garray(0*num.random.rand(1, layerSizes[i])) for i in range(1, len(layerSizes))]
    initialGenBiases = [gnp.garray(0*num.random.rand(1, layerSizes[i])) for i in range(len(layerSizes) - 1)]
    initialWeights = [gnp.garray(initWeightMatrix(shapes[i], scales[i], fanOuts[i], uniforms[i])) \
                      for i in range(len(shapes))]
    
    net = DBN(initialWeights, initialBiases, initialGenBiases, outputActFunct, realValuedVis, useReLU)
    return net

def columnRMS(W):
    return gnp.sqrt(gnp.mean(W*W,axis=0))

def limitColumnRMS(W, rmsLim):
    rmsScale = rmsLim/columnRMS(W)
    return W*(1 + (rmsScale < 1)*(rmsScale-1))
    
class DBN(object):
    def __init__(self, initialWeights, initialBiases, initialGenBiases, outputActFunct, realValuedVis = False, useReLU = False):
        self.realValuedVis = realValuedVis
        self.learnRates = [0.05 for i in range(len(initialWeights))]
        self.momentum = 0.9
        self.L2Costs = [0.0001 for i in range(len(initialWeights))]
        self.dropouts = [0 for i in range(len(initialWeights))]
        self.nesterov = False
        self.nestCompare = False
        self.rmsLims = [None for i in range(len(initialWeights))]
        
        if self.realValuedVis:
            self.learnRates[0] = 0.005
        
        self.weights = initialWeights
        self.biases = initialBiases
        self.genBiases = initialGenBiases
        
        if useReLU:
            self.RBMHidUnitType = RBMReLU()
            self.hidActFuncts = [ReLU() for i in range(len(self.weights) - 1)]
        else:
            self.RBMHidUnitType = RBMBinary()
            self.hidActFuncts = [Sigmoid() for i in range(len(self.weights) - 1)]
        self.outputActFunct = outputActFunct
        self.WGrads = [gnp.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        self.biasGrads = [gnp.zeros(self.biases[i].shape) for i in range(len(self.biases))]
    
    def weightsDict(self):
        d = {}
        if len(self.weights) == 1:
            d['weights'] = num.empty((1,), dtype=num.object)
            d['weights'][0] = numpyify(self.weights)[0]
            d['biases'] = num.empty((1,), dtype=num.object)
            d['biases'][0] = numpyify(self.biases)[0]
        else:
            d['weights'] = num.array(numpyify(self.weights)).flatten()
            d['biases'] = num.array(numpyify(self.biases)).flatten()
            if len(self.genBiases) == 1:
                d['genBiases'] = num.empty((1,), dtype=num.object)
                d['genBiases'][0] = numpyify(self.genBiases)[0]
            else:
                d['genBiases'] = num.array(numpyify(self.genBiases)).flatten()
        return d
    
    def scaleDerivs(self, scale):
        for i in range(len(self.weights)):
            self.WGrads[i] *= scale
            self.biasGrads[i] *= scale
    
    def loadWeights(self, path, layersToLoad = None):
        fd = open(path, 'rb')
        d = num.load(fd)
        if layersToLoad != None:
            self.weights[:layersToLoad] = garrayify(d['weights'].flatten())[:layersToLoad]
            self.biases[:layersToLoad] = garrayify(d['biases'].flatten())[:layersToLoad]
            self.genBiases[:layersToLoad] = garrayify(d['genBiases'].flatten())[:layersToLoad] #this might not be quite right
        else:
            self.weights = garrayify(d['weights'].flatten())
            self.biases = garrayify(d['biases'].flatten())
            if 'genBiases' in d:
                self.genBiases = garrayify(d['genBiases'].flatten())
            else:
                self.genBiases = []
        fd.close()
    
    def saveWeights(self, path):
        num.savez(path, **self.weightsDict())
        
    def preTrainIth(self, i, minibatchStream, epochs, mbPerEpoch):
        self.dW = gnp.zeros(self.weights[i].shape)
        self.dvb = gnp.zeros(self.genBiases[i].shape)
        self.dhb = gnp.zeros(self.biases[i].shape)
        
        for ep in range(epochs):
            recErr = 0
            totalCases = 0
            for j in range(mbPerEpoch):
                inpMB = minibatchStream.next()
                curRecErr = self.CDStep(inpMB, i, self.learnRates[i], self.momentum, self.L2Costs[i])
                recErr += curRecErr
                totalCases += inpMB.shape[0]
            yield recErr/float(totalCases)
    
    def fineTune(self, minibatchStream, epochs, mbPerEpoch, loss = None, progressBar = True, useDropout = False):
        for ep in range(epochs):
            totalCases = 0
            sumErr = 0
            sumLoss = 0
            if self.nesterov:
                step = self.stepNesterov
            else:
                step = self.step
            prog = Progress(mbPerEpoch) if progressBar else DummyProgBar()
            for i in range(mbPerEpoch):
                inpMB, targMB = minibatchStream.next()
                err, outMB = step(inpMB, targMB, self.learnRates, self.momentum, self.L2Costs, useDropout)
                sumErr += err
                if loss != None:
                    sumLoss += loss(targMB, outMB)
                totalCases += inpMB.shape[0]
                prog.tick()
            prog.done()
            yield sumErr/float(totalCases), sumLoss/float(totalCases)            
    
    def totalLoss(self, minibatchStream, lossFuncts):
        totalCases = 0
        sumLosses = num.zeros((1+len(lossFuncts),))
        for inpMB, targMB in minibatchStream:
            inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
            targetBatch = targMB if isinstance(targMB, gnp.garray) else gnp.garray(targMB)

            outputActs = self.fprop(inputBatch)
            sumLosses[0] += self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
            for j,f in enumerate(lossFuncts):
                sumLosses[j+1] += f(targetBatch, outputActs)
            totalCases += inpMB.shape[0]
        return sumLosses / float(totalCases)

    def predictions(self, minibatchStream, asNumpy = False):
        for inpMB in minibatchStream:
            inputBatch = inpMB if isinstance(inpMB, gnp.garray) else gnp.garray(inpMB)
            outputActs = self.fprop(inputBatch)
            yield outputActs.as_numpy_array() if asNumpy else outputActs

    def CDStep(self, inputBatch, layer, learnRate, momentum, L2Cost = 0):
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        mbsz = inputBatch.shape[0]
        vis = self.fprop(inputBatch, layer)
        GRBMFlag = layer==0 and self.realValuedVis
        visType = RBMGaussian() if GRBMFlag else self.RBMHidUnitType
        visHidStats, hidBiasStats, visBiasStats, negVis = \
                     CD1(vis, self.weights[layer], self.genBiases[layer], self.biases[layer], visType, self.RBMHidUnitType)
        factor = 1-momentum if not self.nestCompare else 1
        self.dW = momentum*self.dW + factor*visHidStats
        self.dvb = momentum*self.dvb + factor*visBiasStats
        self.dhb = momentum*self.dhb + factor*hidBiasStats

        if L2Cost > 0:
            self.weights[layer] *= 1-L2Cost*learnRate*factor 
        
        self.weights[layer] += (learnRate/mbsz) * self.dW
        self.genBiases[layer] += (learnRate/mbsz) * self.dvb
        self.biases[layer] += (learnRate/mbsz) * self.dhb
        return gnp.sum((vis-negVis)**2)

    def fpropBprop(self, inputBatch, targetBatch, useDropout):
        if useDropout:
            outputActs = self.fpropDropout(inputBatch)
        else:
            outputActs = self.fprop(inputBatch)
        outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], outputActs)
        error = self.outputActFunct.error(targetBatch, self.state[-1], outputActs)
        errSignals = self.bprop(outputErrSignal)
        return errSignals, outputActs, error
    
    def constrainWeights(self):
        for i in range(len(self.rmsLims)):
            if self.rmsLims[i] != None:
                self.weights[i] = limitColumnRMS(self.weights[i], self.rmsLims[i])
    
    def step(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
        mbsz = inputBatch.shape[0]
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
        
        errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)

        factor = 1-momentum if not self.nestCompare else 1.0
        self.scaleDerivs(momentum)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] += learnRates[i]*factor*(WGrad/mbsz - L2Costs[i]*self.weights[i])
            self.biasGrads[i] += (learnRates[i]*factor/mbsz)*biasGrad
        self.applyUpdates(self.weights, self.biases, self.weights, self.biases, self.WGrads, self.biasGrads)
        self.constrainWeights()
        return error, outputActs
    
    def applyUpdates(self, destWeights, destBiases, curWeights, curBiases, WGrads, biasGrads):
        for i in range(len(destWeights)):
            destWeights[i] = curWeights[i] + WGrads[i]
            destBiases[i] = curBiases[i] + biasGrads[i]
    
    def stepNesterov(self, inputBatch, targetBatch, learnRates, momentum, L2Costs, useDropout = False):
        mbsz = inputBatch.shape[0]
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
        
        curWeights = [w.copy() for w in self.weights]
        curBiases = [b.copy() for b in self.biases]
        self.scaleDerivs(momentum)
        self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)
        
        errSignals, outputActs, error = self.fpropBprop(inputBatch, targetBatch, useDropout)
        
        #self.scaleDerivs(momentum)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] += learnRates[i]*(WGrad/mbsz - L2Costs[i]*self.weights[i])
            self.biasGrads[i] += (learnRates[i]/mbsz)*biasGrad

        self.applyUpdates(self.weights, self.biases, curWeights, curBiases, self.WGrads, self.biasGrads)
        self.constrainWeights()
        return error, outputActs

    def gradDebug(self, inputBatch, targetBatch):
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        targetBatch = targetBatch if isinstance(targetBatch, gnp.garray) else gnp.garray(targetBatch)
        

        mbsz = inputBatch.shape[0]
        outputActs = self.fprop(inputBatch)
        outputErrSignal = -self.outputActFunct.dErrordNetInput(targetBatch, self.state[-1], outputActs)
        errSignals = self.bprop(outputErrSignal)
        for i, (WGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.WGrads[i] = WGrad
            self.biasGrads[i] = biasGrad
        allWeightGrads = itertools.chain(self.WGrads, self.biasGrads)
        return gnp.as_numpy_array(gnp.concatenate([dw.ravel() for dw in allWeightGrads])) 
    def fprop(self, inputBatch, weightsToStopBefore = None ):
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        if weightsToStopBefore == None:
            weightsToStopBefore = len(self.weights)
        self.state = [inputBatch]
        for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
            curActs = self.hidActFuncts[i].activation(gnp.dot(self.state[-1], self.weights[i]) + self.biases[i])
            self.state.append(curActs)
        if weightsToStopBefore >= len(self.weights):
            self.state.append(gnp.dot(self.state[-1], self.weights[-1]) + self.biases[-1])
            self.acts = self.outputActFunct.activation(self.state[-1])
            return self.acts
        return self.state[weightsToStopBefore]
    
    def fpropDropout(self, inputBatch, weightsToStopBefore = None ):
        inputBatch = inputBatch if isinstance(inputBatch, gnp.garray) else gnp.garray(inputBatch)
        if weightsToStopBefore == None:
            weightsToStopBefore = len(self.weights)
        self.state = [inputBatch * (gnp.rand(*inputBatch.shape) > self.dropouts[0])]
        for i in range(min(len(self.weights) - 1, weightsToStopBefore)):
            dropoutMultiplier = 1.0/(1.0-self.dropouts[i])
            curActs = self.hidActFuncts[i].activation(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[i]) + self.biases[i])
            self.state.append(curActs * (gnp.rand(*curActs.shape) > self.dropouts[i+1]) )
        if weightsToStopBefore >= len(self.weights):
            dropoutMultiplier = 1.0/(1.0-self.dropouts[-1])
            self.state.append(gnp.dot(dropoutMultiplier*self.state[-1], self.weights[-1]) + self.biases[-1])
            self.acts = self.outputActFunct.activation(self.state[-1])
            return self.acts
        return self.state[weightsToStopBefore]

    def bprop(self, outputErrSignal, fpropState = None):
        if fpropState == None:
            fpropState = self.state
        assert(len(fpropState) == len(self.weights) + 1)

        errSignals = [None for i in range(len(self.weights))]
        errSignals[-1] = outputErrSignal
        for i in reversed(range(len(self.weights) - 1)):
            errSignals[i] = gnp.dot(errSignals[i+1], self.weights[i+1].T)*self.hidActFuncts[i].dEdNetInput(fpropState[i+1])
        return errSignals
    def gradients(self, fpropState, errSignals):
        assert(len(fpropState) == len(self.weights)+1)
        assert(len(errSignals) == len(self.weights) == len(self.biases))
        for i in range(len(self.weights)):
            yield gnp.dot(fpropState[i].T, errSignals[i]), errSignals[i].sum(axis=0)
from datetime import timedelta
from time import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

warnings.warn("""\
warnings...
""")
class DBN(BaseEstimator):
    def __init__(
        self,
        layer_sizes=None,
        scales=0.05,
        fan_outs=None,
        output_act_funct=None,
        real_valued_vis=True,
        use_re_lu=True,
        uniforms=False,

        learn_rates=0.1,
        learn_rate_decays=1.0,
        learn_rate_minimums=0.0,
        momentum=0.9,
        l2_costs=0.0001,
        dropouts=0,
        nesterov=True,
        nest_compare=True,
        rms_lims=None,

        learn_rates_pretrain=None,
        momentum_pretrain=None,
        l2_costs_pretrain=None,
        nest_compare_pretrain=None,

        epochs=10,
        epochs_pretrain=0,
        loss_funct=None,
        minibatch_size=64,
        minibatches_per_epoch=None,

        pretrain_callback=None,
        fine_tune_callback=None,

        random_state=None,

        verbose=0,
        ):

        if layer_sizes is None:
            layer_sizes = [-1, -1]

        if output_act_funct is None:
            output_act_funct = Softmax()
        if random_state is not None:
            raise ValueError("random_sate must be an int")
        self.layer_sizes = layer_sizes
        self.scales = scales
        self.fan_outs = fan_outs
        self.output_act_funct = output_act_funct
        self.real_valued_vis = real_valued_vis
        self.use_re_lu = use_re_lu
        self.uniforms = uniforms

        self.learn_rates = learn_rates
        self.learn_rate_decays = learn_rate_decays
        self.learn_rate_minimums = learn_rate_minimums
        self.momentum = momentum
        self.l2_costs = l2_costs
        self.dropouts = dropouts
        self.nesterov = nesterov
        self.nest_compare = nest_compare
        self.rms_lims = rms_lims

        self.learn_rates_pretrain = learn_rates_pretrain
        self.momentum_pretrain = momentum_pretrain
        self.l2_costs_pretrain = l2_costs_pretrain
        self.nest_compare_pretrain = nest_compare_pretrain

        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.loss_funct = loss_funct
        self.use_dropout = True if dropouts else False
        self.minibatch_size = minibatch_size
        self.minibatches_per_epoch = minibatches_per_epoch

        self.pretrain_callback = pretrain_callback
        self.fine_tune_callback = fine_tune_callback
        self.random_state = random_state
        self.verbose = verbose

    def _fill_missing_layer_sizes(self, X, y):
        layer_sizes = self.layer_sizes
        if layer_sizes[0] == -1:
            layer_sizes[0] = X.shape[1]
        if layer_sizes[-1] == -1 and y is not None:
            layer_sizes[-1] = y.shape[1]

    def _vp(self, value):
        num_weights = len(self.layer_sizes) - 1
        if not hasattr(value, '__iter__'):
            value = [value] * num_weights
        return list(value)

    def _build_net(self, X, y=None):
        v = self._vp

        self._fill_missing_layer_sizes(X, y)
        if self.verbose:
            print("[DBN] layers {}".format(self.layer_sizes))

        if self.random_state is not None:
            np.random.seed(self.random_state)

        net = buildDBN(
            self.layer_sizes,
            v(self.scales),
            v(self.fan_outs),
            self.output_act_funct,
            self.real_valued_vis,
            self.use_re_lu,
            v(self.uniforms),
            )

        return net

    def _configure_net_pretrain(self, net):
        v = self._vp

        self._configure_net_finetune(net)

        learn_rates = self.learn_rates_pretrain
        momentum = self.momentum_pretrain
        l2_costs = self.l2_costs_pretrain
        nest_compare = self.nest_compare_pretrain

        if learn_rates is None:
            learn_rates = self.learn_rates
        if momentum is None:
            momentum = self.momentum
        if l2_costs is None:
            l2_costs = self.l2_costs
        if nest_compare is None:
            nest_compare = self.nest_compare

        net.learnRates = v(learn_rates)
        net.momentum = momentum
        net.L2Costs = v(l2_costs)
        net.nestCompare = nest_compare

        return net

    def _configure_net_finetune(self, net):
        v = self._vp

        net.learnRates = v(self.learn_rates)
        net.momentum = self.momentum
        net.L2Costs = v(self.l2_costs)
        net.dropouts = v(self.dropouts)
        net.nesterov = self.nesterov
        net.nestCompare = self.nest_compare
        net.rmsLims = v(self.rms_lims)

        return net

    def _minibatches(self, X, y=None):
        while True:
            idx = np.random.randint(X.shape[0], size=(self.minibatch_size,))

            X_batch = X[idx]
            if hasattr(X_batch, 'todense'):
                X_batch = X_batch.todense()

            if y is not None:
                yield (X_batch, y[idx])
            else:
                yield X_batch

    def _onehot(self, y):
        return np.array(
            OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense())

    def _num_mistakes(self, targets, outputs):
        if hasattr(targets, 'as_numpy_array'):
            targets = targets.as_numpy_array()
        if hasattr(outputs, 'as_numpy_array'):
            outputs = outputs.as_numpy_array()
        return np.sum(outputs.argmax(1) != targets.argmax(1))

    def _learn_rate_adjust(self):
        if self.learn_rate_decays == 1.0:
            return

        learn_rate_decays = self._vp(self.learn_rate_decays)
        learn_rate_minimums = self._vp(self.learn_rate_minimums)

        for index, decay in enumerate(learn_rate_decays):
            new_learn_rate = self.net_.learnRates[index] * decay
            if new_learn_rate >= learn_rate_minimums[index]:
                self.net_.learnRates[index] = new_learn_rate

        if self.verbose >= 2:
            print("Learn rates: {}".format(self.net_.learnRates))

    def fit(self, X, y):
        if self.verbose:
            print("[DBN] fitting X.shape=%s" % (X.shape,))
        self._enc = LabelEncoder()
        y = self._enc.fit_transform(y)
        y = self._onehot(y)

        self.net_ = self._build_net(X, y)

        minibatches_per_epoch = self.minibatches_per_epoch
        if minibatches_per_epoch is None:
            minibatches_per_epoch = X.shape[0] / self.minibatch_size

        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        errors_pretrain = self.errors_pretrain_ = []
        losses_fine_tune = self.losses_fine_tune_ = []
        errors_fine_tune = self.errors_fine_tune_ = []

        if self.epochs_pretrain:
            self.epochs_pretrain = self._vp(self.epochs_pretrain)
            self._configure_net_pretrain(self.net_)
            for layer_index in range(len(self.layer_sizes) - 1):
                errors_pretrain.append([])
                if self.verbose:
                    print("[DBN] Pre-train layer {}...".format(layer_index + 1))
                time0 = time()
                for epoch, err in enumerate(
                    self.net_.preTrainIth(
                        layer_index,
                        self._minibatches(X),
                        self.epochs_pretrain[layer_index],
                        minibatches_per_epoch,
                        )):
                    errors_pretrain[-1].append(err)
                    if self.verbose:  # pragma: no cover
                        print("  Epoch {}: err {}".format(epoch + 1, err))
                        elapsed = str(timedelta(seconds=time() - time0))
                        print("  ({})".format(elapsed.split('.')[0]))
                        time0 = time()
                    if self.pretrain_callback is not None:
                        self.pretrain_callback(
                            self, epoch + 1, layer_index)

        self._configure_net_finetune(self.net_)
        if self.verbose:
            print("[DBN] Fine-tune...")
        time0 = time()
        for epoch, (loss, err) in enumerate(
            self.net_.fineTune(
                self._minibatches(X, y),
                self.epochs,
                minibatches_per_epoch,
                loss_funct,
                self.verbose,
                self.use_dropout,
                )):
            losses_fine_tune.append(loss)
            errors_fine_tune.append(err)
            self._learn_rate_adjust()
            if self.verbose:
                print("Epoch {}:".format(epoch + 1))
                print("  loss {}".format(loss))
                print("  err  {}".format(err))
                elapsed = str(timedelta(seconds=time() - time0))
                print("  ({})".format(elapsed.split('.')[0]))
                time0 = time()
            if self.fine_tune_callback is not None:
                self.fine_tune_callback(self, epoch + 1)

    def predict(self, X):
        y_ind = np.argmax(self.predict_proba(X), axis=1)
        return self._enc.inverse_transform(y_ind)

    def predict_proba(self, X):
        if hasattr(X, 'todense'):
            return self._predict_proba_sparse(X)
        res = np.zeros((X.shape[0], self.layer_sizes[-1]))
        for i, el in enumerate(self.net_.predictions(X, asNumpy=True)):
            res[i] = el
        return res

    def _predict_proba_sparse(self, X):
        batch_size = self.minibatch_size
        res = []
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:min(i + batch_size, X.shape[0])].todense()
            res.extend(self.net_.predictions(X_batch))
        return np.array(res).reshape(X.shape[0], -1)

    def score(self, X, y):
        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        outputs = self.predict_proba(X)
        targets = self._onehot(self._enc.transform(y))
        mistakes = loss_funct(outputs, targets)
        return - float(mistakes) / len(y) + 1

    @property
    def classes_(self):
        return self._enc.classes_

!pip install gnumpy
