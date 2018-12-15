# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.

datasetFolder = '/home/shahsavari/dataset_0.1_complete'
frameSize = 0.025
frameStep = 0.010
endToEndSamplingRate = 8000
endToEndFrameSize = 0.100
endToEndFrameStep = 0.020
postprocessIntervalSize = 0.500

import logging
import numpy as np
import os
import pickle
import sys

class Logger():
    folderName = None
    command = None
    csvFileName = None

    def __init__(self, folderName, command):
        self.command = command
        self.folderName = folderName

        logFileName = os.path.join(self.folderName, 'output.log')
        logging.basicConfig(filename=logFileName, level=logging.DEBUG, format='%(asctime)s %(message)s')

        self.csvFileName = os.path.join(self.folderName, 'output.csv')
        if not os.path.isfile(self.csvFileName):
            self.printCSV([command])
            self.printCSV(['epoch','time', 'trainloss', 'validloss', 'validaccuracy'])


    def printInfo(self, a):
        logging.info(a)

    def printCSV(self, plist):
        csvFile = open(self.csvFileName , 'a')
        for i in range(len(plist)):
            plist[i]=str(plist[i])
        out = ','.join(plist)
        csvFile.write(out + '\n')
        csvFile.close()

    def printDynamicCMD(self, a):
        sys.stdout.write("{0}\n".format(a))
        sys.stdout.flush()

    def printErrorCMD(self, a):
        sys.stdout.write("\033[1;31m")
        sys.stdout.write("{0}\n".format(a))
        sys.stdout.write("\033[0m")

    def printSuccessCMD(self, a):
        sys.stdout.write("\033[0;32m")
        sys.stdout.write("{0}\n".format(a))
        sys.stdout.write("\033[0m")

    def saveModel(self, model_params):
        modelFileName = os.path.join(self.folderName, 'best-model.npz')
        np.savez(modelFileName, *model_params)

    def loadModel(self):
        modelFileName = os.path.join(self.folderName, 'best-model.npz')
        with np.load(modelFileName) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        return param_values

    def isFirstEpoch(self):
        initParamsFolder = os.path.join(self.folderName, 'init params')
        if os.path.isdir(initParamsFolder):
            initParamsFile = os.path.join(initParamsFolder, 'params-1.pkl')
            if(os.path.isfile(initParamsFile)):
                return False
        return True

    def saveInitTrainParams(self, epoch,
                            currentModelParams,
                            yellow_cards,
                            learning_rate,
                            best_validation_loss,
                            best_model_params):

        def save_object(filename, obj):
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        initParams = InitParams(epoch, currentModelParams, yellow_cards, learning_rate, best_validation_loss, best_model_params)
        initParamsFolder = os.path.join(self.folderName, 'init params')
        if not os.path.isdir(initParamsFolder):
            os.mkdir(initParamsFolder)
        initParamsFile = os.path.join(initParamsFolder, 'params-' + str(epoch) + '.pkl')
        save_object(initParamsFile, initParams)

    def loadInitTrainParams(self):
        i = 1
        initParamsFolder = os.path.join(self.folderName, 'init params')
        while os.path.isfile( os.path.join(initParamsFolder, 'params-' + str(i) + '.pkl') ):
            i += 1
        initParamsFile = os.path.join(initParamsFolder, 'params-' + str(i-1) + '.pkl')
        with open(initParamsFile, 'rb') as input:
            initParams = pickle.load(input)

        return initParams.epoch, \
               initParams.currentModelParams, \
               initParams.yellow_cards, \
               initParams.learning_rate, \
               initParams.best_validation_loss, \
               initParams.best_model_params

    def saveLabels(self, folder, featureFile, predicted_labels):

        def milisecToSegTime(t):
            def _2zeroPadding(x):
                xStr = str(x) if x >= 10 else '0' + str(x)
                return xStr

            def _3zeroPadding(x):
                xStr = str(x) if x >= 100 else '0' + str(x) if x >= 10 else '00' + str(x)
                return xStr

            hours = int(t / (1000 * 60 * 60))
            hoursStr = _2zeroPadding(hours)

            mins = int(t / (1000 * 60)) - hours * 60
            minsStr = _2zeroPadding(mins)

            secs = int(t / 1000) - hours * 60 * 60 - mins * 60
            secsStr = _2zeroPadding(secs)

            milisecs = int(round(t - hours * 60 * 60 * 1000 - mins * 60 * 1000 - secs * 1000, 3))
            milisecsStr = _3zeroPadding(milisecs)
            return hoursStr + ':' + minsStr + ':' + secsStr + ',' + milisecsStr

        def numericLabelToString(label):
            if (label == 1):
                return 'speech'
            if (label == 0):
                return 'nonspeech'

            return 'none!'

        def printSegment(fileName, index, startTime, endTime, pLabel):
            f = open(fileName, 'a+')
            f.write(str(index) + '\n')
            f.write(milisecToSegTime(startTime*1000))
            f.write(' --> ')
            f.write(milisecToSegTime(endTime*1000))
            f.write('\n')
            f.write(numericLabelToString(pLabel))
            f.write('\n\n')
            f.close()

        def getFileName(file):
            (drive, basename) = os.path.split(file)
            (name, ext) = os.path.splitext(basename)
            return name

        srtFile = os.path.join(folder, getFileName(featureFile) + '.txt')

        startTime = 0
        endTime = 0
        srtIndex = 1
        i = 0

        while i < len(predicted_labels):

            startIndex = i
            i += 1
            endTime += frameStep

            while i < len(predicted_labels) and predicted_labels[i] == predicted_labels[startIndex]:
                endTime += frameStep
                i += 1

            printSegment(srtFile, srtIndex, startTime, endTime, predicted_labels[startIndex])

            startTime = endTime
            srtIndex+=1

        return

class InitParams():
    def __init__(self,  epoch, currentModelParams, yellow_cards, learning_rate, best_validation_loss, best_model_params):
        self.epoch = epoch
        self.currentModelParams = currentModelParams
        self.yellow_cards = yellow_cards
        self.learning_rate = learning_rate
        self.best_validation_loss = best_validation_loss
        self.best_model_params = best_model_params

