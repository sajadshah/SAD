# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.

import json

import scipy.io.wavfile
from srt import parse

from config import *
import os
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import timeit


def loadDataFromFeatureFiles(featuresFolder, n_context, logger):
    logger.printInfo('reading data from: {}'.format(featuresFolder))

    xData = []
    yData = []
    fnData = []
    fileNames = []
    dataSize = 0;
    for featureFile in os.listdir(featuresFolder):
        if featureFile.endswith(".fea"):
            featureFile = os.path.join(featuresFolder, featureFile)
            fileNames.append(featureFile)
            fnIndex = len(fileNames) - 1

            #logger.printInfo('reading data from: {}'.format(featureFile))
            for line in open(featureFile):
                fv = [float(x) if x!='NaN' else 0 for x in line.rstrip().split(" ")]
                dataSize += 1;
                x = fv[1:]
                y = int(fv[0])
                fnData.append(fnIndex)
                xData.append(x)
                yData.append(y)
    # logger.printInfo('{} frames has been read'.format(dataSize))
    xData = np.asarray(xData, dtype=np.float32)
    context = []
    yData = np.asarray(yData, dtype=np.int32)
    fnData = np.asarray(fnData, dtype=np.int32)
    if (n_context == 0):
        return xData, yData, fnData, fileNames

    if (n_context > 0):
        logger.printInfo('...adding context')
        for j in range(-n_context, n_context + 1, 1):

            start_time = timeit.default_timer()
            middle = np.roll(xData, j, axis=0)
            if( j > 0 ):
                middle[:j,:] = 0
            if( j < 0 ):
                for k in range(1, -j+1):
                    middle[-k,:] = 0
            context.append(middle)
            logger.printInfo('context {} took {}'.format(j, timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        xDataWithContext = np.concatenate(context, axis=1)
        logger.printInfo('concatenation took {}'.format(timeit.default_timer() - start_time))
    return xDataWithContext, yData, fnData, fileNames


def normal(x, mean, std):
    x = (x - mean) / std
    return x

def __normalizeData(xTrain, xTest, xValid, logger):
    logger.printInfo('... noramlizing data')
    start_time = timeit.default_timer()

    mean = xTrain.mean(axis=0)
    std = xTrain.std(axis=0)

    xTrain = normal(xTrain, mean, std)
    xTest = normal(xTest, mean, std)
    xValid = normal(xValid, mean, std)

    logger.printInfo('normalization took {}'.format(timeit.default_timer() - start_time))
    return xTrain, xTest, xValid;

def loadSavedData(folder, logger):
    start_time = timeit.default_timer()

    logger.printInfo('... unpickling from {}'.format(folder))
    fileNames = []
    f = open(os.path.join(folder, 'filenames.txt'), 'r')
    for line in f:
        fileNames.append(line)
    f = open(os.path.join(folder, 'others.txt'), 'r')
    n_dim = int(f.readline())
    def loadSavedDataFolder(tName):
        baseFile = os.path.join(folder, tName)
        x = np.load(baseFile + '-x.txt')
        y = np.load(baseFile + '-y.txt')
        fn = np.load(baseFile + '-fn.txt')
        return x, y, fn

    xTrain, yTrain, fnTrain = loadSavedDataFolder('train')
    xTest, yTest, fnTest = loadSavedDataFolder('test')
    xValid, yValid, fnValid = loadSavedDataFolder('valid')
    logger.printInfo('unpickling data took {}s'.format(timeit.default_timer() - start_time))
    return (xTrain, yTrain, fnTrain,
         xTest, yTest, fnTest,
         xValid, yValid, fnValid,
         n_dim, fileNames)
def saveData(folder, xTrain, yTrain, fnTrain, xTest, yTest, fnTest, xValid, yValid, fnValid, n_dim, fileNames):
    os.mkdir(folder)
    f = open(os.path.join(folder, 'filenames.txt'), 'w')
    for x in fileNames:
        f.write(x + '\n')
    f.close()

    f = open(os.path.join(folder, 'others.txt'), 'w')
    f.write(str(n_dim))
    f.close()

    def saveDataFolder(tName, x, y, fn):
        baseFile = os.path.join(folder, tName)
        f = open(baseFile + '-x.txt', 'wb')
        np.save(f, x)
        f = open(baseFile + '-y.txt', 'wb')
        np.save(f, y)
        f = open(baseFile + '-fn.txt', 'wb')
        np.save(f, fn)
        return

    saveDataFolder('train', xTrain, yTrain, fnTrain)
    saveDataFolder('test', xTest, yTest, fnTest)
    saveDataFolder('valid', xValid, yValid, fnValid)


def printInfoDist(y, title, logger):
    logger.printInfo('distribution of {} data:'.format(title))
    total = y.size
    speech = np.count_nonzero(y)
    nonspeech = total - speech
    logger.printInfo('speech   :\t{}/{} ,\t{}% '.format(speech, total, ((speech * 1.0 / total) * 100)))
    logger.printInfo('nonspeech:\t{}/{} ,\t{}% '.format(nonspeech, total, ((nonspeech * 1.0 / total) * 100)))
    logger.printInfo('speech :\t{} seconds'.format(speech * frameStep))
    logger.printInfo('nonspeech :\t{} seconds'.format(nonspeech * frameStep))
    logger.printInfo('\n')

def create_db_info_folder(folder, x, y, n_context):
    infoFile = open(os.path.join(folder, 'data_info.txt'), 'w')
    mean = x.mean(axis=0).tolist()[0:x.shape[1]/(2*n_context+1)]
    std = x.std(axis=0).tolist()[0:x.shape[1]/(2*n_context+1)]
    info = {
        'mean'  : mean,
        'std'   : std
    }
    json.dump(info, infoFile, indent=4)

def loadData(logger, normalize = True, sharedVariable = True, reshapeToChannels = False, n_context_frames = 0):
    start_time = timeit.default_timer()

    def loadDataFolder(folder):
        featuresFolder = os.path.join(folder, 'feature files')
        x, y, fn, fileNames = loadDataFromFeatureFiles(featuresFolder, n_context_frames, logger)
        create_db_info_folder(folder, x, y, n_context_frames)
        return x, y, fn, fileNames

    logger.printInfo('... reading data')
    savedDataFolder = os.path.join(datasetFolder, 'data-cntx' + str(n_context_frames) + '-norm' + str(normalize))
    logger.printInfo(savedDataFolder)
    if os.path.isdir(savedDataFolder):
        (xTrain, yTrain, fnTrain,
         xTest, yTest, fnTest,
         xValid, yValid, fnValid,
         n_dim, fileNames) = loadSavedData(savedDataFolder, logger)
        printInfoDist(yTrain, 'train', logger)
        printInfoDist(yTest, 'test', logger)
        printInfoDist(yValid, 'validation', logger)
        
        return (xTrain, yTrain, fnTrain,
            xTest, yTest, fnTest,
            xValid, yValid, fnValid,
            n_dim, fileNames)

    fileNames = []
    trainFolder = os.path.join(datasetFolder, 'train')
    xTrain, yTrain, fnTrain, fileNamesTrain = loadDataFolder(trainFolder)
    fileNames.extend(fileNamesTrain)
    n_dim = xTrain.shape[1]

    testFolder = os.path.join(datasetFolder, 'test')
    xTest, yTest, fnTest, fileNamesTest = loadDataFolder(testFolder)
    fileNames.extend(fileNamesTest)
    validFolder = os.path.join(datasetFolder, 'validation')
    xValid, yValid, fnValid, fileNamesValid = loadDataFolder(validFolder)
    fileNames.extend(fileNamesValid)

    if(normalize):
        xTrain, xTest, xValid = __normalizeData(xTrain, xTest, xValid, logger)


    printInfoDist(yTrain, 'train', logger)
    printInfoDist(yTest, 'test', logger)
    printInfoDist(yValid, 'validation', logger)

    def shared_dataset(xData, yData, borrow=True):
        shared_x = theano.shared(np.asarray(xData, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(yData, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    if(sharedVariable):
        xTrain, yTrain = shared_dataset(xTrain, yTrain)
        xTest, yTest = shared_dataset(xTest, yTest)
        xValid, yValid= shared_dataset(xValid, yValid)

    if(reshapeToChannels):
        xTrain = xTrain.reshape(-1, 1, 1, n_dim)
        xTest = xTest.reshape(-1, 1, 1, n_dim)
        xValid = xValid.reshape(-1, 1, 1, n_dim)



    p_start_time = timeit.default_timer()
    logger.printInfo('... pickling data')
    saveData(savedDataFolder, xTrain, yTrain, fnTrain, xTest, yTest, fnTest, xValid, yValid, fnValid, n_dim, fileNames)
    logger.printInfo('pickling took:\t%.1fs' % (timeit.default_timer() - p_start_time))

    end_time = timeit.default_timer()
    logger.printInfo('loading data took:\t%.1fs' % (end_time - start_time))
    return (xTrain, yTrain, fnTrain,
            xTest, yTest, fnTest,
            xValid, yValid, fnValid,
            n_dim, fileNames)

def loadDataFile(logger, featureFile, normalize = True, sharedVariable = True, reshapeToChannels = False, n_context_frames = 0 ):
    #print ('loading origin data : {}'.format(featureFile))
    xData = []
    yData = []
    dataSize = 0;
    featureFilename = getFileName(featureFile)
    savedDataFolder = os.path.join(datasetFolder, 'data-cntx' + str(n_context_frames) + '-norm' + str(normalize))
    savedDataFile = os.path.join(savedDataFolder, featureFilename)
    if os.path.isfile(savedDataFile + '-x.txt'):
        start_loading = timeit.default_timer()
        xData = np.load(savedDataFile + '-x.txt')
        yData = np.load(savedDataFile + '-y.txt')
        n_dim = xData.shape[3]
        #print ('loading saved data: {}'.format(timeit.default_timer() - start_loading))
        return (xData, yData, n_dim)

    start_loading = timeit.default_timer()
    if featureFile.endswith(".fea"):
        for line in open(featureFile):
            fv = [float(x) if x != 'NaN' else 0 for x in line.rstrip().split()]
            dataSize += 1;
            x = fv[1:]
            y = int(fv[0])
            xData.append(x)
            yData.append(y)
    else:
        return ([],[],0)
    if (n_context_frames == 0):
        xData = np.asarray(xData, dtype=np.float32)

    if (n_context_frames > 0):
        context = []
        for j in range(-n_context_frames, n_context_frames + 1, 1):

            start_time = timeit.default_timer()
            middle = np.roll(xData, j, axis=0)
            if middle.shape[0] > 0:
                if (j > 0):
                    middle[:j, :] = 0
                if (j < 0):
                    for k in range(1, -j + 1):
                        middle[-k, :] = 0
            context.append(middle)
        xData = np.concatenate(context, axis=1)

    n_dim = xData.shape[1]
    yData = np.asarray(yData, dtype=np.int32)

    if (normalize):
        infoFile = os.path.join(datasetFolder, 'train', 'data_info.txt')
        with open(infoFile) as f:
            info = json.load(f)
        mean = np.asarray(info['mean'], dtype=np.float32).reshape(1,len(info['mean']))
        std = np.asarray(info['std'], dtype=np.float32).reshape(1,len(info['std']))

        meanContext = mean
        stdContext = std
        for i in range(2 * n_context_frames):
            meanContext = np.append(meanContext, mean)
            stdContext = np.append(stdContext, std)
        xData = normal(xData, meanContext, stdContext)

    # printInfoDist(yTrain, 'train', logger)
    # printInfoDist(yTest, 'test', logger)
    # printInfoDist(yValid, 'validation', logger)

    def shared_dataset(xData, yData, borrow=True):
        shared_x = theano.shared(np.asarray(xData, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(yData, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    if (sharedVariable):
        xData, yData = shared_dataset(xData, yData)

    if (reshapeToChannels):
        xData = xData.reshape(-1, 1, 1, n_dim)

    xData = xData.astype(np.float32)

    if not os.path.isdir(savedDataFolder):
        os.mkdir(savedDataFolder)
    f = open(savedDataFile + '-x.txt', 'wb')
    np.save(f, xData)
    f = open(savedDataFile + '-y.txt', 'wb')
    np.save(f, yData)

    return (xData, yData, n_dim)

def loadDataFileEndToEnd(logger, wavFile, normalize = True, sharedVariable = True, reshapeToChannels = False, n_context_frames = 0, frameSize = 100, frameStep = 20 ):
    print wavFile
    wavFilename = getFileName(wavFile)
    savedDataFolder = os.path.join(datasetFolder, 'data-end-to-end-cntx' + str(n_context_frames) + '-norm' + str(normalize) + '-channel' + str(reshapeToChannels) + '-frameSize' + str(frameSize) + '-frameStep' + str(frameStep))
    savedDataFile = os.path.join(savedDataFolder, wavFilename)
    if os.path.isfile(savedDataFile + '-x.txt'):
        start_loading = timeit.default_timer()
        xData = np.load(savedDataFile + '-x.txt')
        yData = np.load(savedDataFile + '-y.txt')
        n_dim = xData.shape[3]
        print ('loading saved data: {} no-frames: {}'.format(timeit.default_timer() - start_loading, xData.shape[0]))
        return (xData, yData, n_dim)

    start_loading = timeit.default_timer()
    rate, signal = scipy.io.wavfile.read(wavFile)

    srtFile = wavFile.replace('wav', 'txt')
    segments = parse(srtFile)
    try :
        dataSize = 0
        n_dim = int(frameSize* endToEndSamplingRate/1000)
        for s in segments:
            time_begin = s[0].ms
            time_end = s[1].ms
            n_frames = int((time_end - time_begin) / frameStep)
            dataSize += n_frames

        xData = np.zeros(shape=(dataSize, n_dim), dtype=np.float32)
        yData = np.zeros(shape=(dataSize), dtype=np.int32)
        dataIndex = 0
        for s in segments:
            time_begin = s[0].ms
            time_end = s[1].ms
            n_frames = int((time_end - time_begin) / frameStep)
            label = 1 if s[2] == 'speech' else 0
            for n in range(n_frames):
                start_ms = time_begin + n * frameStep * 1.0
                start_index = int((start_ms / 1000) * endToEndSamplingRate)
                end_index = start_index + n_dim
                x = signal[start_index:end_index]
                if len(x) == n_dim:
                    xData[dataIndex, :] = x
                    yData[dataIndex] = label
                    dataIndex += 1
                else:
                    logger.printErrorCMD('error loading {}th frame of file : {}'.format(n, wavFile))

        if (n_context_frames > 0):
            context = []
            for j in range(-n_context_frames, n_context_frames + 1, 1):
                middle = np.roll(xData, j, axis=0)
                if middle.shape[0] > 0:
                    if (j > 0):
                        middle[:j,:] = 0
                    if (j < 0):
                        for k in range(1, -j + 1):
                            middle[-k,:] = 0
                context.append(middle)
            xData = np.concatenate(context, axis=1)

        if (normalize):
            mean = xData.mean(axis=0)
            std = xData.std(axis=0)

            xData = normal(xData, mean, std)

        # printInfoDist(yTrain, 'train', logger)
        # printInfoDist(yTest, 'test', logger)
        # printInfoDist(yValid, 'validation', logger)

        def shared_dataset(xData, yData, borrow=True):
            shared_x = theano.shared(np.asarray(xData, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(np.asarray(yData, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        if (sharedVariable):
            xData, yData = shared_dataset(xData, yData)

        if (reshapeToChannels):
            xData = xData.reshape(-1, 1, 1, n_dim)

        xData = xData.astype(np.float32)

        if not os.path.isdir(savedDataFolder):
            os.mkdir(savedDataFolder)
        f = open(savedDataFile + '-x.txt', 'wb')
        np.save(f, xData)
        f = open(savedDataFile + '-y.txt', 'wb')
        np.save(f, yData)
        print ('loading origin data: {} no-frames: {}'.format(timeit.default_timer() - start_loading, dataIndex))
        return (xData, yData, n_dim)
    except Exception as e:
        logger.printErrorCMD('error loading file : {}'.format(wavFile))
        logger.printErrorCMD(e)
        pass


def getFileName(file):
    (drive, basename) = os.path.split(file)
    (name, ext) = os.path.splitext(basename)
    return name

def loadBottleneckData(logger, normalize = True, sharedVariable = True, reshapeToChannels = False, n_context_frames = 0):
    start_time = timeit.default_timer()

    def loadFileNames(folder, postName = 'fea'):
        logger.printInfo('loading file names from {} '.format(folder))
        res = []
        import os
        for file in os.listdir(folder):
            if file.endswith(postName):
                res.append(os.path.join(folder,file))
        return res

    logger.printInfo('... reading data')
    savedDataFolder = os.path.join(datasetFolder, 'data-cntx' + str(n_context_frames) + '-norm' + str(normalize) + '-btlncks')
    logger.printInfo(savedDataFolder)
    if os.path.isdir(savedDataFolder):
        (xTrain, yTrain, fnTrain,
         xTest, yTest, fnTest,
         xValid, yValid, fnValid,
         n_dim, fileNames) = loadSavedData(savedDataFolder, logger)
        printInfoDist(yTrain, 'train', logger)
        printInfoDist(yTest, 'test', logger)
        printInfoDist(yValid, 'validation', logger)

        return (xTrain, yTrain, fnTrain,
            xTest, yTest, fnTest,
            xValid, yValid, fnValid,
            n_dim, fileNames)

    def loadBottleneckDataFolder(featuresFolder, oldFilesList):
        oldFileNamesList = []
        for file in oldFilesList:
            name = getFileName(file)
            oldFileNamesList.append(name)
        btlnkFeatureFileNames = [];
        for file in os.listdir(featuresFolder):
            name = getFileName(file).replace('_', ' ')
            if(name in oldFileNamesList):
                btlnkFeatureFileNames.append(os.path.join(featuresFolder, file))
        middleX = []
        middleY = []
        middleFn = []
        fileNames = []

        for feaFile in btlnkFeatureFileNames:
            if feaFile.endswith(".fea"):
                fh = open(feaFile, 'rb')
                nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIHH", fh.read(12))
                m = np.frombuffer(fh.read(nSamples * sampSize), 'i1')
                m = m.view('>f').reshape(nSamples, sampSize / 4)

                my = []
                mfn = []
                name = getFileName(feaFile).replace('_', ' ')
                for oldFeaFile in oldFilesList:
                    if name == getFileName(oldFeaFile):
                        fileNames.append(oldFeaFile)
                        fnIndex = len(fileNames) - 1
                        for line in open(oldFeaFile):
                            y = int(line[0])
                            mfn.append(fnIndex)
                            my.append(y)
                        my = np.asarray(my, dtype=np.int32)
                        mfn = np.asarray(mfn, dtype=np.int32)
                        sizeData = min(m.shape[0], my.shape[0], mfn.shape[0])
                        m = m[:sizeData]
                        my = my[:sizeData]
                        mfn = mfn[:sizeData]
                        break
                middleX.append(m)
                middleY.append(my)
                middleFn.append(mfn)
        xData = np.concatenate(middleX, axis=0)
        yData = np.concatenate(middleY, axis=0)
        fnData = np.concatenate(middleFn, axis=0)

        return xData, yData, fnData, fileNames

    import struct
    featuresFolder = os.path.join(datasetFolder, 'btlnck_features')

    fileNames = []

    filesInTrainFolder = loadFileNames(os.path.join(datasetFolder, 'feature files'))
    xTrain, yTrain, fnTrain, fileNamesTrain = loadBottleneckDataFolder(featuresFolder, filesInTrainFolder)
    fileNames.extend(fileNamesTrain)

    filesInTestFolder = loadFileNames(os.path.join(datasetFolder, 'test', 'feature files'))
    xTest, yTest, fnTest, fileNamesTest = loadBottleneckDataFolder(featuresFolder, filesInTestFolder)
    fileNames.extend(fileNamesTest)

    filesInValidFolder = loadFileNames(os.path.join(datasetFolder, 'validation', 'feature files'))
    xValid, yValid, fnValid, fileNamesValid = loadBottleneckDataFolder(featuresFolder, filesInValidFolder)
    fileNames.extend(fileNamesValid)

    n_dim = xTrain.shape[1]

    if(normalize):
        xTrain, xTest, xValid = __normalizeData(xTrain, xTest, xValid, logger)


    printInfoDist(yTrain, 'train', logger)
    printInfoDist(yTest, 'test', logger)
    printInfoDist(yValid, 'validation', logger)

    def shared_dataset(xData, yData, borrow=True):
        shared_x = theano.shared(np.asarray(xData, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(yData, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    if(sharedVariable):
        xTrain, yTrain = shared_dataset(xTrain, yTrain)
        xTest, yTest = shared_dataset(xTest, yTest)
        xValid, yValid= shared_dataset(xValid, yValid)

    if(reshapeToChannels):
        xTrain = xTrain.reshape(-1, 1, 1, n_dim)
        xTest = xTest.reshape(-1, 1, 1, n_dim)
        xValid = xValid.reshape(-1, 1, 1, n_dim)

    p_start_time = timeit.default_timer()
    logger.printInfo('... pickling data')
    saveData(savedDataFolder, xTrain, yTrain, fnTrain, xTest, yTest, fnTest, xValid, yValid, fnValid, n_dim, fileNames)
    logger.printInfo('pickling took:\t%.1fs' % (timeit.default_timer() - p_start_time))

    end_time = timeit.default_timer()
    logger.printInfo('loading data took:\t%.1fs' % (end_time - start_time))
    return (xTrain, yTrain, fnTrain,
            xTest, yTest, fnTest,
            xValid, yValid, fnValid,
            n_dim, fileNames)







