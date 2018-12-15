# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.


import sys, getopt

import dataset
from config import *

# xTrain, yTrain, xTest, yTest, xValid, yValid = dataset.loadData(normalize = True, sharedVariable=True)
# from logistic_sgd import sgd_optimization, predict
# sgd_optimization([(xTrain, yTrain), (xValid, yValid), (xTest, yTest)], learning_rate=0.000005, n_epochs=1000, batch_size=100)
# predict(xTest, yTest,10000)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS] [LEARNING_RATE]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        model = 'custom_mlp:1,800,0.2,0.5'
        num_epochs = 500
        learning_rate = 0.0001
        num_context_frames = 0
        n_bottleneck = -1
        resume = False
        btlnkFeatures = False
        normalization = True
        command = ' '.join(sys.argv[1:])
        opts, args = getopt.getopt(sys.argv[1:], "hm:e:l:c:b:t:o:r", ['bottleneck-featrues', 'no-normal'])
        for opt, arg in opts:
            if opt == '-m':
                model = arg
            elif opt == '-e':
                num_epochs = int(arg)
            elif opt == '-l':
                learning_rate = float(arg)
            elif opt == '-c':
                num_context_frames = int(arg)
            elif opt == '-l':
                learning_rate = float(arg)
            elif opt == '-b':
                n_bottleneck = int(arg)
            elif opt == '-t':
                time_limit = int(arg)
            elif opt == '-o':
                outputFolder = arg
            elif opt == '--bottleneck-featrues':
                btlnkFeatures = True
            elif opt == '--additional-train-data':
                datasetFolder = '/home/shahsavari/dataset_0.2_complete_additional_train'
            elif opt == '--no-normal':
                normalization = False


logger = Logger(outputFolder, command)
logger.printInfo('command:\n\t"{}"'.format(command))

if(btlnkFeatures):
    (xTrain, yTrain, fnTrain,
     xTest, yTest, fnTest,
     xValid, yValid, fnValid,
     input_dim, fileNames) = dataset.loadBottleneckData(logger, normalize=True, sharedVariable=False, reshapeToChannels=True, n_context_frames=num_context_frames)

else :
    (xTrain, yTrain, fnTrain,
     xTest, yTest, fnTest,
     xValid, yValid, fnValid,
     input_dim, fileNames) = dataset.loadData(logger, normalize = normalization, sharedVariable=False, reshapeToChannels=True, n_context_frames=num_context_frames)

if logger.isFirstEpoch():
    logger.printInfo('model   :\t{}'.format(model))
    logger.printInfo('n epochs:\t{}'.format(num_epochs))
    logger.printInfo('l rate  :\t{}'.format(learning_rate))
    logger.printInfo('n cntx  :\t{}'.format(num_context_frames))
    logger.printInfo('n btlnk :\t{}'.format(n_bottleneck))
    logger.printInfo('in_dim  :\t{}'.format(input_dim))


from mlp import build_model
build_model([(xTrain, yTrain), (xValid, yValid), (xTest, yTest)], model=model, input_dim=input_dim, learning_rate=learning_rate, num_epochs=num_epochs, batchsize=100, n_bottleneck=n_bottleneck, logger=logger, time_limit=time_limit)





