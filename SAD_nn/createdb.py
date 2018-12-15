import sys, getopt

import dataset
from config import *

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
        num_context_frames = 1
        n_bottleneck = -1
        resume = False
        command = ' '.join(sys.argv[1:])
        opts, args = getopt.getopt(sys.argv[1:], "hm:e:l:c:b:t:o:", ['create-db'])
        for opt, arg in opts:
            if opt == '-m':
                model = arg
            elif opt == '-e':
                num_epochs = int(arg)
            elif opt == '-l':
                learning_rate = float(arg)
            elif opt == '-c':
                num_context_frames = int(arg)
            elif opt == 'l':
                learning_rate = float(arg)
            elif opt == '-b':
                n_bottleneck = int(arg)
            elif opt == '-t':
                time_limit = int(arg)
            elif opt == '-o':
                outputFolder = arg


logger = Logger(outputFolder, command)
logger.printInfo('command:\n\t"{}"'.format(command))

(xTrain, yTrain, fnTrain,
 xTest, yTest, fnTest,
 xValid, yValid, fnValid,
 input_dim, fileNames) = dataset.loadData(logger, normalize = True, sharedVariable=False, reshapeToChannels=True, n_context_frames=num_context_frames)
