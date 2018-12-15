# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.

import sys, getopt

import shutil

import dataset
from config import *

def postprocess(predicted_labels):
    res_labels = []
    n_frames = len(predicted_labels);
    intervals = int((n_frames * frameStep) / postprocessIntervalSize)
    for i in range(intervals):
        beginIndex = int(i * (postprocessIntervalSize / frameStep))
        endIndex = int((i+1) * (postprocessIntervalSize / frameStep))
        if endIndex > n_frames : endIndex = n_frames
        slice_labels = predicted_labels[beginIndex:endIndex]
        speech = np.count_nonzero(slice_labels)
        nonspeech = (endIndex - beginIndex) - speech
        if(speech > nonspeech):
            res_labels.extend([1] * len(slice_labels))
        else:
            res_labels.extend([0] * len(slice_labels))

    return res_labels


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
        opts, args = getopt.getopt(sys.argv[1:], "hm:e:l:c:b:o:pr", ['bottleneck-featrues', 'file-based-train', 'additional-train-data', 'no-normal', 'end-to-end'])
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
            elif opt == '-o':
                outputFolder = arg
            elif opt == '--bottleneck-featrues':
                btlnkFeatures = True
            elif opt == '--additional-train-data':
                datasetFolder = '/home/shahsavari/dataset_0.2_complete_additional_train'
            elif opt == '--no-normal':
                normalization = False


logger = Logger(outputFolder, command)

logger.printInfo('predicting on :')
logger.printInfo('model   :\t{}'.format(model))
logger.printInfo('n epochs:\t{}'.format(num_epochs))
logger.printInfo('l rate  :\t{}'.format(learning_rate))
logger.printInfo('n cntx  :\t{}'.format(num_context_frames))
logger.printInfo('n btlnk :\t{}'.format(n_bottleneck))

from mlp import prepare_predict, predict, build_confusion_matrix

prepared = False

srtOutFolder = os.path.join(logger.folderName, 'labels-test')
if (os.path.isdir(srtOutFolder )):
    shutil.rmtree(srtOutFolder )
os.mkdir(srtOutFolder)

info = []

test_features_folder = os.path.join(datasetFolder, 'test', 'feature files')
featureFilesList = os.listdir(test_features_folder)
for i in range(len(featureFilesList)):
    featureFile = featureFilesList[i]
    logger.printInfo("{}\t/\t{}".format(i, len(featureFilesList)))
    if (btlnkFeatures):
        None
    else:
        x, y, n_dim = dataset.loadDataFile(
            logger,
            featureFile=os.path.join(test_features_folder, featureFile),
            normalize=normalization,
            sharedVariable=False,
            reshapeToChannels=True,
            n_context_frames=num_context_frames)
    if not prepared:
        test_fn = prepare_predict(model, n_bottleneck, n_dim, logger)
        prepared = True

    (predicted_labels, acc, f1Score, loss, falseAcceptance, falseRejection) = predict(featureFile, x, y, test_fn, logger)

    predicted_labels = postprocess(predicted_labels)

    confMatrixAfterPP = build_confusion_matrix(y, predicted_labels)
    tp_tn = confMatrixAfterPP[0][0] + confMatrixAfterPP[1][1]
    fp_fn = confMatrixAfterPP[0][1] + confMatrixAfterPP[1][0]
    accAfterPP = float(tp_tn) / (tp_tn + fp_fn) * 100
    f1ScoreAfterPP  = 100*(float(2 * confMatrixAfterPP[0][0]) / (2 * confMatrixAfterPP[0][0] + confMatrixAfterPP[0][1] + confMatrixAfterPP[1][0]))

    #logger.printInfo("accuracy after pp on \t{}: \t{}".format(featureFile, accAfterPP))
    #logger.printInfo("f1score after pp on \t{}: \t{}".format(featureFile, f1ScoreAfterPP))

    #logger.printInfo("conf_m after pp on \t{}: \n\t\t p_sp \t\t p_nsp \n\t t_sp \t {} \t {} \n\t t_nsp \t {} \t {}"
    #                 .format(featureFile,
    #                         confMatrixAfterPP[0][0], confMatrixAfterPP[0][1],
    #                         confMatrixAfterPP[1][0], confMatrixAfterPP[1][1]
    #                         ))

    falseAcceptanceAfterPP = float(confMatrixAfterPP[0][1]) / (confMatrixAfterPP[0][0] + confMatrixAfterPP[0][1])
    falseRejectionAfterPP = float(confMatrixAfterPP[1][0]) / (confMatrixAfterPP[1][0] + confMatrixAfterPP[1][1])
    #logger.printInfo('false acceptance after pp on \t{}: \t{:.4f}'.format(featureFile, falseAcceptanceAfterPP))
    #logger.printInfo('false rejection after pp on \t{}: \t{:.4f}'.format(featureFile, falseRejectionAfterPP))

    info.append([loss, acc, accAfterPP, f1Score, f1ScoreAfterPP, falseAcceptance, falseAcceptanceAfterPP, falseRejection, falseRejectionAfterPP])

    logger.saveLabels(srtOutFolder, featureFile, predicted_labels)

info = np.asarray(info, dtype=np.float32)
mean = info.mean(axis=0)
logger.printInfo('final results: \nloss \t: \t{:.4f} \nacc \t: \t{}\nacc pp\t: \t{}\nf1score\t: \t{}\nf1score pp\t: \t{}\nfa\t\t: \t{}\nfa pp\t: \t{}\nfr\t\t: \t{}\nfr pp\t: \t{}'.format(mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], mean[6], mean[7], mean[8]))


