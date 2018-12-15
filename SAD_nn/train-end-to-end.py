# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.


import sys, getopt

from config import *
import dataset
import lasagne
import mlp
import time

def iterate_files(filenames, batchsize, shuffle=False):

    if shuffle:
        indices = np.arange(len(filenames))
        np.random.shuffle(indices)
    for start_idx in range(0, len(filenames), batchsize):
        end_idx = min(start_idx + batchsize, len(filenames))
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        res = []
        for i in excerpt:
            res.append(filenames[i])
        yield res

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
        frameSize = 100
        frameStep = 20
        command = ' '.join(sys.argv[1:])
        opts, args = getopt.getopt(sys.argv[1:], "hm:e:l:c:b:t:o:rf:s:", ['bottleneck-featrues', 'file-based-train', 'additional-train-data', 'no-normal', 'end-to-end'])
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
            elif opt == '-f':
                frameSize = int(arg)
            elif opt == '-s':
                frameStep = int(arg)
            elif opt == '--additional-train-data':
                datasetFolder = '/home/shahsavari/dataset_0.2_complete_additional_train'
            elif opt == '--no-normal':
                normalization = False

logger = Logger(outputFolder, command)
logger.printInfo('command:\n\t"{}"'.format(command))

epoch = 0
currentModelParams = None
yellow_cards = 0
best_validation_loss = np.inf
best_model_params = None
batch_size = 500
patience = 5
if not logger.isFirstEpoch():
    epoch, \
    currentModelParams, \
    yellow_cards, \
    learning_rate, \
    best_validation_loss, \
    best_model_params = logger.loadInitTrainParams()



prepared = False

train_audio_folder = os.path.join(datasetFolder, 'train', 'audio files')
trainAudioFilesList = os.listdir(train_audio_folder)

valid_audio_folder = os.path.join(datasetFolder, 'validation', 'audio files')
validAudioFilesList = os.listdir(valid_audio_folder)
start_time_iter = time.time()

while epoch < num_epochs:
    train_file_index = 0
    valid_file_index = 0

    start_time = time.time()
    train_err = 0
    train_batches = 0

    for batch_files in iterate_files(trainAudioFilesList, batchsize=10, shuffle=True):
        x_data = []
        y_data = []
        for wavFile in batch_files:
            if wavFile.endswith('.wav'):
                train_file_index += 1
                logger.printDynamicCMD("epoch: {} loss: {} train[file: {} / {}] validation[file: {} / {}]".format(
                    epoch, train_err / (train_batches+1), train_file_index, len(trainAudioFilesList)/2, valid_file_index, len(validAudioFilesList)/2))
                if (btlnkFeatures):
                    None
                else:
                    try:
                        x, y, n_dim = dataset.loadDataFileEndToEnd(
                        logger,
                        wavFile=os.path.join(train_audio_folder, wavFile),
                        normalize=normalization,
                        sharedVariable=False,
                        reshapeToChannels=True,
                        n_context_frames=num_context_frames,
                        frameSize=frameSize,
                        frameStep=frameStep)

                        x_data.append(x)
                        y_data.append(y)
                    except Exception as e:
                        logger.printErrorCMD('error loading {}'.format(wavFile))
                        logger.printErrorCMD(e)
                        pass
        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        if not prepared:
            network, train_fn, val_fn = mlp.prepare_train(model, n_bottleneck, n_dim, learning_rate, logger)
            if currentModelParams != None:
                lasagne.layers.set_all_param_values(network, currentModelParams)
            prepared = True
        train_err_file, train_batches_file, time_file = mlp.train(x_data, y_data, train_fn, batch_size)

        train_err += train_err_file
        train_batches += train_batches_file
    train_loss = train_err / train_batches
    exec_time = time.time() - start_time

    logger.printInfo("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, exec_time))
    logger.printInfo("  training loss:\t\t{:.6f}".format(train_loss))

    valid_err = 0
    valid_batches = 0
    valid_acc = 0
    valid_file_index = 0
    for wavFile in validAudioFilesList:
        if wavFile.endswith('.wav'):
            valid_file_index += 1
            logger.printDynamicCMD("epoch: {} loss: {} train[file: {} / {}] validation[file: {} / {}]".format(
                epoch, valid_err / (valid_batches+1), train_file_index, len(trainAudioFilesList)/2, valid_file_index, len(validAudioFilesList)/2))
            try:
                x, y, n_dim = dataset.loadDataFileEndToEnd(
                    logger,
                    wavFile=os.path.join(valid_audio_folder, wavFile),
                    normalize=normalization,
                    sharedVariable=False,
                    reshapeToChannels=True,
                    n_context_frames=num_context_frames,
                    frameSize=frameSize,
                    frameStep=frameStep)
            except Exception as e:
                logger.printErrorCMD('error loading {}'.format(wavFile))
                logger.printErrorCMD(e)
                pass
            valid_err_file, valid_acc_file, valid_batches_file = mlp.validate(x,y, val_fn, batch_size)

            valid_err += valid_err_file
            valid_acc += valid_acc_file
            valid_batches += valid_batches_file

    valid_loss = valid_err / valid_batches
    valid_accuracy = (valid_acc / valid_batches) * 100
    logger.printInfo("  validation loss:\t\t{:.6f}".format(valid_loss))
    logger.printInfo("  validation accuracy:\t\t{:.2f} %".format(valid_accuracy))

    if (valid_loss < best_validation_loss):
        best_validation_loss = valid_loss
        yellow_cards = 0

        best_model_params = lasagne.layers.get_all_param_values(network)
        logger.saveModel(best_model_params)
    else:
        yellow_cards += 1
        learning_rate *= 0.9

    logger.printInfo("  yellow cards:\t\t{} ".format(yellow_cards))

    if yellow_cards >= patience:
        break

    epoch+=1

    currentModelParams = lasagne.layers.get_all_param_values(network)
    logger.saveInitTrainParams(epoch, currentModelParams, yellow_cards, learning_rate, best_validation_loss,
                               best_model_params)

    if time.time() - start_time_iter > time_limit:
        logger.printInfo('code has been run for {:.3f}s'.format(time.time() - start_time_iter))
        sys.exit(5)  # means run has not finished yet!

sys.exit(10)  # means run finished successfully!



