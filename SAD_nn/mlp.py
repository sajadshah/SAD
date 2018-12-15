#!/usr/bin/env python

# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""
from lasagne.init import GlorotUniform, Constant
from lasagne.layers.dense import DenseLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.merge import ElemwiseSumLayer
from lasagne.layers.noise import GaussianNoiseLayer
from lasagne.layers.recurrent import RecurrentLayer, LSTMLayer
from lasagne.layers.shape import ReshapeLayer
from lasagne.nonlinearities import softmax, leaky_rectify

from config import *
import time
import sys

import numpy as np
import theano
import theano.tensor as T

import lasagne


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None, input_dim=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 1, input_dim),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    #l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    #l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    #l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_custom_mlp(input_var=None, input_dim=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5, n_bottleneck = -1):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, input_dim),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify #leaky_rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    if(n_bottleneck > 0):
        network = lasagne.layers.DenseLayer(network, n_bottleneck, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 2, nonlinearity=softmax)
    return network

def build_layers_custom_mlp(input_var = None, input_dim = None, widths = [200]):
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, input_dim),
                                        input_var=input_var)

    nonlin = lasagne.nonlinearities.rectify #leaky_rectify
    for w in widths:
        network = lasagne.layers.DenseLayer(
                network, int(w), nonlinearity=nonlin)

    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 2, nonlinearity=softmax)
    return network


def build_cnn(input_var=None, input_dim=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, input_dim),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def create_rnn(input_vars, num_inputs, depth, hidden_layer_size, num_outputs):
    # network = InputLayer((None, None, num_inputs), input_vars)
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, num_inputs),
                                        input_var=input_vars)
    batch_size_theano, _, _, seqlen = network.input_var.shape

    network = GaussianNoiseLayer(network, sigma=0.05)
    for i in range(depth):
		network = RecurrentLayer(network, hidden_layer_size,
                                 W_hid_to_hid = GlorotUniform(),
                                 W_in_to_hid = GlorotUniform(), b = Constant(1.0),
                                 nonlinearity = lasagne.nonlinearities.tanh, learn_init = True)
    network = ReshapeLayer(network, (-1, hidden_layer_size))
    network = DenseLayer(network, num_outputs, nonlinearity=softmax)

    return network

def create_lstm(input_vars, num_inputs, depth, hidden_layer_size, num_outputs):
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, num_inputs), input_var=input_vars)
    #network = GaussianNoiseLayer(network, sigma=0.01)
    nonlin = lasagne.nonlinearities.rectify  # leaky_rectify
    for i in range(depth):
		network = LSTMLayer(network, hidden_layer_size, learn_init = True, nonlinearity=nonlin)

    network = ReshapeLayer(network, (-1, hidden_layer_size))
    network = DenseLayer(network, num_outputs, nonlinearity=softmax)
    return network

def create_blstm(input_vars, mask_vars, num_inputs, depth, hidden_layer_size, num_outputs):
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, num_inputs), input_var=input_vars)
    mask = InputLayer((None, None), mask_vars)
    network = GaussianNoiseLayer(network, sigma=0.01)
    for i in range(depth):
		forward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True)
		backward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True, backwards = True)
		network =  ElemwiseSumLayer([forward, backward])
    network = ReshapeLayer(network, (-1, hidden_layer_size))
    network = DenseLayer(network, num_outputs, nonlinearity=softmax)
    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def build_train_function(network, input_var, target_var, learning_rate):
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=0.9)


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    return train_fn



def build_test_function(network, input_var, target_var):
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_labels = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(test_labels, target_var),
                      dtype=theano.config.floatX)



    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_labels])
    return val_fn

def call_model_builder(model, input_var, input_dim, n_bottleneck, logger):
    if model == 'mlp':
        network = build_mlp(input_var, input_dim)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, input_dim, depth=int(depth), width=int(width),
                                   drop_input=float(drop_in), drop_hidden=float(drop_hid), n_bottleneck=n_bottleneck)
    elif model.startswith('layers_custom_mlp:'):
        widths = model.split(':', 1)[1].split(',')
        network = build_layers_custom_mlp(input_var, input_dim, widths)
    elif model == 'cnn':
        network = build_cnn(input_var, input_dim)
    elif model.startswith('rnn:'):
        depth, width = model.split(':', 1)[1].split(',')
        network = create_rnn(input_var, input_dim, int(depth), int(width), 2)
    elif model.startswith('lstm:'):
        depth, width = model.split(':', 1)[1].split(',')
        network = create_lstm(input_var, input_dim, int(depth), int(width), 2)
    elif model.startswith('blstm:'):
        depth, width = model.split(':', 1)[1].split(',')
        network = create_blstm(input_var, input_dim, int(depth), int(width), 2)
    else:
        logger.printInfo("Unrecognized model type %r." % model)
        exit(1)
    return network

def build_model(dataset, model='mlp', input_dim=None, learning_rate=0.0001, num_epochs=500, batchsize=500, n_bottleneck = -1, logger=None, time_limit=10*24*60*60*60):

    start_time_iter = time.time()

    X_train, y_train = dataset[0]
    X_val, y_val = dataset[1]
    X_test, y_test = dataset[2]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    logger.printInfo("Building model and compiling functions...")

    network = call_model_builder(model, input_var, input_dim, n_bottleneck, logger)

    train_fn = build_train_function(network, input_var, target_var, learning_rate)
    val_fn = build_test_function(network, input_var, target_var)

    # Finally, launch the training loop.
    logger.printInfo("Starting training...")


    best_validation_loss = np.inf
    yellow_cards = 0
    epoch = 0

    if not logger.isFirstEpoch():
        epoch, \
        currentModelParams, \
        yellow_cards, \
        learning_rate, \
        best_validation_loss, \
        best_model_params = logger.loadInitTrainParams()

        lasagne.layers.set_all_param_values(network, currentModelParams)

    patience = 5

    # We iterate over epochs:
    while epoch < num_epochs:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize=batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize=batchsize, shuffle=False):
            inputs, targets = batch
            err, acc, labels = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we logger.printInfo the results for this epoch:
        exec_time = time.time() - start_time
        train_loss = train_err / train_batches
        valid_loss = val_err / val_batches
        valid_accuracy = (val_acc / val_batches) * 100

        logger.printInfo("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, exec_time ))
        logger.printInfo("  training loss:\t\t{:.6f}".format(train_loss))
        logger.printInfo("  validation loss:\t\t{:.6f}".format(valid_loss))
        logger.printInfo("  validation accuracy:\t\t{:.2f} %".format(valid_accuracy))

        logger.printCSV([epoch+1, exec_time , train_loss, valid_loss, valid_accuracy])

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

        epoch += 1

        currentModelParams = lasagne.layers.get_all_param_values(network)
        logger.saveInitTrainParams(epoch, currentModelParams, yellow_cards, learning_rate, best_validation_loss, best_model_params)

        if time.time() - start_time_iter > time_limit:
            logger.printInfo('code has been run for {:.3f}s'.format(time.time() - start_time_iter))
            sys.exit(5) #means run has not finished yet!

    lasagne.layers.set_all_param_values(network, best_model_params)

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batchsize=batchsize, shuffle=False):
        inputs, targets = batch
        err, acc, labels = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    logger.printInfo("Final results:")
    logger.printInfo("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    logger.printInfo("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    sys.exit(10)  # means run finished successfully!


    # After training, we compute and logger.printInfo the test error:

    # lasagne.layers.set_all_param_values(network, best_model_params)
    #
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, batchsize=batchsize, shuffle=False):
    #     inputs, targets = batch
    #     err, acc, labels = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # logger.printInfo("Final results:")
    # logger.printInfo("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # logger.printInfo("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

def prepare_train(model, n_bottleneck, input_dim, learning_rate, logger):

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    logger.printInfo("Building model and compiling functions...")

    network = call_model_builder(model, input_var, input_dim, n_bottleneck, logger)

    train_fn = build_train_function(network, input_var, target_var, learning_rate)
    val_fn = build_test_function(network, input_var, target_var)

    return network, train_fn, val_fn


def train(x, y, train_fn, batchsize=500):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(x, y, batchsize=batchsize, shuffle=False):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
    end_time = time.time()
    return train_err, train_batches, end_time-start_time

def validate(x, y, val_fn, batchsize=500):
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(x, y, batchsize=batchsize, shuffle=False):
        inputs, targets = batch
        err, acc, labels = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    return val_err, val_acc, val_batches

def predict_on(x, y, batchsize, test_fn, data_label, logger):

    # We can test it on some examples from test test
    test_err = 0
    test_acc = 0
    test_batches = 0
    predicted_labels = []
    for batch in iterate_minibatches(x, y, batchsize=batchsize, shuffle=False):
        inputs, targets = batch
        err, acc, labels = test_fn(inputs, targets)
        predicted_labels.extend(labels)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_err /= test_batches
    test_acc = (test_acc / test_batches) * 100

    #logger.printInfo("loss on {}: \t{}".format(data_label, test_err))
    #logger.printInfo("accuracy on {}: \t{}".format(data_label, test_acc))

    start_time = time.time()
    conf_matrix = build_confusion_matrix(y, predicted_labels)
    #logger.printInfo("building confusion matrix on {} took {}".format(data_label, time.time() - start_time))
    #logger.printInfo("conf_m on {}: \n\t\t p_sp \t\t p_nsp \n\t t_sp \t {} \t {} \n\t t_nsp \t {} \t {}"
    #                 .format(data_label,
    #                         conf_matrix[0][0], conf_matrix[0][1],
    #                         conf_matrix[1][0], conf_matrix[1][1]
    #                ))

    false_acceptance = float(conf_matrix[0][1]) / (conf_matrix[0][0] + conf_matrix[0][1])
    false_rejection = float(conf_matrix[1][0]) / (conf_matrix[1][0] + conf_matrix[1][1])
    #logger.printInfo('false acceptance on {}: \t{:.4f}'.format(data_label, false_acceptance))
    #logger.printInfo('false rejection on {}: \t{:.4f}'.format(data_label, false_rejection))

    return predicted_labels

def build_confusion_matrix(true_labels, predicted_labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # speech(positive class): 1, nonspeech(negative class): 0
    for i in range(len(predicted_labels)):
        label = true_labels[i]
        p_label = predicted_labels[i]
        if( label == 1 and p_label == 1):
            tp += 1
        if( label == 1 and p_label == 0):
            fn += 1
        if( label == 0 and p_label == 0):
            tn += 1
        if (label == 0 and p_label == 1):
            fp += 1

    return [[tp, fn],
            [fp, tn]]

def prepare_predict(model, n_bottleneck, input_dim, logger):
    param_values = logger.loadModel()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = call_model_builder(model, input_var, input_dim, n_bottleneck, logger)

    lasagne.layers.set_all_param_values(network, param_values)

    test_fn = build_test_function(network, input_var, target_var)

    return test_fn


def predict(fileName, x, y, test_fn, logger):

    batchsize = 100

    test_err = 0
    test_acc = 0
    test_batches = 0
    predicted_labels = []
    for batch in iterate_minibatches(x, y, batchsize=batchsize, shuffle=False):
        inputs, targets = batch
        err, acc, labels = test_fn(inputs.astype(np.float32), targets)
        predicted_labels.extend(labels)
        test_err += err
        test_acc += acc
        test_batches += 1

    test_err /= test_batches
    test_acc = (test_acc / test_batches) * 100

    conf_matrix = build_confusion_matrix(y, predicted_labels)
    f1Score  = 100 * (float(2*conf_matrix[0][0]) / (2*conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0]))

    #logger.printInfo("loss on \t{}: \t{}".format(fileName, test_err))
    #logger.printInfo("accuracy on \t{}: \t{}".format(fileName, test_acc))
    #logger.printInfo("f1Score on \t{}: \t{}".format(fileName, f1Score))

    #logger.printInfo("conf_m on \t{}: \n\t\t p_sp \t\t p_nsp \n\t t_sp \t {} \t {} \n\t t_nsp \t {} \t {}"
    #                 .format(fileName,
    #                         conf_matrix[0][0], conf_matrix[0][1],
    #                         conf_matrix[1][0], conf_matrix[1][1]
    #                         ))
    false_acceptance = float(conf_matrix[0][1]) / (conf_matrix[0][0] + conf_matrix[0][1])
    false_rejection = float(conf_matrix[1][0]) / (conf_matrix[1][0] + conf_matrix[1][1])
    #logger.printInfo('false acceptance on \t{}: \t{:.4f}'.format(fileName, false_acceptance))
    #logger.printInfo('false rejection on \t{}: \t{:.4f}'.format(fileName, false_rejection))

    return predicted_labels, test_acc, f1Score, test_err, false_acceptance, false_rejection

