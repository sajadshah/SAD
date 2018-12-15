# Copyright 2016 Sajad Shahsavari (Sharif University of Technology)
# Email: sj.shahsavari@gmail.com
#
# This is a skeleton script for running GPU-based ANN training experiments.

import subprocess, sys, getopt, os, time

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
        just_predict = False
        resume = False
        justCreateDB = False
        trainFileBased = False
        additionalTrainData = False
        endToEnd = False
        params = []
        opts, args = getopt.getopt(sys.argv[1:], "hm:e:l:c:b:prf:s:", ['create-db', 'bottleneck-featrues', 'file-based-train', 'additional-train-data', 'no-normal', 'end-to-end'])
        command = ' '.join(sys.argv[1:])
        for opt, arg in opts:
            if opt == '-m':
                model = arg
                params.append(opt)
                params.append(arg)
            elif opt == '-e':
                num_epochs = int(arg)
                params.append(opt)
                params.append(arg)
            elif opt == '-l':
                learning_rate = float(arg)
                params.append(opt)
                params.append(arg)
            elif opt == '-c':
                num_context_frames = int(arg)
                params.append(opt)
                params.append(arg)
            elif opt == 'l':
                learning_rate = float(arg)
                params.append(opt)
                params.append(arg)
            elif opt == '-b':
                n_bottleneck = int(arg)
                params.append(opt)
                params.append(arg)
            elif opt == '-p':
                just_predict = True
            elif opt == '-r':
                resume = True
            elif opt == '-f':
                frameSize = int(arg)
                params.append(opt)
                params.append(arg)
            elif opt == '-s':
                frameStep = int(arg)
                params.append(opt)
                params.append(arg)
            elif opt == '--create-db':
                justCreateDB = True
            elif opt == '--bottleneck-featrues':
                params.append(opt)
            elif opt == '--file-based-train':
                trainFileBased = True
                params.append(opt)
            elif opt == '--additional-train-data':
                additionalTrainData = True
                params.append(opt)
            elif opt == '--no-normal':
                normalization = False
                params.append(opt)
            elif opt == '--end-to-end':
                endToEnd = True
                params.append(opt)

time_limit = 20*60*100

outputFolder = "output" + ''.join(params)
if (not just_predict) and (not resume) and (not justCreateDB) :
    outputFolderTemp = outputFolder
    renameList = []
    while os.path.isdir(outputFolderTemp):
        renameList.append(outputFolderTemp)
        outputFolderTemp = "!" + outputFolderTemp
    for outf in reversed(renameList):
        os.rename(outf, "!" + outf)

    os.mkdir(outputFolder)

if (justCreateDB):
    spcommand = "python createdb.py " + command + " -o \"" + outputFolder + "\""
    print(spcommand)
    return_code = subprocess.call(spcommand, shell=True)
    print("database creation finished with return code: {}".format(return_code))
    exit()

import gpustat
theanoFlag = gpustat.getTheanoFlags()
patience = 5
wrongIters = 0
return_code = 5
i = 1
try:
    while wrongIters<patience and (not just_predict):
        trainFile = "train.py"
        if trainFileBased:
            trainFile = "train-file-based.py"
        if endToEnd:
            trainFile = "train-end-to-end.py"
		#
        spcommand = theanoFlag + " srun --gres=gpu:1 python " + trainFile + " " + command + " -t " + str(time_limit) + " -o \"" + outputFolder + "\""
        print(spcommand)
        return_code = subprocess.call(spcommand, shell=True)
        print("iteration {} done with return code: {}\n".format(i, return_code))
        if return_code == 10:
            break
        if return_code != 5:
            wrongIters += 1
        i += 1
        time.sleep(5)

    if return_code == 10 or just_predict:
        # spcommand = theanoFlag + " srun --gres=gpu:1 python predict.py " + command + " -o \"" + outputFolder + "\""
        spcommand = theanoFlag + " python predict.py " + command + " -o \"" + outputFolder + "\""
        print(spcommand)
        return_code = subprocess.call(spcommand, shell=True)
        print("prediction return code: {}".format(return_code))

    theanoFlag = gpustat.getTheanoFlags()

except KeyboardInterrupt:
    print 'Interrupted'
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
