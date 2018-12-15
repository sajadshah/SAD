import shutil
from config import *

def getFileName(file):
    (drive, basename) = os.path.split(file)
    (name, ext) = os.path.splitext(basename)
    return name


trainAudioFolder = os.path.join(datasetFolder, 'train', 'audio files')
trainAudioFiles = os.listdir(trainAudioFolder)
testAudioFiles = os.listdir(os.path.join(datasetFolder, 'test', 'audio files'))
testFeatureFiles = os.listdir(os.path.join(datasetFolder, 'test', 'feature files'))
validAudioFiles = os.listdir(os.path.join(datasetFolder, 'validation', 'audio files'))
validFeatureFiles = os.listdir(os.path.join(datasetFolder, 'validation', 'feature files'))

tempDir = os.path.join(datasetFolder, 'temp')
from dataset import getFileName
for f in trainAudioFiles:
    if f.endswith('.wav'):
        fn = getFileName(f)
        isTest = False
        for tf in testFeatureFiles:
            tfn = getFileName(tf)
            if tfn == fn:
                isTest = True
        if isTest:
            shutil.move(os.path.join(trainAudioFolder, fn + '.wav'), os.path.join(tempDir, fn + '.wav'))
            shutil.move(os.path.join(trainAudioFolder, fn + '.txt'), os.path.join(tempDir, fn + '.txt'))
            print ('{} was in test  '.format(fn))
            continue

        isValid = False
        for vf in validFeatureFiles:
            vfn = getFileName(vf)
            if vfn == fn:
                isValid = True
        if isValid:
            shutil.move(os.path.join(trainAudioFolder, fn + '.wav'), os.path.join(tempDir, fn + '.wav'))
            shutil.move(os.path.join(trainAudioFolder, fn + '.txt'), os.path.join(tempDir, fn + '.txt'))
            print ('{} was in valid '.format(fn))
            continue


