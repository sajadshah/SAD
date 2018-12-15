import json

import dataset
from config import *

def create_db_info_folder(folder, logger):
    x, y, _, _ = dataset.loadDataFromFeatureFiles(folder, 0, logger)
    infoFile = open(os.path.join(folder, 'data_info.txt'))
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    info = {
        'mean'  : mean,
        'std'   : std
    }
    json.dump(info, infoFile)

def run(logger):
    folder_names = ['train', 'test', 'validation']
    for n in folder_names:
        train_folder = os.path.join(datasetFolder, n)
        create_db_info_folder(train_folder, logger)