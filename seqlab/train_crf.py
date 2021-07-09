from nexus_reborn.nexus_reborn.models.seqlab.boomer import CRFSeqlab
from nexus_reborn.nexus_reborn.utils import CustomLogger

import os
import random
import numpy as np

crf_params =  {
    'algorithm': 'lbfgs',
    'c1': 0.1,
    'c2': 0.1,
    'max_iterations': 100,
    'all_possible_transitions': True
}

config = {
    'crf_params': crf_params,
    'train_data': "seqlab_train_170k.csv",
    'dev_data': "seqlab_dev_10k.csv",
    'test_data': "seqlab_eval_10k.csv"
}

model = CRFSeqlab(log=None, **config)
model.load_from_pickle(".output/model_crf.pickle")
predicted = []
with open("clean_merged.depunct.test.id") as f:
    for line in f:
        predicted.append(" ".join(model.predict([line.strip()])[0][0]))
with open("lab_punctuated.id","w") as o:
    for line in predicted:
    o.write(line+"\n")
