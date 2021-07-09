#Script to predict punctuation labels of text lines of input file
#It saves a file for the predicted labels (for easier accuracy checking)
#It also saves a file of the punctuated input

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

input_file ="./data/input/clean_merged.depunct.test.id"

model = CRFSeqlab(log=None, **config)
model.load_from_pickle("models/model_crf.pickle")
predicted = []
with open(input_file) as f:
    for line in f:
        predicted.append(" ".join(model.predict([line.strip()])[0][0]))
with open("./data/processed/lab_punctuated.id","w") as o: #save punctuation labels
    for line in predicted:
    o.write(line+"\n")

#sequence labelling punctuator, takes labels and unpuncuated text to punctuate it
seqf = open(input_file) #input text
labf = predicted #predicted labels
with open("../data/processed/punctuated.id","w") as o: #punctuated output file
    for seq,lab in zip(seqf,labf):
        res = ""
        for s,l in zip(seq.strip().split(" "), lab.strip().split(" ")):
            punc = ""
            if (l == "DOT"):
                punc = "."
            elif (l == "COMMA"):
                punc = ","
            elif (l == "QUES"):
                punc = "?"
            else:
                punc =""
            res += s+punc +" "
        o.write(res.strip()+"\n")
seqf.close()
labf.close()
print("Saved predicted labels to lab_punctuated.id")
print("Saved punctuated text to punctuated.id")
