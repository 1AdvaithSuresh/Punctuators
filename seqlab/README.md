Data:
The data contained in the data folder is not the full training dataset. The full dataset is available at:
http://opus.nlpl.eu/OpenSubtitles-v2018.php
The three datasets used were en-id, es-id and bs-id datasets for train, eval and dev respectively.
Model:
A pre-trained model is included in the models folder in this repository.


Note: These scripts require that the Nexus_reborn CRF library from:
https://gitlab.com/kata-ai/research/nexus-reborn/-/tree/master/nexus_reborn/models/seqlab

Scripts:

data_processing.py:
Description: It takes a file with punctuation as input, and outputs a .csv file with 2 columns. Col1 = unpunctuated input. Col2= punctuation label corresponding to input. 
Arguments: Arg1 = path to input file, Arg2= path to (and name of) desired output .csv file
Example:

    python data_processing.py ./data/input/OpenSubtitles.en-id.10k ./data/processed/punctuated_os.csv

train_crf.py:
Description: It it uses 3 .csv datasets (training,eval,dev) (not as arguments, files have to be changed within the script) and trains a crf model. Note that this requires the nexus_reborn seqlab model (which is not contained in this respository).
Arguments: None
Example:

    python train_crf.py

punctuate_crf.py:
Description: Uses an already-trained CRF model and uses it predict punctuation in an un-punctuated input.
    It outputs 2 files, one with just the labels, and one with the fully-punctuated text.
Arguments: None (tweak input file in the script)
Example:

    python punctuate_crf.py

accuracy_score.py
Description: Takes 2 inputs as arguments, filepath of predicted tags, and csv containing the gold tags (produced by data_processing.py). It then calculates a convolution matrix and accuracy scores. 
Arguments: Arg1: Predicted tag file, Arg2: gold tag .csv file
Example:

    python accuracy_score.py ./data/processed/pred_labels.id ./data/processed/punctuated_os.csv

