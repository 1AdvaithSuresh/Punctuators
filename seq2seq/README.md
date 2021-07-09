Data:
The data contained in the data folder is not the full training dataset. The full dataset is available at:
http://opus.nlpl.eu/OpenSubtitles-v2018.php
The three datasets used were en-id, es-id and bs-id datasets for train, eval and dev respectively.

Model:
A pre-trained model is available in the VM at: /mnt/data/data-gpu/advaiths_workspace/punc_model/.

Notebook: ./notebooks/Punctuator_Recibrew.ipynb
This notebook contains code for both training and decoding/inference of the seq2seq punctuator model.

Scripts:

data_processing.py:
Description: It takes a file with punctuation as input, and outputs a .csv file with 2 columns. Col1 = unpunctuated input. Col2= punctuation label corresponding to input. 
Arguments: Arg1 = path to input file, Arg2= path to (and name of) desired output .csv file
Example:

    python data_processing.py ./data/input/OpenSubtitles.en-id.10k ./data/processed/punctuated_os.csv

train_BPE.py:
Description: Trains the Byte-Pair-Encoding tokenizer model using a training data-set.
    Saves (by default) to ./data/BPE/.
Arguments: Training dataset file
Example:

    python train_BPE.py ./data/input/OpenSubtitles.en-id.10k

Note: The respository contains a pre-trained BPE tokenizer already. 

