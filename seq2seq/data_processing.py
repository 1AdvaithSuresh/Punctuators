import os
import sys
import spacy
import string
import re
import pandas as pd
import random
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel


#Clean datasets of erroneous punctuation and saves as filename.cleaned
#for the first #dataset_size lines
def clean(input_filename):
    file = input_filename
    output = input_filename+".cleaned"
    with open(file) as f:
        with open(output,'w') as o:
            for idx,line in enumerate(f):
                line = re.sub(r'\([{^})]*\)', '', line) #removes parenthesis and their content
                line = re.sub('!','.',line) #replace ! with .
                line = re.sub('[\.{2,}]$','.',line) #replace ellipses at end of sentence with .
                line = re.sub('\.{2,}',',',line) #replace ellipses with comma
                line = re.sub('[^A-Za-z0-9?.,\' ]+', '', line) #remove all other erroneous chars
                line = line.strip()
                o.write(line+"\n")
    return input_filename+".cleaned"

#Creates sections of a random number between low & high sentences from the cleaned input_file
def sectionize(input_file, low, high):
    size = 0
    with open(input_file) as lc:
        size = len(list(lc))
    with open(input_file) as f:
        with open(input_file+".sectioned", 'w') as o:
            currline = 0
            while (currline < size):
                randNum= int(random.random()*(high-low)+low)
                if (currline+randNum > size):
                    randNum = size - currline
                currline += randNum
                section = [next(f) for x in range(randNum)]
                s = ' '.join(str(e) for e in section)
                s= re.sub('\n','',s)
                s =re.sub(' +', ' ', s)
                o.write(s+"\n")
    return input_file+".sectioned"


#simple tokenize helper function
def tokenize_id(text):
    return [token.text.lower() for token in text]

#creates a .csv training/dev/eval dataset using the data in input_filename
#of size dataset_size and writes to output_filename
def create_csv_dataset(input_filename, output_filename, tokenizer):
    train_tgtTxt= open(input_filename).readlines()
    train_srcTxt= train_tgtTxt.copy()
    train_dataset = []
    for srcLine,tgtLine in zip(train_srcTxt,train_tgtTxt):
      srcLine = srcLine.rstrip('\n')
      tgtLine = tgtLine.rstrip('\n')
      srcLine = srcLine.translate(str.maketrans('', '', string.punctuation)).lower()
      enc_src = tokenizer.encode(srcLine)
      enc_tgt = tokenizer.encode(tgtLine)
      id_src = enc_src.ids #get the BPE encoding ids
      id_tgt = enc_tgt.ids
      for i in range(len(id_src)):
        id_src[i]=str(id_src[i])
      for i in range(len(id_tgt)):
        id_tgt[i]=str(id_tgt[i])
      id_join_src = " ".join(id_src)
      id_join_tgt = " ".join(id_tgt)
      train_dataset.append([id_join_src,id_join_tgt])
    #Save as csv
    train_df = pd.DataFrame(train_dataset, columns=["src", "tgt"])
    train_df.to_csv(output_filename)
    print("Saved .csv to " +output_filename)


def main():
    input_file = sys.argv[1]
    output_filename = sys.argv[2]
    print("Cleaning input "+input_file+" and creating csv dataset")
    cleaned_file = clean(input_file) #cleans bad punction from input
    sectioned_file = sectionize(cleaned_file,2,4) #turns input into sections of 2-4 lines each
    #load BPE tokenizer
    tokenizer = Tokenizer(BPE()) #Byte pair encoding model
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.normalizer = Sequence([ Lowercase()])
    tokenizer.model = BPE('./data/BPE/vocab.json', './data/BPE/merges.txt') #load trained tokenizer
    create_csv_dataset(sectioned_file, output_filename, tokenizer) #creates final .csv dataset for training
    #remove temp files
    os.remove(cleaned_file)
    os.remove(sectioned_file)

if __name__ == "__main__":
    main()


