import os
import sys
import spacy
import string
import re
import pandas as pd
import random


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
def create_csv_dataset(input_filename, output_filename):
    train_tgtTxt = open(input_filename)
    nlp = spacy.blank("id")
    dataset = []
    for tgtLine in train_tgtTxt:
      tgtLine = tgtLine.rstrip('\n').strip()
      tok_tgt = tokenize_id(nlp(tgtLine)) #tokenized line
      seq = tok_tgt.copy() #sequence
      lab = [] #labels
      hit_cnt = 0
      for i,tok in enumerate(tok_tgt): #iterate through each tok in line
        if (tok in string.punctuation):
            seq.pop(i-hit_cnt)
            hit_cnt+=1
            if(i==0):
                continue
            if(not((tok==".")or(tok=="?") or(tok==","))):
                continue
            if (len(lab)>0):
                popped=lab.pop(-1)
            if (tok == "."):
                lab.append("DOT")
            elif (tok == "?"):
                lab.append("QUES")
            elif (tok == ","):
                lab.append("COMMA")
            else:
                pass
        else: #if not punctuation
            lab.append("BLANK")
      if(not(len(seq)==len(lab))): #safety check
        continue
      seq = " ".join(seq)
      lab = " ".join(lab)
      dataset.append([seq,lab])
    train_tgtTxt.close()
    #save as .csv
    df = pd.DataFrame(dataset, columns=["sentence", "label"])
    df.to_csv(output_filename)
    print("Saved dataset as "+ output_filename)


def main():
    input_file = sys.argv[1]
    output_filename = sys.argv[2]
    print("Cleaning input "+input_file+" and creating csv dataset")
    cleaned_file = clean(input_file) #cleans bad punction from input
    sectioned_file = sectionize(cleaned_file,2,4) #turns input into sections of 2-4 lines each
    create_csv_dataset(sectioned_file, output_filename) #creates final .csv dataset for training
    #remove temp files
    os.remove(cleaned_file)
    os.remove(sectioned_file)

if __name__ == "__main__":
    main()


