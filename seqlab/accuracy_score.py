import numpy as np
import sys
import pandas as pd

#Creates convolution matrix for seqlab predicted labels
#format helper function
def format_data(data):
    return str(int(data))+"\t"

#creates a confusion matrix comparing predicted and gold tags
#note that the file for the gold tags must be the csv file
#because the data-processing creates csv files
def confusion_matrix(predicted_file, gold_csv):
    res = open(predicted_file)
    df = pd.read_csv(gold_csv)
    gold = list(df["label"])
    #each line is padded to 66 labels
    punclist = ["BLANK","COMMA","QUES","DOT"]
    conf_mat = np.zeros((4,4))
    total_cnt = 0
    for g,r in zip(gold,res):
        g= g.strip().split(" ")
        g += [''] * (66 - len(g)) #pad to 66 labels
        r = r.strip().split(",")
        non_blank = 0
        for labg,labr in zip(g,r):
            if (labg==""):
                continue
            conf_mat[punclist.index(labr)][punclist.index(labg)] +=1
    print("Predicted, Gold")
    print ("\tBLANK\tCOMMA\tQUES\tDOT")
    print("BLANK\t"+format_data(conf_mat[0][0])+format_data(conf_mat[0][1])+format_data(conf_mat[0][2])+format_data(conf_mat[0][3]))
    print("COMMA\t"+format_data(conf_mat[1][0])+format_data(conf_mat[1][1])+format_data(conf_mat[1][2])+format_data(conf_mat[1][3]))
    print("QUES\t"+format_data(conf_mat[2][0])+format_data(conf_mat[2][1])+format_data(conf_mat[2][2])+format_data(conf_mat[2][3]))
    print("DOT\t"+format_data(conf_mat[3][0])+format_data(conf_mat[3][1])+format_data(conf_mat[3][2])+format_data(conf_mat[3][3]))
    print("Percent errors:")
    b_err = (conf_mat[0][1]+conf_mat[0][2]+conf_mat[0][3])/conf_mat[0,:].sum(0)*100
    print("BLANK: " +str(b_err))
    c_err = (conf_mat[1][0]+conf_mat[1][2]+conf_mat[1][3])/conf_mat[1,:].sum(0)*100
    print("COMMA: " +str(c_err))
    d_err = (conf_mat[2][0]+conf_mat[2][1]+conf_mat[2][3])/conf_mat[2,:].sum(0)*100
    print("DOT: " +str(d_err))
    q_err = (conf_mat[3][0]+conf_mat[3][1]+conf_mat[3][2])/conf_mat[3,:].sum(0)*100
    print("QUES: " +str(q_err))

#helper fn
#function to check if labels are punctuation
def checkPuncLabel(s):
    punclist = ["COMMA", "DOT", "QUES"]
    if s in punclist:
        return True
    else:
        return False

#Accuracy scores True positive, False positive, etc.
def accuracy_score(predicted_file, gold_csv):
    res = open(predicted_file)
    df = pd.read_csv(gold_csv)
    gold = list(df["label"])
    #each line is padded to 66 labels
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg =0
    total_cnt = 0
    for g,r in zip(gold,res):
        g= g.strip().split(" ")
        g += [''] * (66 - len(g)) #pad to 66 labels
        r = r.strip().split(",")
        non_blank = 0
        for labg,labr in zip(g,r):
            if ((labg=="") and (labr=="")):
                continue
            non_blank+=1
            if (not(checkPuncLabel(labg)) and checkPuncLabel(labr)):
                false_pos +=1
            if (checkPuncLabel(labg) and not(checkPuncLabel(labr))):
                false_neg +=1
            if ((labg==labr) and checkPuncLabel(labg)):
                true_pos +=1
            if ((labg==labr) and labg=="BLANK"):
                true_neg +=1
        total_cnt += non_blank
    err_cnt = false_neg + false_pos
    print("# TP: " +str(true_pos))
    print("# TN: " +str(true_neg))
    print("# FN: " +str(false_neg))
    print("# FP: " +str(false_pos))
    print("Errors: " +str(err_cnt))
    print("Total: " +str(total_cnt))
    acc = (true_pos + true_neg)/total_cnt
    print("Accuracy: " +str(acc))
    sens = true_pos/(true_pos+false_neg)
    print("Sensitivity (TP/(TP+FN)) : " + str(sens))
    spec= true_neg/(true_neg+false_pos)
    print("Specificity (TN/(TN+FP)) : " + str(spec))
    prec= true_pos/(true_pos+false_pos)
    print("Precision (TP/(TP+FP)) : " + str(prec))
    npv= true_neg/(true_neg+false_neg)
    print("Negative predictive value (TN/(TN+FN)) : " + str(npv))

def main():
    pred_tags_file = sys.argv[1]
    gold_tags_csv = sys.argv[2]
    print("Confusion matrix:")
    confusion_matrix(pred_tags_file, gold_tags_csv)
    print("\nAccuracy scores:")
    accuracy_score(pred_tags_file, gold_tags_csv)

if __name__ == "__main__":
    main()

