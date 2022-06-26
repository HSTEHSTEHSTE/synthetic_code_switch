import nltk
nltk.download('punkt')
import jieba
from nltk.lm.preprocessing import flatten, pad_both_ends
from nltk.translate import AlignedSent
from nltk.translate.ibm2 import IBMModel2
import pickle
import random

data_file = open("/home/xli257/synthetic_code_switch/parallel.txt", 'r')
# data_file = open("/home/xli257/synthetic_code_switch/parallel_test.txt", 'r')

bitext = []
for line in data_file:
    line = line.split('\t')
    line[1] = line[1].replace('\n', '')
    line[0] = line[0].strip()
    line[1] = line[1].strip()

    source_tokenised = list(pad_both_ends(nltk.tokenize.word_tokenize(line[0]), n = 2))
    target_tokenised = list(pad_both_ends(jieba.lcut(line[1]), n = 2))

    bitext.append(AlignedSent(source_tokenised, target_tokenised))

iterations = 20
ibm2_model = IBMModel2(bitext, iterations)
pickle.dump(bitext, open('/home/xli257/synthetic_code_switch/bitext', 'wb'))