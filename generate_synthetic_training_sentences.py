from nltk.translate import AlignedSent
from nltk.translate.ibm2 import IBMModel2
import pickle
import random

def replace_words(input_list, target_list, replacements):
    for item in input_list:
        if input_list.index(item) in replacements:
            for replacement_item in replacements[input_list.index(item)]:
                yield target_list[replacement_item]
        else:
            yield item


bitext = pickle.load(open('/home/xli257/synthetic_code_switch/bitext', 'rb'))
generated_data_file = open('/home/xli257/synthetic_code_switch/generated.txt', 'w')
codeswitch_ratio = .5
sigma = .4
for bitext_element in bitext:
    alignments = bitext_element.alignment
    replacement_chance = random.gauss(codeswitch_ratio, sigma)
    alignment_dictionary = {}
    generated_sentence = bitext_element.mots
    for alignment in alignments:
        if alignment[1] in alignment_dictionary:
            alignment_dictionary[alignment[1]].append(alignment[0])
        else:
            alignment_dictionary[alignment[1]] = [alignment[0]]
    replacement_dictionary = {}
    for key in alignment_dictionary:
        replacement_dice = random.random()
        if replacement_dice > replacement_chance:
            replacement_dictionary[key] = alignment_dictionary[key]
            replacement_dictionary[key].sort()
    codeswitched_sentence = list(replace_words(bitext_element.mots, bitext_element.words, replacement_dictionary))
    generated_data_file.write('  '.join(codeswitched_sentence) + '\n')

generated_data_file.close()