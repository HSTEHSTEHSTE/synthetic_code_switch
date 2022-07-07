import pickle
import random
import tqdm

def replace_words(input_list, target_list, replacements):
    for item_index, item in enumerate(input_list):
        if str(item_index) in replacements:
            for replacement_item in replacements[str(item_index)]:
                yield target_list[int(replacement_item)]
        else:
            yield item


bitext = open('/home/xli257/synthetic_code_switch/data/parallel.txt', 'r')
generated_data_file = open('/home/xli257/synthetic_code_switch/data/parallel_generated.txt', 'w')
alignments = open('/home/xli257/synthetic_code_switch/data/parallel_align.txt', 'r')
codeswitch_ratio = .5
sigma = .4
repeat = 8
for bitext_element in tqdm.tqdm(bitext, total = 162026):
    sentences = bitext_element.strip().split('|||')
    sentence_zh = sentences[0].split()
    sentence_en = sentences[1].split()

    alignment_sentence = alignments.readline().strip().split()

    for repeat_index in range(repeat):
        replacement_chance = random.gauss(codeswitch_ratio, sigma)
        alignment_dictionary = {}
        if random.random() > .5:
            generated_sentence = sentence_en
            target_sentence = sentence_zh
            original_language_id = 1
            target_language_id = 0
        else:
            generated_sentence = sentence_zh
            target_sentence = sentence_en
            original_language_id = 0
            target_language_id = 1
        for alignment in alignment_sentence:
            alignment = alignment.split('-')
            if alignment[original_language_id] in alignment_dictionary:
                alignment_dictionary[alignment[original_language_id]].append(alignment[target_language_id])
            else:
                alignment_dictionary[alignment[original_language_id]] = [alignment[target_language_id]]
        replacement_dictionary = {}
        for key in alignment_dictionary:
            replacement_dice = random.random()
            if replacement_dice > replacement_chance:
                replacement_dictionary[key] = alignment_dictionary[key]
                replacement_dictionary[key].sort()
        codeswitched_sentence = list(replace_words(generated_sentence, target_sentence, replacement_dictionary))
        generated_data_file.write(''.join(codeswitched_sentence) + '\n')

generated_data_file.close()