import numpy as np
import os


# get path of test folder
glove_folder = os.path.join(os.getcwd(), 'glove_file')

# get path of glove.6B.300d.txt file in test folder
glove_file = os.path.join(glove_folder, 'glove.6B.300d.txt')

def load_glove_vectors(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return words, word_to_vec_map

words, word_to_vec_map = load_glove_vectors(glove_file)
print(len(words))
print(len(word_to_vec_map))
print(word_to_vec_map['the'])
print(word_to_vec_map['the'].shape)
