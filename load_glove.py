import numpy as np
import os


# get path of test folder
glove_folder = os.path.join(os.getcwd(), 'glove_file')

# get path of glove.6B.300d.txt file in test folder
glove_file = os.path.join(glove_folder, 'glove.6B.300d.txt')


# # Path: load_glove.py
# def cosine_similarity(u, v):
#     distance = 0.0
#     dot = np.dot(u, v)
#     norm_u = np.sqrt(np.sum(u ** 2))
#     norm_v = np.sqrt(np.sum(v ** 2))
#     cosine_similarity = dot / (norm_u * norm_v)
#     return cosine_similarity

# # Path: load_glove.py
# def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
#     """
#     Performs the word analogy task as explained above: a is to b as c is to ____. 
#     """
#     word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
#     e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
#     words = word_to_vec_map.keys()
#     max_cosine_sim = -100                
#     best_word = None                   
    
#     input_words_set = set([word_a, word_b, word_c])
    
#     for w in words:        
#         if w in input_words_set:
#             continue
        
#         cosine_sim = cosine_similarity(word_to_vec_map[w], e_b - e_a + e_c)
        
#         if cosine_sim > max_cosine_sim:
#             max_cosine_sim = cosine_sim
#             best_word = w
            
#     return best_word

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
