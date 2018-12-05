from __future__ import print_function
from gensim.models import KeyedVectors  # gensim needs to be installed

en_model = KeyedVectors.load_word2vec_format(en_model = KeyedVectors.load_word2vec_format('/home/herokwon/downloads/wiki-news-300d-1M.vec')

# Getting the tokens
words = []
for word in en_model.vocab:
    words.append(word)

# two input scenario

input_a, input_b = YOLO()  ### returned values(words) from object detection
print(en_model.similarity(input_a, input_b))

# one input scenario

# input = YOLO()
# for similar_word in en_model.similar_by_word(input):
#     if similar_word[0] in YOLO_dataset      # check if similar word is in YOLO datasets. if not, next similar word is supposed to be checked
#         Image_output(similar_word[0])
#         break


# two input pairs scenario (two inputs per image)

# input_a1, input_a2, input_b1, input_b2 = YOLO()
# a1b1 = en_model.similarity(input_a1,input_b1)
# a1b2 = en_model.similarity(input_a1,input_b2)
# a2b1 = en_model.similarity(input_a2,input_b1)
# a2b2 = en_model.similarity(input_a2,input_b2)
# print((a1b1+a1b2+a2b1+a2b2)/4)