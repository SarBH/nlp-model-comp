import pandas as pd
import numpy as np
# from scipy import spatial
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import nltk
import string
# from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import csv

#-*- coding: utf-8 -*-

def import_pretrained_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

 

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

# print(find_closest_embeddings(embeddings_dict["king"])[1:6])


def import_data(filepaths_dict, embedding):
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    lst_all = []
    for author, filepath in filepaths_dict.items():
        # get rows from .txt file
        with open(filepath) as f:
            reader = f.read()
            reader = reader.split("\n\n")

            tokenized_paragraphs = []
        
            # iterate through every text row to clean it up
            for sample_idx, paragraph in enumerate(reader):
                # 1. remove punctuations
                paragraph = paragraph.translate(str.maketrans('','',string.punctuation))
                paragraph = paragraph.replace('\n', ' ')
                # 2. tokenize
                tokens = nltk.word_tokenize(paragraph)
                # 3. remove stop words

                vectors = []
                
                for token in tokens:
                    if not token in stop_words:
                        try:
                            vector = embedding[token.lower()]
                            vectors.append(vector)
                        except KeyError:
                            continue
                        

            
                # tokenized_paragraphs.append(list(vectors))
                # assert(len(tokenized_paragraphs[sample_idx]) == len(vectors))
                thisrow = [vectors, author]
                
                lst_all.append(thisrow)
    
    return lst_all

def vectorize_data(tokenized_data, embeddings):
    tokens = tokenized_data[:]['paragraph']

    for sample_idx, sentence in enumerate(tokens):

        tokenized_sentence = []

        for word in sentence:
            print(word)
            print(embeddings[word])
            tokenized_sentence.append(embeddings[word])
    
        tokenized_data[sample_idx]['tokenized'] = tokenized_sentence

        assert(len(tokenized_data[sample_idx]['paragraph']) == len(tokenized_data[sample_idx]['tokenized']))

    assert(len(tokenized_data.columns) == 3)
    return tokenized_data


if "__main__" == __name__:
    # print(tf.config.list_physical_devices('GPU'))
    embeddings_dict = import_pretrained_embeddings('glove.6B.50d.txt')

    filepaths_dict = {'fd': './a4-data/q1/fd.txt',
                'acd': './a4-data/q1/acd.txt',
                'ja': './a4-data/q1/ja.txt'}

    tokenized_data = import_data(filepaths_dict, embeddings_dict)



