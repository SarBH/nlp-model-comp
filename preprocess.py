import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
import string
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import csv


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



def import_data(filepaths_dict):
    
    df_all = pd.DataFrame(columns=['paragraph', 'author'])

    stop_words = set(nltk.corpus.stopwords.words('english'))

    for author, filepath in filepaths_dict.items():
        # get rows from .txt file
        with open(filepath, newline='\n\n') as f:
            reader = csv.reader(f)
            books_sentences = list(reader)
        
            tokenized_paragraphs = []
        
            # iterate through every text row to clean it up
            for sample_idx, paragraph in df_author_rows['paragraph']:
                # 1. remove punctuations
                paragraph = paragraph.translate(str.maketrans('','',string.punctuation))
                # 2. tokenize
                tokens = nltk.word_tokenize(paragraph)
                # 3. remove stop words
                tokens = [w.lower() for w in tokens if not w in stop_words]
                
                tokenized_paragraphs.append(tokens)

        df_all['paragraph'].append(tokenized_paragraphs)
        df_all['author'].append(author)
        print("df all has rows:", len(df_all))
    
    
    df = pd.concat(df_list)
    
    return df


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
    embeddings_dict = import_pretrained_embeddings('../glove.6B.50d.txt')

    filepaths_dict = {'fd': './a4-data/q1/fd.txt',
                'acd': './a4-data/q1/acd.txt',
                'ja': './a4-data/q1/ja.txt'}

    tokenized_data = import_data(filepaths_dict)

    print(tokenized_data)
    vectorized_data = vectorize_data(tokenized_data, embeddings_dict)


