import gensim
import pandas as pd
import numpy as np
import ast
import re
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from gensim.models import Word2Vec

def f_learyt(file_second):
    listich = []
    products = pd.read_csv('result.csv')
    print(products)
    length = len(products.index)
    for i in range(length):
        dd = ast.literal_eval(products['comments'][i])
        listich.append(dd)

    products2 = pd.read_csv(file_second, sep=',')
    print(products2)
    length2 = len(products2.index)
    for i in range(length2):
        dd = ast.literal_eval(products2['comments'][i])
        listich.append(dd)
    
    print(len(listich))
    # Load the saved model
    #model = Word2Vec.load("word2vec.model")
    # Train the model
    model = Word2Vec(listich, vector_size=100, window=5, min_count=1, workers=4, sg = 1)

    # Save the model
    model.save("word2vec.model")
    # Get the most similar words to a given word
    similar_words2 = model.wv.most_similar("отличный")
    print("отличный ", similar_words2)
    # pp = model.wv.get_vector("папа")
    # pp2 = model.wv.get_vector("мама")
    # pp3 = model.wv.get_vector("дочь")
    # pp4 = pp+pp2-pp3
    # similar_word = model.wv.most_similar([pp4])[0][0]
    # print(similar_word)

    # Load the CSV file with tokenized text
    df = pd.read_csv(file_second)
    # Define a function to clean the text
    def clean_text(text):
        text = re.sub('[\[\]\',]', '', text)
        return text

    # Clean the tokenized text column
    df['tokenized_text'] = df['comments'].apply(clean_text)
    # Create a new column for the vector representation of each word
    # для нахождения средних значений векторов
    def document_vector(x):
        bb = []
        result = []
        for word in x.split():
            bb.append(model.wv[word])
            vector = np.array(bb).mean(axis=0) 
            result = ','.join(vector.astype(str))
        return result

    df['word_vectors'] = df['tokenized_text'].apply(document_vector)# для обученного ворд2век
    # Save the updated CSV file 
    df.to_csv('some_learyted.csv', index=False)

    df2 = pd.read_csv("result_learyted.csv")
    # Define a function to clean the text
    def clean_text(text):
        text = re.sub('[\[\]\',]', '', text)
        return text

        # Clean the tokenized text column
    df2['tokenized_text'] = df2['comments'].apply(clean_text)
        # Create a new column for the vector representation of each word
        # для нахождения средних значений векторов
    def document_vector(x):
        bb = []
        result = []
        for word in x.split():
            bb.append(model.wv[word])
            vector = np.array(bb).mean(axis=0) 
            result = ','.join(vector.astype(str))
        return result

    df2['word_vectors'] = df2['tokenized_text'].apply(document_vector)# для обученного ворд2век
    # Save the updated CSV file 
    df2.to_csv('result_learyted.csv', index=False)