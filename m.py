import tensorflow as tf
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI
import numpy as np
import nltk
stemmer = PorterStemmer()
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
all_words = []
f1 = open("all_words.txt", "r")
all_words = f1.read().splitlines()
f1.close()
m = tf.keras.models.load_model('diseases.h5')
label=[]
f1 = open("labels.txt", "r")
label = f1.read().splitlines()
label = sorted(set(label),key=label.index)
app = FastAPI()
@app.get('/')
def index():
    return {'message': 'Welcome to Our Model'}

@app.get('/predict_disease')
def predict_disease(phrase:str):
    sentence = nltk.word_tokenize(phrase)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    output = m.predict(X)
    return {
        'prediction': label[np.argmax(output)]
    }
