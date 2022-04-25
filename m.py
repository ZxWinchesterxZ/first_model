import tensorflow as tf
import nltk
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import uvicorn
from fastapi import FastAPI

diseases=pd.read_csv("Diseases_master.csv")
diseases = diseases['name of symptoms-causes']
label=[]
phrases=[]
for i in diseases:
    st = 'can you tell me information about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'can you tell me anything about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'could you tell me information about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'could you tell me anything about '+i+'?'
    label.append(i)
    phrases.append(st)
    
    st = "I'd like to know information about "+i+'?'
    label.append(i)
    phrases.append(st)
    st = "I'd like to know anything about "+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'Do you know information about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'Do you know anything about '+i+'?'
    label.append(i)
    phrases.append(st)
    
    
    st = " Have you any idea about "+i+'?'
    label.append(i)
    phrases.append(st)
    st = "Do you happen to know anything about "+i+'?'
    label.append(i)
    phrases.append(st)
    st = ' Would you happen to know information about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'I wonder if you could tell me anything about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'tell me information about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'i want to know about '+i+'?'
    label.append(i)
    phrases.append(st)
    st = i+' do you know anything about this disease'+'?'
    label.append(i)
    phrases.append(st)
    st = 'what is the symptoms of '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'do you know the symptoms of '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'what is the cure for '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'do you know the cure of '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'what is the treatment for '+i+'?'
    label.append(i)
    phrases.append(st)
    st = 'do you know the treatment of '+i+'?'
    label.append(i)
    phrases.append(st)

tags = sorted(set(label),key=label.index)
all_words = []
i = 0
for pattern in phrases:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        i=i+1

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag    
stemmer = PorterStemmer()
ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words),key=all_words.index)

m = tf.keras.models.load_model('diseases.h5')
app = FastAPI()
@app.get('/')
def index():
    return {'message': 'Welcome to Our Model'}
# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.get('/predict_disease')
def predict_disease(phrase:str):
    sentence = nltk.word_tokenize(phrase)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    #X = torch.from_numpy(X)
    output = m.predict(X)
    return {
        'prediction': tags[np.argmax(output)]
    }
