
import streamlit as st
from keras.models import load_model
import numpy as np
import tensorflow as tf
import heapq


st.set_page_config(page_title='Next Word Prediction Model', page_icon=None, layout='centered', initial_sidebar_state='auto')

#Importing RegexTeokenizer module from NLTK library
#Loading the data
path = '1661-0.txt'
text = open(path, "r", encoding='utf-8').read().lower()
#print ('Corpus length: ',len(text))



from nltk.tokenize import RegexpTokenizer
#Tokenizing the data and converting to tokens
tokenizer = RegexpTokenizer(r'\w+')
#lowering the case
wo = text.lower() 
words = tokenizer.tokenize(wo)
#print(words)

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))







# Model saved with Keras model.save()
MODEL_PATH ='./next_word_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#pickle.dump(history, open("history.p", "wb"))
#model = load_model('next_word_model.h5')
#history = pickle.load(open("history.p", "rb"))

LENGTH = 5



#Defining a function to prepare input for using in model
def prep(text):
    x = np.zeros((1, LENGTH, len(unique_words)))
    for a, word in enumerate(text.split()):
        print(word)
        x[0, a, unique_word_index[word]] = 1
    return x

#using the prep funtion we are transforming the data
prep("It is not a lack".lower())


#Defining a function called sample
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

#Defining a function predict_completion to predict the data using model
def completion(text):
    original_text = text
    generated = text
    completion = ''
    indices_char={}
    while True:
        x = prep(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion
def completions(text, n=3):
    x = prep(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] + completion(text[1:] + completion[idx]) for idx in next_indices]

def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prep(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]


#q =  input("Enter sentence")
q = st.text_area("Enter your text here")
print("correct sentence: ",q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence: ",seq)
print("next possible words: ", predict_completions(seq, 5))

st.text_area("Predicted word is here",predict_completions(seq, 5),key="predicted_list")