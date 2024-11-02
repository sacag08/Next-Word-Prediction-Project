import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load LSTM model

model = load_model('next_word_lstm.h5')

#load tokenizer
with open("tokenize.pickel",'rb') as tok:
    tokenizer = pickle.load(tok)



#create the function 
def predict_next_word(model,input_text,max_token_length,tockenize):
    input_sequence = tockenize.texts_to_sequences([input_text])[0]
    if len(input_sequence)>= max_token_length:
        input_sequence = input_sequence[-(max_token_length-1):]
    input_sequence = pad_sequences([input_sequence],maxlen=max_token_length-1,padding='pre')
    pred = model.predict(input_sequence,verbose=0)
    predected_index = np.argmax(pred,axis=1)
    for word,index in tockenize.word_index.items():
        if predected_index==index:
            return word
    return None

#streamlit app
st.title("Next Word Prediction with LSTM")

input_text = st.text_input("Enter the sequence")
if st.button("predict next word"):
    max_length = model.input_shape[1] + 1
    next_word = predict_next_word(model,input_text,max_length,tokenizer)
    st.write(f' Next word: {next_word}')