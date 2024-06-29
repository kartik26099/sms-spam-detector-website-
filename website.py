import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
nltk.download("stopwords")
from nltk.corpus import stopwords
import string
def NLP_preprocessor(text):
    text=text.lower() #lowering all the text
    text=nltk.word_tokenize(text)# tokenizing words
    y=[]
    for i in text:
        if i.isalnum():# this teels if string is a special character or not an if not than we will append it in y
            y.append(i)
    text=y[:] # copying text
    y.clear()# removing all the elemnt in y

    for i in text:
        if i not in (set(stopwords.words("english"))) and i not in string.punctuation:# checking for puncutations and stopwords
            y.append(i)
    text=y[:]
    y.clear()
    for word in text:
        steamer=PorterStemmer.stem(word=word,self=PorterStemmer())
        y.append(steamer)



    return" ".join(y)
tfdif=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))
st.title("SMS/email classifier")
input_sms=st.text_input("enter the message")
if st.button("predict"):
            #preprocessing
            transformed_text=NLP_preprocessor(input_sms)
            #vectorizing
            tfidf_transformed=tfdif.transform([transformed_text])
            #prediction
            result=model.predict(tfidf_transformed)
            #display
            if(result==1):
                st.header("Spam teaxt")
            else:
                st.header("not a spam")