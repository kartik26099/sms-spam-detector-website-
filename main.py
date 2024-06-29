import pandas as pd
import numpy as np
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\projects\spam detection\spam.csv",encoding='latin-1')
print(df.head())
#stages of project
#data preprocessing
#EDA(Evaluation of data)
#text preprocessing
#model buliding
#evaluation
#improvment

#1] Data cleaning
print(df.info())# i have discovered almost all the values in coloumn 2,3,4 are zero so removing them
df=df.iloc[:, [0,1]]
print(df.shape)# checking shape
#converting the data in encoded data in form of 0 and 1
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df["v1"]=encoder.fit_transform(df["v1"])
print(df["v1"])
#checking for null value
print(df.isna().sum(axis=0))
#checking for duplicate value
print("toal number of duplicate value: ",df.duplicated().sum(axis=0))# we have checked data if there are any duplicate value and we have found 403 duplicate value
#removing duplicate value
df=df.drop_duplicates(keep="first") # will drop all the duplicated value
print("toal number of duplicate value after pre procesing: ",df.duplicated().sum(axis=0))# we now 0 duplicate value

#2]EDA(evaluation data)
print("\nnumber of ham and spam ",df["v1"].value_counts()) # to find the number of ham and spam
import matplotlib.pyplot as plt
plt.pie(df["v1"].value_counts(),labels=["ham","spam"],autopct="%0.2f")#pie plot on number of ham spam and autopct will give vales in percent
plt.show()
#conclusion:-from this ananlysis we can tell that the data is imbalance
import nltk
nltk.download("punkt")
#we will be findng the number of word,character adn sentence in each coloumn
df["num_character"]=df["v2"].apply(len)# this will return number of character in coloumn
df["num_words"]=df["v2"].apply(lambda x:len(nltk.word_tokenize(x)))#nltk will word_tokanize ever word and by finding its lenght we will get number of words
df["num_sentence"]=df["v2"].apply(lambda x:len(nltk.sent_tokenize(x)))#nltk will sent_tokanize ever sentence and by finding its lenght we will get number of  sentence
print(df.columns) # introduce with new columns
#we will find the discription from each of the new coloumn
print(df[["num_character","num_words","num_sentence"]].describe())
#from this i have found that num_character are avg 78, num sent 18,num sentence 2
# to find the number of words in a particular ham and spam data set
print("\ndiscription of only ham",df[df["v1"]==0][["num_character","num_words","num_sentence"]].describe())
print("\ndiscription of only spam",df[df["v1"]==1][["num_character","num_words","num_sentence"]].describe())
#conclusion:-we have found that the data spam has more words and sentence than ham as spam mean=137.891271 ham mean=70.459256
#we will see it by using histograme
import seaborn as sns
#checking hisogram for character
sns.histplot(df[df["v1"]==0][["num_character"]])
sns.histplot(df[df["v1"]==1][["num_character"]],color="red")
plt.show()
#similarly we can do for every words and snetence
sns.pairplot(df,hue="v1")
plt.show() # words chracter senternce have linear relation
sns.heatmap(df.iloc[:, [0,2,3,4]].corr(),annot=True)
plt.show()
#conclusion:- as collinearlity between all the 3 num_words,sentence character is very high(multi_collinearlity) so we will always take any one of them
#in this case we will take character because of its high rrelation with target v1

#data preprocessing of sentence
# it involves to remove puncutions , lower letter steam etc...
import re
nltk.download("stopwords")
from nltk.stem import PorterStemmer
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


df["transformed_text"]=df["v2"].apply(NLP_preprocessor) # we will be creating a new coloumn direct with transformed and modified text

#now we will develop a wordcloud which will tell some most occured  words used in ham and spam

from wordcloud import WordCloud#importing word cloud
wc=WordCloud(background_color="white")# setting the word cloud
word_cloud_spam=wc.generate(df[df["v1"]==1]["transformed_text"].str.cat(sep=" "))#as we will only take the string and concat it with space
plt.imshow(word_cloud_spam)# for spam message
plt.show()
word_cloud_ham=wc.generate(df[df["v1"]==0]["transformed_text"].str.cat(sep=" "))#as we will only take the string and concat it with space
plt.imshow(word_cloud_ham)# for ham message
plt.show()

#to get a top 30 most frequent words
def all_words(data):
    collection_word=[]
    for msg in data:
        for word in msg.split():
            collection_word.append(word)
    return collection_word
from collections import Counter

num_spam=pd.DataFrame(Counter(all_words(df[df["v1"]==0]["transformed_text"])).most_common(30)) #will give the most common 30 words in spam
num_ham=pd.DataFrame(Counter(all_words(df[df["v1"]==1]["transformed_text"])).most_common(30))# will give most common 30 owrds in haam
sns.barplot(x=num_spam[0],y=num_spam[1])# barplot for maximum amount of spam words
plt.show()
sns.barplot(x=num_ham[0],y=num_ham[1])# barplot for maximum amount of spam words
plt.show()

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf=TfidfVectorizer(max_features=3000)
x=tfidf.fit_transform(df["transformed_text"]).toarray()
print(x.shape)
y=df["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
gb=GaussianNB()
mb=MultinomialNB()
bb=BernoulliNB()
gb.fit(x_train,y_train)
mb.fit(x_train,y_train)
bb.fit(x_train,y_train)
y_predict1=gb.predict(x_test)
y_predict2=mb.predict(x_test)
y_predict3=bb.predict(x_test)
from sklearn.metrics import accuracy_score,precision_score
print("gaussian NB:",accuracy_score(y_test,y_predict1),"precision:",precision_score(y_test,y_predict1))
print("multinomai NB:",accuracy_score(y_test,y_predict2),"precision:",precision_score(y_test,y_predict2))
print("bb NB:",accuracy_score(y_test,y_predict3),"precision:",precision_score(y_test,y_predict3))
# we have seen that most  accuracy and precision is in multinomial data set
#to store the file
import pickle # to store my vectorizeer and model in the desktop
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)
with open("model.pkl", "wb") as model_file:
    pickle.dump(mb, model_file)




