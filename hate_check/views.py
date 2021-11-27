
from django.shortcuts import render, redirect
from django.urls import path
from django.http import HttpResponse
import re
import numpy as np
import pickle
from string import punctuation
import nltk
nltk.download('punkt')
from preprocessor.api import clean
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer
import tweepy as tw
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')


def index(request):
    if request.method == 'POST':
        url = request.POST.get('username')
        b = check_validtweet(url)
        if(b):
            text = get_tweet_text(url)
            pred = predict(text)
            keywords = ""
            print(pred)
            if(pred != "Not Hate"):
               keywords = ""
            return render(request, "page.html", {'username': pred, 'text': text, 'keywords': keywords})
        else:
            return HttpResponse("inavlid url..sorry")
    else:
        return render(request, "index.html")


def page(request):
    return render(request, "page.html")




def check_validtweet(url):
    pattern = "^https://twitter.com/"
    x = re.findall(pattern, url)
    if x:
        file = url.split("/")
        if(len(file) == 6):
            return True
        else:
            return False
    else:
        return False


def get_tweet_text(url):
    consumer_key = "FvquxZOzpY8T4J0Q01XQFhaeI"
    consumer_secret = 'TRwUZ7zLOVl2aNDTNEjB3LYQo0jTv6jdPV7vZnNHETmkOtGuPS'
    access_token = '1802379589-oN2X9Fc85KNoOIpLbv7RNayqbsw8dvXkXGk6gXi'
    access_token_secret = 'snwKx7pfBF0Tlww1ONxjFVBVzTjpr9PW1x95femY6cTaW'
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    url = url.split("/")
    tweetid = np.uint64(url[-1])
    text = api.get_status(tweetid, tweet_mode='extended').full_text
    return text


def preprocess(text):
    loaded_vect = pickle.load(open("hate_check/vect.pkl", 'rb'))    
    text1 = clean(text.lower())      
    text1 = word_tokenize(text1)
    text1 = ' '.join([word for word in text1 if word not in punctuation])
    words = text1.split()
    words = [word for word in words if word not in STOPWORDS and "'" not in word]
    ps = PorterStemmer()
    words = " ".join([ps.stem(word) for word in words])
    a = loaded_vect.transform([words])
    print(loaded_vect.inverse_transform(a))
    return a;
    

def predict(text):
    #loaded_model = pickle.load(open("predict.sav", 'rb'))
    loaded_model = pickle.load(open("hate_check/random_forest.pkl", 'rb'))
    vect = 0
    with open('hate_check/vect.pkl', 'rb') as read:
        vect = pickle.load(read)
    test = preprocess(text)
    pred = (loaded_model.predict(test))[0]
    d1 = {0:'Not Hate', 2:'Religion', 1:'OtherHate',3:'Sexist'}
    return d1[pred]
