import streamlit as st
import os
import random
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import json
from autocorrect import Speller
from termcolor import colored, cprint
from datetime import datetime
from PIL import Image
spell = Speller(lang ='en')
lem = WordNetLemmatizer()
image = Image.open('chatbot.png')

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(dir_path[:-4]+"data") #Make sure directory is good


intents = json.loads(open('intents.json').read())
words  = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('brain.h5')
gamesort = pd.read_csv('gamecategory.csv')
gamecategory = list(gamesort.Genres.unique())
game = pd.read_csv('game.csv')
def funfact():
    r = requests.get('http://randomfactgenerator.net/')
    soup = bs(r.content)
    filtered = soup.find_all('div', attrs= {'id':'z'})
    filtered = str(filtered[0])[12:]
    filtered = filtered.split('<br/>')
    fact = filtered[0]
    return fact.lower()
def inputclean(_input):
    phrase = spell(_input)
    ind_words = nltk.word_tokenize(phrase)
    words = [lem.lemmatize(word) for word in ind_words]
    return words

def bag_of_words(_input):
    _words = inputclean(_input)
    bag= [0]* len(words)
    #print(_words)
    for x in _words:
        for i, word in enumerate(words):
            if word == x:
                bag[i]= 1
    #print(bag)
    return np.array(bag)

def classpicker(_input): #this predicts what the user is saying/asking using our model
    bow = bag_of_words(_input)
    #print(bow)
    result = model.predict(np.array([bow]))[0]
    threshold = 0.25 # cap for result
    final = [[i,r] for i, r in enumerate(result) if r > threshold]
    
    final.sort(key=lambda x : x[1], reverse =True)
    return_list = []
    for r in final:
        return_list.append({"intent":classes[r[0]], "prob":str([r[1]])})
    return return_list

def response(intents_list,json_file): # outputs the correct response based on the prediction 
    neutral = ['greetings','goodbye','feel?','who?']
    tag = intents_list[0]['intent']
    _list = json_file['intents']
    for i in _list:
        if i['tag'] == tag:
            #print(tag)
            if tag in neutral:# outputs from json response bracket
                reply = random.choice(i['response'])
                break
            if tag == 'game':# outputs all games categories
                gamestr = "\ncasino | simulation | card | board | role playing | music | education | strategy | sports | adventure | puzzle | racing | casual | action | arcade. "
                c = random.choice(i['response'])
                reply = str(c) + gamestr + '\nDoes anything interest you?'
                break
            if tag in gamecategory: # if a category is recognized it prints a random game based on rating 
                a = (random.choice(i['response']))
                b = (random.choice(gamefindr(tag)))
                ans = (a+" "+b.upper())
                reply = (str(ans) + '. \nAnything else I can help you with?')
                break
            if tag == 'time':# Out puts current time
                atime = (random.choice(i['response']))
                now = datetime.now()
                reply = atime + str(now.strftime("%H:%M"))
                break
            if tag == 'fact':# Out puts current time
                afact = (random.choice(i['response']))
                reply = afact + funfact()
                break            
            
    return reply

def gamefindr(category):# sorts games by an acceptable rating 
    tempdf = game.loc[(game.Genres == category) & (game.Rating > 4)]
    return list(tempdf.App)



    
#_input = input()
#inputclean(_input)

def main():
    st.title(" GooglePlay Chatbot")
    st.image(image, caption=' ')
    st.text('''The purpose of this Chatbot it to recommend a game from the GooglePlayStore, based on what \nyou like. It can determined this by what you say.

For example you can say "I like to play games related to music." The bot then recommends a \ngame related to music with a good rating.

It can also answer basic question as well as have some fun features.

Things you can ask :
•About a game from a specific category
•Ask about time
•Ask about a fun fact
    
    ''')
    
    with st.form(key ='chatbox'):
        say = st.text_input("Ask something!")
        submit = st.form_submit_button(label='Send')
        
    if submit:
        message = say
        ints = classpicker(message)
        res = "ChatBot : " + response(ints, intents)
        st.header(res)
            


if __name__ == "__main__":
    main()