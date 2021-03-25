import tkinter
from tkinter import *

import pickle
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import json
import random
with open('mhc_intents_data_2.json') as json_data:

    intents = json.load(json_data)

stemmer =  LancasterStemmer()


model = load_model('my_model.h5')    
his = []
meaning = {1:'all or nothing', 2: 'overgeneralization', 3:'discounting the positive', 4: 'catastrophizing', 5:'fallacy'}
cbt_dists = list(meaning.values())  
 
global graph
graph = tf.compat.v1.get_default_graph()

filename = 'final_intents_data.pkl'
model_2 = pickle.load(open(filename, 'rb'))

words = model_2['words']
classes = model_2['classes']


def classify_resp(sentence_in):
    
    sentence_words_1 = nltk.word_tokenize(sentence_in)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words_1]
    # bag of words
    bag = [0]*len(words)  

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    
        # generate probabilities from the model
    input_data = pd.DataFrame([np.array(bag)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
        
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]]})
        # return tuple of intent and probability
    intnt = return_list[0]['intent']

    #resp = [d['responses'][np.random.randint(0,len(d['responses']))]  for d in intents['intents'] if d['tag'] == intnt]
    for d in intents['intents']:

      if d['tag'] == intnt:

        l = len(d['responses'])

        ind = np.random.randint(0,l)

        response = d['responses'][ind]

        his.append(intnt)

    return response

def get_max(h):
    d = {}
  

    new_h = [f for f in his if f in cbt_dists]
  
    for f in new_h:
        if f not in d.keys():
            d[f] =1
   
        else:
            d[f]+=1


    vals = list(d.values())
    indx = vals.index(max(vals))

    return list(d.keys())[indx]

def acknowledge(new_his):

    dist = get_max(new_his)

    remedy = {'all or nothing': "My analysis tells me you're showing signs of 'ALL OR NOTHING' thinking in which an individual displays a polarized, black and white thinking. Let me tell you a small anecdote that may help you. Roger recently went through a terrible breakup. He decided one day to finally go out there again and face his anxiety and ask a woman out on a date. He left her a voicemail message. A few days go by and Roger hasn't heard back from her. He thinks, 'I'm a total loser with nothing to offer ... No one wants to go out with me ... I will never find the right person, so why bother?' He starts to feel nervous and upset as he considers a future alone. Do you see the problem here dear? Rather than finding the middle ground in ​this scenario, Roger is thinking in extremes. He can replace his negative self-defeating thoughts with more realistic ones. Roger could consider the possibility that the woman didn’t get his message or is out of town. Even if she forgot about the message or decided to ignore it, Roger can choose to think that he is still a worthwhile person. He can remind himself that this particular person may just not have been right for him. Hence, in any case, it is important to avoid thinking in negative, absolute terms, find the positives and acknowledge that setbacks happen rather than focus on faults and self - defeating thoughts.",
       'overgeneralization': "My analysis tells me you're showing signs of 'OVERGENERALIZING' in which an individual tends to assume certain results for a large number of scenarios. Let me tell you a small anecdote that you may relate to. Karen has a very important business meeting today. She was up all night preparing for it. She organises everything for the next morning hoping for a smooth lead up to the meeting. The next morning when she is going through her notes for the meeting, she spills coffee on them and ruins them. By this point she is late. She gets out of the house and drives to the office already stopping at 3 red lights. When the fourth one stops her as well she exclaims out of frustration, 'Why me? Why do I always hit a red light?' The problem with her statement is that although she may have coincidentally encountered many red lights, she doesn't encounter all of them. In this way she may wrongly start responding to a pattern of events instead of just the one event she has faced. Her tendency to overgeneralize will pile her anger up and eventually lead her to get frustrated by every little thing. So it is best for her to use more situation accurate and realistic terms.",
       'discounting the positive' : "My analysis tells me you're showing signs of 'DISCOUNTING THE POSITIVE' in which an individual ignores the positives and adapts to a pessimistic behaviour. Let me tell you a small anecdote that may help you. Alex is always first in all her classes. She is appreciated for her intelligence both by family and teachers at school. She works relentlessly to be consistent in her efforts but never takes a moment to appreciate herself for her hard work. Recently, when she aced her algebra test, her teacher Ms. Nolan applauded her for her score. Alex acknowledged her but felt restless secretly looking for faults in her questions. She saw Ms. Nolan's praise as a way to overlook her mistakes and beat herself up over a minute mistake. If Alex follows the same pattern, she may do more damage to her mental health and confidence. To avoid any such negative thoughts, Alex should appreciate herself more and reward herself for all the rights than punishing herself for the wrongs.",
       'catastrophizing': "My analysis tells me you're showing signs of 'CATASTROPHIZING' in which an individual tends to think of worst of outcomes of all. Let me tell you an instance that you may relate to. Luke was the senior quarterback on the high school football team. He was an excellent player and coach's favourite. In the last game, he had a lapse of judgement when he passed the ball to Phil and not Jay who was nearer to the post. They lost the game failing to score that point. Luke blamed himself or this loss when they were destined to loose from the very beginning. He magnified the importance of that mistake and labelled himself as a terrible teammate. Luke was clearly wrong to do make such a huge claim for a small error. He overestimated the magnitude of it and made it a bigger deal than it was. Luke should have instead acknowledged that although it was a misjudged move, it was not the reason they lost. Losing the game was not the end of the world for Luke and not something that would define him. He should still see himself as a good team player despite a minute loss.",
       'fallacy': "My analysis tells me you're showing signs of 'FALLACIES' in which an individual becomes delusional. Let me tell you about a time that you may relate to. Phil is a middle aged realtor who gave up on his dream to be a magician. He has a son who has shown a little interest in magic. Phil, thinking that he can live his dream through his son, encourages him to go to Magic Camp for the summer. His son, although having a talent for the same, returns early as he lost interest. Phil is very disappointed in his son and fails to understand that his happiness should not depend on his son's success. He should not try to fulfil his dreams through his son as forcing him to be someone he is not will not be the way he gets what he wants. He will not be a better father by bonding with his son over an activity that brings his son no joy. Phil needs to understand that his happiness should be dependent on his son's ability to resonate with his own interests and likings."
       }

    return remedy[dist]


def reply(msg):

    sentence_in = msg
    
    new_h = [f for f in his if f in cbt_dists]

    if len(new_h) >=5:

        res = acknowledge(new_h)

    else:
        res = classify_resp(sentence_in)

    return res


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = reply(msg)
        ChatLog.insert(END, "Kayra: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("KAYRA")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()