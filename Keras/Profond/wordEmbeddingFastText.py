import tensorflow as tf
import xml.etree.ElementTree as ET
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical
import gensim, re, sys, pickle
import numpy as np
from pathlib import Path
from sentimentWord import evaluteSentimentSentence
from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()
word2vec = gensim.models.FastText.load("Models/fasttext_dev.model")
regexFindSpecialCarac = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ-\s0-9.']+")
regexSmiley = re.compile(r"(\:\)|\:\(|\:\/|\:-\/|\:\||\:p\s)")


def prepareData(filename, coef):
    listReview = []
    if Path('PreparedData/preparedDataFastText_'+filename+'.pickle').is_file():
        f = open('PreparedData/preparedDataFastText_'+filename+'.pickle', 'rb')
        (listReview,listNote) = pickle.load(f)
        f.close()
    else: 
        f = open('PreparedData/preparedDataFastText_'+filename+'.pickle', 'wb')
        root = loadFile(filename)
        (listReview,listNote) = read(root)
        pickle.dump((listReview,listNote), f)
        f.close()
    numberOfDataTrain = int(len(listReview) * coef)
    numberOfDataValidation = int(len(listReview) * (1-coef))
    endOfDataValidation = numberOfDataValidation + numberOfDataTrain
    dataTrain = (np.asarray(listReview[0:numberOfDataTrain]),np.asarray(listNote[0:numberOfDataTrain]))
    dataValidation = (np.asarray(listReview[numberOfDataTrain: endOfDataValidation]),np.asarray(listNote[numberOfDataTrain: endOfDataValidation]))
    return dataTrain, dataValidation


def loadFile(nameFile):
    '''load the xml file into the root'''
    pathFile = "../Data/" + nameFile + ".xml"
    root = ET.parse(pathFile).getroot()
    return root
    

def parseSmiley(comment):
    matches = re.finditer(regexFindSpecialCarac, comment)
    for matchNum, match in enumerate(matches, start=1):
        if not(regexSmiley.search(match.group())):
            comment = comment.replace(match.group(), '', 1)
    return comment

def replacePonctuation(comment):
    regex = r"(\.{2})(\.{1,})|(\!{2})(\!{1,})"
    subst = "\\1\\3"
    comment = re.sub(regex, subst, comment, 0, re.MULTILINE)
    return comment


def read(root):
    listReview = []
    listNote = []
    index = 0
    for comment in root.findall('comment'):
        index += 1
        score = [0]*100
        noteVec = [0]*10
        note = int(float(comment.find('note').text.replace(',', '.'))*2)
        # meta_donnees = [int(comment.find('movie').text)] # , comment.find('user_id').text[:1]
        noteVec[note-1] = 1
        commentReview = comment.find('commentaire').text
        if(commentReview != '' and commentReview != ' ' and commentReview != None):
            commentReview = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', commentReview)
            commentReview = parseSmiley(commentReview)
            commentReview = replacePonctuation(commentReview)
            words = list(filter(None, commentReview.split(' ')))
            for word in words:
                if stemmer.stem(word.lower()) in word2vec.wv:
                    score = [a+b for a, b in zip(score, list(word2vec.wv[stemmer.stem(word.lower())]))]
        score = list(evaluteSentimentSentence(commentReview))+score
        #score = meta_donnees + score
        listReview.append(score)
        listNote.append(noteVec)
        if( index % 100 == 0):
            print(index, end="\r", flush=True)
      
    return (listReview,listNote)
      
        
model = Sequential()
model.add(Dense(150, input_shape=(102,), activation='relu'))
model.add(Dense(150, activation='tanh'))
model.add(Dense(80, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','categorical_accuracy'])


checkpointAccuracy = ModelCheckpoint("./bestAccuracyFastText.h5", monitor="val_categorical_accuracy", verbose=0,  save_best_only=True, save_weights_only=False)


(x_train, y_train), (x_test, y_test) = prepareData(sys.argv[1], 0.8)


model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpointAccuracy])

score = model.evaluate(x_test, y_test, verbose=0)



print('Test score:', score[0])
print('Test accuracy:', score[1])