import tensorflow as tf
import xml.etree.ElementTree as ET
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical
import numpy as np
import gensim, re
from sentimentWord import evaluteSentimentSentence
from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()

regexFindSpecialCarac = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ-\s0-9.]+")
regexSmiley = re.compile(r"(\:\)|\:\(|\:\/|\:-\/|\:\||\:p\s)")

# Load model fasttext_train
fasttext = gensim.models.FastText.load("../Models/fasttext_train.model") 

model = load_model("./bestAccuracyFastText.h5")


def loadFile():
    '''load the xml file into the root'''
    pathFile = "../../Data/test.xml"
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

# Fonction to predict Polarity With the neural network
def read(root):

    f = open("Results/result.txt","w+",encoding="utf8")
    print("Process Predict...",end="",flush=True)
    result = { }
    #Load model NN
    listPredictComment = []
    reviewsId = []
    for comment in root.findall('comment'):
        text = comment.find('commentaire').text
        reviewsId.append(comment.find('review_id').text)
        meta_donnees = [int(comment.find('movie').text)]
        score = [0]*100
        if(text != '' and text != None):
            text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
            text = text.replace("'", ' ')
            textReview = parseSmiley(text)
            words = list(filter(None, replacePonctuation(textReview).split(' ')))
            for word in words:
                if stemmer.stem(word.lower()) in fasttext.wv:
                    score = [a+b for a, b in zip(score, list(fasttext.wv[stemmer.stem(word.lower())]))]
        score = list(evaluteSentimentSentence(text)) + score
        score = meta_donnees + score
        listPredictComment.append(np.array(score))



    resultModel = list(model.predict(np.array(listPredictComment)))
    for i in range(0,len(resultModel)):
        line = list(resultModel[i])
        index = line.index(max(line))
        note = (index +1)/2
        f.write(reviewsId[i] +" "+ str(note).replace(".",",")+'\n')
    f.close()
    
    #reviewId = comment.find('movie').text

root = loadFile()
read(root)