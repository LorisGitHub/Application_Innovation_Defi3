import tensorflow as tf
import xml.etree.ElementTree as ET
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Softmax
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import activations, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import gensim, re, sys, pickle
import numpy as np
from pathlib import Path
from sentimentWord import evaluteSentimentSentence
from nltk.stem.snowball import FrenchStemmer

nb_words = 480268
stemmer = FrenchStemmer()
tokenizer = Tokenizer(num_words=nb_words, lower=True, char_level=False)
word2vec = gensim.models.FastText.load("Models/fasttext_train.model")
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
        noteVec[note-1] = 1
        commentReview = comment.find('commentaire').text
        splitedComment = []
        if(commentReview != '' and commentReview != ' ' and commentReview != None):
            commentReview = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', commentReview)
            commentReview = parseSmiley(commentReview)
            commentReview = replacePonctuation(commentReview)
            words = list(filter(None, commentReview.split(' ')))
            for word in words:
                if stemmer.stem(word.lower()) in word2vec.wv:
                    splitedComment.append(word)
        sentence = " ".join(splitedComment)
        listNote.append(noteVec)
        listReview.append(sentence)
    
    tokenizer.fit_on_texts(listReview)
    word_seq_train = tokenizer.texts_to_sequences(listReview)
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=400)
    return (word_seq_train,listNote)


def prepare_matrix():
    embedding_matrix = np.zeros((nb_words, 100))
    i = 0
    for word, i in tokenizer.word_index:
        if i >= nb_words:
            continue
        embedding_vector = word2vec.wv[word]
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        i += 1
    return embedding_matrix
      
(x_train, y_train), (x_test, y_test) = prepareData(sys.argv[1], 0.8)

embedding_matrix = prepare_matrix()

model = Sequential()
model.add(Embedding(nb_words, 100,
          weights=[embedding_matrix], input_length=500, trainable=False))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','categorical_accuracy'])


checkpointAccuracy = ModelCheckpoint("./bestAccuracyFastText.h5", monitor="val_categorical_accuracy", verbose=0,  save_best_only=True, save_weights_only=False)

model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpointAccuracy])

score = model.evaluate(x_test, y_test, verbose=0)



#print('Test score:', score[0])
#print('Test accuracy:', score[1])