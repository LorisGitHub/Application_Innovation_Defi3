import tensorflow as tf
import xml.etree.ElementTree as ET
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, Activation, Softmax, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import activations, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import gensim, re, sys, pickle
import numpy as np
from pathlib import Path
from statistics import mean
from sentimentWord import evaluteSentimentWord
from nltk.stem.snowball import FrenchStemmer

MAX_NB_WORDS = 800000
nb_words = 0
max_seq_len = 200
stemmer = FrenchStemmer()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
word2vec = gensim.models.FastText.load("Models/fasttext_train.model")
regexFindSpecialCarac = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ-\s0-9.']+")
regexSmiley = re.compile(r"(\:\)|\:\(|\:\/|\:-\/|\:\||\:p\s)")
reviewsId = None


def prepareData(filename, coef):
    global nb_words
    global reviewsId
    global max_seq_len
    listReview = []

    # Reading or saving data from pickle (so we doesn't have to parse the corpus every time) to prepare the model
    if Path('PreparedData/preparedDataFastTextCNN_'+filename+'.pickle').is_file():
        print('Loading train pickle...')
        (listReview,y_train) = loadPickle('preparedDataFastTextCNN_'+filename)
    else: 
        root = loadFile(filename)
        (listReview, y_train) = read(root)
        saveToPickle('preparedDataFastTextCNN_'+filename, listReview, y_train)
    if Path('PreparedData/preparedCNN_Test.pickle').is_file():
        print('Loading test pickle...')
        (listReviewTest,reviewsId) = loadPickle('preparedCNN_Test')
    else: 
        testRoot = loadFile("test")
        (listReviewTest,reviewsId) = readTest(testRoot)
        saveTestToPickle(listReviewTest, reviewsId)

    # Initializing keras tokenizer by fitting every review from train and test to get the same indexes   
    tokenizer.fit_on_texts(listReview + listReviewTest)
    x_train = tokenizer.texts_to_sequences(listReview)

    # Preparing max_seq_len by adding the mean length and the standard deviation of the length of each sentence
    lengths = [len(i) for i in x_train]
    mean = (float(sum(lengths)) / len(lengths))
    std = np.std(lengths)
    max_seq_len = np.round(mean+std).astype(int)
    print("max_seq_len:", max_seq_len)

    # Generating the tokenizer
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
    data_to_test = tokenizer.texts_to_sequences(listReviewTest)
    data_to_test = sequence.pad_sequences(data_to_test, maxlen=max_seq_len)
    data_to_test = (np.asarray(data_to_test))
    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index) + 1)
    print("nb_words:", nb_words)

    # split data for training and validation
    numberOfDataTrain = int(len(x_train) * coef)
    numberOfDataValidation = int(len(x_train) * (1-coef))
    endOfDataValidation = numberOfDataValidation + numberOfDataTrain
    dataTrain = (np.asarray(x_train[0:numberOfDataTrain]),np.asarray(y_train[0:numberOfDataTrain]))
    dataValidation = (np.asarray(x_train[numberOfDataTrain: endOfDataValidation]),np.asarray(y_train[numberOfDataTrain: endOfDataValidation]))

    return dataTrain, dataValidation, data_to_test


def loadFile(nameFile):
    '''load the xml file into the root'''
    pathFile = "../Data/" + nameFile + ".xml"
    root = ET.parse(pathFile).getroot()
    return root  

def loadPickle(namefile):
    f = open('PreparedData/' +namefile + '.pickle', 'rb')
    (x_train,y_train) = pickle.load(f)
    f.close()
    return (x_train,y_train)


def saveToPickle(namefile, listReview, y_train):
    f = open('PreparedData/' + namefile + '.pickle', 'wb')
    pickle.dump((listReview, y_train), f)
    f.close()

def saveTestToPickle(listReviewTest, reviewsId):
    ftest = open('PreparedData/preparedCNN_Test.pickle', 'wb')
    pickle.dump((listReviewTest, reviewsId), ftest)
    ftest.close()
    return (listReviewTest,reviewsId)

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
    print("Process Train")
    listReview = []
    y_train = []
    index = 0
    for comment in root.findall('comment'):
        index += 1
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
        y_train.append(noteVec)
        listReview.append(sentence)
        if( index % 100 == 0):
            print(index, end="\r", flush=True)
        
    return (listReview, y_train)


def readTest(root):
    print("Process Test")
    listReview = []
    reviewsId = []
    index = 0
    for comment in root.findall('comment'):
        index += 1
        reviewsId.append(comment.find('review_id').text)
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
        listReview.append(sentence)
        if( index % 100 == 0):
            print(index, end="\r", flush=True)
        
    return (listReview, reviewsId)


def prepare_matrix():
    print("Process embedding matrix...")
    embedding_matrix = np.zeros((nb_words, 101))
    i = 0
    for word in tokenizer.word_index:
        if i >= nb_words:
            continue
        embedding_vector = word2vec.wv[word]
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = np.append(embedding_vector, float(evaluteSentimentWord(word)))
        i += 1
    return embedding_matrix


def writeToFile(resultModel):
    f = open("Results/resultCNN.txt","w+",encoding="utf8")
    for i in range(0,len(resultModel)):
        line = list(resultModel[i])
        index = line.index(max(line))
        note = (index +1)/2
        f.write(reviewsId[i] +" "+ str(note).replace(".",",")+'\n')
    f.close()
      
(x_train, y_train), (x_test, y_test), test_data = prepareData(sys.argv[1], 0.8)

embedding_matrix = prepare_matrix()

model = Sequential()
model.add(Embedding(nb_words, 101, weights=[embedding_matrix], input_length=max_seq_len, trainable=True))
#model.add(BatchNormalization())
#model.add(Conv1D(64, 7, activation='relu', padding='same'))
#model.add(MaxPooling1D(2))
#model.add(Conv1D(64, 7, activation='relu', padding='same'))
model.add(Conv1D(64, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
#model.add(Flatten())
model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','categorical_accuracy'])
model.summary()

checkpointAccuracy = ModelCheckpoint("./bestAccuracyFastTextCNN.h5", monitor="val_categorical_accuracy", verbose=0,  save_best_only=True, save_weights_only=False)
model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpointAccuracy])

model.load_weights("./bestAccuracyFastTextCNN.h5")

#model = load_model("./bestAccuracyFastTextCNN.h5")

print("Process predict...")
resultModel = list(model.predict(test_data))
writeToFile(resultModel)