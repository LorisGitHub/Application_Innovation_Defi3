import sys, string, json, math, re
import xml.etree.ElementTree as ET
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords

class SvmBuilder:

    def __init__(self, nameFile, train, algo, stem = False, stopWords = False, links = False, punctuation = False):
        self.root = self.loadFile(nameFile)
        self.nameFile = nameFile
        self.train = train
        self.algo = algo
        self.stem = stem
        self.stopWords = stopWords
        self.links = links
        self.punctuation = punctuation
        self.wordIndex = 1
        self.xmlLength = 0
        self.dictionnary = {}
        self.dictionnary_count = {}
        self.lines = []
        self.french_stopwords = list(set(stopwords.words('french')))
        self.stemmer = FrenchStemmer()

        print("Class initialized!")


    def loadFile(self, nameFile):
        '''load the xml file into the root'''
        pathFile = "./Data/" + nameFile + ".xml"
        root = ET.parse(pathFile).getroot()
        print('Longueur du corpus', len(root.findall('comment')))
        return root


    def splitReview(self, review):
        '''split the user review based on different filters'''
        if(review == None or len(review) == 0):
            return ' '
        if(self.links):
            review = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', review)
        if(self.punctuation):
            #review = re.sub('(@[A-Za-z0-9_]+)','', review)
            review = review.translate(str.maketrans('', '', string.punctuation))
        return review.split(' ')


    def initBase(self):
        '''init the base algorithm, use median score to give a base value'''
        index = 0
        for comment in self.root.findall('comment'):
            index += 1
            note = "7" if self.train else "0"
            splitedReview = self.splitReview(comment.find('commentaire').text)
            baseBuildedReview = self.buildBaseReview(splitedReview)
            self.lines.append( note + baseBuildedReview )
            if(index % 1000 == 0):
                print(index, flush=True, end='\r')
        self.writeToFile()


    def init(self):
        '''init the frequency algorithm, use frequency of word in sentence'''
        index = 0
        self.xmlLength = len(self.root.findall('comment'))
        if self.algo == 'i':
            print("preparing Tf-Idf")
            for comment in self.root.findall('comment'):
                index += 1
                splitedReview = self.splitReview(comment.find('commentaire').text)
                self.prepareDictionnaryForIdf(splitedReview)
                if(index % 1000 == 0):
                    print(index, flush=True, end='\r')
        index = 0
        print("executing normal loop")
        for comment in self.root.findall('comment'):
            index += 1
            note = str(int(float(comment.find('note').text.replace(',', '.'))*2)) if self.train else "0"
            splitedReview = self.splitReview(comment.find('commentaire').text)
            if self.algo == 'f':
                baseBuildedReview = self.buildBaseReview(splitedReview)
            elif self.algo == 'i':
                baseBuildedReview = self.buildIdfReview(splitedReview)
            self.lines.append( note + baseBuildedReview )
            if(index % 1000 == 0):
                print(index, flush=True, end='\r')
        self.writeToFile()


    def buildBaseReview(self, splitedReview):
        if splitedReview == ' ':
            return splitedReview

        svmReview = ''
        localIndexes = []
        for item in splitedReview:
            if (self.stopWords and item not in self.french_stopwords) or self.stopWords == False :
                if self.stem:
                    itemCopy = self.stemmer.stem(item.lower())
                else: 
                    itemCopy = item.lower()
                if itemCopy not in self.dictionnary:
                    self.dictionnary[itemCopy] = self.wordIndex
                    self.wordIndex += 1
                localIndexes.append(self.dictionnary[itemCopy])

        uniqLocalIndex = list(set(localIndexes))
        uniqLocalIndex.sort()
        for word in uniqLocalIndex:
            svmReview += " " + str(word) + ":" + str(localIndexes.count(word))
        return svmReview


    def prepareDictionnaryForIdf(self, splitedReview):
        if splitedReview == ' ' or splitedReview == None:
            return None

        localDictionary = []
        for item in splitedReview:
            if (self.stopWords and item not in self.french_stopwords) or self.stopWords == False :
                if self.stem:
                    itemCopy = self.stemmer.stem(item.lower())
                else: 
                    itemCopy = item.lower()
                if itemCopy not in self.dictionnary:
                    self.dictionnary[itemCopy] = self.wordIndex
                    self.dictionnary_count[self.wordIndex] = 1
                    self.wordIndex += 1
                    localDictionary.append(itemCopy)
                elif itemCopy not in localDictionary:
                    self.dictionnary_count[self.dictionnary[itemCopy]] += 1
                    localDictionary.append(itemCopy)


    def buildIdfReview(self, splitedReview):
        if splitedReview == ' ' or splitedReview == None:
            return ' '

        svmReview = ''
        localIndexes = []
        for item in splitedReview:
            if (self.stopWords and item not in self.french_stopwords) or self.stopWords == False :
                if self.stem:
                    itemCopy = self.stemmer.stem(item.lower())
                else: 
                    itemCopy = item.lower()
                localIndexes.append(self.dictionnary[itemCopy])

        uniqLocalIndex = list(set(localIndexes))
        uniqLocalIndex.sort()
        for word in uniqLocalIndex:
            TF = localIndexes.count(word) / len(localIndexes)
            IDF = math.log(self.xmlLength / self.dictionnary_count[word])
            svmReview += " " + str(word) + ":" + str(TF*IDF)
        return svmReview


    def writeToFile(self):
        outF = open(self.nameFile +".svm", "w", encoding='utf8')
        for line in self.lines:
            outF.write(line)
            outF.write('\n')
        outF.close()
        print("File is complete!")