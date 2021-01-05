import sys, string, json, math, re
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from gensim.models.fasttext import FastText
import re

review_lines = []
french_stopwords = stopwords.words('french')
french_stopwords.remove("pas")
regexFindSpecialCarac = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ-\s0-9.']+")
regexSmiley = re.compile(r"(\:\)|\:\(|\:\/|\:-\/|\:\||\:p\s)")
stemmer = FrenchStemmer()

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
    print(len(root.findall('comment')))
    index = 0
    for comment in root.findall('comment'):
        try:
            if(comment.find('commentaire').text != '' and comment.find('commentaire').text != None):
                commentClear = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', comment.find('commentaire').text)
                commentClear = parseSmiley(commentClear)
                commentClear = replacePonctuation(commentClear)
                tokens = list(filter(None, commentClear.split(' ')))
                tokens = [stemmer.stem(w.lower()) for w in tokens]
                tokens = [w for w in tokens if not w in french_stopwords]
                review_lines.append(tokens)
        except Exception as e:
            print("Une ligne en moins:", e)

        # if( index % 100 == 0):
        #     print(index, end="\r", flush=True)
        index += 1

    return review_lines


def convertLinesToFastTextFormat(lines, name):
    print("converts")
    model = FastText(sentences = lines, size = 100, window = 5, workers = 4, min_count = 1)
    words = list(model.wv.vocab)
    print('Vocabulary size: %d' % len(words))
    model.save("Models/fasttext_" + name + ".model")


root = loadFile(sys.argv[1])
lines = read(root)
convertLinesToFastTextFormat(lines, sys.argv[1])