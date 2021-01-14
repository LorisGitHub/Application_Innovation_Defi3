import spacy

"""print("Load Lemmatizer....")
nlp = spacy.load("fr_core_news_md")
print("\tDone")"""
from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()


listPositiveWord = ["superbement", "merci", "touchant", "perfection", "leçon", "remarquable", "excellent", "parfait", "bonheur", "émotion", "touchant", "parfait", "sublime", "superbe", "excellent", "oeuvre", "extraordinaire", ":)", "justesse", "magnifiques", "réussite", "magnifique", "bravo", "super", "bijou", "émouvant", "recommande", "magnifique", "larmes", "vivement", "claque", "adoré", "merveille", "magnifiquement", "gags", "formidable", "intense", "génial"]
listNegativeWord = ["navet", "ennui", "intérêt", "nul", "mauvais", "téléfilm", "ennuyeux", "mauvais", "pseudo", "ridicule", "éviter", "décevant", "déception", "étoile", "aucun", "ridicule", "raté", "pire", "plat", "déception", "intérêt", "vide", "pauvre", "sauve", "rien", "passez"]

#listPositiveWord = [nlp(i.lower())[0].lemma_ for i in listPositiveWord]
#listNegativeWord = [nlp(i.lower())[0].lemma_ for i in listNegativeWord]
listPositiveWord = [stemmer.stem(i.lower()) for i in listPositiveWord]
listNegativeWord = [stemmer.stem(i.lower()) for i in listNegativeWord]


dicMem = {}

def evaluteSentimentSentence(text):
    #[positive, negative]
    result = [0,0]
    if(text != '' and text != None):
        words = []
        for k in text.split(' '):
            k = k.lower()
            if(len(k)>0):
                if(k in dicMem):
                    words.append(dicMem[k])
                else:
                    lemma = stemmer.stem(k)
                    dicMem[k] = lemma
                    words.append(lemma)
        #words = [nlp(i.lower())[0].lemma_ for i in text.split(' ') if len(i)>0] B====>
        """for word in words:
            if( word in listPositiveWord):
                result[0] += 1
            if( word in listNegativeWord):
                result[1] += 1"""
        for word in listPositiveWord:
            if(word in words):
                result[0] += 1

        for word in listNegativeWord:
            if(word in words):
                result[1] += 1
    return tuple(result)

def evaluteSentimentWord(word):
    if(word != '' and word != None):
        if word in listPositiveWord:
            return 1

        if word in listNegativeWord:
            return -1
        
    return 0
