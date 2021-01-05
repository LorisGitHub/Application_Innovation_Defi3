import sys
import json
import xml.etree.ElementTree as ET

blockList = ['le', 'la', 'l\'', 'un', 'une', 'du', 'de', 'de la', 'les', 'des', 'ce', 'cet', 'cette', 'mon', 'ton', 'son', 'notre', 'votre', 'leur', 'ces', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'et', 'en', 'à', 'qui', 'que', 'avec', 'au', 'aux', 'pour']
ponctuList = ['.', '!', '?']

def loadFile(nameFile):
    pathFile = "Data/" + nameFile + ".xml"
    root = ET.parse(pathFile).getroot()
    return root

def read(root):
    index = 1
    lines = []
    for comment in root.findall('comment'):
        lines.append({"index":{"_index":"data_movie","_id": index}})
        lines.append({'user_id': comment.find('user_id').text, 'movie_id': comment.find('movie').text, 'note': float(comment.find('note').text.replace(',','.')), 'commentaire': comment.find('commentaire').text, 'review_id': comment.find('review_id').text, 'username': comment.find('name').text, 'lst_mots': splitCommentaire(comment.find('commentaire').text)})
        index += 1
    return lines

def splitCommentaire(commentaire):
    if(commentaire == None):
        return []
    splitedCommentaires = commentaire.split(' ')
    listMots = []
    for item in splitedCommentaires:
        if(len(item) > 0):
            if not(item.lower() in blockList):
                if(item[0] in ponctuList and len(item) >= 2):
                    listMots.append({"text": str(item[0]*3)})
                else:
                    listMots.append({"text":item})
    return listMots

def writeToJson(lines):
    indexName = 1
    fileName = "out" + str(indexName) + ".json"
    outF = open(fileName, "w", encoding='utf8')
    for index, line in enumerate(lines, start=0):
        if(index == indexName*70000):
            outF.close()
            indexName += 1
            fileName = "out" + str(indexName) + ".json"
            outF = open(fileName, "w", encoding='utf8')
        json.dump(line, outF, ensure_ascii=False)
        outF.write('\n')
    outF.close()

root = loadFile(sys.argv[1])
lines = read(root)
writeToJson(lines)