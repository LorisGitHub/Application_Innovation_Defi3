import sys
import json
import xml.etree.ElementTree as ET

def loadFile(nameFile):
    pathFile = "Data/" + nameFile + ".xml"
    root = ET.parse(pathFile).getroot()
    return root

def read(root):
    index = 1
    lines = []
    for comment in root.findall('comment'):
        lines.append({"index":{"_index":"data_movie","_id": index}})
        lines.append({'user_id': comment.find('user_id').text, 'movie_id': comment.find('movie').text, 'commentaire': comment.find('commentaire').text, 'review_id': comment.find('review_id').text, 'username': comment.find('name').text, 'lst_mots': splitCommentaire(comment.find('commentaire').text)})
        index += 1
    return lines

def splitCommentaire(commentaire):
    listMots = []
    if commentaire: 
        splitedCommentaires = commentaire.split(' ')
        for item in splitedCommentaires:
            listMots.append({"text":item})
    return listMots

def writeToJson(lines):
    outF = open("mapping_movie.json", "w", encoding='utf8')
    for line in lines:
        json.dump(line, outF, ensure_ascii=False)
        outF.write('\n')
    outF.close()

root = loadFile(sys.argv[1])
lines = read(root)
writeToJson(lines)