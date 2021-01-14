from random import shuffle
import sys, string, json, math, re
from math import floor
import xml.etree.ElementTree as ET

def loadFile(nameFile):
    '''load the xml file into the root'''
    pathFile = "./Data/" + nameFile + ".xml"
    root = ET.parse(pathFile).getroot()
    return root

def read(root):
    notes = {}
    print(len(root.findall('comment')))
    for comment in root.findall('comment'):
        note = comment.find('note').text
        if note not in notes:
            notes[note] = [comment]
        else: 
            notes[note].append(comment)

    mostFrequentNote = 0
    mostFrequentItem = None
    result = []

    for item in notes:
        if(len(notes[item]) > mostFrequentNote):
            mostFrequentNote = len(notes[item])
            mostFrequentItem = item

    for item in notes:
        coef = round(mostFrequentNote/ len(notes[item]))
        result += notes[item]*coef
        print(item, len(notes[item]), "->", len(notes[item]*coef))

    shuffle(result)

    return result

def writeToXml(lines, filename):
    data = ET.Element('comments')

    for line in lines:
        data.append(line)
    tree = ET.ElementTree(data)
    tree.write(filename + "_refactored.xml")

root = loadFile(sys.argv[1])
result = read(root)
#writeToXml(result, sys.argv[1])