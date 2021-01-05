import sys
import json
import xml.etree.ElementTree as ET

def loadFile(nameFile):
    pathFile = "Data/" + nameFile + ".xml"
    root = ET.parse(pathFile).getroot()
    return root

def loadOut(nameFile):
    pathFile = "Data/" + nameFile + ".txt"
    file = open(pathFile, "r")
    outlines = file.read().split('\n')
    file.close()
    return outlines

def read(root, outlines):
    lines = []
    for index, comment in enumerate(root.findall('comment'), start=0):
        lines.append(comment.find('review_id').text + " " + str(float(outlines[index])/2).replace('.',','))
        if(index % 1000 == 0):
            print(index, flush=True, end='\r')
    return lines

def writeToOutput(lines):
    outF = open("realOutput.txt", "w", encoding='utf8')
    for line in lines:
        outF.write(line)
        outF.write('\n')
    outF.close()

root = loadFile(sys.argv[1])
outlines = loadOut(sys.argv[2]) 
lines = read(root, outlines)
writeToOutput(lines)