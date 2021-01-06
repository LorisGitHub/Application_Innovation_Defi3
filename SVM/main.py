import argparse
from argparse import RawTextHelpFormatter
from src.svmBuilder import *

def arguments_parse():
	parser = argparse.ArgumentParser(description="Algorithme de document", formatter_class=RawTextHelpFormatter)

	parser.add_argument("file", help="Nom du document")
	parser.add_argument("-t", "--type", help="Type de traitement (train/test)", type=str, default="train")
	parser.add_argument("-a", "--algo", type=str, default="base", help="Choix de l'algorithme: \n"
											"b = base\n"
											"f = frequency\n"
											"i = tf - idf\n")
	args = parser.parse_args()

	# Optionnal parameters
	useStemmer = False
	useStopWords = False
	removeLinks = False
	removePunctuation = False

	svmBuilder = SvmBuilder(args.file, args.type == "train", args.algo, useStemmer, useStopWords, removeLinks, removePunctuation)
	if args.algo == "b":
		svmBuilder.initBase()
	else:
		svmBuilder.init()

arguments_parse()