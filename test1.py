import nltk.data
import urllib
import os
import glob
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
from urllib.request import urlopen
from textstat import textstat
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.tokenize import MWETokenizer
from statistics import mean
from statistics import median

import re
from PyDictionary import PyDictionary

posListAdjectives = ["JJ", "JJR", "JJS"]
posListAdverbs = ["RB", "RBR", "RBS", "WRB"]
posListCommonNoun = ["NN", "NNS"]

obligationWords = ['must','shall','should','have to']
permissionWords = ['can','could','may', 'might', 'ought', 'will', 'would']
prohibitionWords = ['must_not', 'shall_not', 'shan\'t',  'should_not',
                    'shouldn\'t', 'can_not', 'can\'t', 'could_not', 'couldn\'t', 'may_not', 'might_not',
                    'ought_not', 'will_not', 'won\'t', 'would_not', 'wouldn\'t',]

path = 'C:/Users/anantaa/Desktop/ALDA/OPP-115/sanitized_policies'
outputpath = 'C:/Users/anantaa/Desktop/python/AmbiguityDetection/result'
fileinitialiser = 'file:///'

def createNewFileName(fileName) :
    newFileName = fileName.strip(path)
    newFileName = newFileName.strip("\\")
    newFileName = newFileName.strip(".com.html")

    return newFileName

def preProcessFile (filename) :
    page = urlopen(fileinitialiser + filename)
    pageRead = page.read().decode('utf-8', 'ignore')
    page.close()
    soup = BeautifulSoup(pageRead, 'lxml')

    for script in soup.findAll('strong'):
        script.extract()  # rip it out

    for script in soup.findAll("(\\d|\\W)+"):
        script.extract()

    text = soup.get_text()
    text = re.sub("  ", ".", text)
    text = re.sub("[.]+", ". ", text)
    text = re.sub("[|]+", " ", text)

    return text

def tokenizeSentence (text) :
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenized_sent = sent_detector.tokenize(text.strip())

    return tokenized_sent


def pre_process(text):
    # lowercase
    text = text.lower()
    # remove tags
    text = re.sub("<!--?.*?-->", "", text)
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    text = re.sub("<!--?.*?-->", "", text)
    text = re.sub("[0-9]+", "", text)
    # text = re.sub("[.]+", ".", text)


    return text


def extractAmbiguousWords(tokenized_sent, countWords, f):
    ambwordlist = set()
    wordCount = 0

    for sent in tokenized_sent:
        # f.write("\n" + sent + " Words :: ")
        posTagged = []
        words = nltk.word_tokenize(sent)
        for word in words:
            word = pre_process(word)

        posTagged.append(nltk.pos_tag(words))

        extractAdjectivess = extractWords(posTagged, posListAdjectives)
        wordCount += len(extractAdjectivess)
        for eachAdjective in extractAdjectivess:
            amb_msr = 0
            for syn in wn.synsets(str(eachAdjective)):
                if syn.pos() == "a":
                    amb_msr += 1
            if amb_msr > 1:
                ambwordlist.add(eachAdjective)
                # f.write(eachAdjective)

        extractAdverbs = extractWords(posTagged, posListAdverbs)
        wordCount += len(extractAdverbs)

        for eachAdverb in extractAdverbs:
            amb_msr = 0
            for syn in wn.synsets(str(eachAdverb)):
                if syn.pos() == "r":
                    amb_msr += 1
            if amb_msr > 1:
                ambwordlist.add(eachAdverb)
                # f.write(eachAdverb)
    # f.write(str(ambwordlist).strip('[]') + "\n")

    f.write("Count of Ambiguous Words ::" + str(len(ambwordlist)) + "\n")
    f.write("Normalized Count of Ambiguous Words ::" + str(len(ambwordlist)/wordCount) + "\n")

    return len(ambwordlist)/wordCount


def findReadingEase (text, f, sentClass):
    # f.write(text + "\n")
    # f.write("flesch_reading_ease Score for " + sentClass + " is: " + str(textstat.flesch_reading_ease(text)) + "\n")
    # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
    f.write("flesch_kincaid_grade Score: " + sentClass + " is: " + str(textstat.flesch_kincaid_grade(text)) + "\n")
    # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level
    f.write("linsear_write_formula Score: " + sentClass + " is: " + str(textstat.linsear_write_formula(text)) + "\n")
    # https://en.wikipedia.org/wiki/Linsear_Write

    return textstat.linsear_write_formula(text)

def findMostComplexSentence(tokenized_sent, f):
    linsear_write_formula = 0;
    complex_sent = ''

    for sent in tokenized_sent:
        if (textstat.linsear_write_formula(sent) > linsear_write_formula):
            linsear_write_formula = textstat.linsear_write_formula(sent)
            complex_sent = sent

    f.write("Most Complex sentence is: " + complex_sent + "\n")


def deonticLogicComparison(tokenized_sent, countWords, f, prohibition_sent, obligation_sent, permission_sent):
    obligationFactor = 0
    permissionFactor = 0
    prohibitionFactor = 0

    tokenizer = MWETokenizer([('must', 'not'), ('shall', 'not'), ('should', 'not'), ('can', 'not'),
                              ('could', 'not'), ('may', 'not'), ('might', 'not'), ('ought', 'not'),
                              ('will', 'not'), ('would', 'not')])

    for sent in tokenized_sent:
        words = tokenizer.tokenize(sent.split())

        for word in words:
            if word in prohibitionWords:
                # f.write(word + "::" + str(sent) + "\n")
                prohibition_sent.append(sent)
                prohibitionFactor += 1

            elif word in permissionWords:
                # f.write(word + "::" + str(sent) + "\n")
                permission_sent.append(sent)
                permissionFactor += 1

            elif word in obligationWords:
                # f.write(word + "::" + str(sent) + "\n")
                obligation_sent.append(sent)
                obligationFactor += 1

    f.write("obligationFactor: " + str(len(obligation_sent)/len(tokenized_sent)) + "\n")
    f.write("permissionFactor: " + str(len(permission_sent)/len(tokenized_sent)) + "\n")
    f.write("prohibitionFactor: " + str(len(permission_sent)/len(tokenized_sent)) + "\n")

    return prohibition_sent, permission_sent, obligation_sent


def extractWords(posTagged, posList):
    extractedWords = []
    for eachPOSTag in posList:
        for posTaggedEach in posTagged:
            for word, pos in posTaggedEach:
                if pos == eachPOSTag:
                    extractedWords.append(word)
    return extractedWords


def countWords(tokenized_sent):
    words_in_text = set()
    for sent in tokenized_sent:
        words = nltk.word_tokenize(sent)
        for word in words:
            word = pre_process(word)
            if word not in STOPWORDS:
                words_in_text.add(word)

    return len(words_in_text)


def generateWordCloud(theWords):
    print("\nGenerating the Word Cloud")
    plt.imshow(theWords)
    plt.axis("off")
    plt.show()

def getTextFromSent (tokenized_sent):
    text = ''
    for sent in tokenized_sent:
        text += (sent)

    return text


readingEaseProhibition = []
readingEaseObligation = []
readingEasePermission = []
readingEase = []
amb_count = []
obligationFactor = []
permissionFactor = []
prohibitionFactor = []
zero_obligation = []
zero_permission = []

obligation_to_permission_ratio =[]

for filename in glob.glob(os.path.join(path, '*.html')):
    prohibition_sent = []
    obligation_sent = []
    permission_sent = []

    text = preProcessFile(filename)

    tokenized_sent = tokenizeSentence(text)
    f = open(os.path.join(outputpath, createNewFileName(filename)) + ".txt", "w")

    countOfWords = countWords(tokenized_sent)
    prohibition_sent, obligation_sent, permission_sent = \
        deonticLogicComparison(tokenized_sent, countOfWords, f, prohibition_sent, obligation_sent, permission_sent)

    obligation = len(obligation_sent) / len(tokenized_sent)
    permission = len(permission_sent) / len(tokenized_sent)

    obligationFactor.append(obligation)
    permissionFactor.append(permission)
    prohibitionFactor.append(len(prohibition_sent)/len(tokenized_sent))

    if permission == 0:
        permission = 1
        zero_permission.append(filename)

    if obligation == 0:
        zero_obligation.append(filename)

    obligation_to_permission_ratio.append(obligation / permission)

    prohibition_text = getTextFromSent(prohibition_sent)
    obligation_text = getTextFromSent(obligation_sent)
    permission_text = getTextFromSent(permission_sent)

    readingEaseProhibition.append(findReadingEase(prohibition_text, f, "Prohibition Sentences"))
    readingEaseObligation.append(findReadingEase(obligation_text, f, "Obligation Sentences"))
    readingEasePermission.append(findReadingEase(permission_text, f, "Permission Sentences"))
    readingEase.append(findReadingEase(text, f, "Whole text"))

    findMostComplexSentence(tokenized_sent, f)

    amb_count.append(extractAmbiguousWords(tokenized_sent, countOfWords, f))
    f.close()

f2 = open(os.path.join(outputpath, createNewFileName("123")) + ".txt", "w")

f2.write("Mean readingEase for Prohibition Sentences : " + str(mean(readingEaseProhibition)) + "\n")
f2.write("Median readingEase for Prohibition Sentences : " + str(median(readingEaseProhibition)) + "\n")
f2.write("Max readingEase for Prohibition Sentences : " + str(max(readingEaseProhibition)) + "\n")

f2.write("Mean readingEase for Obligation Sentences : " + str(mean(readingEaseObligation)) + "\n")
f2.write("Median readingEase for Obligation Sentences : " + str(median(readingEaseObligation)) + "\n")
f2.write("Max readingEase for Obligation Sentences : " + str(max(readingEaseObligation)) + "\n")

f2.write("Mean readingEase for Permission Sentences : " + str(mean(readingEasePermission)) + "\n")
f2.write("Median readingEase for Permission Sentences : " + str(median(readingEasePermission)) + "\n")
f2.write("Max readingEase for Permission Sentences : " + str(max(readingEasePermission)) + "\n")

f2.write("Mean readingEase for All Sentences : " + str(mean(readingEase)) + "\n")
f2.write("Median readingEase for All Sentences : " + str(median(readingEase)) + "\n")
f2.write("Max readingEase for All Sentences : " + str(max(readingEase)) + "\n")

f2.write("Mean Ambiguity Count : " + str(mean(amb_count)) + "\n")
f2.write("Median Ambiguity Count : " + str(median(amb_count)) + "\n")
f2.write("Max Ambiguity Count : " + str(max(amb_count)) + "\n")
f2.write("Min Ambiguity Count : " + str(min(amb_count)) + "\n")

f2.write("Mean obligationFactor : " + str(mean(obligationFactor)) + "\n")
f2.write("Median obligationFactor : " + str(median(obligationFactor)) + "\n")
f2.write("Max obligationFactor : " + str(max(obligationFactor)) + "\n")

f2.write("Mean permissionFactor : " + str(mean(permissionFactor)) + "\n")
f2.write("Median permissionFactor : " + str(median(permissionFactor)) + "\n")
f2.write("Max permissionFactor : " + str(max(permissionFactor)) + "\n")

f2.write("Mean prohibitionFactor : " + str(mean(prohibitionFactor)) + "\n")
f2.write("Median prohibitionFactor : " + str(median(prohibitionFactor)) + "\n")
f2.write("Max prohibitionFactor : " + str(max(prohibitionFactor)) + "\n")

f2.write("Mean obligation_to_permission_ratio : " + str(mean(obligation_to_permission_ratio)) + "\n")
f2.write("Median obligation_to_permission_ratio : " + str(median(obligation_to_permission_ratio)) + "\n")
f2.write("Max obligation_to_permission_ratio : " + str(max(obligation_to_permission_ratio)) + "\n")
f2.write("Min obligation_to_permission_ratio : " + str(min(obligation_to_permission_ratio)) + "\n")

f2.write("No obligation:\n")
for f in zero_obligation:
    f2.write(f)

f2.write("No permission:\n")
for f in zero_permission:
    f2.write(f)

f2.close()
