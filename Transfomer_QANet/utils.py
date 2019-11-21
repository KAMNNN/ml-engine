import pandas as pd
import _pickle as cPickle
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")

def dumpPickle(filename, buf):
    file = open(filename, 'wb')
    cPickle.dump(buf, file, -1)
    file.close()

def loadPickle(filename):
    file = open(filename, 'rb')
    content = cPickle.load(file)
    file.close()
    return content

def pickleEists(filename):
    file = Path (file)
    if (file.is_file()):
        return True
    return False


def getNEStartIndexs(doc):
    neStarts = {}
    for ne in doc.ents:
        neStarts[ne.start] = ne
        
    return neStarts


def getSentenceStartIndexes(doc):
    senStarts = []    
    for sentence in doc.sents:
        senStarts.append(sentence[0].i)
    return senStarts
 

def getSentenceForWordPosition(wordPos, senStarts):
    for i in range(1, len(senStarts)):
        if (wordPos < senStarts[i]):
            return i - 1


def tokenIsAnswer(token, sentenceId, answers):
    for i in range(len(answers)):
        if (answers[i]['sentenceId'] == sentenceId):
            if (answers[i]['text'] == token):
                return True
    return False

def extractAnswers(qas, doc):
    answers = []
    senStart = 0
    senId = 0
    for sentence in doc.sents:
        senLen = len(sentence.text)
        for answer in qas:
            answerStart = answer['answers'][0]['answer_start']

            if (answerStart >= senStart and answerStart < (senStart + senLen)):
                answers.append({'sentenceId': senId, 'text': answer['answers'][0]['text']})

        senStart += senLen
        senId += 1
    
    return answers


def addWordsForParagrapghs(newWords, titleId, paragraphId, df):
    text = df['data'][titleId]['paragraphs'][paragraphId]['context']
    qas = df['data'][titleId]['paragraphs'][paragraphId]['qas']
    doc = nlp(text)
    answers = extractAnswers(qas, doc)
    neStarts = getNEStartIndexs(doc)
    senStarts = getSentenceStartIndexes(doc)
    i = 0
    while(i < len(doc)):
        if(i in neStarts):
            word = neStarts[i]
            currentSentence = getSentenceForWordPosition(word.start, senStarts)
            ta = tokenIsAnswer(word.text, currentSentence, answers)
            wordLen = word.end - word.start
            shape = ''
            for wordIdx in range(word.start, word.end):
                shape += (' ' + doc[wordIdx].shape_)
            newWords.append([word.text, ta, titleId, paragraphId, currentSentence, wordLen, word.label_, None, None, None, shape])
            i = neStarts[i].end-1
        else:
            if(doc[i].is_stop == False and doc[i].is_alpha == True):
                word = doc[i]
                currentSentence = getSentenceForWordPosition(i, senStarts)
                ta = tokenIsAnswer(word.text, currentSentence, answers)
                wordLen = 1
                newWords.append([word.text, ta, titleId, paragraphId, currentSentence, wordLen, None, word.pos_, word.tag_, word.dep_, word.shape_])
                i += 1
