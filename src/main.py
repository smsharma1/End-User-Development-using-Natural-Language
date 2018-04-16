import random
import json
import string
from gensim.models import doc2vec
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
import re
from nltk import pos_tag
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def extract_phrases(my_tree, phrase):
    my_phrases = []
    if my_tree.label() == phrase:
        my_phrases.append(my_tree.copy(True))
    for child in my_tree:
        if type(child) is nltk.Tree:
            list_of_phrases = extract_phrases(child, phrase)
            if len(list_of_phrases) > 0:
                my_phrases.extend(list_of_phrases)
    return my_phrases

grammar = "NP: {<VB>(<[(CC)(CD)(DT)(EX)(FW)(IN)(JJ)(JJR)(JJS)(LS)(MD)(NN)(NNS)(NNP)(NNPS)(PDT)(POS)(PRP)(PRP$)(RB)(RBR)(RBS)(RP)(SYM)(TO)(UH)(VBD)(VBG)(VBN)(VBP)(VBZ)(WDT)(WP)(WP$)(WRB)]*>)*?(?=<VB>)}"
cp = nltk.RegexpParser(grammar)

verbs = ['Collect','Get','Call','call','Show','List','check','turn','Turn','open','create','Create','submit','include','Look','Capture','find','Open','Include','Enter','Catch','Print','translate','Reply','reply','set','Add','Set','Register','Schedule','Post','Upload','Publish','Message','forward','share','Share','Write','publish','Contact','Find','Search','write','Send','send','save','pay','Pay','receive','Wait','Check','seek','Translate','Save','Extract','Convert']

def get_phrases(sentence):
    sentence = re.sub('['+string.punctuation+']', ' ', sentence)
    global verbs,cp
    sentence = pos_tag(word_tokenize(sentence))
    j = 0
    for sent in sentence:
        if sent[0] in verbs:
            sentence[j] = (sent[0],'VB')
        j+=1
    tree = cp.parse(sentence)
    list_of_noun_phrases = extract_phrases(tree, 'NP')
    ph = []
    nl_phrase = []
    for phrase in list_of_noun_phrases:
        ph = " ".join([x[0] for x in phrase.leaves()])
        ph = word_tokenize(ph)
        ph = " ".join([w for w in ph if w not in stop_words]).lower()
        nl_phrase.append(ph)
    return nl_phrase

translator = str.maketrans('', '', string.punctuation + "\n\t")

factionkb   =   open("../data/actionkbv3.json")
fmapping    =   open("../data/mappingv3_discourse_tagger.json")

actionkb    =   json.load(factionkb)
mapping     =   json.load(fmapping)

stop_words = set(stopwords.words('english'))
stop_words.add('I')

actionvectors = {}
for action in actionkb:
    actionvectors[action['id']] = {}
    actionvectors[action['id']]['action'] = []
    try:
        if action['desc'] == None:
            action['desc'] = 'AAAAA'
    except:
        action['desc'] = 'AAAAAAAA'
    try:
        if action['provider'] == None:
            action['provider'] = 'AAAAAAAAAA'
    except:
        action['provider'] = 'AAAAAAAAA'
    actionvectors[action['id']]['actionname'] = action['name']
    actionname = action['name'] + " " + action["desc"] + " " + action["provider"]
    actionname = re.sub('['+string.punctuation+']', ' ', actionname)
    actionname = actionname.replace("e-mail","email")
    actionname = actionname.replace("-"," ").replace("_"," ")
    actionname = word_tokenize(actionname)
    actionname = " ".join([w for w in actionname if w not in stop_words]).lower()
    actionvectors[action['id']]['action'].append(actionname)

nlvectors = {}
for mapp in mapping[0:1]:
    nlvectors[mapp['id']] = {}
    nlvectors[mapp['id']]['actions'] = []
    for action in mapp['action_instances'] :
        try:
            action['options']
            for ac in action['options']:
                nlvectors[mapp['id']]['actions'].append(ac['id'])
        except:
            pass
        try:
            nlvectors[mapp['id']]['actions'].append(action['id'])
        except:
            pass
    nlvectors[mapp['id']]['phrases'] = mapp['nl_phrases_sw_remove']

actionindexmap = {}
tfidfinputdocs = []
i = 0
for key in actionvectors.keys():
    actionvectors[key]['tfidfvectorsindex'] = len(tfidfinputdocs)
    actionindexmap[len(tfidfinputdocs)] = key
    tfidfinputdocs.append(actionvectors[key]['action'][0])
    i+=1

actionoffset = len(tfidfinputdocs)
actionoffset1 = actionoffset

for key in nlvectors.keys():
    nlvectors[key]['tfidfvectoroffset'] = actionoffset1
    for phrase in nlvectors[key]['phrases']:
        tfidfinputdocs.append(phrase)
        actionoffset1+=1

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(tfidfinputdocs)

i = actionoffset
for key in nlvectors.keys():
    index = []
    for action in nlvectors[key]['actions']:
        index.append(actionvectors[action]['tfidfvectorsindex'])
    if (i != nlvectors[key]['tfidfvectoroffset']):
        print("Code is wrong")
    for phrase in nlvectors[key]['phrases']:
        cosine_similarities = linear_kernel(tfidf[i], tfidf[index] ).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        actionvectors[actionindexmap[index[related_docs_indices[0]]]]['action'] = [actionvectors[actionindexmap[index[related_docs_indices[0]]]]['action'][0] + " " + phrase]
        i+=1

del tfidfinputdocs, actionindexmap, nlvectors


fvalcommands    =   open("../data/val_commands_discourse_tagger.json")
valcommands     =   json.load(fvalcommands)
fvalcommands.close()

tfidfinputdocs = []
actionindexmap = {}
i = 0
for key in actionvectors.keys():
    actionvectors[key]['tfidfvectorsindex'] = len(tfidfinputdocs)
    actionindexmap[len(tfidfinputdocs)] = key
    tfidfinputdocs.append(actionvectors[key]['action'][0])
    i+=1
actionoffset = len(tfidfinputdocs)
actionoffset1 = actionoffset

for command in valcommands:
    phrases = command['nl_phrases_sw_remove']
    for phrase in phrases:
        tfidfinputdocs.append(phrase)

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(tfidfinputdocs)


commandactionmap = {}
j=0
for command in valcommands:
    phrases = command['nl_phrases_sw_remove']
    commandactionmap[command['id']] = []
    for phrase in phrases:
        cosine_similarities = linear_kernel(tfidf[actionoffset1], tfidf[:(actionoffset-1)]).flatten()
        related_docs_indices = cosine_similarities.argsort()[-1:-5:-1]
        valcommands[j]['action_instances'].append(actionvectors[actionindexmap[related_docs_indices[0]]]['actionname'])
        commandactionmap[command['id']].append(actionindexmap[related_docs_indices[0]])
        # print(actionvectors[actionindexmap[related_docs_indices[0]]]['actionname'])
        actionoffset1+=1
    j+=1

fvalcommands    =   open("../data/val_commands_discourse_tagger.json","w+")
json.dump(valcommands, fvalcommands, indent=4, sort_keys=True)
fvalcommands.close()

fvalmapping     =   open("../data/val_mappingv3.json")   
val_mapping     =   json.load(fvalmapping)

commandactionmapgiven = {}
for actioninstances in val_mapping:
    commandactionmapgiven[actioninstances['id']]=[]
    # print(actioninstances)
    for action in actioninstances['action_instances']:
        try:
            commandactionmapgiven[actioninstances['id']].append(action['id'])
        except:
            pass
        try:
            commandactionmapgiven[actioninstances['id']].append(action['condition']['id'])
        except:
            pass
        try:
            for op in action['options']:
                commandactionmapgiven[actioninstances['id']].append(op['id'])
        except:
            pass
        try:
            for cons in action['consequent']:
                try:
                    commandactionmapgiven[actioninstances['id']].append(cons['id'])
                except:
                    pass
                try:
                    for ops in cons['options']:
                        commandactionmapgiven[actioninstances['id']].append(ops['id'])
                except:
                    pass
        except:
            pass

pos = 0
neg = 0
for key in commandactionmapgiven.keys():
    for act in commandactionmap[key]:
        if act in commandactionmapgiven[key]:
            pos+=1
        else:
            neg+=1 

print("The l1 accuracy is ", neg/(pos+neg))

pos = 0
neg = 0
for key in commandactionmapgiven.keys():
    for act in commandactionmapgiven[key]:
        if act in commandactionmap[key]:
            pos+=1
        else:
            neg+=1

print("The l2 accuracy is ",pos/(pos + neg))
exit()
while(1):
    tfidfinputdocs = []
    actionindexmap = {}
    phraseindexmap = {}
    i = 0
    for key in actionvectors.keys():
        actionindexmap[i] = key
        tfidfinputdocs.append(actionvectors[key]['action'][0])
        i+=1
    nlcommand = input("Please enter the new NL command\n")
    nlcommand = nlcommand + " call my name"
    phrases = get_phrases(nlcommand)
    tfidfvectoroffset = i
    for phrase in phrases:
        tfidfinputdocs.append(phrase)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(tfidfinputdocs)
    tfidfvectoroffset1 = tfidfvectoroffset
    for phrase in phrases:
        cosine_similarities = linear_kernel(tfidf[tfidfvectoroffset1], tfidf[:(tfidfvectoroffset-1)]).flatten()
        print("Choose action for phrase: " + phrase)
        related_docs_indices = cosine_similarities.argsort()[-1:-5:-1]
        t = 1
        localindexmap = {}
        for ind in related_docs_indices:
            localindexmap[t] = actionindexmap[ind]
            print(str(t) + ") " + actionvectors[actionindexmap[ind]]['actionname'])
            t+=1
        inputindex = int(input())
        actionvectors[localindexmap[inputindex]]['action'] = [actionvectors[localindexmap[inputindex]]['action'][0] + phrase]
        tfidfvectoroffset1 += 1

