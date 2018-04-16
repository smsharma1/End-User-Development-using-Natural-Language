from __future__ import unicode_literals
import json
import string
from gensim import corpora, models, similarities
import nltk
from nltk import pos_tag
from nltk import tokenize
import re
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

stop_words = set(stopwords.words('english'))
stop_words.add('I')
print(stop_words)
grammar = "NP: {<VB>(<[(CC)(CD)(DT)(EX)(FW)(IN)(JJ)(JJR)(JJS)(LS)(MD)(NN)(NNS)(NNP)(NNPS)(PDT)(POS)(PRP)(PRP$)(RB)(RBR)(RBS)(RP)(SYM)(TO)(UH)(VBD)(VBG)(VBN)(VBP)(VBZ)(WDT)(WP)(WP$)(WRB)]*>)*?(?=<VB>)}"
cp = nltk.RegexpParser(grammar)

fmapping    =   open("../data/val_commands.json")
mapping     =   json.load(fmapping)
sentences = []
for mapp in mapping:
    sentences.append(mapp['nl_command_statment'])

i = 0
for x in sentences:
    x = re.sub(r'(\")([^\"]*)(\")', r'', x);
    translator = str.maketrans('', '', string.punctuation + "\n\t")
    x = x.translate(translator) + " call my name"
    mapping[i]["nl_command_statment_clean"] = x
    sentence = pos_tag(tokenize.word_tokenize(x))
    verbs = ['Finish','suspend','Pause','End','Collect','Get','Call','call','Show','List','check','turn','Turn','open','create','Create','submit','include','Look','Capture','find','Open','Include','Enter','Catch','Print','translate','Reply','reply','set','Add','Set','Register','Schedule','Post','Upload','Publish','Message','forward','share','Share','Write','publish','Contact','Find','Search','write','Send','send','save','pay','Pay','receive','Wait','Check','seek','Translate','Save','Extract','Convert']
    j = 0
    for sent in sentence:
        if sent[0] in verbs:
            sentence[j] = (sent[0],'VB')
        j+=1
    tree = cp.parse(sentence)
    list_of_noun_phrases = extract_phrases(tree, 'NP')
    ph = []
    nl_phrase = []
    nl_phrase_sw_remove = []
    for phrase in list_of_noun_phrases:
        ph = " ".join([x[0] for x in phrase.leaves()])
        nl_phrase.append(ph)
        ph = word_tokenize(ph)
        ph = " ".join([w for w in ph if w not in stop_words]).lower()
        nl_phrase_sw_remove.append(ph)
    mapping[i]['nl_phrases_sw_remove'] = nl_phrase_sw_remove
    mapping[i]['nl_phrases'] = nl_phrase
    i+=1
    
fmapping_updated    =   open("../data/val_commands_discourse_tagger.json","w+")
json.dump(mapping,fmapping_updated,indent=4, sort_keys=True)

# import json
# import nltk
# from nltk import word_tokenize, ne_chunk

# fmapping    =   open("../data/mappingv3.json")
# mapping     =   json.load(fmapping)
# fmapping.close()

# print(len(mapping))

# for i in range(len(mapping)):
#     text = word_tokenize(mapping[i]['nl_command_statment'])
#     postext = nltk.pos_tag(text)
#     mapping[i]['nl_command_statment_postags'] = postext

# fmappingprocessed   =   open("../data/mappingv3processed.json","w+")
# json.dump(mapping,fmappingprocessed,indent=4, sort_keys=True)
# fmappingprocessed.close()



# import json
# from gensim import corpora, models, similarities
# import nltk
# from nltk import pos_tag
# from nltk import tokenize

# def extract_phrases(my_tree, phrase):
#     my_phrases = []
#     if my_tree.label() == phrase:
#         my_phrases.append(my_tree.copy(True))
#     for child in my_tree:
#         if type(child) is nltk.Tree:
#             list_of_phrases = extract_phrases(child, phrase)
#             if len(list_of_phrases) > 0:
#                 my_phrases.extend(list_of_phrases)
#     return my_phrases

# grammar = "NP: {(<VB>|<VBP>)+(<IN>|<PRP>|<TO>|<DT>|<JJ>|<NN>|<IN>|<VBZ>)*(<NN>|<NNP>)+}"
# cp = nltk.RegexpParser(grammar)

# print(related_docs_indices)

# # print(docs[related_docs_indices])
# fmapping    =   open("../data/mappingv3.json")
# mapping     =   json.load(fmapping)
# sentences = []
# for mapp in mapping:
#     sentences.append(mapp['nl_command_statment'])
# i = 0
# for x in sentences:
#     sentence = pos_tag(tokenize.word_tokenize(x))
#     # print sentence
#     tree = cp.parse(sentence)
#     print("Noun phrases:")
#     list_of_noun_phrases = extract_phrases(tree, 'NP')
#     print(sentence)
#     ph = []
#     nl_phrase = []
#     for phrase in list_of_noun_phrases:
#         print(phrase, " ".join([x[0] for x in phrase.leaves()]))
#         #doc2.append(" ".join([x[0] for x in phrase.leaves()]))
#         ph = " ".join([x[0] for x in phrase.leaves()])
#         nl_phrase.append(ph)
#     mapping[i]['nl_phrase'] = nl_phrase
#     i+=1

# fmapping_updated    =   open("../data/mappingv3_discourse_tagger.json","w+")
# json.dump(mapping,fmapping_updated,indent=4, sort_keys=True)