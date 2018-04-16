import random
import json
from gensim.models import doc2vec
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk import pos_tag
from nltk import tokenize


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

factionkb   =   open("../data/actionkbv3.json")

actionkb    =   json.load(factionkb)

doc1 = []
for action in actionkb:
    # print(action)
    try:
        if action['desc'] == None:
            action['desc'] = ''
    except:
        action['desc'] = ''
    try:
        if action['provider'] == None:
            action['provider'] = ''
    except:
        action['provider'] = ''
    doc1.append(action['name'])
    # doc1.append(action['name'] + " " + action['desc'] + " " + action['provider'] )

# alpha_val = 0.025        # Initial learning rate
# min_alpha_val = 1e-4     # Minimum for linear learning rate decay
# passes = 15              # Number of passes of one document during training

# alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

# # doc1 = ["This is a sentence", "This is another sentence"]

# # Transform data (you can add more data preprocessing steps) 

# docs = []
# analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
# for i, text in enumerate(doc1):
#     words = text.lower().split()
#     tags = [i]
#     docs.append(analyzedDocument(words, tags))



# model = doc2vec.Doc2Vec( vector_size = 100 # Model initialization
#     , window = 300
#     , min_count = 1
#     , workers = 4)

# model.build_vocab(docs) # Building vocabulary

# # print(docs)

# for epoch in range(passes):

#     # Shuffling gets better results

#     random.shuffle(docs)

#     # Train

#     model.alpha, model.min_alpha = alpha_val, alpha_val
#     print(model.corpus_count)
#     model.train(docs,total_examples=model.corpus_count,epochs=1)

#     # Logs

#     print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))

#     # Next run alpha

#     alpha_val -= alpha_delta

doc2 = []
# doc3 = []
# for i, text in enumerate(doc2):
#     words = text.lower().split()
#     tags = [i]
#     doc3.append(analyzedDocument(words, tags))

# print(doc3)
# new_vector = model.infer_vector(doc2[0])
# sims = model.docvecs.most_similar([new_vector])
# print(sims)
# tags = []
# for elem in sims:
#     tags.append(elem[0])
# for doc in docs:
#     if doc[1][0] in tags:
#         print(doc[0])
#         print("\n")

grammar = "NP: {(<VB>|<VBP>)+(<IN>|<PRP>|<TO>|<DT>|<JJ>|<NN>|<IN>|<VBZ>)*(<NN>|<NNP>)+}"
cp = nltk.RegexpParser(grammar)


# print(related_docs_indices)

# print(docs[related_docs_indices])
fmapping    =   open("../data/mappingv3.json")
mapping     =   json.load(fmapping)
sentences = []
for mapp in mapping:
    sentences.append(mapp['nl_command_statment'])
for x in sentences:
    sentence = pos_tag(tokenize.word_tokenize(x))
    # print sentence
    tree = cp.parse(sentence)
    print("\nNoun phrases:")
    list_of_noun_phrases = extract_phrases(tree, 'NP')
    print(sentence)
    for phrase in list_of_noun_phrases:
        print(phrase, " ".join([x[0] for x in phrase.leaves()]))
        doc2.append(" ".join([x[0] for x in phrase.leaves()]))

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(doc1 + doc2)

# tfidf = vectorizer.fit_transform(doc1 + doc2)
# print(tfidf)
cosine_similarities = linear_kernel(tfidf[len(doc1) + 1], tfidf[:len(doc1)-1]).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1]
print(doc2[1])
for i in related_docs_indices:
    print(doc1[i])