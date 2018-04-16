import json
from gensim import corpora, models, similarities

factionkb   =   open("../data/actionkbv3.json")
fmapping    =   open("../data/mappingv3.json")


actionkb    =   json.load(factionkb)
mapping     =   json.load(fmapping)

print(len(actionkb))
exit()
processeddata = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']] 

dictionary = corpora.Dictionary(processeddata)
dictionary.save('processdata.dict')
print("We create a dictionary, an index of all unique values: %s"%type(dictionary))

raw_corpus = [dictionary.doc2bow(t) for t in processeddata]
print("Then convert convert tokenized documents to vectors: %s"% type(raw_corpus))
corpora.MmCorpus.serialize('processdata.mm', raw_corpus)
corpus = corpora.MmCorpus('processdata.mm')

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
index = similarities.MatrixSimilarity(tfidf[corpus])
sims = index[corpus_tfidf]

similarish = {}
for k in range(9):
    similarish[k] = sims.argsort()[k][::-1][1]

