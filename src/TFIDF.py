import json
from gensim import corpora, models, similarities

factionkb   =   open("../data/actionkbv3.json")
fmapping    =   open("../data/mappingv3.json")


actionkb    =   json.load(factionkb)
mapping     =   json.load(fmapping)