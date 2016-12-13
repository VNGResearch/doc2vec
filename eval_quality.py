import os

import gensim
import pickle

import numpy as np
from collections import defaultdict
import random


from web_run.measures import Similarity

#load the best model for severing
model_path = './web_run/models/pass0/'
model_file = os.path.join(model_path, 'BEST.doc2vec.model')
classifier_file = os.path.join(model_path, 'BEST.class.model')
scaler_file = os.path.join(model_path, 'BEST.scaler.model')
doc2vec = gensim.models.doc2vec.Doc2Vec.load(model_file)
doc_infos = None
with open(model_file + '.info', 'rb') as f:
    doc_infos = pickle.load(f)

#preparation
cat_docs = defaultdict(list)
doc2cat = {}
for doc_id in doc_infos:
    info = doc_infos[doc_id]
    tag = info[1][0]
    cat = info[2]
    cat_docs[cat].append(tag)
    doc2cat[tag] = cat 

#evaluate
N = 100000
max_cat = 59;
acc = 0
for i in range(N):
    cat1, cat2 = random.sample(range(0, max_cat), 2)
    doc1, doc2 = random.sample(cat_docs[cat1], 2)
    doc3= random.sample(cat_docs[cat2], 1)[0]
    vec1 = doc2vec.docvecs[doc1]
    vec2 = doc2vec.docvecs[doc2]
    vec3 = doc2vec.docvecs[doc3]
    cos12 = Similarity.cosine_similarity(vec1, vec2)
    cos13 = Similarity.cosine_similarity(vec1, vec3)
    cos23 = Similarity.cosine_similarity(vec2, vec3)

    acc += 1 if cos12>cos13 and cos12>cos23 else 0
    if i%5000==0:
        print(i)

print('Accuracy {}'.format(acc/float(N)))
