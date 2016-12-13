import os
import gensim
import pickle
import numpy as np
from collections import defaultdict
import random
from sklearn.cluster import KMeans
from sklearn import metrics

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
lst_doc = []
labels = []
for doc_id in doc_infos:
    info = doc_infos[doc_id]
    tag = info[1][0]
    cat = info[2]
    cat_docs[cat].append(tag)
    doc2cat[tag] = cat
    lst_doc.append(doc2vec.docvecs[tag])
    labels.append(cat)

#fit kmeans
print('Clustering...')
X = np.array(lst_doc)
labels = np.array(labels)
cat_num = 59
kmeans = KMeans(n_clusters=cat_num, verbose=0)
kmeans.fit(X)

#score
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmeans.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, kmeans.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmeans.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, kmeans.labels_))
print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, kmeans.labels_, sample_size=1000))
