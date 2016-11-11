from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple, defaultdict
import random, numpy as np
import os, glob
import gensim
import re

#from config import config

import pdb

Document = namedtuple('Document', 'url topic_id doc_no words tags topic')

class Log(object):
    @staticmethod
    def info(sender, message):
        print('---INFO: ', sender, message, sep='\t')

class Step(BaseEstimator):
    def __init__(self):
        super(Step, self).__init__()

    def fit(self, X, Y, *args, **kwargs):
        warnings.warn('!!!call to DefaultStep.fit from ' + self.__class__.__name__)
        return self

    def transform(self, X, *args, **kwargs):
        warnings.warn('!!!call to DefaultStep.transform from ' + self.__class__.__name__)
        return X

    def predict(self, X):
        warnings.warn('!!!call to DefaultStep.predict from ' + self.__class__.__name__)
        #return selfi

class Doc2Vec(Step):
    
    def __init__(self, dm=1, size=100, window=8, min_count=5, passes=1, batch_size=0, shuffle=False):
        super(Doc2Vec, self).__init__()

        self.dm = dm
        self.size = size
        self.window = window
        self.min_count = min_count

        self.passes = passes
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        self.model = gensim.models.Doc2Vec(size = self.size, window = self.window, min_count = min_count, workers = 4)

    def load(self):
        model_file = self.auto_name()
        if os.path.isfile(model_file):
            warnings.warn('Load trained model from: ' + model_file)
            return gensim.models.doc2vec.Doc2Vec.load(model_file)

    def plot_with_color(self, low_dim_embs, labels, classes, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        Log.info(self, 'Start drawing...')
        #plt.figure(figsize=(18, 18))  #in inches
        unique_classes = list(set(classes))
        colors = cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i,:]
            plt.scatter(x, y, color=colors[unique_classes.index(classes[i])])
            plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        #plt.savefig(filename + '.png')
        Log.info(self, '######showing...')
        plt.show()

    def visualize(self, docvecs, docs, filename, plot_only=None):
        Log.info(self, 'Start reduce docvec dimention')
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        if plot_only is not None:
            ids  = random.sample(range(0, len(docvecs)), plot_only)
            docvecs = docvecs[ids]
            docs = [docs[i] for i in ids]
        try:
            low_dim_embs = tsne.fit_transform(docvecs)
        except:
            print('!!!Cant reduce the docvecs dementions:', sys.exc_info())
            return

        #labels = [str(doc.topic) + '.' + str(doc.doc_no) for doc in docs]
        #labels = [str(doc.topic) + '.' + str(doc.tags[0]) for doc in docs]
        labels = [str(doc.topic_id) for doc in docs]
        classes = [doc.topic_id for doc in docs]

        #plot_with_labels(low_dim_embs, labels) 
        self.plot_with_color(low_dim_embs, labels, classes, filename)

    def fit(self, X, *args, **kwargs):
        '''Train doc2vec model.

        Args:
            X: Document iterator.
        '''
        print('docvec.fit')
        #pdb.set_trace()
        model = self.load()
        if model is not None:
            self.model = model
            return self

        Log.info(self, '--build vocabs')
        self.model.build_vocab(X)

        #mutiple passes, mini-batch training
        for rep in range(0, self.passes):
            Log.info(self, '==========pass num{}'.format(rep))
            corpus = []
            for doc in X:
                corpus.append(doc)
                if self.batch_size > 0 and len(corpus)>=self.batch_size:
                    if self.shuffle:
                        random.shuffle(corpus)
                    self.model.train(corpus)
                    Log.info(self, '---train a mini-batch')
            if len(corpus) !=0:
                Log.info(self, '---train the last')
                self.model.train(corpus) 
        self.model.save(self.auto_name())

        self.visualize(self.model.docvecs, corpus, 'new_code',500) 
        pdb.set_trace() 
        return self

    def auto_name(self):
        return os.path.join(model_dir, re.sub('[\W_]+', '.', str(self.model)).lower() + 'model')

    def transform(self, X, *args, **kwargs):
        #X = np.zeros(shape=(num, self._vector_size))
        #Y = np.zeros(shape=(num,))
        print('docvec.transform')
        #pdb.set_trace()
        Log.info(self, 'Get vectors for documents...')
        data = []
        target = []
        infos = []
        for doc in X:
            #data.append(self.model.infer_vector(doc.words))
            #TODO: must change back
            data.append(self.model.docvecs[doc.doc_no])
            target.append(doc.topic_id)
            infos.append((doc.doc_no, doc.tags, doc.topic_id, doc.topic, doc.url))
        #pdb.set_trace()
        return np.array(data), target, infos, self.model

    def predict(self, X):
        '''Never been used.'''
        return self.transofrm(X)


class KNN(Step):
    def __init__(self, k=10):
        self.K = k
        self._correct = -1

    def fit(self, X, Y, *args, **kwargs):
        print('knn.fit')
        self.get_data(X)

    def get_data(self, X):
        self.data = X[0]
        self.target = X[1]
        self.infos = X[2]
        self.model = X[3]

    def classify_by_count(self, sim_docs):
        count = defaultdict(int)
        ret_cat = -1
        max_count = -1
        for doc in sim_docs:
            cat_id = self.infos[doc[0]][2]
            count[cat_id] +=1
            if count[cat_id]>max_count:
                ret_cat = cat_id
                max_count = count[cat_id]
        return ret_cat

    def predict(self, X):
        data, target, infos, model = X 
        print('call predict predict')
        self._correct = 0
        wrong_list = []
        for i, doc_vec in enumerate(data):
            sim_docs = self.model.docvecs.most_similar([doc_vec], topn=self.K)
            cat_id = self.classify_by_count(sim_docs)
            if cat_id == infos[i][2]:
                self._correct +=1
            else:
                wrong_list.append(i)
                #pdb.set_trace()
        self._score = self._correct*100 / float(len(infos))
        print('****score= {}%'.format(self._score))
        pdb.set_trace()

    def score(self):
        return self._score
       
#template class for build a fitable classification
class PredictCategory(Step):
    def __init__(self):
        pass

    def fit(self, X, Y, *args, **kwargs):
        print('call predict fit')
        pdb.set_trace()

    def transform(self, X, *args, **kwargs):
        print('call predict transform')
        pdb.set_trace()

    def predict(self, X):
        print('call predict predict')
        pdb.set_trace()


#return multiple generator
def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

@multigen
def read_corpus(data_dir, from_percent, to_percent):
    #corpus = []
    topic_id = -1
    doc_id = -1
    doc_count = -1
    for filename in glob.iglob(data_dir + '*.tsv'):
        topic_id += 1
        doc_count = -1
        with open(filename) as f:
            Log.info('read_corpus', '{}\t{}'.format(topic_id, os.path.basename(filename)))
            #TODO seek to the right part for reading
            docs = f.readlines()
            doc_len = float(len(docs))
            for doc in docs:
                doc_count +=1
                percent = (doc_count+1)/doc_len
                if percent>=to_percent:
                    break
                if percent<from_percent:
                    continue

                doc_id += 1
                parts = doc.split('\t')
                words = ' '.join(part.strip() for part in parts[1:])
                #words = gensim.utils.to_unicode(words).split()
                #pdb.set_trace()
                #corpus.append(Document(parts[0], topic_id, doc_count, words, [doc_id]))
                yield Document(parts[0], topic_id, doc_count, words, [doc_id], os.path.basename(filename))
    #return corpus

def make_pipeline():
    #read data
    #train modeli with different params
    #read valid data
    #
    chain = [('doc2vec', Doc2Vec()),
            ('predict', KNN()),
        ]

    pline = Pipeline(chain)
    return pline

def search_params():
    pass 

#global configuration
data_dir = '../crawl_news/data/zing/'
train_percent = 0.6
valid_percent = 0.2
test_percent = 0.2
token_type = 'char'#word, vn_token
model_dir = './models/'


def main():
    pipeline = make_pipeline()
    train_docs = read_corpus(data_dir, 0, train_percent)
    #pdb.set_trace()
    #pipeline.set_params(doc2vec__computed=True)#need need to be in the init method
    pipeline.fit(train_docs)#only pass to the fit method

    Log.info('main', '============call predict')
    #TODO read validation test
    pipeline.predict(train_docs)
    print('DONE')
    pdb.set_trace()

if __name__=='__main__':
    main()
