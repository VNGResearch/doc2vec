from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from collections import namedtuple
import gensim
import glob
import random
import re
import os

#from config import config

import pdb

Document = namedtuple('Document', 'url topic doc_no words tags')

class Step(object):
    def fit(self, X, Y, *args, **kwargs):
        warnings.warn('!!!call to DefaultStep.fit from ' + self.__class__.__name__)
        return self

    def transform(self, X, *args, **kwargs):
        warnings.warn('!!!call to DefaultStep.transform from ' + self.__class.__name__)
        return X

    def predict(self, X):
        warnings.warn('!!!call to DefaultStep.predict from ' + self.__class__.__name__)
        #return selfi

class Doc2Vec(Step):
    
    def __init__(self, dm=1, size=100, window=8, min_count=5, passes=1, batch_size=0, shuffle=False):
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

    def fit(self, X, *args, **kwargs):
        '''Train doc2vec model.

        Args:
            X: Document iterator.
        '''
        model = self.load()
        if model is not None:
            self.model = model
            return

        #TODO: Logging
        print('build vocabs')
        self.model.build_vocab(X)
        #TODO should save the model as building vocabs take time

        #mutiple passes, mini-batch training
        for rep in range(0, self.passes):
            print('==========pass num{}'.format(rep))
            corpus = []
            for doc in X:
                corpus.append(doc)
                if self.batch_size > 0 and len(corpus)>=self.batch_size:
                    if self.shuffle:
                        random.shuffle(corpus)
                    self.model.train(corpus)
                    print('mini-batch training')
            if len(corpus) !=0:
                print('train the last')
                self.model.train(corpus) 
        pdb.set_trace()
        self.model.save(self.auto_name())
        return self

    def auto_name(self):
        return re.sub('[\W_]+', '.', str(self.model)) + 'model'

    def transform(self, X, *args, **kwargs):
        print('call doc2vec transform')
        X = [4,5,6]
        return X

    def predict(self, X):
        print('call doc2vec predict')
        #pdb.set_trace()
        

class PredictCategory(Step):
    def __init__(self):
        pass

    def fit(self, X, Y, *args, **kwargs):
        print('call predict fit')

    def transform(self, X, *args, **kwargs):
        print('call predict transform')

    def predict(self, X):
        print('call predict predict')


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
            #print('{}\t{}'.format(topic_id, os.path.basename(filename)))
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
                yield Document(parts[0], topic_id, doc_count, words, [doc_id])
    #return corpus

def make_pipeline():
    #read data
    #train modeli with different params
    #read valid data
    #
    chain = [('doc2vec', Doc2Vec()),
            ('predict', PredictCategory()),
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
model_prefix = 'char_'


def main():
    pipeline = make_pipeline()
    train_docs = read_corpus(data_dir, 0, train_percent)
    pipeline.fit(train_docs)

    print('---call predict')
    pipeline.predict([1,2,3])
    print('DONE')
    pdb.set_trace()

if __name__=='__main__':
    main()
