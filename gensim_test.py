from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import os, time, sys
from timeit import default_timer
import warnings

from random import shuffle
from collections import namedtuple, OrderedDict
import numpy as np
import re
import multiprocessing

import wikipedia as wiki
#from gensim.models import doc2vec
import gensim
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import pdb

data_file = './data/wiki.txt'
Document = namedtuple('Document', 'topic doc_no words tags')

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace xxx with spaces
    norm_text = norm_text.replace('<br />', ' ')
    norm_text = norm_text.replace('\n', ' ')
    norm_text = norm_text.replace('==', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':', '\'s']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    norm_text = re.sub(' +', ' ', norm_text)    
    return norm_text

def crawl_wiki():
    if os.path.isfile(data_file):
        #print('Using data from', data_file)	
        warnings.warn('Using data from: ' + data_file)	
        return
    f_triplets = './corpora/wikipedia/wikipedia-hand-triplets-release.txt'
    topic = ''
    topic_count = -1
    page_count = -1
    titles = []
    with open(data_file, 'w') as fw:
        with open(f_triplets) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    topic = line[2:]
                    topic_count += 1
                    page_count = -1
                    continue
                #if topic_count>2:
                    #break
                urls = line.split()[0:2]#the last url is not in the topic; not correct
                for url in urls:
                    title = url[url.rindex('/')+1:].replace('_', ' ').lower()
                    if title in titles:
                        print('---repeat: ' + url)    
                        continue
                    print('...get url: ' + url)
                    page_count +=1
                    titles.append(title)
                    try:
                        page = wiki.page(title)
                        text = page.content
                    except:
                        print('xxxCant read from wiki')
                        continue
                    text = normalize_text(text)
                    fw.write('%d\t%d\t%s\n'%(topic_count, page_count, text))

def read_data():
    docs = []
    with open(data_file) as f:
        for line_no, line in enumerate(f):
            topic, no, doc = line.split('\t')
            words = gensim.utils.to_unicode(doc).split()
            #tags = [int(topic + '' + no)]
            tags = [line_no]
            docs.append(Document(int(topic), int(no), words, tags))
    return docs 

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)
    plt.show()

def plot_with_color(low_dim_embs, labels, classes, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    #plt.figure(figsize=(18, 18))  #in inches
    colors = cm.rainbow(np.linspace(0, 1, len(set(classes))))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y, color=colors[classes[i]])
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    #plt.savefig(filename + '.png')
    plt.show()

def visualize(docvecs, docs, filename):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #pdb.set_trace()
    try:
        low_dim_embs = tsne.fit_transform(docvecs)
    except:
        print('!!!Cant reduce the docvecs dementions:', sys.exc_info())
        return

    labels = [str(doc.topic) + '.' + str(doc.doc_no) for doc in docs]
    classes = [doc.topic for doc in docs]
    
    #plot_with_labels(low_dim_embs, labels) 
    plot_with_color(low_dim_embs, labels, classes, filename)

@contextmanager
def time_it(msg_in, msg_out):
    #print(msg_in)
    sys.stderr.write(msg_in + '\n')
    start = time.time()
    yield
    duration = time.time() - start
    #print(msg_out % duration)
    sys.stderr.write(msg_out % duration + '\n')

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def train_models(docs, passes):
    cores = multiprocessing.cpu_count()
    models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        gensim.models.Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        gensim.models.Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/average
        gensim.models.Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
        #PV-DM, concatinate
        gensim.models.Doc2Vec(size=100, window=8, min_count=5, workers=4),
    ]

    print('Initilizing all models, build vocabs...')
    '''
    models[0].build_vocab(docs)
    for model in models[1:]:
        model.reset_from(models[0])
    '''
    for model in models:
        print('---', model)
        model.build_vocab(docs)
    
    #models_by_name = OrderedDict((str(model), model) for model in models)
    models_by_name = OrderedDict()
    models_by_name['default'] = models[3]
    models_by_name['dmc'] = models[0]
    models_by_name['dbow'] = models[1]
    models_by_name['dmm'] = models[2]
    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([models[1], models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([models[1], models[0]])

    #learning rate decreaseing
    alpha, min_alpha= (0.025, 0.001)
    alpha_delta = (alpha - min_alpha) / passes

    for epoch in range(passes):
        print('====================pass', epoch)
        #shuffle(docs)
        for name, model in models_by_name.items():

            model.alpha, model.min_alpha = alpha, alpha#set learning rate

            print('---training model', name, 'at alpha', alpha)
            with elapsed_timer() as elapsed:
                model.train(docs)
                print('------finished after %0.1fs'%elapsed())
            if epoch%3==0:
                visualize(model.docvecs, docs, name + '_' + str(epoch))
    
        alpha -= alpha_delta

def debug(docs):
    model = gensim.models.Doc2Vec(docs, size=100, window=8, min_count=5, workers=4)
    shuffle(docs)
    visualize(model.docvecs, docs, 'dm_with_docs')
    return
    for i in range (10):
        #shuffle(docs)
        model.train(docs)
    visualize(model.docvecs, docs, 'dm_with_docs' + str(i))

    
def main():
    crawl_wiki()
    #read_data()
    docs = read_data()

    #model = doc2vec.Doc2Vec(docs, size=100, window=8, min_count=5, workers=4)
    #print('---default model')
    #model = gensim.models.Doc2Vec(docs, size=100, window=8, min_count=5, workers=4)
    #visualize(model.docvecs, docs, 'dm_with_docs')

    debug(docs)
    #train_models(docs, 10)
    print('DONE')

if __name__=='__main__':
    main()
