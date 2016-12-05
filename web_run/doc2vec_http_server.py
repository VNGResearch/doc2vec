'''
A HTTP Server for the simple Doc2Vec
'''

'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
'''

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import jsonlib
from datetime import datetime

import gensim
import pickle

import numpy

import pdb

#load the best model for severing
model_file = 'BEST.doc2vec.model'
classifier_file = 'BEST.class.model'
scaler_file = 'BEST.scaler.model'
doc2vec = gensim.models.doc2vec.Doc2Vec.load(model_file)
doc_infos = None
with open(model_file + '.info', 'rb') as f:
    doc_infos = pickle.load(f)
with open(classifier_file, 'rb') as f:
    classifier = pickle.load(f)
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

id2cat= {}
with open('zing_topics.txt', 'r') as f:
    for line in f:
        line = line.strip()
        parts = line.split()
        id2cat[int(parts[0])] = parts[1][0:parts[1].rindex('.')]

def preprocess(content):
   words = gensim.utils.to_unicode(content).split()
   return words

class StatefulHandler(SimpleHTTPRequestHandler):
    '''
    def __init__(self, *args, **kwargs):
        super(StatefulHandler, self).__init__(*args, **kwargs)
        self.load_doc2vec() 
        pdb.set_trace()
    '''
    
    def do_POST(self):
        print('---Have a request from {} at {}'.format(self.address_string(), self.date_time_string()))
        output_text_inner = self.do_POST_inner()
        #add http header
        output_text = 'HTTP/1.0 200 OK\nContent-Type: application/json\n\n' + output_text_inner + '\n'
        
        output_text = output_text.encode('utf8')
        #serve result
        self.wfile.write(output_text)
        #print('---Handled request:\n%s'%(output_text))
        print('------FINISH the request from {} at {}'.format(self.address_string(), self.date_time_string()))
        
    def do_POST_inner(self):
        path = self.path[1:]
        content = self.rfile.readline()
        
        message = jsonlib.read(content,use_float = True)
        
        ret = ''
        if path == 'dm':
            ret = self._dm_process(message)
        else:
            pass
        
        return ret

    def _dm_process(self, message):
        content = message['content']
        tokens = preprocess(content)

        docvec = doc2vec.infer_vector(tokens)
       	docvec_scaled = docvec.reshape(1, -1)
        docvec_scaled = scaler.transform(docvec_scaled)

        #cats = classifier.predict(docvec_scaled)[0]
        cats = classifier.predict_proba(docvec_scaled)[0]
        cat_rank = np.argsort(cats)[::-1][:5]
        pdb.set_trace()

        related = doc2vec.docvecs.most_similar(positive=[docvec], topn=10)
        rels = []
        for rel in related:
            rels.append(doc_infos[rel[0]][4])

        ret = {'category': str(cats), 'related': rels, 'docvec': docvec.tolist()}
        return jsonlib.write(ret).decode('utf8')
        #return jsonlib.write(ret)
    
def main():
    #uuid = 'D24CB19B8EAF11E4ACAFC1AA6AEC2530'
    port = 8080
    
    server_address=('',port)
    httpd = HTTPServer(server_address, StatefulHandler)
    print('Ready for serving...')
    httpd.serve_forever()

if (__name__ == '__main__'):
    main()
