'''
A HTTP Server for the simple VPA
'''

'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
'''
#from BaseHTTPServer import HTTPServer
#from CGIHTTPServer import CGIHTTPRequestHandler

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import jsonlib
'''
from logging import getLogger

from utils.globalconfig import *
from utils.design_pattern import *
from asr.watson_asr_result import WatsonASRResult
from asr.transcript_asr_result import TranscriptASRResult
from task_classifier.simple_task_classifier import SimpleTaskClassifier
from nlg.simple_nlg import SimpleNLG
'''

import pdb

#class StatefulHandler(CGIHTTPRequestHandler):
#class StatefulHandler(BaseHTTPRequestHandler):
class StatefulHandler(SimpleHTTPRequestHandler):
    MY_ID = 'HTTPServer'

##    def __init__(self, request, client_address, server):#visit every single request
##        #super.__init__(self, p1, p2, p3)
##        #super(StatefulHandler, self).__init__(p1, p2, p3)
##        CGIHTTPRequestHandler.__init__(self,request,client_address,server)
##        self.config = get_config()
##        self.logger = getLogger(self.MY_ID)
##        global dms
##        dms = {}

#    def do_GET(self):
#        super
    
    def do_POST(self):
        print('---Have a request')
        
        output_text = self.do_POST_inner()
        #add http header
        output_text = 'HTTP/1.0 200 OK\nContent-Type: application/json\n\n' + output_text + '\n'

        #serve result
        self.wfile.write(output_text)
        print('---Handled request:\n%s'%(output_text))

    def do_POST_inner(self):
        path = self.path[1:]
        content = self.rfile.readline()
        print('get message:', content)
        pdb.set_trace()
        
        content = jsonlib.read(content,use_float = True)
        content = content['content']
        print(content)
        
        ret = ''
        if path == 'dm':
            ret = self._dm_process(session, message)
        else:
            pass
        
        return ret

    def _dm_process(self, content):
        ret = {'results': 'slfdjs;f'}

        return jsonlib.write(ret)
    
def main():
    #uuid = 'D24CB19B8EAF11E4ACAFC1AA6AEC2530'
    port = 8080
    
    server_address=('',port)
    httpd = HTTPServer(server_address, StatefulHandler)
    print('Ready for serving...')
    httpd.serve_forever()

if (__name__ == '__main__'):
    main()
