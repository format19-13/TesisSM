import glob
import os
import xml.etree.ElementTree as XML
import string
import itertools as IT
import io

path = '/home/vero/Dropbox/TesisVT/pan16-author-profiling-training-dataset-2016-04-25/pan16-author-profiling-training-dataset-english-2016-04-25'

pathConfig= '/home/vero/proyectos/TesisVT/twitter_timeline_extractor/twitter_timeline_extractor/configs/settings.py'
screen_names=''

for filename in glob.glob(os.path.join(path, '*.xml')):
    try:
        e = XML.parse(filename).getroot()
    except XML.ParseError as err:
        lineno, column = err.position
        line = next(IT.islice(io.BytesIO(filename), lineno))
        caret = '{:=>{}}'.format('^', column)
        err.msg = '{}\n{}\n{}'.format(err, line, caret)
        raise 

    if (len(screen_names)>0):
        screen_names=screen_names+','+ string.replace(e.attrib['url'], 'https://twitter.com/', '')
    else:
        screen_names=string.replace(e.attrib['url'], 'https://twitter.com/', '')


f = open(pathConfig,'a')

f.write('SCREEN_NAMES='+'"'+screen_names+'"'+'\n') 

f.close() 
