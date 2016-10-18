import glob
import os
import xml.etree.ElementTree as XML
import string
import itertools as IT
import io

path = '/home/vtortorella/Dropbox/TesisVT/pan16-author-profiling-training-dataset-2016-04-25/pan16-author-profiling-training-dataset-english-2016-04-25'
pathConfig= '/home/vtortorella/proyectos/tesisVT/twitter_timeline_extractor/twitter_timeline_extractor/configs/settings.py'
screen_names1=''
screen_names2=''
screen_names3=''
screen_names4=''
screen_names5=''
cont=1

for filename in glob.glob(os.path.join(path, '*.xml')):
    try:
        e = XML.parse(filename).getroot()
        cont=cont+1
    except XML.ParseError as err:
        lineno, column = err.position
        line = next(IT.islice(io.BytesIO(filename), lineno))
        caret = '{:=>{}}'.format('^', column)
        err.msg = '{}\n{}\n{}'.format(err, line, caret)
        raise 
    if (cont<100):
        if (len(screen_names1)>0):
            screen_names1=screen_names1+','+ string.replace(e.attrib['url'], 'https://twitter.com/', '')
        else:
            screen_names1=string.replace(e.attrib['url'], 'https://twitter.com/', '')
    elif (cont<200):
        if (len(screen_names2)>0):
            screen_names2=screen_names2+','+ string.replace(e.attrib['url'], 'https://twitter.com/', '')
        else:
            screen_names2=string.replace(e.attrib['url'], 'https://twitter.com/', '')
    elif (cont<300):
        if (len(screen_names3)>0):
            screen_names3=screen_names3+','+ string.replace(e.attrib['url'], 'https://twitter.com/', '')
        else:
            screen_names3=string.replace(e.attrib['url'], 'https://twitter.com/', '')
    elif (cont<400):
        if (len(screen_names4)>0):
            screen_names4=screen_names4+','+ string.replace(e.attrib['url'], 'https://twitter.com/', '')
        else:
            screen_names4=string.replace(e.attrib['url'], 'https://twitter.com/', '')
    else:
        if (len(screen_names5)>0):
            screen_names5=screen_names5+','+ string.replace(e.attrib['url'], 'https://twitter.com/', '')
        else:
            screen_names5=string.replace(e.attrib['url'], 'https://twitter.com/', '')

f = open(pathConfig,'a')

f.write('SCREEN_NAMES1='+'"'+screen_names1+'"' + '\n') 
f.write('SCREEN_NAMES2='+'"'+screen_names2+'"' + '\n') 
f.write('SCREEN_NAMES3='+'"'+screen_names3+'"' + '\n') 
f.write('SCREEN_NAMES4='+'"'+screen_names4+'"' + '\n') 
f.write('SCREEN_NAMES5='+'"'+screen_names5+'"' + '\n') 
f.close() 
