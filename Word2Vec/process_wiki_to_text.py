#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: process_wiki_to_text.py 
@desc: 将xml的维基数据转换为text格式
@time: 2017/09/19 
"""

from __future__ import  print_function

import logging
import os.path
import six
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki_to_text.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp,outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if six.PY3:
            output.write(b' '.join(text).decode('utf-8') + '\n')
        else:
            output.write(space.join(text) + '\n')
        i = i + 1
        if (i % 100000 == 0):
            logger.info("Saved" + str(i) + "articles")

    output.close()
    logger.info("Finished Saved" + str(i) + "articles")
