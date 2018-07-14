import os
import re

def strip_bad_line(line):
    '''
    Params : line - sentences from the file

    This function stripes out all the lines containing extra details such as header information, footer information and
    other unwanted information such as line stating( mailto: , Fax: , Telephone: etc)

    '''
    bad_line_contains=[':','>','|']
    flag =1
    #checks the line containing symbols such as ':','>','|'
    for char in line:
        if char in bad_line_contains:
            flag=0
    if(flag):
        return line

def build_corpus(path, corpus_filename):
    '''
    Params
    Path: Path of the directory containing all files
    corpus_filename : Name for corpus_filename

    Performs preprocessing and builts corpus of raw text on which word2vec model was going to be trained
    '''
    content_raw = []
    for file in os.listdir(path):
        file = os.path.join(path, file)
        f = open(file, 'r', encoding="Latin-1")
        content = f.read()
        #list of sentences present the file
        content = content.splitlines()
        #list to contain important raw text of the file.
        good_content = []
        for line in content:
            #removing unwanted lines
            good_line = strip_bad_line(line)
            if (good_line):
                good_line = re.sub("[^A-Z a-z.]", '', good_line)
                good_content.append(good_line.lstrip())

        for i in good_content:
            j = i.split()
            #feeding the sentence which has more than 10 words
            if (len(j) > 10):
                content_raw.append([i])
    #creating a file to store raw text or creating a file for text corpus
    corpus_file = open(corpus_filename, 'w')
    for i in content_raw:
        corpus_file.write(i[0] + " ")
    corpus_file.close()
