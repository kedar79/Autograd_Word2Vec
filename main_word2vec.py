#Dependencies
from autograd import numpy as np
import autograd_word2vec
import preprocessing_and_corpus_building
from tempfile import TemporaryFile

#Path of dataset
PATH = "20news-bydate-train"
#Corpus Filename
corpus_file_name = 'corpus.txt'

#Preprocessing of Data and building raw text corpus
preprocessing_and_corpus_building.build_corpus(PATH, corpus_filename=corpus_file_name)
#Training Word2Vec model
word2vec = autograd_word2vec.word2vec_model(corpus_file_name,dims=100,window_size=5,validation_split=0.2,epochs=3000,performance=True)

vectors = TemporaryFile()
#Saving trained Word2Vec vectors
np.save(vectors,word2vec[0][0])