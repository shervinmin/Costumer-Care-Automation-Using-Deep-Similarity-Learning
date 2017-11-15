#### word2vec from gensim 
#### shervin minaee

import os, sys, math, nltk, json, pickle, pprint, time
import numpy as np
import gensim, logging
from nltk.tokenize import sent_tokenize
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import LabeledSentence
start = time.time()


##################### Reading the data
##### sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

labeled_ques_insurance = pickle.load( open( "labeled_ques_insurance100.p", "rb" ) )
#labeled_ans_insurance  = pickle.load( open( "labeled_ans_insurance.p", "rb" ) )
#labeled_all_insurance= labeled_ques_insurance+labeled_ans_insurance


######################   Training the model for both question and answer together
## model parameters here: https://radimrehurek.com/gensim/models/word2vec.html
## model = Doc2Vec( labeled_sentences, size= 300, window=9, min_count= 1, workers=5, iter= 20 )

'''
model = Doc2Vec( size=200, window= 5, min_count=2, workers= 5, alpha=0.025, min_alpha=0.025,  batch_words= 2000) # use fixed learning rate
num_epochs= 10
model.build_vocab(labeled_all_insurance)
for epoch in range(num_epochs):
    print("%d epoch of training" % (epoch+1) )
    model.train(labeled_all_insurance)
    model.alpha -= 0.001 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay
    model.train(labeled_all_insurance)
    
new_sent= "ice is cold"
new_sent= 'Please wait while I transfer this chat to the tech department.'
new_sent_token = sent_tokenize(new_sent)
#print(model.vocab.keys())
new_doc_vec = model.infer_vector(new_sent_token)
print( model.docvecs.most_similar([new_doc_vec]) )


model.save('doc2vec_insurance_all.doc2vec')
#model_loaded = Doc2Vec.load('doc2vec_20days.doc2vec')
'''


########################## Training the model for questions

model = Doc2Vec( size=200, window= 5, min_count=2, workers= 5, alpha=0.025, min_alpha=0.025,  batch_words= 2000) # use fixed learning rate
num_epochs= 10
model.build_vocab(labeled_ques_insurance)
for epoch in range(num_epochs):
    print("%d epoch of training" % (epoch+1) )
    model.train(labeled_ques_insurance)
    model.alpha -= 0.001 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay
    model.train(labeled_ques_insurance)
    
new_sent= "ice is cold"
new_sent= 'Please wait while I transfer this chat to the tech department.'
new_sent_token = sent_tokenize(new_sent)
#print(model.vocab.keys())
new_doc_vec = model.infer_vector(new_sent_token)
print( model.docvecs.most_similar([new_doc_vec]) )

model.save('doc2vec_insurance_ques100.doc2vec')



########################## Training the model for answers

model = Doc2Vec( size=200, window= 5, min_count=2, workers= 5, alpha=0.025, min_alpha=0.025,  batch_words= 2000) # use fixed learning rate
num_epochs= 10
model.build_vocab(labeled_ans_insurance)
for epoch in range(num_epochs):
    print("%d epoch of training" % (epoch+1) )
    model.train(labeled_ans_insurance)
    model.alpha -= 0.001 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay
    model.train(labeled_ans_insurance)
    
new_sent= "ice is cold"
new_sent= 'Please wait while I transfer this chat to the tech department.'
new_sent_token = sent_tokenize(new_sent)
#print(model.vocab.keys())
new_doc_vec = model.infer_vector(new_sent_token)
print( model.docvecs.most_similar([new_doc_vec]) )

model.save('doc2vec_insurance_ans.doc2vec')

##################################################




end = time.time()
tot_time_word= end-start
print("time of word2vec:", tot_time_word)



