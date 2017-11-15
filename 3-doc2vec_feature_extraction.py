"""
Shervin Minaee, July 27, 2016
Feature Extraction
"""

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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

ques_train = pickle.load( open( "question_training_text100.p", "rb" ) )
ques_valid = pickle.load( open( "question_validation_text100.p", "rb" ) )
ques_test  = pickle.load( open( "question_test_text100.p", "rb" ) )

#answer_all  = pickle.load( open( "answer_texts.p", "rb" ) )


############################################################  question feature extraction
model_ques = Doc2Vec.load('doc2vec_insurance_ques100.doc2vec')

ques_train_dec2vec= list()
ques_valid_dec2vec= list()
ques_test_dec2vec=  list()

for i in range(len(ques_train)):
    print("%d-th training feature" % (i+1))
    curr_ques_str  = ques_train[i]
    curr_ques_list = sent_tokenize(curr_ques_str)
    curr_ques_feat = model_ques.infer_vector(curr_ques_list)
    ques_train_dec2vec.append(curr_ques_feat)

for i in range(len(ques_valid)):
    print("%d-th validation feature" % (i+1))
    curr_ques_str  = ques_valid[i]
    curr_ques_list = sent_tokenize(curr_ques_str)
    curr_ques_feat = model_ques.infer_vector(curr_ques_list)
    ques_valid_dec2vec.append(curr_ques_feat)    
    
for i in range(len(ques_test)):
    print("%d-th testing feature" % (i+1))
    curr_ques_str  = ques_test[i]
    curr_ques_list = sent_tokenize(curr_ques_str)
    curr_ques_feat = model_ques.infer_vector(curr_ques_list)
    ques_test_dec2vec.append(curr_ques_feat) 

#print(model_ques.vocab.keys())
#print( model_ques.docvecs.most_similar([new_doc_vec]) )

pickle.dump(ques_train_dec2vec , open( "ques_train_dec2vec100.p", "wb" ), protocol=2)
pickle.dump(ques_valid_dec2vec , open( "ques_valid_dec2vec100.p", "wb" ), protocol=2)
pickle.dump(ques_test_dec2vec , open( "ques_test_dec2vec100.p", "wb" ), protocol=2)

############################################################  answer feature extraction

model_ans = Doc2Vec.load('doc2vec_insurance_ans.doc2vec')

ans_dec2vec= list()

for i in range(len(answer_all)):
    print("%d-th answer feature" % (i+1))
    curr_ans_str  = answer_all[i]
    curr_ans_list = sent_tokenize(curr_ans_str)
    curr_ans_feat = model_ans.infer_vector(curr_ans_list)
    ans_dec2vec.append(curr_ans_feat)

pickle.dump(ans_dec2vec , open( "ans_dec2vec.p", "wb" ))

##########################################################  end of feature extraction



end = time.time()
tot_time_doc= end-start
print("time of doc2vec:", tot_time_doc)
