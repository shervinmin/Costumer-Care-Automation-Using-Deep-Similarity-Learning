#### shervin minaee, document preparation for doc2vec
#### June 20, 2016

import math, nltk, pickle, time
import numpy as np
from nltk.tokenize import sent_tokenize
import gensim, logging
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import LabeledSentence
start = time.time()

##### labeled sentence object, in order to prepare data for doc2vec
##### sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])


################################################## Question Processing
question_training_text = pickle.load( open( "question_training_text100.p", "rb" ) )
question_validation_text = pickle.load( open( "question_validation_text100.p", "rb" ) )
question_test_text = pickle.load( open( "question_test_text100.p", "rb" ) )

labeled_ques_insurance= list()
## Training data labeling
for i in range( len(question_training_text) ): 
    curr_ques= question_training_text[i]       ## an string which has i-th question (is multiple sentences)
    ff= curr_ques.split()
    question_number= i+1
    labeled_ques_insurance.append( LabeledSentence( ff, ['SENT_ques_%d' % question_number ]) )

## Validation data labeling
for i in range( len(question_validation_text) ): 
    curr_ques= question_validation_text[i]       ## an string which has i-th question (is multiple sentences)
    ff= curr_ques.split()
    question_number= len(question_training_text)+i+1
    labeled_ques_insurance.append( LabeledSentence( ff, ['SENT_ques_%d' % question_number ]) )
    
## Test data labeling
for i in range( len(question_test_text) ): 
    curr_ques= question_test_text[i]       ## an string which has i-th question (is multiple sentences)
    ff= curr_ques.split()
    question_number= len(question_training_text)+len(question_validation_text)+i+1
    labeled_ques_insurance.append( LabeledSentence( ff, ['SENT_ques_%d' % question_number ]) )
    
   
pickle.dump(labeled_ques_insurance, open( "labeled_ques_insurance100.p", "wb" ))
 

################################################## Answer Processing
answer_texts = pickle.load( open( "answer_texts.p", "rb" ) )  

labeled_ans_insurance=  list()

## All Answers Processing
for i in range( len(answer_texts) ): 
    curr_ans= answer_texts[i]       ## an string which has i-th question (is multiple sentences)
    ff= curr_ans.split()
    answer_number= i+1
    labeled_ans_insurance.append( LabeledSentence( ff, ['SENT_ans_%d' % answer_number ]) ) 
        

pickle.dump(labeled_ans_insurance , open( "labeled_ans_insurance.p", "wb" ))


########################################### end of answer processing


end = time.time()
tot_time_word= end-start
print("time of word2vec:", tot_time_word)



