#### Shervin Minaee
#### July 12, 2016

import math, nltk, pickle, time, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



###  Reading data
question_training_all = pickle.load( open( "question_training_all100.p", "rb" ) )
question_validation_all = pickle.load( open( "question_validation_all100.p", "rb" ) )
question_test_all = pickle.load( open( "question_test_all100.p", "rb" ) )
answer_texts = pickle.load( open( "answer_texts.p", "rb" ) )

ques_train_dec2vec = pickle.load( open( "ques_train_dec2vec100.p", "rb" ) )
ques_valid_dec2vec = pickle.load( open( "ques_valid_dec2vec100.p", "rb" ) )
ques_test_dec2vec = pickle.load( open( "ques_test_dec2vec100.p", "rb" ) )  
ans_dec2vec = pickle.load( open( "ans_dec2vec.p", "rb" ) )  


n_input= 200   ### feature dimension for question and asnwer representation
num_train_sim= 21325
num_valid_sim= 3354
num_test_sim = 3308
pool_size= 100

num_train_pair= len(question_training_all)*30  #### we also take 15 positive samples from answers by over-sampling to overcome imbalanced training problem 
train_ques_feat= np.random.randn( num_train_pair, n_input)
train_ans_feat = np.random.randn( num_train_pair, n_input)
train_label    = np.random.randn( num_train_pair, 2)

num_valid_pair= num_valid_sim+ num_valid_nonsim
num_test_pair=  num_test_sim+  num_test_nonsim
num_valid_nonsim= len(question_validation_all)*pool_size
num_test_nonsim = len(question_test_all)*pool_size
valid_ques_feat= np.random.randn( num_valid_pair, n_input)
valid_ans_feat = np.random.randn( num_valid_pair, n_input)
valid_label    = np.random.randn( num_valid_pair, 2)
test_ques_feat= np.random.randn( num_test_pair, n_input)
test_ans_feat = np.random.randn( num_test_pair, n_input)
test_label    = np.random.randn( num_test_pair, 2)


############################################## training pairs preparation
#### positive pairs
curr_ind= 0  ### shows the index of current sample in paired data

for i in range(len(question_training_all)):
    print("\n")
    num_positive_answer= 0
    curr_GT= question_training_all[i]['ground_truth']
    for j in range(len(curr_GT)):
        curr_matched_ans_ind= int( curr_GT[j] )
        print("%d-th question in training, matched answer" % (i+1), ", num in GT=", len(curr_GT))
        #print("matched answer=", curr_matched_ans_ind)
        curr_ques_feat = ques_train_dec2vec[i]
        curr_ques_feat = (curr_ques_feat - curr_ques_feat.mean(axis=0)) / curr_ques_feat.std(axis=0)
        curr_ans_feat = ans_dec2vec[curr_matched_ans_ind-1]
        curr_ans_feat = (curr_ans_feat - curr_ans_feat.mean(axis=0)) / curr_ans_feat.std(axis=0)
        train_ques_feat[ i*15+num_positive_answer, :]= np.reshape( curr_ques_feat, (1, curr_ques_feat.size))    
        train_ans_feat[  i*15+num_positive_answer, :]= np.reshape( curr_ans_feat, (1, curr_ans_feat.size))
        train_label[ i*15+num_positive_answer, :]= [1,0]  ### for matched question and answer
        curr_ind= curr_ind+1
        num_positive_answer= num_positive_answer+1
        #print("num_positive= ", num_positive_answer, ", num_all=", curr_ind-1)
    if ( num_positive_answer< 15 ):  ## meaning we need to over-sample answers in ground-truth
        num_ans_wanted= 15- num_positive_answer
        num_ans_have= num_positive_answer
        over_sampled_indices= np.random.choice( num_ans_have, num_ans_wanted, replace=True)
        #print("over_sampled_ind=", over_sampled_indices, ", number have=", num_ans_have)
        for over_ind in range(len(over_sampled_indices)):
            train_ques_feat[ i*15+num_positive_answer, :]= train_ques_feat[ i*15+over_sampled_indices[over_ind],:]
            train_ans_feat[ i*15+num_positive_answer, :]= train_ans_feat[ i*15+over_sampled_indices[over_ind],:]
            train_label[ i*15+num_positive_answer, :]= train_label[ i*15+over_sampled_indices[over_ind],:]
            #print("new index is", i*15+num_positive_answer, ", used index:", i*15+over_sampled_indices[over_ind])
            num_positive_answer= num_positive_answer+1
            curr_ind= curr_ind+1
            
tot_num_pos_ans= curr_ind
            

        
#### negative pairs

for i in range(len(question_training_all)):
    num_negative_answer= 0
    curr_pool= question_training_all[i]['answer_pool']
    curr_GT_list= question_training_all[i]['ground_truth']
    neg_ans_ind= list( set(curr_pool) - set(curr_GT_list))
    neg_sampled_indices= random.sample( neg_ans_ind, 15)
    #print("\n", i+1,"-th negative answer,", "negative indices= ", neg_sampled_indices)
    #print("current_index=", tot_num_pos_ans+ i*15+ num_negative_answer)
    for j in range( len(neg_sampled_indices)):
        curr_nonmatched_ans_ind= int( neg_sampled_indices[j] )
        print("%d-th question in training, non-matched answer index= %d" % (i+1, int(neg_sampled_indices[j])) )
        curr_ques_feat = ques_train_dec2vec[i]
        curr_ques_feat = (curr_ques_feat - curr_ques_feat.mean(axis=0)) / curr_ques_feat.std(axis=0)
        curr_ans_feat = ans_dec2vec[curr_nonmatched_ans_ind-1]
        curr_ans_feat = (curr_ans_feat - curr_ans_feat.mean(axis=0)) / curr_ans_feat.std(axis=0)
        train_ques_feat[ tot_num_pos_ans+ i*15+ num_negative_answer,:]= np.reshape( curr_ques_feat, (1, curr_ques_feat.size))    
        train_ans_feat[  tot_num_pos_ans+ i*15+ num_negative_answer,:]= np.reshape( curr_ans_feat, (1, curr_ans_feat.size))
        train_label[ tot_num_pos_ans+ i*15+ num_negative_answer,:]= [0,1]  ### for non-matched question and answer
        num_negative_answer= num_negative_answer+1
        curr_ind= curr_ind+1
        


pickle.dump(train_ques_feat, open( "train_ques_siames_pool100.p", "wb" ), protocol=2)
pickle.dump(train_ans_feat, open( "train_ans_siames_pool100.p", "wb" ), protocol=2)
pickle.dump(train_label, open( "train_label_siames_pool100.p", "wb" ), protocol=2)




############################################## validation pairs preparation
#### positive pairs
curr_ind= 0  ### shows the index of current sample in paired data
for i in range(len(question_validation_all)):
    curr_GT= question_validation_all[i]['ground_truth']
    for j in range(len(curr_GT)):
        curr_matched_ans_ind= int( curr_GT[j] )
        print("%d-th question in validation, matched answer" % (i+1) )
        #print("matched answer=", curr_matched_ans_ind)
        curr_ques_feat = ques_valid_dec2vec[i]
        curr_ques_feat = (curr_ques_feat - curr_ques_feat.mean(axis=0)) / curr_ques_feat.std(axis=0)
        curr_ans_feat = ans_dec2vec[curr_matched_ans_ind-1]
        curr_ans_feat = (curr_ans_feat - curr_ans_feat.mean(axis=0)) / curr_ans_feat.std(axis=0)
        valid_ques_feat[curr_ind,:]= np.reshape( curr_ques_feat, (1, curr_ques_feat.size))    
        valid_ans_feat[ curr_ind,:]= np.reshape( curr_ans_feat, (1, curr_ans_feat.size))
        valid_label[curr_ind,:]= [1,0]  ### for matched question and answer
        curr_ind= curr_ind+1
        
#### negative pairs
for i in range(len(question_validation_all)):
    curr_pool= question_validation_all[i]['answer_pool']
    for j in range(10):
        curr_nonmatched_ans_ind= int( curr_pool[j] )
        print("%d-th question in validation, non-matched answer" % (i+1) )
        curr_ques_feat = ques_valid_dec2vec[i]
        curr_ques_feat = (curr_ques_feat - curr_ques_feat.mean(axis=0)) / curr_ques_feat.std(axis=0)
        curr_ans_feat = ans_dec2vec[curr_nonmatched_ans_ind-1]
        curr_ans_feat = (curr_ans_feat - curr_ans_feat.mean(axis=0)) / curr_ans_feat.std(axis=0)
        valid_ques_feat[curr_ind,:]= np.reshape( curr_ques_feat, (1, curr_ques_feat.size))    
        valid_ans_feat[ curr_ind,:]= np.reshape( curr_ans_feat, (1, curr_ans_feat.size))
        valid_label[curr_ind,:]= [0,1]  ### for non-matched question and answer
        curr_ind= curr_ind+1

pickle.dump(valid_ques_feat, open( "valid_ques_feat_10pool.p", "wb" ))
pickle.dump(valid_ans_feat, open( "valid_ans_feat_10pool.p", "wb" ))
pickle.dump(valid_label, open( "valid_label_10pool.p", "wb" ))




############################################## Test pairs preparation
#### positive pairs
curr_ind= 0  ### shows the index of current sample in paired data
for i in range(len(question_test_all)):
    curr_GT= question_test_all[i]['ground_truth']
    for j in range(len(curr_GT)):
        curr_matched_ans_ind= int( curr_GT[j] )
        print("%d-th question in test set, matched answer" % (i+1) )
        #print("matched answer=", curr_matched_ans_ind)
        curr_ques_feat = ques_test_dec2vec[i]
        curr_ques_feat = (curr_ques_feat - curr_ques_feat.mean(axis=0)) / curr_ques_feat.std(axis=0)
        curr_ans_feat = ans_dec2vec[curr_matched_ans_ind-1]
        curr_ans_feat = (curr_ans_feat - curr_ans_feat.mean(axis=0)) / curr_ans_feat.std(axis=0)
        test_ques_feat[curr_ind,:]= np.reshape( curr_ques_feat, (1, curr_ques_feat.size))    
        test_ans_feat[ curr_ind,:]= np.reshape( curr_ans_feat, (1, curr_ans_feat.size))
        test_label[curr_ind,:]= [1,0]  ### for matched question and answer
        curr_ind= curr_ind+1
        
#### negative pairs
for i in range(len(question_test_all)):
    curr_pool= question_test_all[i]['answer_pool']
    for j in range(10):
        curr_nonmatched_ans_ind= int( curr_pool[j] )
        print("%d-th question in test set, non-matched answer" % (i+1) )
        curr_ques_feat = ques_test_dec2vec[i]
        curr_ques_feat = (curr_ques_feat - curr_ques_feat.mean(axis=0)) / curr_ques_feat.std(axis=0)
        curr_ans_feat = ans_dec2vec[curr_nonmatched_ans_ind-1]
        curr_ans_feat = (curr_ans_feat - curr_ans_feat.mean(axis=0)) / curr_ans_feat.std(axis=0)
        test_ques_feat[curr_ind,:]= np.reshape( curr_ques_feat, (1, curr_ques_feat.size))    
        test_ans_feat[ curr_ind,:]= np.reshape( curr_ans_feat, (1, curr_ans_feat.size))
        test_label[curr_ind,:]= [0,1]  ### for non-matched question and answer
        curr_ind= curr_ind+1

pickle.dump(test_ques_feat, open( "test_ques_feat_10pool.p", "wb" ))
pickle.dump(test_ans_feat, open( "test_ans_feat_10pool.p", "wb" ))
pickle.dump(test_label, open( "test_label_10pool.p", "wb" ))





###################### finding number of paired question and answers, used above
max_gt_train= 0  ## 15
max_gt_valid= 0  ## 11
max_gt_test= 0   ## 10
num_train_sim= 0
for i in range(len(question_training_all)):
    curr_gt= question_training_all[i]['ground_truth']
    num_train_sim= num_train_sim+ len(curr_gt)
    if len(curr_gt)>max_gt_train:
        max_gt_train= len(curr_gt)
    
num_valid_sim= 0
for i in range(len(question_validation_all)):
    curr_gt= question_validation_all[i]['ground_truth']
    num_valid_sim= num_valid_sim+ len(curr_gt)
    if len(curr_gt)>max_gt_valid:
        max_gt_valid= len(curr_gt)

num_test_sim= 0
for i in range(len(question_test_all)):
    curr_gt= question_test_all[i]['ground_truth']
    num_test_sim= num_test_sim+ len(curr_gt)
    if len(curr_gt)>max_gt_test:
        max_gt_test= len(curr_gt)





