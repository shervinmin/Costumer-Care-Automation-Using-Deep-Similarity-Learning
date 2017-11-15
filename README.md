# Customer-Care-Automation-Using-Deep-Similarity-Learning

This is a Python implementation of our work on customer care automation using a deep learning framework. 
We treat this problem as a question-answering problem, where the goal is to retrieve/generate an answer for a given question from customer. 

We first learn the sentence embedding for questions and answers using Doc2Vec (an extension of word2vec), and then use a deep similarity network which gets a pair of question and answer and generates a similarity score.

This code uses Gensim package for Doc2Vec training, and Tensorflow library for Similarity Neural Network training.
