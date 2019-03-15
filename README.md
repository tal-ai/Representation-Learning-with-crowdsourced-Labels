# Representation-Learning-with-crowdsourced-Labels
The source code for "Learning Effective Embeddings From Crowdsourced Labels: An Educational Case Study", ICDE 2019

Dependency:  
Python3   
TensorFlow  
Numpy   
Pandas 

RLL.py implements the RLL framework and its variants  
utils.py creates groups and confidence estimates based on bayesian or MLE inference  
train.py is the main function to create groups, confidene scores and trains a neuralNet for representation learning

To run:
python train.py

train, validation and test data are removed due to privacy issues  
Feel free to contact xuguowei@100tal.com should you have any questions. 
