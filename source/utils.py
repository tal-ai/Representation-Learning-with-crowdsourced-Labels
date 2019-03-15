import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def getQueryDoc(pos, neg, neg_doc_num=3):
    query_pos_doc = [(i, j) for i in pos['id'] for j in pos['id'] if i!=j]
    n = len(query_pos_doc)
    query_pos_doc = pd.DataFrame(query_pos_doc, columns = ['query', 'positive_doc'])
    neg_doc = np.zeros(shape=[n, neg_doc_num])
    for i in range(n):
        neg_doc[i] = neg['id'].sample(neg_doc_num)
    
    neg_doc_header = ['negative_doc{}'.format(x) for x in range(neg_doc_num)]
    neg_doc = pd.DataFrame(neg_doc, columns =neg_doc_header)
    pair = pd.concat([query_pos_doc, neg_doc], axis=1)
    pair = shuffle(pair)
    return pair

def lookupFeature(data, pair, save_path, which):
    feature_lst = data.columns[3:]
    feature = data.drop(columns=['votes','mv_fluency'])
    for m in pair.columns:
        matrix = pair[[m]].merge(right=feature, left_on=m, right_on='id', how='left')[feature_lst]
        matrix.to_csv(os.path.join(save_path, '{}_{}.csv'.format(m, which)), index=False)
        
        
def createVoteConfidence(data, pair, save_path, which, method=None, alpha=None, beta=None):
    votes = data[['id', 'votes']]
    if(method=='Bayesian'):
        confidence = inferenceBayesian(votes, alpha, beta)
    elif(method=='MLE'):
        confidence = inferenceMLE(votes)
    else:
        print('Please specify confidence inference method')
        return
    for m in pair.columns:
        weights = pair[[m]].merge(right=confidence, left_on=m, right_on='id', how='left')['confidence']
        weights.to_csv(os.path.join(save_path, 'weight_{}_{}.csv'.format(m, which)), index=False)
    
def inferenceMLE(votes):
    max_vote = max(votes['votes'].values)
    confidence_lst = []
    for i in range(votes.shape[0]):
        this_vote = votes['votes'].iloc[i]
        if(this_vote>=3):
            confidence_lst.append(float(this_vote/max_vote))
        else:
            confidence_lst.append(1-float(this_vote/max_vote))
    confidence = pd.DataFrame()
    confidence['id'] = [x for x in votes['id'].values]
    confidence['confidence'] = confidence_lst
    return confidence

def inferenceBayesian(votes, alpha, beta):
    max_vote = max(votes['votes'].values)
    confidence_lst = []
    for i in range(votes.shape[0]):
        this_vote = votes['votes'].iloc[i]
        if(this_vote>=3):
            confidence_lst.append(float((this_vote+alpha)/(max_vote+alpha+beta)))
        else:
            confidence_lst.append(float((max_vote-this_vote+alpha)/(max_vote+alpha+beta)))
    confidence = pd.DataFrame()
    confidence['id'] = [x for x in votes['id'].values]
    confidence['confidence'] = confidence_lst
    return confidence

def createInput(train, validation, save_path, method=None, alpha=None, beta=None):
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    train_pos = train.loc[train['mv_fluency']==1]
    train_neg = train.loc[train['mv_fluency']==0]
    val_pos = validation.loc[validation['mv_fluency']==1]
    val_pos = validation.loc[validation['mv_fluency']==0]
    
    tr_pair = getQueryDoc(train_pos, train_neg)
    val_pair = getQueryDoc(val_pos, val_pos)
    lookupFeature(train, tr_pair, save_path, 'train')
    lookupFeature(validation, val_pair, save_path, 'validation')
    if(method!=None):
        weight_path = os.path.join(save_path, 'weight')
        if(not os.path.exists(weight_path)):
            os.makedirs(weight_path)
        print('creating example confidence using {}'.format(method))
        createVoteConfidence(train, tr_pair, weight_path, 'train', method, alpha, beta)
        createVoteConfidence(validation, val_pair, weight_path, 'validation', method, alpha, beta)