from RLL import *
from utils import *


train = pd.read_csv('../raw_data/train.csv')
validation = pd.read_csv('../raw_data/validation.csv')

# This is where group instances and example confidence are saved
feature_path = '../data/input/'
weight_path = '../data/input/weight'
alpha, beta = 3, 2 # Bayesian prior

#create neural net input and example confidence using bayesian inference, e.e. RLL-Bayesian
#createInput(train, validation, feature_path, 'Bayesian', alpha, beta)

#create neural net input and example confidence using MLE inference, e.e. RLL-MLE
createInput(train, validation, feature_path, 'MLE')

#In each path above you should have the following files created, which will be fed to the neural network
trainFileList = ['query_train.csv', 'positive_doc_train.csv', 'negative_doc0_train.csv', 'negative_doc1_train.csv', \
            'negative_doc2_train.csv', 'weight_positive_doc_train.csv', 'weight_negative_doc0_train.csv',\
            'weight_negative_doc1_train.csv', 'weight_negative_doc2_train.csv']

testFileList = ['query_validation.csv', 'positive_doc_validation.csv', 'negative_doc0_validation.csv', 'negative_doc1_validation.csv', \
            'negative_doc2_validation.csv', 'weight_positive_doc_validation.csv', 'weight_negative_doc0_validation.csv',\
            'weight_negative_doc1_validation.csv', 'weight_negative_doc2_validation.csv']

# load data
train_data = read_input_feature(trainFileList, feature_path, weight_path)
test_data = read_input_feature(testFileList, feature_path, weight_path)


# do grid search and train
l1_n_lst = [64, 32]
l2_n_lst = [32, 16]
dropout_rate_lst = [0.5, 0.6, 0.7]
reg_scale_lst = [0.7, 0.9, 1.0, 1.5]
lr_rate_lst = [0.05, 0.01, 0.1]
bs_lst = [512, 1024]
gamma_lst = [1.0, 10.0]
max_iter = 150

for l1_n, l2_n, dropout_rate, reg_scale, lr_rate, bs, gamma in [(l1_n, l2_n, dropout_rate, \
    reg_scale, lr_rate, bs, gamma) for l1_n in l1_n_lst for l2_n in l2_n_lst for dropout_rate in dropout_rate_lst  \
    for reg_scale in reg_scale_lst for lr_rate in lr_rate_lst for bs in bs_lst for gamma in gamma_lst]:

    print('dropout{}, regularization{} learning rate{} batch size {}'.format(dropout_rate, reg_scale, lr_rate, bs))
    model_name = '_'.join([str(l1_n), str(l2_n), str(dropout_rate), str(reg_scale), str(lr_rate), str(bs), str(gamma)])
    save_path = '../model/weighted/'
    log_path = "../log/weighted/"
    save_path = os.path.join(save_path, model_name)
    log_path = os.path.join(log_path, model_name)

    if(not os.path.exists(log_path)):
        os.makedirs(log_path)

    if(not os.path.exists(save_path)):
        os.makedirs(save_path)

    logging.basicConfig(
        filename=os.path.join(log_path,'{}.log'.format(model_name)),
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
        )

    RLL(train_data, test_data, bs, lr_rate, l1_n, l2_n, max_iter, reg_scale, dropout_rate, gamma, True, save_path, model_name+'.ckpt')
