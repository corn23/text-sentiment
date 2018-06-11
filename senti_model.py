import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from gensim import models


def from_txt_to_embedding(txt, emb_dict,emb_dim=50):
    txt_embedding = []
    for line in txt:
        line_result = []
        for tokens in line.strip():
            if tokens in emb_dict.vocab:
                line_result.append(emb_dict[tokens])
            else:
                line_result.append(np.zeros(emb_dim))
        txt_embedding.append(line_result)
    return txt_embedding


# # load dict keys
# f = open('vocab_cut.txt','r')
# lines = f.readlines()
# dict_keys = [line.strip() for line in lines]
# f.close()
#
# # load dict values
# emb = np.load('embeddings.npz')
# emb_arr = emb['arr_0']
# emb_dict = dict(zip(dict_keys,emb_arr))

emb_dict = models.KeyedVectors.load_word2vec_format('glove.6B.50d.txt')
print("embed_dict loading complete")

# load txt
f_pos = open('train_pos.txt', 'r', encoding='utf8')
f_neg = open('train_neg.txt', 'r', encoding='utf8')
txt_pos = f_pos.readlines()
txt_neg = f_neg.readlines()
txt = txt_pos + txt_neg
f_pos.close()
f_neg.close()
label = np.concatenate([np.ones(len(txt_pos)),np.zeros(len(txt_neg))])
txt_embedding = from_txt_to_embedding(txt,emb_dict,50)
# extract the mean features and split
ave_txt_embedding = np.array([np.mean(iemb,axis=0) for iemb in txt_embedding])
X_train, X_test, y_train, y_test = train_test_split(ave_txt_embedding,label)
print('text transformed')

# build logistic regression
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
train_err =lgr.score(X_train,y_train)
test_err = lgr.score(X_test,y_test)
print("the training error:{} test error:{}".format(train_err,test_err))


# visualize the wording embedding
