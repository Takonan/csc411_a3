from elm import *
import numpy as np
from utils import *

def elmclassifier():
    train_in, train_tar, val_in, val_tar = load_data()
    ELMClas = ELMClassifier()
    ELMClas.fit(train_in, train_tar)

    accuracy = ELMClas.score(val_in, val_tar)

    print 'elm unit accuracy', accuracy

def elmclass_list(size):
    train_in, train_tar, val_in, val_tar = load_data()
    train_len = train_in.shape[0]
    val_len = val_in.shape[0]
    clas_list = [ELMClassifier() for x in xrange(size)]
    predict_list = np.zeros((val_len, size), dtype=np.int64)
    agg_pred = np.zeros((val_len, 1))
    for x in range(size):
        # Partition data into non-overlapping segments
        #x_in = train_in[x*train_len/size:(x+1)*train_len/size]
        #x_tar = train_tar[x*train_len/size:(x+1)*train_len/size])
        # Fit each classifier with all data
        #x_in = train_in
        #x_tar = train_tar
        # Sample N of the total training set
        N = train_len/3
        sample = np.random.choice(train_len, N, replace=False)
        x_in = train_in[sample]
        x_tar = train_tar[sample]
        clas_list[x].fit(x_in, x_tar)
        predict_list[:, x] = (clas_list[x].predict(val_in)).astype(np.int64)

    # Use the mode as the prediction
    for n in xrange(val_len):
        #print predict_list[n, :], predict_list[n, :].dtype
        agg_pred[n] = (np.bincount(predict_list[n, :])).argmax()
    print train_in.shape, val_tar.shape 
    accuracy = np.count_nonzero(np.equal(agg_pred, val_tar))
    print 'list accuracy', accuracy*1./val_len

if __name__ == '__main__':
    elmclassifier()
    # Run 20 classifiers
    elmclass_list(20)
