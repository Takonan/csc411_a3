from utils import *
# import csutils
import theano
import yaml

import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2

#from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LabelKFold

def build_cnn():
    # Use CNN, from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    # input image dimensions
    img_rows, img_cols = 32, 32
    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 5
    nb_conv2 = 3
    # number of classes
    nb_classes = 7

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Activation('relu'))
    #model.add(Convolution2D(nb_filters, nb_conv2, nb_conv2))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(256,W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(64, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def build_mlp():
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.

    # model = Sequential()
    # model.add(Dense(64, input_dim=1024, init='uniform', activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, init='uniform', activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(7, init='uniform', activation='softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # model = Sequential()
    # model.add(Dense(128,input_dim=1024))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(7))
    # model.add(Activation('softmax'))

    # rms = RMSprop()
    # model.compile(loss='categorical_crossentropy', optimizer=rms)

    # # Here's a Deep Dumb MLP (DDMLP) # Accuacy with 3 fold labelKFold = 0.54905982906
    model = Sequential()
    model.add(Dense(512, input_shape=(1024,), init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(256, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(64, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(7, init='lecun_uniform'))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    return model

def main(model_type='CNN', model_checkpoint='model.yaml', weights_checkpoint='NNweights_', useBagging=True):
    inputs, targets, identities = load_data_with_identity(True)
    lkf = LabelKFold(identities, n_folds=10)
    if model_type == 'CNN':
        inputs = inputs.reshape(inputs.shape[0], 1, 32,32) # For CNN model
        inputs = preprocess_images(inputs)

    print "Loaded the data..."

    # Load the unlabeled data
    unlabeled_inputs = load_unlabeled_data(include_mirror=False)
    unlabeled_inputs = unlabeled_inputs.reshape(unlabeled_inputs.shape[0], 1, 32,32)
    unlabeled_inputs = preprocess_images(unlabeled_inputs)

    mega_inputs = np.append(unlabeled_inputs, inputs, axis=0)
    ZCAMatrix = zca_whitening(mega_inputs, epsilon=10e-2)
    print "Done computing ZCAMatrix on unlabeled + labeled input...."
    print "ZCAMatrix shape: ", ZCAMatrix.shape

    outfile = open('ZCAMatrix.npy','w')
    np.save(outfile,ZCAMatrix)
    outfile.close()
    print "Saved ZCAMatrix as ZCAMatrix.npy..."

    n_folds = 10
    lkf = LabelKFold(identities, n_folds)
    nn_list = []
    score_list = np.zeros(len(lkf))

    index = 0
    val_loss = 1e7
    val_acc = 0
    batch_size = 1024
    nb_classes = 7
    nb_epoch = 100
    training_stats = np.zeros((nb_epoch, 4))

    for train_index, test_index in lkf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        print X_train.shape
        #print y_train.shape

        #print "Transforming X_train, X_test with ZCA"
        #X_train = np.dot(X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]),ZCAMatrix)
        X_train = X_train.reshape(X_train.shape[0], 1, 32,32)
        #X_test = np.dot(X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]),ZCAMatrix)
        X_test = X_test.reshape(X_test.shape[0], 1, 32,32)

        # ShowMeans(X_train[2000:2004]) # Debug: Show faces after being processed

        # convert class vectors to binary class matrices
        y_train_oneOfK = np_utils.to_categorical(y_train-1, nb_classes)
        y_test_oneOfK = np_utils.to_categorical(y_test-1, nb_classes)

        # y_train = np_utils.to_categorical(y_train-1)
        # print y_train
        # y_test_oneOfK = np_utils.to_categorical(y_test-1)
        # print y_test
        # print y_test.shape

        if model_type == 'CNN':
            model = build_cnn()


        else:
            if model_type == 'MLP':
                model = build_mlp()
        # model.fit(X_train, y_train, nb_epoch=20, batch_size=100, show_accuracy=True)
        # score = model.evaluate(X_test, y_test_oneOfK, batch_size=100, show_accuracy=True)
        val_loss = 1e7
        val_acc = 0
        patience = 0
        for epoch_i in np.arange(nb_epoch):
            model.fit(X_train, y_train_oneOfK,
          batch_size=batch_size, nb_epoch=1,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, y_test_oneOfK))
            score = model.evaluate(X_test, y_test_oneOfK,
                       show_accuracy=True, verbose=0)
            print "Score:", score
            #print X_test.shape
            if (score[0] < val_loss):
                patience = 0
                model.save_weights(weights_checkpoint+"{:d}.h5".format(index), overwrite=True)
                print "Saved weights to "+weights_checkpoint+"{:d}.h5".format(index)
                val_loss = score[0]
                val_acc = score[1]
            else:
                patience += 1
            if patience > 20:
                print "Running out of patience...at {:d}".format(epoch_i)
                break
            pred = model.predict_classes(X_test)
            # print "Prediction: ", pred
            # print "y_test - 1: ", y_test-1
            print "Manual score, fold {:d}".format(index), (pred == (y_test-1)).mean()
            score_list[index] = (pred == (y_test-1)).mean()
            # Randomly choose a fold to record
            if index==7:
                train_score = model.evaluate(X_train, y_train_oneOfK,
                                show_accuracy=True, verbose=0)
                training_stats[epoch_i, :2] = train_score
                training_stats[epoch_i, 2:] = score
        outfile = open('training_stats.npy','w')
        np.save(outfile, training_stats)
        outfile.close()
        print "Saved training stats for fold"



        # Save model and weights

        yaml_string = model.to_yaml()
        with open(model_checkpoint, 'w+') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        # Only save the weights when the current index score is equal to the best so far
        #if (index > 0 and score_list[index] == score_list.max()):
        #    model.save_weights(weights_checkpoint, overwrite=True)
        #    print "Saved weights"

        nn_list.append(model)

        index += 1

    # use the NN model to classify test data
    print score_list
    print score_list.mean()
    print "Last weights validation loss {:0.4f} accuracy {:0.4f}".format(val_loss, val_acc)
    # Saving validation accuracies for fold
    outfile = open('{:d}fold_val_acc.npy'.format(n_folds), 'w')
    np.save(outfile, score_list)
    outfile.close()
    return nn_list

def test_model(model_checkpoint='model.yaml', weights_checkpoint='NNweights_8.h5', useZCA=False):
    model_stream = file(model_checkpoint, 'r')
    test_model = model_from_yaml(yaml.safe_load(model_stream))
    test_model.load_weights(weights_checkpoint)

    # Load and preprocess test set
    x_test = load_public_test()
    x_test = preprocess_images(x_test)

    if useZCA:
        ZCAMatrix = np.load('ZCAMatrix.npy')
        x_test = np.dot(x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]),ZCAMatrix)
        x_test = x_test.reshape(x_test.shape[0], 1, 32,32)
        print "Processed test input with ZCAMatrix"

    print "Finished loading test model"

    predictions = test_model.predict_classes(x_test)
    print predictions+1
    save_output_csv("test_predictions.csv", predictions+1)
    return

# This function is not done yet
def validate_model(model_checkpoint='model.yaml', weights_checkpoint='NNweights_5.h5', useZCA=True, Folds=10):
    model_stream = file(model_checkpoint, 'r')
    test_model = model_from_yaml(yaml.safe_load(model_stream))
    test_model.load_weights(weights_checkpoint)

    # Load and preprocess test set
    x_test, y_test, identities = load_data_with_identity(True)
    x_test = x_test.reshape(x_test.shape[0], 1, 32, 32)
    x_test = preprocess_images(x_test)
    lkf = LabelKFold(identities, n_folds=10)

    if useZCA:
        ZCAMatrix = np.load('ZCAMatrix.npy')
        x_test = np.dot(x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]),ZCAMatrix)
        x_test = x_test.reshape(x_test.shape[0], 1, 32,32)
        print "Processed test input with ZCAMatrix"

    print "Finished loading test model"
    
    predictions = test_model.predict_classes(x_test)
    return 

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.nan)
    #print "Using board {:d}".format(csutils.get_board())
    #main('CNN')

    #test_model(useZCA=True)
    NN_bag_predict_unlabeled()
