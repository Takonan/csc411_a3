from utils import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LabelKFold

def build_cnn():
    # Use CNN, from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    # input image dimensions
    img_rows, img_cols = 32, 32
    # number of convolutional filters to use
    nb_filters = 5
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
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
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

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

def main(model_type='CNN'):
    inputs, targets, identities = load_data_with_identity(True)
    if model_type == 'CNN':
        inputs = inputs.reshape(inputs.shape[0], 1, 32,32) # For CNN model

    inputs = inputs.astype("float32")
    inputs /= 255

    print "Loaded the data..."

    lkf = LabelKFold(identities, n_folds=3)
    nn_list = []
    score_list = np.zeros(len(lkf))
    index = 0

    batch_size = 128
    nb_classes = 7
    nb_epoch = 30

    for train_index, test_index in lkf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        print X_train.shape
        print y_train.shape

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
        model.fit(X_train, y_train_oneOfK,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, y_test_oneOfK))
        score = model.evaluate(X_test, y_test_oneOfK,
                       show_accuracy=True, verbose=0)
        print "Score:", score

        pred = model.predict_classes(X_test)
        # print "Prediction: ", pred
        # print "y_test - 1: ", y_test-1
        print "Manual score", (pred == (y_test-1)).mean()

        score_list[index] = (pred == (y_test-1)).mean()

        nn_list.append(model)

        index += 1

    # use the NN model to classify test data
    print score_list
    print score_list.mean()

    return nn_list

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.nan)
    main('CNN')
